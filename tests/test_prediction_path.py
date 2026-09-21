"""The full prediction path, with estimators in place.

models/ is empty and stays that way until the NHANES artefacts land
(CLAUDE.md Finding 5), so these install stand-in estimators implementing the
threshold rules the quarantined models encoded -- reproduced on 7,196 of
7,196 rows, CLAUDE.md Finding 1. What is under test is the spine: request to
canonicalisation to feature vector to estimator to response to stored row to
log record. The stand-ins make that path runnable; they are not the subject.
"""

import json

import pytest

from tests.test_units import RULES


class Encoder:
    """Stand-in for the pickled LabelEncoder."""

    classes_ = ("High", "Low", "Medium")

    @staticmethod
    def inverse_transform(values):
        return list(values)


@pytest.fixture
def with_models(flask_app):
    """Install the stand-in estimators for the duration of one test."""
    import app as app_module

    app_module.models.update(RULES)
    app_module.encoders.update({name: Encoder() for name in RULES})
    app_module.model_versions.update({name: "sha256:testdouble" for name in RULES})
    yield
    app_module.models.clear()
    app_module.encoders.clear()
    app_module.model_versions.clear()


@pytest.fixture
def logged_in(client, csrf_token):
    """A registered, logged-in client, so predictions are persisted."""
    client.post(
        "/api/auth/register",
        json={
            "username": "units_tester",
            "email": "units_tester@example.com",
            "password": "correct-horse-battery",
            "full_name": "Units Tester",
        },
        headers={"X-CSRFToken": csrf_token},
    )
    client.post(
        "/api/auth/login",
        json={"username": "units_tester", "password": "correct-horse-battery"},
        headers={"X-CSRFToken": csrf_token},
    )
    return client


# test_2.jpg, the mixed-unit page.
TEST_2 = {
    "rbc": {"value": 4.40, "unit": "millions/uL"},
    "wbc": {"value": 5800, "unit": "cells/uL"},
    "platelets": {"value": 85000, "unit": "cells/uL"},
    "anc": {"value": 3500, "unit": "cells/uL"},
    "neutrophils_percent": {"value": 60, "unit": "%"},
}


def predict(client, csrf_token, body):
    return client.post(
        "/api/predict", json=body, headers={"X-CSRFToken": csrf_token}
    )


def test_a_full_prediction_succeeds_and_echoes_units(with_models, client, csrf_token):
    response = predict(client, csrf_token, TEST_2)

    assert response.status_code == 200, response.data
    body = response.get_json()
    assert body["input_values"]["platelets"]["canonical"] == {
        "value": 85000.0, "unit": "cells/uL",
    }
    assert body["input_values"]["platelets"]["as_reported"] == {
        "value": 85000.0, "unit": "cells/uL",
    }


def test_the_response_states_which_units_each_model_was_fed(
    with_models, client, csrf_token
):
    body = predict(client, csrf_token, TEST_2).get_json()

    assert body["predictions"]["thrombocytopenia"]["feature_units"] == [
        {"analyte": "rbc", "unit": "million/uL"},
        {"analyte": "wbc", "unit": "thousand/uL"},
        {"analyte": "platelets", "unit": "thousand/uL"},
    ]


@pytest.mark.parametrize(
    "platelets,expected",
    [
        ({"value": 900, "unit": "cells/uL"}, "High"),
        ({"value": 900, "unit": "thousand/uL"}, "Low"),
    ],
)
def test_the_two_platelet_readings_differ_over_http(
    with_models, client, csrf_token, platelets, expected
):
    """The regression, end to end, through the real request path."""
    body = predict(client, csrf_token, dict(TEST_2, platelets=platelets)).get_json()

    assert body["predictions"]["thrombocytopenia"]["risk"] == expected


def test_combined_severity_does_not_default_a_missing_model_to_low(
    with_models, client, csrf_token
):
    """Neutropenia cannot be predicted with no ANC and no neutrophil %.

    The code this replaces scored that absence as Low, pulling the combined
    severity down. It is now reported as not predicted.
    """
    body = {k: v for k, v in TEST_2.items() if k not in ("anc", "neutrophils_percent")}
    result = predict(client, csrf_token, body).get_json()

    assert result["predictions"]["neutropenia"]["risk"] is None
    assert result["combined"]["not_predicted"] == ["neutropenia"]
    assert "neutropenia" not in result["combined"]["basis"]


def test_flags_accompany_the_prediction(with_models, client, csrf_token):
    body = predict(client, csrf_token, TEST_2).get_json()
    conditions = {f["condition"]: f for f in body["flags"]}

    assert conditions["thrombocytopenia"]["grade"] == 1
    assert conditions["thrombocytopenia"]["compared_as"] == {
        "value": 85.0, "unit": "thousand/uL",
    }


# --- storage ----------------------------------------------------------------

def test_a_stored_prediction_keeps_canonical_values_and_the_original(
    with_models, flask_app, logged_in, csrf_token
):
    response = predict(logged_in, csrf_token, TEST_2)
    prediction_id = response.get_json()["prediction_id"]
    assert prediction_id is not None, response.data

    from database import Prediction, db

    with flask_app.app_context():
        row = db.session.get(Prediction, prediction_id)

        assert row.input_platelets_cells_per_ul == 85000.0
        assert row.input_wbc_cells_per_ul == 5800.0
        assert row.input_rbc_million_per_ul == 4.40
        assert row.input_reported_quantities["platelets"] == {
            "value": 85000.0, "unit": "cells/uL", "unit_source": "operator",
        }
        assert row.model_feature_units["thrombocytopenia"][2] == {
            "analyte": "platelets", "unit": "thousand/uL",
        }
        assert row.request_id == response.get_json()["request_id"]


def test_reading_a_stored_prediction_back_returns_units(
    with_models, logged_in, csrf_token
):
    prediction_id = predict(logged_in, csrf_token, TEST_2).get_json()["prediction_id"]

    body = logged_in.get(f"/api/predictions/{prediction_id}").get_json()
    stored = body["prediction"]["input_values"]["platelets"]

    assert stored["canonical"] == {"value": 85000.0, "unit": "cells/uL"}


# --- the structured log -----------------------------------------------------

def test_a_prediction_is_logged_with_inputs_units_and_model_versions(
    with_models, client, csrf_token, caplog
):
    import prediction_log

    with caplog.at_level("INFO", logger=prediction_log.LOGGER_NAME):
        predict(client, csrf_token, TEST_2)

    records = [json.loads(r.message) for r in caplog.records]
    entry = next(r for r in records if r.get("outcome") == "predicted")

    assert entry["inputs"]["platelets"]["canonical"]["unit"] == "cells/uL"
    assert entry["model_inputs"]["thrombocytopenia"][2]["fed_to_model"] == {
        "value": 85.0, "unit": "thousand/uL",
    }
    assert entry["model_versions"]["thrombocytopenia"] == "sha256:testdouble"
    assert entry["outputs"]["thrombocytopenia"]["risk"] == "Medium"
    assert entry["flags"]
    assert entry["timestamp"]


def test_the_log_record_is_json_and_ascii(with_models, client, csrf_token, caplog):
    import prediction_log

    with caplog.at_level("INFO", logger=prediction_log.LOGGER_NAME):
        predict(client, csrf_token, TEST_2)

    for record in caplog.records:
        json.loads(record.message)
        record.message.encode("ascii")


# --- the contract guard -----------------------------------------------------

def test_an_estimator_of_the_wrong_arity_is_refused(flask_app):
    from model_contract import ContractError, check_estimator

    class WrongArity:
        n_features_in_ = 7

    with pytest.raises(ContractError, match="takes 7 features"):
        check_estimator("thrombocytopenia", WrongArity())


# --- the PDF a clinician reads ---------------------------------------------

def test_the_pdf_prints_the_unit_the_value_is_actually_in(
    with_models, flask_app, logged_in, csrf_token
):
    """The table used to hardcode "thousand/uL" beside whatever float it had.

    Fed test_2.jpg's platelet count that produced "85000.00 thousand/uL" on
    the page a clinician reads -- eleven times the upper limit of normal,
    printed as though measured.
    """
    pytest.importorskip("reportlab")

    prediction_id = predict(logged_in, csrf_token, TEST_2).get_json()["prediction_id"]

    from pdf_text import extract_text   # local helper, defined below
    from export import generate_pdf_report
    from database import User, db

    with flask_app.app_context():
        user = User.query.filter_by(username="units_tester").first()
        buffer = generate_pdf_report(prediction_id, user)

    text = extract_text(buffer.getvalue())

    assert "85,000" in text
    assert "85000.00 thousand" not in text
    # The as-reported column repeats what the operator entered.
    assert "cells/µL" in text or "cells/uL" in text
