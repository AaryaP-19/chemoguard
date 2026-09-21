"""The HTTP boundary: a bare number is refused before anything else happens.

These run with models/ empty, which is the expected state (CLAUDE.md Finding
5). That is useful rather than limiting: the unit check has to happen before
the model lookup, so a request with no units must come back 422 and not 503.
If these ever return 503 for a unitless body, the order has been inverted and
a deployment with models would be predicting from guesses again.
"""

import json

import pytest

WITH_UNITS = {
    "rbc": {"value": 4.40, "unit": "millions/uL"},
    "wbc": {"value": 5800, "unit": "cells/uL"},
    "platelets": {"value": 85000, "unit": "cells/uL"},
    "anc": {"value": 3500, "unit": "cells/uL"},
}

BARE_NUMBERS = {"rbc": 4.40, "wbc": 5800, "platelets": 85000, "anc": 3500}


def post(client, csrf_token, path, body):
    return client.post(path, json=body, headers={"X-CSRFToken": csrf_token})


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_bare_numbers_are_refused_with_422(client, csrf_token, path):
    response = post(client, csrf_token, path, BARE_NUMBERS)

    assert response.status_code == 422, response.data
    body = response.get_json()
    assert body["reason"] == "missing_or_unknown_unit"
    assert "will not infer a unit" in body["error"]


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_a_unitless_body_is_refused_before_the_models_are_consulted(
    client, csrf_token, path
):
    """422, not 503, even though there are no models to serve."""
    response = post(client, csrf_token, path, BARE_NUMBERS)

    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_a_body_with_units_gets_past_the_boundary(client, csrf_token, path):
    """With units supplied the request is well formed; only models are missing."""
    response = post(client, csrf_token, path, WITH_UNITS)

    assert response.status_code == 503, response.data
    assert response.get_json()["reason"] == "models_unavailable"


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_an_unknown_unit_is_refused_not_defaulted(client, csrf_token, path):
    body = dict(WITH_UNITS, platelets={"value": 85000, "unit": "sparkles"})
    response = post(client, csrf_token, path, body)

    assert response.status_code == 422
    assert "Unrecognised unit" in response.get_json()["error"]


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_a_missing_required_analyte_is_400_not_422(client, csrf_token, path):
    """Absent and uninterpretable are different failures."""
    body = {k: v for k, v in WITH_UNITS.items() if k != "platelets"}
    response = post(client, csrf_token, path, body)

    assert response.status_code == 400
    assert response.get_json()["missing"] == ["platelets"]


@pytest.mark.parametrize("path", ["/predict", "/api/predict"])
def test_a_wrongly_chosen_unit_is_refused_as_implausible(client, csrf_token, path):
    body = dict(WITH_UNITS, platelets={"value": 85000, "unit": "thousand/uL"})
    response = post(client, csrf_token, path, body)

    assert response.status_code == 422
    assert response.get_json()["reason"] == "implausible_value"


def test_every_refusal_carries_a_request_id(client, csrf_token):
    response = post(client, csrf_token, "/api/predict", BARE_NUMBERS)

    assert len(response.get_json()["request_id"]) == 32


# --- the structured log -----------------------------------------------------

def test_a_refusal_is_logged_as_json_with_the_offending_input(client, csrf_token, caplog):
    import prediction_log

    with caplog.at_level("INFO", logger=prediction_log.LOGGER_NAME):
        post(client, csrf_token, "/api/predict", BARE_NUMBERS)

    records = [json.loads(r.message) for r in caplog.records]
    refusals = [r for r in records if r.get("outcome") == "refused"]

    assert len(refusals) == 1
    entry = refusals[0]
    assert entry["refusal"]["reason"] == "missing_or_unknown_unit"
    assert entry["raw_input"] == BARE_NUMBERS
    assert entry["schema_version"] == prediction_log.LOG_SCHEMA_VERSION
    assert entry["timestamp"].endswith("+00:00")


def test_log_records_are_ascii_so_a_cp1252_console_cannot_break_them(caplog):
    """CLAUDE.md records a UnicodeEncodeError read as a model-loading error."""
    import prediction_log
    from cbc import build_panel

    panel = build_panel({"wbc": (5800, "cells/µL")})

    with caplog.at_level("INFO", logger=prediction_log.LOGGER_NAME):
        prediction_log.log_prediction(
            request_id="test",
            panel=panel,
            predictions={},
            feature_records={},
            model_versions={},
        )

    message = caplog.records[-1].message
    message.encode("cp1252")   # raises if a micro sign survived
    assert "cells/uL" in message


# --- health -----------------------------------------------------------------

def test_health_publishes_the_declared_feature_units(client):
    body = client.get("/health").get_json()

    assert body["feature_contracts"]["thrombocytopenia"] == [
        {"analyte": "rbc", "unit": "million/uL"},
        {"analyte": "wbc", "unit": "thousand/uL"},
        {"analyte": "platelets", "unit": "thousand/uL"},
    ]
