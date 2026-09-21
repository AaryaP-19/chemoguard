"""The unit boundary: no bare number reaches a model.

The cases in this file are the ones CLAUDE.md Finding 4 records as producing
a wrong answer that pointed the safe way. Each failed before the change.
"""

import pytest

from cbc import (
    ANALYTES,
    ImplausibleValueError,
    build_panel,
    derive_anc,
    panel_from_extraction,
    panel_from_mapping,
)
from clinical_flags import flag_for, flags_for
from model_contract import contract_for, features_for
from units import (
    IncompatibleUnitsError,
    MissingUnitError,
    Quantity,
    UnknownUnitError,
    parse_unit,
)


# --- the type ---------------------------------------------------------------

@pytest.mark.parametrize(
    "spelling",
    ["cells/uL", "cells/µL", "cells/μL", "CELLS/UL", "/cumm", "/mm3", "cells / ul"],
)
def test_every_spelling_of_cells_per_ul_parses_to_one_unit(spelling):
    assert parse_unit(spelling) is parse_unit("cells/uL")


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("K/uL", "thousand/uL"),
        ("10^3/uL", "thousand/uL"),
        ("10^9/L", "thousand/uL"),
        ("M/uL", "million/uL"),
        ("10^6/uL", "million/uL"),
        ("10^12/L", "million/uL"),
        ("gm/dL", "g/dL"),
    ],
)
def test_report_spellings_resolve_to_the_right_unit(spelling, expected):
    assert parse_unit(spelling).symbol == expected


def test_conversion_is_exact_in_both_directions():
    assert Quantity.of(900, "cells/uL").magnitude_in("thousand/uL") == 0.9
    assert Quantity.of(900, "thousand/uL").magnitude_in("cells/uL") == 900_000


def test_dimensions_cannot_be_crossed():
    with pytest.raises(IncompatibleUnitsError):
        Quantity.of(13.0, "g/dL").to("cells/uL")


def test_an_unrecognised_unit_is_not_treated_as_a_default():
    with pytest.raises(UnknownUnitError):
        Quantity.of(5800, "widgets per furlong")


# --- the five regression cases ----------------------------------------------
#
# A stand-in estimator, because models/ is empty and expected to stay so until
# the NHANES artefacts land (CLAUDE.md Finding 5). It implements the threshold
# rules the quarantined models encoded -- reproduced exactly on 7,196 of 7,196
# rows, CLAUDE.md Finding 1 -- so these tests assert the behaviour of the
# boundary feeding a model, not the behaviour of a pickle that no longer
# exists. The point of each case is that the SAME estimator returns different
# answers for 900 cells/uL and 900 thousand/uL, which is only possible if the
# unit survived the journey.

class ThresholdEstimator:
    """The documented decision rule of one quarantined model."""

    def __init__(self, rule, n_features):
        self._rule = rule
        self.n_features_in_ = n_features

    def predict(self, rows):
        return [self._rule(*row) for row in rows]


def _anemia_rule(rbc, wbc_k, plt_k):
    # CLAUDE.md Finding 1: count of (RBC<3.5, WBC<4.0, PLATELETS<150).
    # int() because a feature row that came through numpy yields np.bool_.
    score = int(rbc < 3.5) + int(wbc_k < 4.0) + int(plt_k < 150)
    return ("Low", "Medium", "High", "High")[score]


def _neutropenia_rule(rbc, wbc_k, plt_k, anc_k):
    # CLAUDE.md Finding 1: ANC >=1500 Low, >=1000 Medium, else High.
    anc_cells = anc_k * 1000
    if anc_cells >= 1500:
        return "Low"
    return "Medium" if anc_cells >= 1000 else "High"


def _thrombocytopenia_rule(rbc, wbc_k, plt_k):
    # CLAUDE.md Finding 1: Platelets >=150 Low, >=75 Medium, else High.
    if plt_k >= 150:
        return "Low"
    return "Medium" if plt_k >= 75 else "High"


RULES = {
    "anemia": ThresholdEstimator(_anemia_rule, 3),
    "neutropenia": ThresholdEstimator(_neutropenia_rule, 4),
    "thrombocytopenia": ThresholdEstimator(_thrombocytopenia_rule, 3),
}


def risk(model, panel):
    """Run one stand-in estimator through the real feature boundary."""
    row, _record = features_for(model, panel.with_derived_anc())
    return RULES[model].predict([row])[0]


def panel(**values):
    """Build a panel, defaulting the analytes a model needs but a case is not about."""
    supplied = {
        "rbc": (4.5, "million/uL"),
        "wbc": (7000, "cells/uL"),
        "platelets": (250_000, "cells/uL"),
    }
    supplied.update(values)
    return build_panel(supplied)


def test_platelets_900_cells_per_ul_is_high_thrombocytopenia_risk():
    """The headline case. Read as 900 thousand/uL this returned Low."""
    p = panel(platelets=(900, "cells/uL"))

    assert p.get("platelets").magnitude_in("thousand/uL") == 0.9
    assert risk("thrombocytopenia", p) == "High"


def test_platelets_900_thousand_per_ul_is_low_risk():
    """The same digits, the other unit. Both have to work."""
    p = panel(platelets=(900, "thousand/uL"))

    assert p.get("platelets").magnitude_in("thousand/uL") == 900
    assert risk("thrombocytopenia", p) == "Low"


def test_the_two_platelet_readings_disagree():
    """Guards the whole change in one line.

    Under magnitude inference these two produced an identical feature value
    and therefore an identical answer. If this ever passes by returning the
    same risk twice, the boundary has collapsed again.
    """
    assert risk("thrombocytopenia", panel(platelets=(900, "cells/uL"))) != \
           risk("thrombocytopenia", panel(platelets=(900, "thousand/uL")))


def test_wbc_800_cells_per_ul_is_flagged_severe_leukopenia():
    p = panel(wbc=(800, "cells/uL"))
    flag = flag_for("wbc", p.get("wbc"))

    assert flag.condition == "leukopenia"
    assert flag.severe
    assert flag.grade == 4


def test_anc_800_cells_per_ul_is_high_neutropenia_risk():
    p = panel(anc=(800, "cells/uL"))

    assert p.get("anc").magnitude_in("cells/uL") == 800
    assert risk("neutropenia", p) == "High"
    assert flag_for("anc", p.get("anc")).severe


def test_a_value_with_no_unit_raises_and_does_not_predict():
    with pytest.raises(MissingUnitError):
        build_panel({"platelets": (900, None)})

    with pytest.raises(MissingUnitError):
        panel_from_mapping({"rbc": 4.5, "wbc": 7000, "platelets": 900})


def test_the_missing_unit_error_names_the_analyte():
    with pytest.raises(MissingUnitError, match="Platelets"):
        build_panel({"platelets": (900, None)})


# --- every accepted request shape requires a unit ---------------------------

@pytest.mark.parametrize(
    "body",
    [
        {"platelets": {"value": 900, "unit": "cells/uL"}},
        {"platelets": 900, "platelets_unit": "cells/uL"},
        {"platelets": 900, "units": {"platelets": "cells/uL"}},
    ],
)
def test_all_three_request_shapes_carry_the_unit(body):
    p = panel_from_mapping(body)

    assert p.get("platelets").magnitude_in("cells/uL") == 900


@pytest.mark.parametrize(
    "body",
    [
        {"platelets": {"value": 900}},
        {"platelets": {"value": 900, "unit": None}},
        {"platelets": 900},
        {"platelets": 900, "units": {}},
    ],
)
def test_no_request_shape_lets_a_bare_number_through(body):
    with pytest.raises(MissingUnitError):
        panel_from_mapping(body)


# --- derived values keep their units ----------------------------------------

def test_anc_derived_from_wbc_and_neutrophil_percent():
    anc = derive_anc(Quantity.of(5800, "cells/uL"), Quantity.of(60, "%"))

    assert anc.unit.symbol == "cells/uL"
    assert anc.value == pytest.approx(3480)


def test_derived_anc_is_the_same_whichever_unit_wbc_arrived_in():
    from_cells = derive_anc(Quantity.of(5800, "cells/uL"), Quantity.of(60, "%"))
    from_thousands = derive_anc(Quantity.of(5.8, "thousand/uL"), Quantity.of(60, "%"))

    assert from_cells.magnitude_in("cells/uL") == pytest.approx(
        from_thousands.magnitude_in("cells/uL")
    )


def test_a_reported_anc_is_not_overwritten_by_a_derived_one():
    p = build_panel({
        "wbc": (5800, "cells/uL"),
        "neutrophils_percent": (60, "%"),
        "anc": (3500, "cells/uL"),
    }).with_derived_anc()

    assert p.get("anc").magnitude_in("cells/uL") == 3500
    assert p.unit_provenance["anc"] == "operator"


# --- the model contract -----------------------------------------------------

def test_features_are_built_in_the_declared_unit():
    p = panel(platelets=(85_000, "cells/uL"), wbc=(5800, "cells/uL"))
    row, record = features_for("thrombocytopenia", p)

    assert row == [4.5, 5.8, 85.0]
    assert [f["fed_to_model"]["unit"] for f in record] == [
        "million/uL", "thousand/uL", "thousand/uL",
    ]


def test_the_feature_record_reports_both_scales():
    p = panel(platelets=(85_000, "cells/uL"))
    _row, record = features_for("thrombocytopenia", p)
    platelets = next(f for f in record if f["analyte"] == "platelets")

    assert platelets["canonical"] == {"value": 85_000.0, "unit": "cells/uL"}
    assert platelets["fed_to_model"] == {"value": 85.0, "unit": "thousand/uL"}


def test_a_contract_is_declared_for_every_served_model():
    for model in ("anemia", "neutropenia", "thrombocytopenia"):
        assert len(contract_for(model)) > 0


# --- plausibility rejects, it does not rescale ------------------------------

def test_a_wrongly_chosen_unit_is_refused_not_silently_corrected():
    """85000 thousand/uL is 8.5e10 platelets per litre. Refuse it."""
    with pytest.raises(ImplausibleValueError):
        build_panel({"platelets": (85_000, "thousand/uL")})


def test_a_genuine_cells_per_ul_reading_is_not_rejected():
    """The old _validate_value bound of (10, 2000) discarded exactly this."""
    p = build_panel({"platelets": (85_000, "cells/uL"), "wbc": (5800, "cells/uL")})

    assert p.get("platelets").magnitude_in("thousand/uL") == 85
    assert p.get("wbc").magnitude_in("thousand/uL") == 5.8


def test_hyperleukocytosis_is_plausible():
    """500,000 cells/uL is real in acute leukaemia and must not be refused."""
    p = build_panel({"wbc": (500_000, "cells/uL")})

    assert p.get("wbc").magnitude_in("thousand/uL") == 500


# --- extraction output ------------------------------------------------------

def test_extraction_skips_a_value_whose_unit_was_not_read():
    p, skipped = panel_from_extraction({
        "wbc": {"value": 5800, "unit": "cells/uL"},
        "platelets": {"value": 85000, "unit": None},
    })

    assert p.get("wbc") is not None
    assert p.get("platelets") is None
    assert skipped["platelets"]["reason"] == "MissingUnitError"


def test_every_analyte_declares_a_canonical_unit():
    for key, analyte in ANALYTES.items():
        assert analyte.canonical.dimension is analyte.canonical.dimension
        assert analyte.plausible_min < analyte.plausible_max
