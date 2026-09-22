"""AUC intervals, the AUC-scored leakage margin, and age-normalised MCV.

The interval maths is here because subgroup AUCs get read side by side as if
they were comparable, and on a rare positive class they are not.
"""

import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")

from chemoguard import training  # noqa: E402
from chemoguard.features import FEATURES, PROFILES, compute_one  # noqa: E402
from chemoguard.labels import (  # noqa: E402
    ADULT_MCV_LLN_AGE,
    MICROCYTIC_BELOW,
    Morphology,
    mcv_lower_limit,
    morphology_for,
)
from units import Quantity  # noqa: E402


# --- Hanley-McNeil ----------------------------------------------------------

def test_hanley_mcneil_matches_a_hand_computed_value():
    """AUC 0.75, 50 cases, 50 controls.

        Q1 = 0.75/1.25 = 0.6           Q2 = 2(0.5625)/1.75 = 0.642857
        var = [0.1875 + 49(0.0375) + 49(0.080357)] / 2500 = 0.0023850
        SE  = 0.04884
    """
    assert training.hanley_mcneil_se(0.75, 50, 50) == pytest.approx(
        0.04884, abs=1e-4)


def test_the_interval_is_driven_by_cases_not_rows():
    """The whole reason the interval is reported at all.

    Ten times the controls barely moves it; ten times the cases shrinks it.
    """
    few_cases = training.hanley_mcneil_se(0.8, 37, 646)
    more_controls = training.hanley_mcneil_se(0.8, 37, 6460)
    more_cases = training.hanley_mcneil_se(0.8, 370, 6460)

    assert more_controls > 0.5 * few_cases      # hardly changes
    assert more_cases < 0.4 * few_cases         # collapses


def test_se_is_undefined_without_both_classes():
    assert training.hanley_mcneil_se(0.8, 0, 100) != \
        training.hanley_mcneil_se(0.8, 0, 100)   # NaN


def test_auc_with_interval_brackets_the_estimate():
    import numpy as np

    rng = np.random.default_rng(0)
    y = np.array([0] * 200 + [1] * 40)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.2, 1, 40)])

    result = training.auc_with_interval(y, scores)
    assert result["n_positive"] == 40
    assert result["n_negative"] == 200
    assert result["ci_low"] < result["auc"] < result["ci_high"]
    assert 0.0 <= result["ci_low"] and result["ci_high"] <= 1.0


def test_intervals_overlap_detects_indistinguishable_groups():
    wide = {"ci_low": 0.63, "ci_high": 0.82}
    narrow = {"ci_low": 0.88, "ci_high": 0.91}
    touching = {"ci_low": 0.80, "ci_high": 0.90}

    assert training.intervals_overlap(wide, narrow) is False
    assert training.intervals_overlap(wide, touching) is True


# --- the leakage margin -----------------------------------------------------

def test_the_leakage_margin_is_on_auc_not_accuracy():
    """CLAUDE.md's hard rule changed scale on 2026-09-21.

    0.02 has to sit above fold-to-fold AUC noise (~0.014) and below the gap
    a learnable target shows (~0.06).
    """
    assert training.LEAKAGE_AUC_MARGIN == 0.02
    assert 0.014 < training.LEAKAGE_AUC_MARGIN < 0.06


# --- age-banded MCV ---------------------------------------------------------

@pytest.mark.parametrize("age,expected", [
    (1, 71.0), (2, 72.0), (5, 75.0), (9, 79.0),
    (10, 80.0), (30, 80.0), (80, 80.0), (None, 80.0),
])
def test_mcv_lower_limit_follows_70_plus_age_below_ten(age, expected):
    assert mcv_lower_limit(age).magnitude_in("fL") == pytest.approx(expected)


def test_the_paediatric_limit_joins_the_adult_one_without_a_step():
    at_nine = mcv_lower_limit(ADULT_MCV_LLN_AGE - 1).magnitude_in("fL")
    at_ten = mcv_lower_limit(ADULT_MCV_LLN_AGE).magnitude_in("fL")
    assert at_ten - at_nine == pytest.approx(1.0)
    assert at_ten == MICROCYTIC_BELOW.magnitude_in("fL")


def test_a_toddler_is_not_microcytic_at_an_adult_cutpoint():
    """MCV 75 fL: normal for a three-year-old, microcytic in an adult."""
    mcv = Quantity.of(75.0, "fL")
    assert morphology_for(mcv, age_years=3) is Morphology.NORMOCYTIC
    assert morphology_for(mcv, age_years=30) is Morphology.MICROCYTIC
    # No age supplied keeps the old behaviour, so nothing changed silently.
    assert morphology_for(mcv) is Morphology.MICROCYTIC


def test_mcv_vs_age_lln_is_the_distance_from_the_age_specific_limit():
    values = FEATURES["mcv_vs_age_lln"].fn(
        {"mcv": [75.0, 75.0], "age_years": [3, 30]})
    assert list(values) == pytest.approx([2.0, -5.0])


def test_the_age_aware_feature_declares_its_demographic_input():
    feature = FEATURES["mcv_vs_age_lln"]
    assert feature.meta_inputs == ("age_years",)
    assert feature.analytes == ("mcv",)
    assert feature.reads_haemoglobin is False


def test_compute_one_still_refuses_a_bare_mcv():
    with pytest.raises(TypeError):
        compute_one("mentzer_index", {"mcv": 90.0, "rbc": 4.5})


def test_only_the_age_aware_profile_carries_the_feature():
    assert "mcv_vs_age_lln" not in PROFILES["iron-deficiency"][1]
    assert "mcv_vs_age_lln" in PROFILES["iron-deficiency-age-aware"][1]
    # Everything else about the two profiles is identical, so a difference
    # between them is attributable to the one feature.
    assert PROFILES["iron-deficiency"][0] == \
        PROFILES["iron-deficiency-age-aware"][0]
    assert set(PROFILES["iron-deficiency"][1]) < \
        set(PROFILES["iron-deficiency-age-aware"][1])
