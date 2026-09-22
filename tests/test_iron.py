"""The iron-deficiency target: WHO 2020 thresholds, inflammation, and leakage.

The point of this target is that it is *not* in the CBC. The tests that
matter are the ones pinning that: no closed-form route is declared, and the
three columns that define the label can never become features of it.
"""

import pytest

from chemoguard import leakage
from chemoguard.features import NEVER_A_FEATURE, PROFILES, build_matrix
from chemoguard.iron import (
    INFLAMMATION_CRP_THRESHOLD,
    STFR_CAVEAT,
    WHO_2020_FERRITIN_BANDS,
    IronStatus,
    UnlabelledIron,
    band_for_age,
    is_inflamed,
    iron_status_for,
)
from units import MICROGRAM_PER_L, Quantity


def ferritin(value, unit="ug/L"):
    return Quantity.of(value, unit)


def crp(value, unit="mg/L"):
    return Quantity.of(value, unit)


# --- the transcription ------------------------------------------------------

#: WHO 2020 technical brief Table 1, in ug/L, retyped independently.
EXPECTED = {
    "infants_0_23_months": (12, 30),
    "children_24_59_months": (12, 30),
    "children_5_9_years": (15, 70),
    "adolescents_10_19_years": (15, 70),
    "adults_20_59_years": (15, 70),
    "older_persons_60_plus": (15, 70),
}


@pytest.mark.parametrize("key,expected", sorted(EXPECTED.items()))
def test_bands_match_who_2020_table_1(key, expected):
    band = WHO_2020_FERRITIN_BANDS[key]
    actual = (band.healthy.magnitude_in(MICROGRAM_PER_L),
              band.inflamed.magnitude_in(MICROGRAM_PER_L))
    assert actual == pytest.approx(expected)


def test_there_is_an_over_60_band_unlike_the_anaemia_guideline():
    """The reason the over-65 exclusion does not carry over to this target.

    WHO's anaemia table stops at 65 and calls older adults a research gap.
    Its ferritin table has an explicit 60+ row at the same cut-off.
    """
    older = WHO_2020_FERRITIN_BANDS["older_persons_60_plus"]
    adult = WHO_2020_FERRITIN_BANDS["adults_20_59_years"]
    assert older.healthy.magnitude_in(MICROGRAM_PER_L) == \
        adult.healthy.magnitude_in(MICROGRAM_PER_L)

    assert band_for_age(80).key == "older_persons_60_plus"
    assert iron_status_for(
        ferritin(9), 80, crp(1.0),
        adjust_for_inflammation=True) is IronStatus.DEFICIENT


@pytest.mark.parametrize("age,key", [
    (1, "infants_0_23_months"),
    (2, "children_24_59_months"),
    (4, "children_24_59_months"),
    (5, "children_5_9_years"),
    (9, "children_5_9_years"),
    (10, "adolescents_10_19_years"),
    (19, "adolescents_10_19_years"),
    (20, "adults_20_59_years"),
    (59, "adults_20_59_years"),
    (60, "older_persons_60_plus"),
])
def test_age_selects_the_right_band(age, key):
    assert band_for_age(age).key == key


def test_the_under_five_threshold_is_lower():
    """12 ug/L below five, 15 from five up. A ferritin of 13 flips."""
    adjust = dict(adjust_for_inflammation=True)
    under_five = iron_status_for(ferritin(13), 3, crp(1.0), **adjust)
    over_five = iron_status_for(ferritin(13), 7, crp(1.0), **adjust)
    assert under_five is IronStatus.REPLETE
    assert over_five is IronStatus.DEFICIENT


# --- inflammation -----------------------------------------------------------

def test_inflammation_threshold_is_whos_worked_example():
    assert INFLAMMATION_CRP_THRESHOLD.magnitude_in("mg/L") == 5.0
    assert is_inflamed(crp(5.1)) is True
    assert is_inflamed(crp(4.9)) is False


def test_inflammation_raises_the_cutoff():
    """A ferritin of 40 is replete in a healthy adult, deficient if inflamed.

    This is the whole reason ferritin needs an inflammation marker: the same
    number means different things, and the raw threshold under-calls.
    """
    adjust = dict(adjust_for_inflammation=True)
    healthy = iron_status_for(ferritin(40), 30, crp(1.0), **adjust)
    inflamed = iron_status_for(ferritin(40), 30, crp(9.0), **adjust)
    assert healthy is IronStatus.REPLETE
    assert inflamed is IronStatus.DEFICIENT


def test_absent_crp_is_unlabelled_not_assumed_healthy():
    """Assuming "not inflamed" applies the *lower* cutoff, which under-calls.

    Every error from that assumption points abnormal -> normal, which is the
    direction Finding 4 is about. So the row is excluded instead.
    """
    assert iron_status_for(
        ferritin(40), 30, None,
        adjust_for_inflammation=True) is UnlabelledIron.NO_CRP


def test_pregnancy_is_excluded():
    assert iron_status_for(
        ferritin(10), 30, crp(1.0), adjust_for_inflammation=True,
        pregnant=True) is UnlabelledIron.PREGNANT
    assert iron_status_for(
        ferritin(10), 30, crp(1.0), adjust_for_inflammation=True,
        pregnancy_ascertained=False) is UnlabelledIron.PREGNANCY_UNASCERTAINED


def test_absent_ferritin_is_its_own_reason():
    """The NHANES eligibility subsample, which is most of the exclusions."""
    assert iron_status_for(
        None, 30, crp(1.0),
        adjust_for_inflammation=True) is UnlabelledIron.NO_FERRITIN


# --- no bare numbers --------------------------------------------------------

def test_bare_numbers_are_refused():
    with pytest.raises(TypeError):
        iron_status_for(15.0, 30, crp(1.0), adjust_for_inflammation=True)
    with pytest.raises(TypeError):
        is_inflamed(5.0)


def test_ferritin_units_convert_correctly():
    """ng/mL is exactly ug/L, which is why NHANES publishes both."""
    assert ferritin(15, "ng/mL").magnitude_in("ug/L") == pytest.approx(15.0)
    assert iron_status_for(
        ferritin(14, "ng/mL"), 30, crp(1.0),
        adjust_for_inflammation=True) is IronStatus.DEFICIENT


def test_a_ferritin_sized_number_in_the_wrong_unit_changes_the_answer():
    """15 mg/L is 15,000 ug/L, not 15. The unit is not decoration."""
    assert iron_status_for(
        ferritin(15, "mg/L"), 30, crp(1.0),
        adjust_for_inflammation=True) is IronStatus.REPLETE
    assert iron_status_for(
        ferritin(14, "ug/L"), 30, crp(1.0),
        adjust_for_inflammation=True) is IronStatus.DEFICIENT


# --- the target is not in the features --------------------------------------

def test_label_inputs_can_never_be_features():
    assert set(NEVER_A_FEATURE) == {"ferritin", "stfr", "crp"}
    for profile, (analytes, _) in PROFILES.items():
        for banned in NEVER_A_FEATURE:
            assert banned not in analytes, profile


def test_a_profile_smuggling_a_label_input_is_refused():
    """The guard is in build_matrix, not only in the profile lists."""
    import pandas as pd

    from chemoguard.data.analytes import DATASET_ANALYTES
    from chemoguard.data.nhanes import Dataset

    frame = pd.DataFrame({"rbc": [4.5], "ferritin": [12.0]})
    dataset = Dataset(
        name="t", values=frame,
        units={c: DATASET_ANALYTES[c].canonical.symbol for c in frame.columns},
        meta=None)

    PROFILES["__smuggled__"] = (("rbc", "ferritin"), ())
    try:
        with pytest.raises(ValueError, match="define a target"):
            build_matrix(dataset, "__smuggled__", include_demographics=False)
    finally:
        del PROFILES["__smuggled__"]


def test_no_closed_form_route_to_ferritin_is_declared():
    """Not an omission: no CBC identity produces ferritin.

    If someone ever adds one, this fails and the pivot needs re-examining.
    """
    assert leakage.FERRITIN_PATHS == ()


def test_the_stfr_definition_declares_that_it_is_not_a_who_value():
    assert "WHO sets none" in STFR_CAVEAT


# --- the inflammation adjustment is an explicit choice ----------------------

def test_the_adjustment_flag_has_no_default():
    """Neither setting is authoritative, so neither is the default.

    WHO reports high variability across adjustment methods and BRINDA is
    unavailable without AGP. A silent default would present one of two
    defensible numbers as the answer.
    """
    with pytest.raises(TypeError):
        iron_status_for(ferritin(40), 30, crp(9.0))


def test_unadjusted_keeps_the_healthy_threshold_for_inflamed_rows():
    assert iron_status_for(
        ferritin(40), 30, crp(9.0),
        adjust_for_inflammation=False) is IronStatus.REPLETE
    assert iron_status_for(
        ferritin(40), 30, crp(9.0),
        adjust_for_inflammation=True) is IronStatus.DEFICIENT


def test_crp_is_required_even_when_the_adjustment_is_off():
    """So both prevalences are computed over exactly the same rows.

    Without this the unadjusted rule would silently run on 288 more rows
    than the adjusted one, and the difference between them would no longer
    measure only the adjustment.
    """
    assert iron_status_for(
        ferritin(40), 30, None,
        adjust_for_inflammation=False) is UnlabelledIron.NO_CRP
