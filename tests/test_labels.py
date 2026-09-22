"""WHO 2024 anaemia bands, and the decisions taken at their edges.

The numeric assertions lock the transcription of WHO 2024 Table 3. If a
threshold here changes, it should be because WHO changed it and someone
updated the citation, not because a comparison was flipped.
"""

import pytest

from chemoguard.labels import (
    MACROCYTIC_ABOVE,
    MICROCYTIC_BELOW,
    UNADJUSTED_FOR,
    WHO_2024_BANDS,
    Morphology,
    Severity,
    Unlabelled,
    band_for,
    morphology_for,
    severity_for,
)
from units import G_PER_DL, G_PER_L, Quantity


def hb(value, unit="g/dL"):
    return Quantity.of(value, unit)


# --- the transcription ------------------------------------------------------

#: WHO 2024 Table 3, in g/L, independently retyped from the guideline so a
#: single-character slip in the module is caught rather than mirrored.
EXPECTED_G_PER_L = {
    "children_6_23_months": (105, 95, 70),
    "children_24_59_months": (110, 100, 70),
    "children_5_11_years": (115, 110, 80),
    "children_12_14_years": (120, 110, 80),
    "adult_female": (120, 110, 80),
    "adult_male": (130, 110, 80),
    "pregnant_first_trimester": (110, 100, 70),
    "pregnant_second_trimester": (105, 95, 70),
    "pregnant_third_trimester": (110, 100, 70),
}


@pytest.mark.parametrize("key,expected", sorted(EXPECTED_G_PER_L.items()))
def test_bands_match_who_2024_table_3(key, expected):
    band = WHO_2024_BANDS[key]
    actual = (
        band.anaemia.magnitude_in(G_PER_L),
        band.mild.magnitude_in(G_PER_L),
        band.moderate.magnitude_in(G_PER_L),
    )
    assert actual == pytest.approx(expected)


def test_thresholds_are_declared_in_grams_per_litre():
    """WHO writes g/L; the module must not hand-convert."""
    for band in WHO_2024_BANDS.values():
        assert band.anaemia.unit is G_PER_L
        assert band.mild.unit is G_PER_L
        assert band.moderate.unit is G_PER_L


def test_conversion_to_the_canonical_unit_is_exact():
    men = WHO_2024_BANDS["adult_male"]
    assert men.anaemia.magnitude_in(G_PER_DL) == pytest.approx(13.0)
    assert men.mild.magnitude_in(G_PER_DL) == pytest.approx(11.0)
    assert men.moderate.magnitude_in(G_PER_DL) == pytest.approx(8.0)


def test_2024_kept_the_adult_bands_of_2011():
    """The brief's adult numbers; unchanged by the 2024 revision."""
    men = WHO_2024_BANDS["adult_male"]
    assert men.classify(hb(13.0)) is Severity.NONE
    assert men.classify(hb(12.9)) is Severity.MILD
    assert men.classify(hb(11.0)) is Severity.MILD
    assert men.classify(hb(10.9)) is Severity.MODERATE
    assert men.classify(hb(8.0)) is Severity.MODERATE
    assert men.classify(hb(7.9)) is Severity.SEVERE

    women = WHO_2024_BANDS["adult_female"]
    assert women.classify(hb(12.0)) is Severity.NONE
    assert women.classify(hb(11.9)) is Severity.MILD
    assert women.classify(hb(11.0)) is Severity.MILD
    assert women.classify(hb(10.9)) is Severity.MODERATE


def test_the_sex_adjustment_is_real():
    """11.5 g/dL is mild anaemia in a man and no anaemia in a woman... no.

    It is mild in both, but 12.5 separates them: below the male cutoff of 13
    and above the female cutoff of 12.
    """
    assert severity_for(hb(12.5), 40, "male") is Severity.MILD
    assert severity_for(hb(12.5), 40, "female") is Severity.NONE


def test_classification_is_unit_independent():
    """The same haemoglobin in g/L classifies identically."""
    assert severity_for(hb(125, "g/L"), 40, "male") is Severity.MILD
    assert severity_for(hb(12.5), 40, "male") is Severity.MILD


# --- paediatric bands -------------------------------------------------------

@pytest.mark.parametrize("age,expected_key", [
    (1, "children_6_23_months"),
    (2, "children_24_59_months"),
    (4, "children_24_59_months"),
    (5, "children_5_11_years"),
    (11, "children_5_11_years"),
    (12, "children_12_14_years"),
    (14, "children_12_14_years"),
    (15, "adult_female"),
])
def test_age_selects_the_right_band(age, expected_key):
    band, _ = band_for(age, "female")
    assert band.key == expected_key


def test_the_new_2024_infant_band_is_lower_than_the_2011_one():
    """6-23 months moved from 110 to 105 g/L. A toddler at 10.8 g/dL is not
    anaemic under 2024 and was under 2011."""
    assert severity_for(hb(10.8), 1, "female") is Severity.NONE
    assert severity_for(hb(10.8), 3, "female") is Severity.MILD


def test_a_child_is_not_labelled_with_an_adult_cutoff():
    """13.0 g/dL: not anaemic at any age, but 11.4 separates the bands."""
    assert severity_for(hb(11.4), 8, "male") is Severity.MILD      # <11.5
    assert severity_for(hb(11.4), 30, "male") is Severity.MILD     # <13.0
    assert severity_for(hb(11.4), 3, "male") is Severity.NONE      # >=11.0


# --- the stated decisions ---------------------------------------------------

def test_over_65_is_unlabelled_by_default():
    """WHO scopes the adult rows to 15-65 and names >65 a research gap."""
    assert severity_for(hb(11.0), 70, "male") is Unlabelled.OVER_65
    assert severity_for(hb(11.0), 65, "male") is Severity.MILD


def test_over_65_can_be_labelled_deliberately_and_is_marked():
    band, extrapolated = band_for(70, "male", allow_over_65=True)
    assert band.key == "adult_male"
    assert extrapolated is True

    band, extrapolated = band_for(60, "male", allow_over_65=True)
    assert extrapolated is False


def test_pregnancy_is_excluded_because_trimester_is_unknown():
    assert severity_for(hb(10.0), 30, "female", pregnant=True) is (
        Unlabelled.PREGNANT_TRIMESTER_UNKNOWN)
    assert severity_for(
        hb(10.0), 30, "female", pregnancy_ascertained=False) is (
        Unlabelled.PREGNANCY_UNASCERTAINED)


def test_the_trimester_choice_would_have_mattered():
    """Half a severity band apart, which is why it is not guessed."""
    first = WHO_2024_BANDS["pregnant_first_trimester"]
    second = WHO_2024_BANDS["pregnant_second_trimester"]
    assert first.classify(hb(10.7)) is Severity.MILD
    assert second.classify(hb(10.7)) is Severity.NONE


def test_unadjusted_factors_are_declared():
    assert "elevation of residence" in UNADJUSTED_FOR
    assert "smoking" in UNADJUSTED_FOR


# --- no bare numbers --------------------------------------------------------

def test_a_bare_number_is_refused():
    with pytest.raises(TypeError):
        severity_for(12.5, 40, "male")
    with pytest.raises(TypeError):
        morphology_for(85.0)


# --- morphology -------------------------------------------------------------

@pytest.mark.parametrize("mcv,expected", [
    (79.9, Morphology.MICROCYTIC),
    (80.0, Morphology.NORMOCYTIC),
    (100.0, Morphology.NORMOCYTIC),
    (100.1, Morphology.MACROCYTIC),
])
def test_morphology_boundaries(mcv, expected):
    assert morphology_for(Quantity.of(mcv, "fL")) is expected


def test_morphology_thresholds_are_quantities():
    assert MICROCYTIC_BELOW.magnitude_in("fL") == 80.0
    assert MACROCYTIC_ABOVE.magnitude_in("fL") == 100.0


def test_morphology_does_not_read_haemoglobin():
    """It survives the Hb hold-out; the anaemia severity label does not."""
    import inspect

    source = inspect.getsource(morphology_for)
    assert "hb" not in source
