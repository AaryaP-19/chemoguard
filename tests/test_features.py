"""Derived features, the anaemia hold-out, and whether the hold-out is real.

The load-bearing test here is not ``test_haemoglobin_is_not_in_the_matrix``.
That one passes trivially. It is
``test_the_default_matrix_has_no_closed_form_route_to_haemoglobin``, which
fails on a matrix that merely drops the column name -- because MCH x RBC / 10
puts haemoglobin straight back.
"""

import pytest

pd = pytest.importorskip("pandas")

from chemoguard import leakage  # noqa: E402
from chemoguard.data.analytes import DATASET_ANALYTES  # noqa: E402
from chemoguard.data.nhanes import Dataset  # noqa: E402
from chemoguard.features import (  # noqa: E402
    ANAEMIA_PROFILES,
    FEATURES,
    HAEMOGLOBIN_FEATURES,
    HELD_OUT_FOR_ANAEMIA,
    build_anaemia_matrix,
    compute_one,
)
from chemoguard.labels import label_dataset  # noqa: E402
from units import Quantity  # noqa: E402


def coherent_dataset(n=200):
    """A synthetic cohort that satisfies the CBC identities exactly.

    The identities are the leak, so a test of the leak has to be run on data
    where they hold -- which on NHANES is 19,747 rows out of 19,749.
    """
    rbc = pd.Series([3.5 + (i % 40) * 0.05 for i in range(n)])
    hb = pd.Series([9.0 + (i % 50) * 0.12 for i in range(n)])
    hct = hb / 33.0 * 100.0                    # MCHC pinned at a normal 33
    platelets = pd.Series([150_000.0 + (i % 30) * 4000 for i in range(n)])
    lymphocytes = pd.Series([1000.0 + (i % 25) * 60 for i in range(n)])

    values = pd.DataFrame({
        "rbc": rbc,
        "hb": hb,
        "hematocrit": hct,
        "mcv": hct / rbc * 10,
        "mch": hb / rbc * 10,
        "mchc": hb / hct * 100,
        "rdw": pd.Series([12.0 + (i % 20) * 0.2 for i in range(n)]),
        "platelets": platelets,
        "mpv": pd.Series([8.0 + (i % 15) * 0.1 for i in range(n)]),
        "wbc": pd.Series([5000.0 + (i % 35) * 100 for i in range(n)]),
        "anc": pd.Series([3000.0 + (i % 28) * 80 for i in range(n)]),
        "lymphocytes": lymphocytes,
        "neutrophils_percent": pd.Series([55.0 + (i % 18) for i in range(n)]),
    })
    units = {c: DATASET_ANALYTES[c].canonical.symbol for c in values.columns}
    meta = pd.DataFrame({
        "age_years": [20 + (i % 40) for i in range(n)],
        "sex": ["male" if i % 2 else "female" for i in range(n)],
        "pregnancy_code": [None] * n,
    })
    return Dataset(name="synthetic", values=values, units=units, meta=meta)


# --- the unit contract ------------------------------------------------------

def test_every_feature_declares_a_unit_for_every_input():
    for feature in FEATURES.values():
        assert feature.inputs
        for item in feature.inputs:
            assert item.analyte in DATASET_ANALYTES
            assert item.unit


def test_compute_one_refuses_a_bare_number():
    with pytest.raises(TypeError):
        compute_one("mentzer_index", {"mcv": 90.0, "rbc": 4.5})


def test_a_feature_converts_its_inputs_rather_than_trusting_them():
    """The same measurement in a different unit gives the same feature.

    Mentzer is defined for RBC in million/uL. Handing it cells/uL without
    conversion would be 10^6 times wrong and raise nothing.
    """
    in_millions = compute_one("mentzer_index", {
        "mcv": Quantity.of(90.0, "fL"),
        "rbc": Quantity.of(4.5, "million/uL"),
    })
    in_cells = compute_one("mentzer_index", {
        "mcv": Quantity.of(90.0, "fL"),
        "rbc": Quantity.of(4_500_000.0, "cells/uL"),
    })
    assert in_millions == pytest.approx(20.0)
    assert in_cells == pytest.approx(in_millions)


def test_hb_per_rbc_is_mch_over_ten():
    """Stated in the docstring; asserted here so it cannot drift."""
    value = compute_one("hb_per_rbc", {
        "hb": Quantity.of(13.5, "g/dL"),
        "rbc": Quantity.of(4.5, "million/uL"),
    })
    mch = 13.5 / 4.5 * 10
    assert value == pytest.approx(mch / 10)


def test_features_that_read_haemoglobin_say_so():
    # green_king joined this set on 2026-09-21: the published 1989 index
    # reads Hb, unlike the rdw_weighted_mentzer that used to carry its name.
    assert set(HAEMOGLOBIN_FEATURES) == {
        "hb_per_rbc", "cytopenia_count", "green_king"}


# --- the hold-out, by name --------------------------------------------------

@pytest.mark.parametrize("profile", sorted(ANAEMIA_PROFILES))
def test_haemoglobin_is_not_in_the_matrix(profile):
    """The test the brief asks for. Necessary, and nowhere near sufficient."""
    matrix = build_anaemia_matrix(coherent_dataset(), profile=profile)
    for held_out in HELD_OUT_FOR_ANAEMIA:
        assert held_out not in matrix.columns
    assert "hb" not in matrix.columns
    assert "hematocrit" not in matrix.columns


@pytest.mark.parametrize("profile", sorted(ANAEMIA_PROFILES))
def test_no_feature_reading_haemoglobin_is_in_the_matrix(profile):
    matrix = build_anaemia_matrix(coherent_dataset(), profile=profile)
    for name in HAEMOGLOBIN_FEATURES:
        assert name not in matrix.columns


def test_every_exclusion_carries_a_reason():
    matrix = build_anaemia_matrix(coherent_dataset())
    for name, reason in matrix.excluded.items():
        assert reason, name


# --- the hold-out, algebraically -------------------------------------------

def test_the_default_matrix_has_no_closed_form_route_to_haemoglobin():
    """The real test. A name-only hold-out fails this one."""
    dataset = coherent_dataset()
    matrix = build_anaemia_matrix(dataset, profile="reconstruction-safe")
    recoveries = leakage.closed_form(matrix, dataset, dataset.values["hb"])

    leaking = [r.name for r in recoveries if r.leaks]
    assert not leaking, "haemoglobin is reconstructible from {0}".format(leaking)


def test_the_name_only_hold_out_is_demonstrably_leaked():
    """Pinned as a fact, not a regression to fix.

    If this ever stops failing to hold out haemoglobin, either the identities
    stopped holding -- which would mean the data got worse -- or the profile
    quietly changed.
    """
    dataset = coherent_dataset()
    matrix = build_anaemia_matrix(dataset, profile="name-excluded")
    recoveries = leakage.closed_form(matrix, dataset, dataset.values["hb"])

    by_name = {r.name: r for r in recoveries}
    assert by_name["Hb from MCH x RBC"].leaks
    assert by_name["Hb from MCH x RBC"].rate == pytest.approx(1.0)


def test_haematocrit_is_reconstructible_wherever_mcv_and_rbc_survive():
    """The HCT hold-out is cosmetic in both non-trivial profiles."""
    dataset = coherent_dataset()
    for profile in ("name-excluded", "reconstruction-safe"):
        matrix = build_anaemia_matrix(dataset, profile=profile)
        recoveries = leakage.closed_form(
            matrix, dataset, dataset.values["hematocrit"],
            paths=leakage.HCT_PATHS)
        assert any(r.leaks for r in recoveries), profile


def test_red_cell_free_profile_breaks_every_closed_form_route():
    dataset = coherent_dataset()
    matrix = build_anaemia_matrix(dataset, profile="red-cell-free")
    for paths, truth in (
        (leakage.PATHS, dataset.values["hb"]),
        (leakage.HCT_PATHS, dataset.values["hematocrit"]),
    ):
        for recovery in leakage.closed_form(matrix, dataset, truth, paths):
            assert not recovery.available or not recovery.leaks


def test_reconstruction_tolerance_matches_the_coherence_filter():
    """A looser bound here would let a leak pass a test the data would fail."""
    from chemoguard.data.coherence import EXACT_IDENTITY_TOLERANCE

    assert leakage.RECONSTRUCTION_TOLERANCE == EXACT_IDENTITY_TOLERANCE


# --- labels line up with the matrix ----------------------------------------

def test_labelling_a_dataset_reports_what_it_could_not_label():
    dataset = coherent_dataset()
    report = label_dataset(dataset)
    assert report.counts["rows"] == len(dataset.values)
    assert report.counts["labelled"] > 0
    assert report.assumptions
