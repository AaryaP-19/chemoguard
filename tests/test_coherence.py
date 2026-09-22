"""The staged coherence checks, and the three ways a flat filter gets it wrong.

These lock the *design* decisions, not the survivor counts. The counts move
whenever the data does; the reasons MCHC is excluded, the HCT rule is
excluded, and an absent column is not a pass do not.

No network and no quarantined files: every case here is built from a handful
of rows constructed to satisfy or violate one identity at a time.
"""

import pytest

pd = pytest.importorskip("pandas")

from chemoguard.data.analytes import (  # noqa: E402
    DATASET_ANALYTES,
    Column,
    conversion_factor,
    rescale_frame,
)
from chemoguard.data.coherence import (  # noqa: E402
    APPROXIMATE_IDENTITY_TOLERANCE,
    EXACT_IDENTITY_TOLERANCE,
    IDENTITIES,
    JOINT_IDENTITIES,
    evaluate,
)
from chemoguard.data.nhanes import (  # noqa: E402
    ABSENT_FROM_NHANES,
    NHANES_CBC_COLUMNS,
    Dataset,
)
from units import IncompatibleUnitsError  # noqa: E402


def coherent_rows(n=50):
    """Rows that satisfy every identity exactly, in canonical units.

    Built forwards from the four independent measurements an analyser
    actually makes -- RBC, Hb, HCT, platelets, MPV -- with the five derived
    columns computed from them. Nothing here is fitted to the checks; the
    checks recompute the same arithmetic.
    """
    rbc = pd.Series([4.0 + i * 0.02 for i in range(n)])
    hb = pd.Series([12.0 + i * 0.05 for i in range(n)])
    hct = pd.Series([36.0 + i * 0.15 for i in range(n)])
    platelets = pd.Series([150_000.0 + i * 2000 for i in range(n)])
    mpv = pd.Series([8.0 + i * 0.05 for i in range(n)])

    to_thousand = conversion_factor("cells/uL", "thousand/uL")
    return pd.DataFrame({
        "rbc": rbc,
        "hb": hb,
        "hematocrit": hct,
        "platelets": platelets,
        "mpv": mpv,
        "mcv": hct / rbc * 10,
        "mch": hb / rbc * 10,
        "mchc": hb / hct * 100,
        "pct": (platelets * to_thousand) * mpv / 10000.0,
    })


def as_dataset(frame, name="synthetic", absent=()):
    units = {c: DATASET_ANALYTES[c].canonical.symbol for c in frame.columns}
    return Dataset(name=name, values=frame, units=units, absent=absent)


# --- the constants are constants -------------------------------------------

def test_tolerances_are_named_module_constants():
    """Finding 2: a tolerance chosen per run makes the count unreproducible."""
    assert EXACT_IDENTITY_TOLERANCE == 0.02
    assert APPROXIMATE_IDENTITY_TOLERANCE == 0.15
    for identity in IDENTITIES:
        expected = (EXACT_IDENTITY_TOLERANCE if identity.kind == "exact"
                    else APPROXIMATE_IDENTITY_TOLERANCE)
        assert identity.tolerance == expected, identity.key


def test_evaluate_takes_no_tolerance_argument():
    """There is no per-run override, by construction."""
    import inspect

    parameters = inspect.signature(evaluate).parameters
    assert list(parameters) == ["dataset"]


# --- the identities themselves ---------------------------------------------

def test_coherent_rows_pass_every_identity():
    report = evaluate(as_dataset(coherent_rows()))
    for result in report.identities:
        assert result.is_evaluable, result.identity.key
        assert result.pass_rate == 1.0, result.identity.key


@pytest.mark.parametrize("identity_key,column,factor", [
    ("mcv", "mcv", 1.5),
    ("mch", "mch", 0.5),
    ("pct", "pct", 3.0),
])
def test_one_broken_column_fails_only_its_own_identity(identity_key, column, factor):
    """An identity is a local check; breaking one column is not contagious.

    Except where the algebra says it must be: perturbing MCV or MCH also
    breaks the MCHC relation, which is the point of the next test.
    """
    frame = coherent_rows()
    frame[column] = frame[column] * factor
    report = evaluate(as_dataset(frame))

    assert report.identity(identity_key).pass_rate == 0.0
    for other in ("mcv", "mch", "pct"):
        if other != identity_key:
            assert report.identity(other).pass_rate == 1.0, other


def test_hct_rule_fails_on_correct_data_when_mchc_is_not_33():
    """The rule of thumb is wrong on good data, which is why it cannot filter.

    HCT = Hb/MCHC x 100, so HCT = 3 x Hb only at MCHC = 33.3. These rows are
    internally perfect at MCHC 28, a value a real patient has.
    """
    rbc = pd.Series([4.0] * 20)
    hb = pd.Series([10.0 + i * 0.1 for i in range(20)])
    hct = hb / 28.0 * 100.0          # MCHC pinned at 28 g/dL
    frame = pd.DataFrame({
        "rbc": rbc, "hb": hb, "hematocrit": hct,
        "mcv": hct / rbc * 10, "mch": hb / rbc * 10, "mchc": hb / hct * 100,
    })
    report = evaluate(as_dataset(frame))

    assert report.identity("mcv").pass_rate == 1.0
    assert report.identity("mch").pass_rate == 1.0
    assert report.identity("mchc").pass_rate == 1.0
    # ...and the rule of thumb rejects every one of them.
    assert report.identity("hct_rule").pass_rate == 0.0


# --- why MCHC is excluded ---------------------------------------------------

def test_mchc_is_implied_by_mcv_and_mch():
    """MCH/MCV x 100 == Hb/HCT x 100, so MCHC adds no independent constraint.

    Asserted on the arithmetic, not on a pass rate: RBC cancels exactly.
    """
    frame = coherent_rows()
    implied = frame["mch"] / frame["mcv"] * 100.0
    assert (implied - frame["mchc"]).abs().max() < 1e-9


def test_mchc_conditional_pass_rate_is_total_on_coherent_data():
    report = evaluate(as_dataset(coherent_rows()))
    assert report.mchc_redundancy["conditional_pass_rate"] == 1.0


def test_mchc_and_hct_rule_are_not_in_the_joint_filter():
    joint = {i.key for i in JOINT_IDENTITIES}
    assert joint == {"mcv", "mch", "pct"}
    assert "mchc" not in joint
    assert "hct_rule" not in joint


def test_excluding_mchc_cannot_lower_the_survivor_count():
    """The exclusion only ever moves the count up, never down.

    Adding a redundant constraint can only remove rows -- it is the
    deflation Finding 6 describes. This pins the direction.
    """
    frame = coherent_rows()
    # Break MCHC alone, leaving MCV and MCH intact. Only reachable by
    # contradicting the algebra, which is exactly what a synthesiser does.
    frame.loc[:9, "mchc"] = frame.loc[:9, "mchc"] * 1.4
    report = evaluate(as_dataset(frame))

    assert report.stage("exact-identities").survivors == len(frame)
    assert report.identity("mchc").passed == len(frame) - 10


# --- an absent column is not a pass ----------------------------------------

def test_absent_plateletcrit_is_reported_not_counted_as_passing():
    """The failure mode this guards: NHANES has no PCT column.

    A filter that skips a missing identity silently gives NHANES an easier
    test than the legacy data and makes the comparison meaningless.
    """
    frame = coherent_rows().drop(columns=["pct", "mpv"])
    report = evaluate(as_dataset(frame, absent=("pct",)))

    result = report.identity("pct")
    assert not result.is_evaluable
    assert result.passed == 0
    assert "does not report" in result.not_evaluable_reason
    assert "NOT APPLIED" in report.stage("exact-evaluable").detail
    assert any("not evaluable" in n for n in report.notes)


def test_nhanes_declares_plateletcrit_absent():
    assert ABSENT_FROM_NHANES == ("pct",)
    mapped = {c.analyte for c in NHANES_CBC_COLUMNS}
    assert "pct" not in mapped
    assert "mpv" in mapped


# --- plausibility is a separate stage --------------------------------------

def test_plausibility_is_reported_separately_from_the_identities():
    """A row can be perfectly coherent and still impossible.

    Scaling Hb and HCT together with the three columns that follow from them
    leaves every identity exactly satisfied -- MCHC is untouched because both
    of its inputs move -- while putting Hb, HCT, MCV and MCH far outside any
    physiological range. Coherence and possibility are different questions,
    and this row answers them differently.
    """
    frame = coherent_rows(10)
    frame["hb"] = frame["hb"] * 4             # ~48-50 g/dL
    frame["hematocrit"] = frame["hematocrit"] * 4
    frame["mcv"] = frame["mcv"] * 4           # follows HCT
    frame["mch"] = frame["mch"] * 4           # follows Hb
    report = evaluate(as_dataset(frame))

    assert report.identity("mcv").pass_rate == 1.0
    assert report.identity("mch").pass_rate == 1.0
    assert report.identity("mchc").pass_rate == 1.0
    assert report.stage("exact-identities").survivors == 10
    assert report.stage("plausibility").survivors == 0
    assert report.stage("coherent-and-plausible").survivors == 0
    assert report.plausibility["hb"]["violations"] == 10


# --- units are declared, never inferred ------------------------------------

def test_rescale_frame_converts_through_declared_units():
    frame = pd.DataFrame({"WBC_K": [5.8, 0.9]})
    values, units = rescale_frame(
        frame, [Column("WBC_K", "wbc", "thousand/uL", "test")])

    assert units == {"wbc": "cells/uL"}
    assert list(values["wbc"]) == [5800.0, 900.0]


def test_a_wrongly_dimensioned_declaration_raises():
    """A unit of the wrong dimension is caught, not applied."""
    frame = pd.DataFrame({"HB": [13.0]})
    with pytest.raises(IncompatibleUnitsError):
        rescale_frame(frame, [Column("HB", "hb", "fL", "deliberately wrong")])


def test_column_rejects_an_unknown_analyte():
    with pytest.raises(KeyError):
        Column("X", "not_an_analyte", "g/dL", "test")


def test_evaluate_refuses_non_canonical_values():
    """Values must already be canonical; coherence does not rescale."""
    frame = coherent_rows(5)
    dataset = as_dataset(frame)
    dataset.units["platelets"] = "thousand/uL"      # a lie about the column
    with pytest.raises(ValueError, match="canonical"):
        evaluate(dataset)


def test_the_two_registries_agree_where_they_overlap():
    """The dataset registry extends cbc.ANALYTES; it never redefines it."""
    from cbc import ANALYTES

    for key, analyte in ANALYTES.items():
        assert DATASET_ANALYTES[key] is analyte
    # Research-only analytes: needed by the coherence identities (mpv, pct)
    # or by the derived features (rdw, lymphocytes), and deliberately kept
    # out of cbc.ANALYTES because database.REPORT_ANALYTES derives the
    # MedicalReport columns from it.
    assert set(DATASET_ANALYTES) - set(ANALYTES) == {
        "mpv", "pct", "rdw", "lymphocytes",     # CBC identities and features
        "ferritin", "stfr", "crp",              # the iron-status target
    }
