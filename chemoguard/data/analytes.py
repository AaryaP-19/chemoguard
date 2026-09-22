"""The analytes the dataset layer reads, and the one way it rescales a column.

``cbc.ANALYTES`` is the application's registry: what the API accepts, what the
prediction path uses, and -- via ``database.REPORT_ANALYTES`` -- what columns a
``MedicalReport`` row has. It deliberately does not carry MPV or plateletcrit,
because the app neither asks for them nor stores them.

The coherence checks do need them: the plateletcrit identity
(``PCT = PLATELETS x MPV / 10000``) is one of the five, and it reads two
columns the app never sees. So this module extends the registry for dataset
work only, reusing ``cbc.Analyte`` and the ``units`` objects so that a
canonical unit is still declared exactly once per analyte and in the same
shape. Adding these to ``cbc.ANALYTES`` instead would add two columns to the
``MedicalReport`` schema for values nothing in the app produces.

Column conversion goes through ``Quantity``, never through a literal factor.
``rescale_frame`` computes each column's factor once with
``Quantity(1.0, source).magnitude_in(target)`` -- the same construction as
``train_risk_prediction_models.to_contract_units`` -- and multiplies the column
by it. The factor comes from the two declared units and from nothing else; no
step anywhere inspects a value to decide what it means.
"""

from __future__ import annotations

from dataclasses import dataclass

from cbc import ANALYTES as APP_ANALYTES, Analyte
from units import (
    CELLS_PER_UL,
    FEMTOLITRE,
    MICROGRAM_PER_L,
    MILLIGRAM_PER_L,
    PERCENT,
    THOUSAND_PER_UL,
    Quantity,
    Unit,
    parse_unit,
)

#: Analytes the dataset layer reads that the application does not model.
#:
#: Plausibility bounds follow the convention set in ``cbc.py``: far wider than
#: a reference range, narrow enough that nothing outside them is a real
#: measurement. They are used only by the plausibility stage of
#: ``coherence.py``, which is reported separately from the identity stages.
_RESEARCH_ANALYTES = (
    # Mean platelet volume. Reference range is about 7.5-11.5 fL; the bounds
    # here admit the full range any analyser will report before flagging.
    Analyte("mpv", "Mean Platelet Volume (MPV)", FEMTOLITRE,
            3.0, 30.0, (FEMTOLITRE,)),
    # Plateletcrit: the fraction of blood volume occupied by platelets,
    # reported as a percentage. Reference range is about 0.15-0.40%.
    #
    # The 2.0% ceiling is derived from the two columns PCT follows from,
    # rather than from the declared bounds on those columns. Essential
    # thrombocythaemia at a platelet count of 2,000,000/uL with a normal MPV
    # of 8 fL gives 2000 x 8 / 10000 = 1.6%, and that is already a rare
    # extreme. Composing the *declared* platelet and MPV maxima instead
    # (5,000,000/uL x 30 fL) would permit 15%, which no blood sample reaches
    # -- the two maxima cannot both be attained by one specimen.
    #
    # This bound therefore does real work, and is meant to: it is what flags
    # the reported PCT column of the withdrawn dataset, whose p75 is 2.59 and
    # whose maximum is exactly 3.000 with 30 rows pinned there (CLAUDE.md
    # Finding 6). A PCT that is merely *inconsistent* with its own platelet
    # count still fails on the identity rather than here.
    Analyte("pct", "Plateletcrit (PCT)", PERCENT,
            0.0, 2.0, (PERCENT,)),
    # Red cell distribution width, as the coefficient of variation (RDW-CV),
    # which is what both NHANES and the withdrawn dataset report. Reference
    # range is about 11.5-14.5%; it rises in iron deficiency and in
    # anisocytosis generally. Needed by the Green & King discriminant index.
    Analyte("rdw", "Red Cell Distribution Width (RDW)", PERCENT,
            5.0, 60.0, (PERCENT,)),
    # Absolute lymphocyte count, for the neutrophil-to-lymphocyte and
    # platelet-to-lymphocyte ratios. Bounded like the other leukocyte counts:
    # chronic lymphocytic leukaemia reaches several hundred thousand per uL
    # and that is a real reading, not an error.
    Analyte("lymphocytes", "Absolute Lymphocyte Count", CELLS_PER_UL,
            0.0, 500_000.0, (CELLS_PER_UL, THOUSAND_PER_UL)),
    # --- iron status and inflammation ---------------------------------
    # Not CBC analytes. They are here so that a target built from them
    # declares its units the same way a feature does, and so the
    # plausibility stage bounds them. They are joined only on request; the
    # CBC loaders do not return them.
    #
    # Ferritin's ceiling is deliberately enormous: haemophagocytic
    # lymphohistiocytosis and adult-onset Still's disease reach five and six
    # figures, and those are real measurements that must not be rejected as
    # impossible. NHANES tops out at 5,190.
    Analyte("ferritin", "Serum Ferritin", MICROGRAM_PER_L,
            0.0, 100_000.0, (MICROGRAM_PER_L,)),
    # Soluble transferrin receptor. Assay-dependent -- WHO declines to set a
    # universal cut-off for exactly this reason -- so the bounds here admit
    # any plausible assay's range.
    Analyte("stfr", "Soluble Transferrin Receptor (sTfR)", MILLIGRAM_PER_L,
            0.0, 100.0, (MILLIGRAM_PER_L,)),
    # High-sensitivity C-reactive protein, the inflammation marker the
    # ferritin threshold depends on. Severe sepsis exceeds 500 mg/L.
    Analyte("crp", "High-Sensitivity C-Reactive Protein", MILLIGRAM_PER_L,
            0.0, 1_000.0, (MILLIGRAM_PER_L,)),
)

#: The application's analytes plus the research-only ones. Keys match
#: ``cbc.ANALYTES`` wherever both define one, so a column mapped to ``"hb"``
#: means the same thing in either registry.
DATASET_ANALYTES = dict(APP_ANALYTES, **{a.key: a for a in _RESEARCH_ANALYTES})


def canonical_unit(key: str) -> Unit:
    """The declared canonical unit for a dataset analyte."""
    try:
        return DATASET_ANALYTES[key].canonical
    except KeyError:
        raise KeyError(
            "{0!r} is not a dataset analyte. Known: {1}".format(
                key, ", ".join(sorted(DATASET_ANALYTES))
            )
        ) from None


@dataclass(frozen=True)
class Column:
    """One source column: the analyte it holds and the unit it holds it in.

    ``unit`` is read off the dataset's own documentation and written down
    here. It is never derived from the values in the column.
    """

    source: str
    analyte: str
    unit: str
    #: Where the unit statement came from, quoted, so the claim is auditable.
    documented_as: str = ""

    def __post_init__(self) -> None:
        if self.analyte not in DATASET_ANALYTES:
            raise KeyError(
                "Column {0!r} maps to {1!r}, which is not a dataset analyte."
                .format(self.source, self.analyte)
            )
        parse_unit(self.unit)          # fail at import if the spelling is wrong


def conversion_factor(source, target) -> float:
    """The factor taking a value in ``source`` to one in ``target``.

    One ``Quantity.to`` call, whose result is then reused across a whole
    column. Raises ``IncompatibleUnitsError`` if the dimensions differ, which
    is how a mis-declared unit is caught rather than silently applied.
    """
    return Quantity(1.0, parse_unit(source)).magnitude_in(target)


def rescale_frame(frame, columns, rename=True):
    """Convert declared source columns to their analytes' canonical units.

    ``columns`` is an iterable of ``Column``. Returns ``(frame, units)`` where
    every returned column is in the canonical unit of its analyte and
    ``units`` maps analyte key to that unit's symbol, so the result is never
    a bare number without a recorded unit.

    Columns absent from ``frame`` are skipped and simply do not appear in
    ``units`` -- an absent measurement is absent, not zero and not imputed.
    """
    out, unit_symbols = {}, {}

    for column in columns:
        if column.source not in frame.columns:
            continue
        target = canonical_unit(column.analyte)
        factor = conversion_factor(column.unit, target)
        name = column.analyte if rename else column.source
        out[name] = frame[column.source] * factor
        unit_symbols[name] = target.symbol

    result = frame.__class__(out, index=frame.index)
    return result, unit_symbols
