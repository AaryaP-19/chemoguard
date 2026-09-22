"""Audit-only reader for the four quarantined CBC files.

**This is not a training loader and must never become one.** CLAUDE.md
Finding 5 quarantined the withdrawn Mendeley dataset and its three
derivatives outside the repo, and says plainly: do not copy them back, do not
point a loader at the quarantine path, nothing trains on these again. The same
finding also says they are *retained so the audit trail is reproducible*.

This module is that audit trail and nothing else. It exists so the coherence
report can put the legacy data beside NHANES under one identical set of
checks, which is the evidence for the rebuild. It:

* reads from the quarantine path, never from the repo;
* returns analyte values only, with no label column, so there is nothing here
  to fit against even by accident;
* raises if the files have moved, rather than searching for them.

``LEGACY_SOURCE_UNITS`` declares the unit of every column of every file. The
three derived CSVs share column *names* with the source but not column
*meanings*: ``cbc_with_anemia_risk.csv`` holds WBC and PLATELETS in
thousand/uL where the other three hold cells/uL, with no rename. That is
Finding 3, and it is the reason this table is written out per file instead of
once. Declaring the units correctly makes the four files produce identical
coherence results, which is the check that the declaration is right.
"""

from __future__ import annotations

import os

from .analytes import Column, rescale_frame
from .nhanes import Dataset

#: Finding 5: quarantined outside the repo, retained for audit.
QUARANTINE = os.path.join(
    os.path.expanduser("~"), "Downloads", "chemoguard-withdrawn-dataset"
)

_SOURCE_CSV = os.path.join(
    "Clinically Validated CBC Dataset of 7,196 Samples",
    "BDCBC7196_Hematology_Dataset.csv",
)

#: Columns shared by all four files, with the units they are in everywhere
#: except where a per-file override below says otherwise.
#:
#: Units here are inferred from *provenance*, not from magnitudes: the three
#: derivatives are row-identical to the source on these columns (Finding 5),
#: so whatever the source holds, they hold. The source itself carries no
#: codebook -- the Mendeley record was withdrawn with its documentation
#: (Finding 5) -- which is precisely why the numbers below cannot be treated
#: as documented the way the NHANES ones are. They are a reconstruction.
_COMMON = (
    Column("Hb", "hb", "g/dL", "reconstructed: no codebook survives"),
    Column("RBC", "rbc", "million/uL", "reconstructed"),
    Column("HCT", "hematocrit", "%", "reconstructed"),
    Column("MCV", "mcv", "fL", "reconstructed"),
    Column("MCH", "mch", "pg", "reconstructed"),
    Column("MCHC", "mchc", "g/dL", "reconstructed"),
    Column("MPV", "mpv", "fL", "reconstructed"),
    Column("PCT", "pct", "%", "reconstructed"),
    Column("RDW", "rdw", "%", "reconstructed"),
)

#: Columns read but not mapped: LYMP and MONO (lymphocyte and monocyte
#: *percent*; the analyte is an absolute count, and deriving one would mean
#: multiplying by a WBC column whose unit differs between these four files
#: -- Finding 3 -- for no gain, since nothing trains on this data), PDW
#: (platelet distribution width, reported as either % or fL and undocumented
#: here, so no unit can honestly be declared), Gender, Age, and every derived
#: label and flag.

#: Per file: the columns whose unit is not the common one, plus extras.
LEGACY_SOURCE_UNITS = {
    _SOURCE_CSV: _COMMON + (
        Column("WBC", "wbc", "cells/uL", "reconstructed: source holds cells/uL"),
        Column("PLATELETS", "platelets", "cells/uL", "reconstructed"),
    ),
    # Finding 3: WBC and PLATELETS divided by 1000 during derivation, with
    # the column names left unchanged. The only difference from the source.
    "cbc_with_anemia_risk.csv": _COMMON + (
        Column("WBC", "wbc", "thousand/uL",
               "Finding 3: divided by 1000 with no rename"),
        Column("PLATELETS", "platelets", "thousand/uL",
               "Finding 3: divided by 1000 with no rename"),
    ),
    "cbc_with_neutropenia_risk.csv": _COMMON + (
        Column("WBC", "wbc", "cells/uL", "row-identical to source"),
        Column("PLATELETS", "platelets", "cells/uL", "row-identical to source"),
        Column("ANC", "anc", "cells/uL", "derived in-file from WBC x Neutrophils%"),
        Column("Neutrophils_percent", "neutrophils_percent", "%", "reconstructed"),
    ),
    "cbc_with_thrombocytopenia_risk.csv": _COMMON + (
        Column("WBC", "wbc", "cells/uL", "row-identical to source"),
        Column("PLATELETS", "platelets", "cells/uL", "row-identical to source"),
    ),
}

#: Label and flag columns. Named so the reader can assert it drops them all.
_LABEL_COLUMNS = (
    "anemia_risk", "neutropenia_risk", "thrombocytopenia_risk",
    "anemia_score", "RBC_flag", "WBC_flag", "PLT_flag", "ANC_flag",
    "Platelets_thousands",
)


def available(quarantine=QUARANTINE):
    """The quarantined files that are actually present, in declared order."""
    return [
        name for name in LEGACY_SOURCE_UNITS
        if os.path.exists(os.path.join(quarantine, name))
    ]


def load_legacy(name, quarantine=QUARANTINE):
    """Read one quarantined CSV into canonical units, labels discarded.

    ``name`` must be a key of ``LEGACY_SOURCE_UNITS``. A file with no declared
    units does not load -- the rule that governs training
    (``train_risk_prediction_models.SOURCE_UNITS``) governs the audit path
    too, because a coherence result computed on undeclared units would mean
    nothing.
    """
    import pandas as pd

    try:
        columns = LEGACY_SOURCE_UNITS[name]
    except KeyError:
        raise KeyError(
            "No source units declared for {0!r}. Declare them in "
            "LEGACY_SOURCE_UNITS before reading it.".format(name)
        ) from None

    path = os.path.join(quarantine, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{0} is not at the quarantine path {1}. The quarantined copies are "
            "retained for audit (CLAUDE.md Finding 5); they are not in the "
            "repo and must not be copied back into it.".format(name, quarantine)
        )

    frame = pd.read_csv(path)
    values, units = rescale_frame(frame, columns)

    dropped = [c for c in _LABEL_COLUMNS if c in frame.columns]
    absent = tuple(
        a for a in ("pct", "mpv", "anc", "neutrophils_percent")
        if a not in values.columns
    )

    return Dataset(
        name=os.path.basename(name),
        values=values,
        units=units,
        meta=None,
        absent=absent,
        notes=(
            "quarantined copy, read for audit only; not a training source",
            "units reconstructed from provenance -- the upstream record was "
            "withdrawn with its documentation (Finding 5)",
            "label and flag columns discarded on read: {0}".format(
                ", ".join(dropped) if dropped else "none present"),
        ),
    )


def load_all(quarantine=QUARANTINE):
    """Every quarantined file that is present, each as its own Dataset.

    All four carry the same 7,196 rows (Finding 5), so their coherence
    results should agree exactly. Where they do, the unit declarations above
    are right -- including the divide-by-1000 on the anaemia file, which is
    the only thing standing between it and a wrong answer.
    """
    return [load_legacy(name, quarantine) for name in available(quarantine)]
