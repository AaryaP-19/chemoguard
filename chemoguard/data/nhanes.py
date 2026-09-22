"""NHANES CBC ingestion: explicit column map, explicit units, join on SEQN.

NHANES is public, de-identified, and needs no data-use application. It
replaces the withdrawn Mendeley record of CLAUDE.md Finding 5.

Two things this module refuses to do:

* **Infer a unit from a magnitude.** Every column's unit below is copied from
  the NHANES codebook and recorded in ``documented_as`` alongside it, so the
  claim can be checked against the CDC page without rerunning anything. The
  variable label *is* the unit statement: NHANES writes
  ``LBXWBCSI - White blood cell count (1000 cells/uL)``.
* **Match columns by pattern.** ``NHANES_CBC_COLUMNS`` is an explicit list.
  ``LBXMC`` (mean cell haemoglobin *concentration*, g/dL) against
  ``LBXMCHSI`` (mean cell haemoglobin, pg) is exactly the pair a substring
  rule gets wrong, and the two do not even share a dimension.

Sources, checked 2026-09-21:

* 2021-2023  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt
* 2017-2020  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_CBC.xpt

The 2017-2020 file is the *pre-pandemic* release: NHANES 2019-2020 was cut
short by COVID-19, so CDC combined 2017-2018 with the partial 2019-March 2020
sample into one file carrying a ``P_`` prefix rather than a cycle suffix. It
is not a two-year cycle and its sample weights are named differently
(``WTMECPRP``, not ``WTMEC2YR``); both are declared per cycle below rather
than assumed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .analytes import Column, rescale_frame

#: Where XPT downloads live. Already gitignored; never commit these.
DEFAULT_CACHE = os.path.join("data", "cache")


@dataclass(frozen=True)
class Cycle:
    """One NHANES release, and the two files it takes to build a row."""

    key: str
    label: str
    lab_file: str
    demo_file: str
    #: First year of the release; the segment CDC files the download under.
    year_dir: str
    #: Examination sample weight for this release. Named, not guessed --
    #: the pre-pandemic file does not use the two-year names.
    mec_weight: str
    #: Iron-status and inflammation files, joined only on request.
    ferritin_file: str = ""
    tfr_file: str = ""
    crp_file: str = ""
    #: Who CDC actually ran the ferritin assay on, quoted from the codebook.
    #: This is *not* the examined sample, and it differs between releases.
    #: See ``FERRITIN_ELIGIBILITY_WARNING``.
    ferritin_eligible: str = ""

    def urls(self, stem):
        """Candidate download URLs for a file stem, in preference order.

        CDC reorganised the download paths; the older ``/Nchs/Nhanes/<cycle>/``
        form still resolves. Both are tried so a further reorganisation shows
        up as a clear failure rather than a silent one.
        """
        return (
            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{0}/DataFiles/{1}.xpt"
            .format(self.year_dir, stem),
            "https://wwwn.cdc.gov/Nchs/Nhanes/{0}/{1}.xpt".format(self.key, stem),
        )


#: Ordered: 2021-2023 first, then 2017-2020.
CYCLES = {
    c.key: c
    for c in (
        Cycle("2021-2023", "NHANES 2021-August 2023",
              "CBC_L", "DEMO_L", "2021", "WTMEC2YR",
              "FERTIN_L", "TFR_L", "HSCRP_L",
              "participants 1-5 years old and 12-49 years old females "
              "were eligible"),
        Cycle("2017-2020", "NHANES 2017-March 2020 pre-pandemic",
              "P_CBC", "P_DEMO", "2017", "WTMECPRP",
              "P_FERTIN", "P_TFR", "P_HSCRP",
              "participants 1-5 years old and 12 years and older were "
              "eligible"),
    )
}

#: The join key. One row per examined participant, per release.
SEQN = "SEQN"

#: NHANES CBC variable -> our analyte, with the documented unit.
#:
#: ``documented_as`` is the codebook's own variable label, quoted. Where
#: NHANES writes "1000 cells/uL" we declare ``thousand/uL``, which the alias
#: table in ``units.py`` already recognises as the same thing.
NHANES_CBC_COLUMNS = (
    Column("LBXRBCSI", "rbc", "million/uL",
           "LBXRBCSI - Red blood cell count (million cells/uL)"),
    Column("LBXWBCSI", "wbc", "thousand/uL",
           "LBXWBCSI - White blood cell count (1000 cells/uL)"),
    Column("LBXPLTSI", "platelets", "thousand/uL",
           "LBXPLTSI - Platelet count (1000 cells/uL)"),
    # NHANES derives this itself as LBXWBCSI * LBXNEPCT/100, rounded to one
    # decimal, and says so in the codebook. It counts segmented neutrophils
    # only -- bands are not reported separately -- so it slightly
    # underestimates a clinical ANC. Recorded here, not silently equated.
    Column("LBDNENO", "anc", "thousand/uL",
           "LBDNENO - Segmented neutrophils num (1000 cell/uL)"),
    Column("LBXNEPCT", "neutrophils_percent", "%",
           "LBXNEPCT - Segmented neutrophils percent (%)"),
    Column("LBXHGB", "hb", "g/dL",
           "LBXHGB - Hemoglobin (g/dL)"),
    Column("LBXHCT", "hematocrit", "%",
           "LBXHCT - Hematocrit (%)"),
    Column("LBXMCVSI", "mcv", "fL",
           "LBXMCVSI - Mean cell volume (fL)"),
    Column("LBXMCHSI", "mch", "pg",
           "LBXMCHSI - Mean cell hemoglobin (pg)"),
    # Concentration, not content. One character from LBXMCHSI and a
    # different dimension.
    Column("LBXMC", "mchc", "g/dL",
           "LBXMC - Mean Cell Hgb Conc. (g/dL)"),
    Column("LBXMPSI", "mpv", "fL",
           "LBXMPSI - Mean platelet volume (fL)"),
    Column("LBXRDW", "rdw", "%",
           "LBXRDW - Red cell distribution width (%)"),
    # Absolute lymphocyte count. NHANES derives it the same way it derives
    # LBDNENO, as LBXWBCSI x LBXLYPCT/100 rounded to one decimal, so it
    # shares that column's null pattern: present only where the five-part
    # differential ran.
    Column("LBDLYMNO", "lymphocytes", "thousand/uL",
           "LBDLYMNO - Lymphocyte number (1000 cells/uL)"),
)

#: Dataset analytes NHANES does not report at all.
#:
#: Plateletcrit is not in the NHANES CBC file. The panel is a Beckman Coulter
#: DxH 800 five-part differential, which computes PCT, but CDC does not
#: release it. So the PCT identity is *not evaluable* on NHANES -- which is
#: not the same as passing, and the coherence report says so rather than
#: quietly filtering on fewer identities than the legacy data gets.
ABSENT_FROM_NHANES = ("pct",)

#: Present in the file but deliberately unmapped: LBXNRBC (nucleated RBC,
#: reported per 100 WBC -- a ratio to a count, which is not a unit this
#: module models), and the monocyte, eosinophil and basophil channels of the
#: five-part differential. None takes part in an identity check and none is a
#: model feature. Left out rather than given a canonical unit nothing
#: consumes. LBXLYPCT is also unmapped: LBDLYMNO already carries the same
#: information as an absolute count, which is what the ratios need.

#: Demographic variables, joined on SEQN. Not analytes: these carry no
#: laboratory unit, so they travel beside the canonical frame rather than
#: through ``rescale_frame``. Age names its unit in the column, following the
#: storage convention of CLAUDE.md Finding 4.
NHANES_DEMOGRAPHIC_COLUMNS = {
    "RIAGENDR": "sex_code",
    "RIDAGEYR": "age_years",
    "RIDRETH3": "race_ethnicity_code",
    # 1 = pregnant at exam. WHO anaemia cut-offs differ in pregnancy, so this
    # is carried now for the labelling step rather than rejoined later.
    "RIDEXPRG": "pregnancy_code",
}

#: NHANES codings, for the labelling step. Written down, not inferred.
SEX_LABELS = {1: "male", 2: "female"}
PREGNANCY_LABELS = {1: "pregnant", 2: "not pregnant", 3: "cannot determine"}


# ---------------------------------------------------------------------------
# Iron status and inflammation
# ---------------------------------------------------------------------------

#: Iron-status and inflammation columns, with their documented units.
#:
#: NHANES publishes ferritin twice, LBXFER in ng/mL and LBDFERSI in ug/L, and
#: states the conversion factor is 1.00 -- they are the same number because a
#: nanogram per millilitre *is* a microgram per litre. We read LBDFERSI, the
#: SI column; the alias table knows both spellings anyway.
#:
#: sTfR is the opposite trap. The ``SI`` suffix means nmol/L here, **not**
#: mg/L: LBDTFRSI has median 37.3 against LBXTFR's 3.16, a ratio of 11.8,
#: which is the ~85 kDa molar mass. Every published sTfR cut-off is written
#: in mg/L, so we read LBXTFR. Taking ``SI`` to mean "the one we want",
#: as we do for ferritin, would be wrong by an order of magnitude here --
#: which is why each column is declared rather than matched on a suffix.
NHANES_IRON_COLUMNS = (
    Column("LBDFERSI", "ferritin", "ug/L",
           "LBDFERSI - Ferritin (ug/L)"),
    Column("LBXTFR", "stfr", "mg/L",
           "LBXTFR - Transferrin receptor (mg/L)"),
    Column("LBXHSCRP", "crp", "mg/L",
           "LBXHSCRP - HS C-Reactive Protein (mg/L)"),
)

#: Which file each iron column comes from, by cycle attribute name.
_IRON_SOURCE_FILE = {
    "ferritin": "ferritin_file",
    "stfr": "tfr_file",
    "crp": "crp_file",
}

#: The single most important caveat about an NHANES iron-status cohort.
#:
#: Ferritin is not run on the examined sample. It is run on an eligibility
#: subsample, and **the subsample is defined differently in the two releases**
#: (see ``Cycle.ferritin_eligible``). In 2021-2023 it is children 1-5 and
#: females 12-49 only -- 93% female, nobody over 49. In 2017-2020 it is
#: everybody from 12 up, plus children 1-5.
#:
#: Pooling the two without saying so builds a cohort whose demographic mix is
#: an artefact of which cycle a row came from. ``iron_cohort`` reports the
#: composition per cycle and will restrict to one release on request.
FERRITIN_ELIGIBILITY_WARNING = (
    "NHANES ferritin is measured on a per-cycle eligibility subsample, not on "
    "the examined sample; the two loaded releases use different subsamples."
)


def load_iron_markers(cycle, cache_dir=DEFAULT_CACHE, refresh=False):
    """Ferritin, sTfR and CRP for one cycle, canonicalised, indexed by SEQN.

    Returns ``(values, units)``. Each marker comes from its own file, so a
    participant may have one and not another -- sTfR in particular is a
    smaller subsample than ferritin in 2017-2020.
    """
    import pandas as pd

    if not isinstance(cycle, Cycle):
        cycle = CYCLES[cycle]

    frames, units = {}, {}
    for column in NHANES_IRON_COLUMNS:
        stem = getattr(cycle, _IRON_SOURCE_FILE[column.analyte])
        if not stem:
            continue
        table = _read_xpt(download(stem, cycle, cache_dir, refresh))
        if column.source not in table.columns:
            raise KeyError(
                "{0}.xpt has no {1}; the codebook changed.".format(
                    stem, column.source)
            )
        values, unit = rescale_frame(table, [column])
        frames[column.analyte] = values[column.analyte]
        units.update(unit)

    return pd.DataFrame(frames), units


def with_iron_markers(dataset, cache_dir=DEFAULT_CACHE, refresh=False):
    """Return a copy of a stacked NHANES Dataset with iron markers joined.

    Joins on the ``(cycle, SEQN)`` index the stacked loader produces, so a
    SEQN reused across releases cannot pick up the wrong participant's
    ferritin.
    """
    import pandas as pd

    per_cycle, units = [], {}
    for name in dataset.values.index.get_level_values("cycle").unique():
        key = str(name).replace("NHANES ", "")
        values, cycle_units = load_iron_markers(key, cache_dir, refresh)
        values = pd.concat([values], keys=[name], names=["cycle", SEQN])
        per_cycle.append(values)
        units.update(cycle_units)

    markers = pd.concat(per_cycle).reindex(dataset.values.index)
    combined = pd.concat([dataset.values, markers], axis=1)

    return Dataset(
        name=dataset.name + " + iron markers",
        values=combined,
        units=dict(dataset.units, **units),
        meta=dataset.meta,
        absent=dataset.absent,
        notes=tuple(dataset.notes) + (FERRITIN_ELIGIBILITY_WARNING,),
    )


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset: canonical values, their units, and its provenance.

    ``values`` holds one column per analyte, each in that analyte's canonical
    unit. ``units`` records which unit that is, per column, so no consumer
    ever has to assume. ``absent`` names analytes the source does not report
    at all, which is distinct from analytes present and null.
    """

    name: str
    values: object
    units: dict
    meta: object = None
    absent: tuple = ()
    notes: tuple = field(default_factory=tuple)

    def __len__(self):
        return len(self.values)


def cached_path(stem, cache_dir=DEFAULT_CACHE):
    return os.path.join(cache_dir, stem + ".xpt")


def download(stem, cycle, cache_dir=DEFAULT_CACHE, refresh=False):
    """Fetch one XPT into the cache, returning its path.

    Skips the network entirely when the file is already cached, so reruns
    cost nothing and the survivor counts stay reproducible offline.
    """
    import urllib.error
    import urllib.request

    path = cached_path(stem, cache_dir)
    if os.path.exists(path) and not refresh and os.path.getsize(path) > 0:
        return path

    os.makedirs(cache_dir, exist_ok=True)
    errors = []
    for url in cycle.urls(stem):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
        except (urllib.error.URLError, OSError) as exc:
            errors.append("{0}: {1}".format(url, exc))
            continue
        # An XPT begins with a fixed 80-character library header. Anything
        # else is an error page that arrived as HTTP 200 -- writing it would
        # cache a file that fails later, somewhere less obvious.
        if not payload.startswith(b"HEADER RECORD"):
            errors.append("{0}: not an XPT ({1} bytes starting {2!r})"
                          .format(url, len(payload), payload[:32]))
            continue
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    raise RuntimeError(
        "Could not download {0}.xpt. Tried:\n  {1}".format(stem, "\n  ".join(errors))
    )


def _read_xpt(path):
    import pandas as pd

    frame = pd.read_sas(path, format="xport")
    if SEQN not in frame.columns:
        raise KeyError("{0} has no {1} column; cannot join.".format(path, SEQN))
    frame[SEQN] = frame[SEQN].astype("int64")
    return frame.set_index(SEQN)


def load_cycle(cycle, cache_dir=DEFAULT_CACHE, refresh=False):
    """Download if needed, join CBC to demographics on SEQN, canonicalise.

    Returns a ``Dataset`` whose ``values`` carry only analytes, in canonical
    units, and whose ``meta`` carries demographics and the sample weight.
    """
    import pandas as pd

    if not isinstance(cycle, Cycle):
        cycle = CYCLES[cycle]

    lab = _read_xpt(download(cycle.lab_file, cycle, cache_dir, refresh))
    demo = _read_xpt(download(cycle.demo_file, cycle, cache_dir, refresh))

    missing = [c.source for c in NHANES_CBC_COLUMNS if c.source not in lab.columns]
    if missing:
        raise KeyError(
            "{0} is missing declared columns {1}. The codebook changed; update "
            "NHANES_CBC_COLUMNS rather than falling back to pattern matching."
            .format(cycle.lab_file, ", ".join(missing))
        )

    values, units = rescale_frame(lab, NHANES_CBC_COLUMNS)

    meta = pd.DataFrame(index=lab.index)
    for source, name in NHANES_DEMOGRAPHIC_COLUMNS.items():
        meta[name] = demo[source] if source in demo.columns else None
    meta["sex"] = meta["sex_code"].map(SEX_LABELS)
    meta["cycle"] = cycle.key
    # Declared per cycle: the pre-pandemic release does not use WTMEC2YR.
    meta["mec_weight"] = (
        demo[cycle.mec_weight] if cycle.mec_weight in demo.columns else None
    )
    meta["mec_weight_variable"] = cycle.mec_weight
    meta = meta.reindex(values.index)

    # Participants with a row in the lab file but no result at all gave no
    # blood. Dropping them is not filtering: there is no measurement to
    # filter, and leaving them in would understate every pass rate below.
    measured = values.notna().any(axis=1)
    dropped = int((~measured).sum())
    values, meta = values[measured], meta[measured]

    return Dataset(
        name="NHANES {0}".format(cycle.key),
        values=values,
        units=units,
        meta=meta,
        absent=ABSENT_FROM_NHANES,
        notes=(
            "{0}; lab {1}.xpt + demographics {2}.xpt, joined on SEQN".format(
                cycle.label, cycle.lab_file, cycle.demo_file),
            "{0} rows carried no CBC result and were dropped before any "
            "stage".format(dropped),
        ),
    )


def load_nhanes(cycles=None, cache_dir=DEFAULT_CACHE, refresh=False):
    """Load and stack cycles into one Dataset.

    The stacked index is ``(cycle, SEQN)``. SEQN is unique within a release
    but is reused across releases, so concatenating on SEQN alone would
    silently collide.
    """
    import pandas as pd

    keys = list(CYCLES) if cycles is None else list(cycles)
    loaded = [load_cycle(k, cache_dir, refresh) for k in keys]

    units = {}
    for dataset in loaded:
        units.update(dataset.units)

    values = pd.concat(
        [d.values for d in loaded], keys=[d.name for d in loaded],
        names=["cycle", SEQN],
    )
    meta = pd.concat(
        [d.meta for d in loaded], keys=[d.name for d in loaded],
        names=["cycle", SEQN],
    )

    notes = []
    for dataset in loaded:
        notes.extend("{0}: {1}".format(dataset.name, n) for n in dataset.notes)

    return Dataset(
        name="NHANES " + "+".join(keys),
        values=values,
        units=units,
        meta=meta,
        absent=ABSENT_FROM_NHANES,
        notes=tuple(notes),
    )
