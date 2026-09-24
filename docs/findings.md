# ChemoGuard — the full record

The complete audit trail: evidence tables, resolution notes, superseded
numbers and the reasoning behind each relabelling. `CLAUDE.md` carries the
working summary and is what loads at the start of a session; this file is
what you read when a summary is not enough.

Split out of `CLAUDE.md` on 2026-09-24. **Nothing was deleted in the split** —
every section below is the text that used to sit in `CLAUDE.md`, moved
verbatim. Withdrawn figures are kept with their withdrawal notes attached,
because the reconciliation between a wrong number and its replacement is
itself part of the record.

## Contents

| Section | What it establishes |
|---|---|
| [Finding 1](#finding-1--the-labels-are-threshold-rules-over-the-training-features) | Labels are threshold rules over the training features |
| [Finding 2](#finding-2--roughly-half-the-dataset-is-synthetic) | Roughly half the legacy dataset is synthetic |
| [Finding 3](#finding-3--a-single-1000-on-wbc-and-platelets-with-no-column-rename) | A ÷1000 with no column rename |
| [Finding 4](#finding-4--unit-inference-fails-toward-normal) | Magnitude inference fails toward "normal" — resolved |
| [Finding 5](#finding-5--provenance-the-source-file-and-its-withdrawal) | Provenance could not be established; everything quarantined |
| [Finding 6](#finding-6--a-fifth-failed-identity-and-clamped-extremes) | A fifth failed identity, and clamped extremes |
| [Finding 7](#finding-7--unit-conventions-inferred-at-fit-time-never-persisted) | Unit conventions inferred at fit time — resolved |
| [NHANES ingestion](#nhanes-ingestion--landed-2026-09-21) | The replacement dataset and its unit declarations |
| [Coherence: the staged filter](#coherence-the-staged-filter) | Four stages, and why the old single number was wrong |
| [Labels and features](#labels-and-features--landed-2026-09-21) | WHO 2024 anaemia thresholds, exclusions, assumptions |
| [Finding 8](#finding-8--the-anaemia-hold-out-cannot-work-on-a-cbc-panel) | The anaemia hold-out is not a well-posed task |
| [Iron deficiency](#iron-deficiency--the-replacement-target-landed-2026-09-21) | The replacement target, audited clean |
| [Training the iron classifier](#training-the-iron-classifier--2026-09-21) | AUC 0.8658 ± 0.0139, subgroups, no operating point |
| [Finding 9](#finding-9--the-extraction-benchmark-and-what-it-measured) | The extraction benchmark; M1–M4 |
| [Standing limitation](#standing-limitation--the-unit-boundary-does-not-guarantee-correct-digits) | The unit boundary does not guarantee correct digits |
| [Fixes that look done and aren't](#fixes-that-look-done-and-arent) | Three times the symptom pointed at the wrong cause |
| [Write down why something works](#write-down-why-something-works-not-just-that-it-does) | A property that held by accident, and the near miss |
| [Direction, before the split](#direction--the-work-order-as-it-stood-before-the-split) | The completed work order and why its order was chosen |
| [Hard rules, Environment, Conventions](#hard-rules-environment-and-conventions--as-they-read-before-the-split) | The three sections condensed rather than moved |
| [Resolved issues — archive](#resolved-issues--archive) | Known issues struck off, with what verified them |

---

## Finding 1 — the labels are threshold rules over the training features

Each reconstruction reproduces the published label on **100.00% of 7,196
rows** — exactly, not approximately:

- `anemia_risk`: count of (RBC<3.5, WBC<4.0, PLATELETS<150) → 0=Low, 1=Medium, 2+=High
- `neutropenia_risk`: ANC ≥1500 Low, ≥1000 Medium, else High
- `thrombocytopenia_risk`: Platelets ≥150 Low, ≥75 Medium, else High

A depth-3 decision tree matches or beats the 100-tree Random Forest on every
task:

| Task | RF(100) | Tree(depth 3) | Majority baseline |
|---|---|---|---|
| Anemia (score/flags included, audit variant) | 100.00% | 100.00% | 69.38% |
| Anemia (score/flags dropped) | 99.72% | 99.93% | 69.38% |
| Neutropenia | 100.00% | 99.93% | 92.29% |
| Thrombocytopenia | 100.00% | 100.00% | 90.90% |

The headline "99.93%" sits about 7 points above always answering "Low".

The anemia model never uses `Hb`: High-risk rows average Hb 11.82 and reach
24.8 (polycythaemia), while Low-risk rows include Hb 5.1 (transfusion
threshold).

**Resolved 2026-09-21 — the score and flag columns never reach `.fit()`.**
The feature matrices are hardcoded column selections:

| Model | Line | Features passed to `.fit()` |
|---|---|---|
| Anemia | `train_risk_prediction_models.py:53` | `['RBC', 'WBC', 'PLATELETS']` |
| Neutropenia | `train_risk_prediction_models.py:76` | `['RBC', 'WBC', 'PLATELETS', 'ANC']` |
| Thrombocytopenia | `train_risk_prediction_models.py:104` | `['RBC', 'WBC', 'PLATELETS']` |

Nothing else reaches the estimator. Each `X` is `df[features].copy()`, and the
one `.fit()` in the file (`train_risk_prediction_models.py:206`) is called with
that matrix — standardised for Logistic Regression, raw for the tree models.
`anemia_score`, `RBC_flag`, `WBC_flag` and `PLT_flag` are present as columns in
the CSVs but are never selected.

So this is **"model learns the rule from its inputs"**, not "model is handed
the answer as a feature" — the less egregious of the two defects. It does not
soften Finding 1. Every label is a deterministic threshold function of columns
that sit in its own feature matrix:

- `anemia_risk` thresholds RBC, WBC and PLATELETS — exactly the three features
- `neutropenia_risk` thresholds ANC — a feature
- `thrombocytopenia_risk` thresholds PLATELETS — a feature

100.00% is arithmetic, not learning.

**Table note.** The row now reading "Anemia (score/flags included, audit
variant)" was originally labelled "as trained". It does not describe this
script. It came from an audit run that deliberately trained a variant with
the score and flag columns added, in order to separate the two defects.
Relabelled 2026-09-21 so the table is not read as describing the repo's
actual training path; the numbers are unchanged.

## Finding 2 — roughly half the dataset is synthetic

Identities that hold in any real CBC do not hold here:

| Identity | corr(reported, derived) |
|---|---|
| HCT ≈ 3×Hb | 0.408 |
| MCV = HCT/RBC×10 | 0.206 |
| MCH = Hb/RBC×10 | 0.160 |
| MCHC = Hb/HCT×100 | 0.049 |

**Resolved 2026-09-21 — the staged filter, and the numbers it produces.**
The old figure here ("around 3,000–3,800 rows survive, depending on
tolerance") and Finding 6's ("709 of 7,196") have both been withdrawn. They
measured different things and neither was reproducible; see *Coherence: the
staged filter* below for what replaces them and why the two disagreed.

Under the filter now implemented in `chemoguard/data/coherence.py` —
three exact analyser identities, jointly, at a 2% tolerance — **1,963 of
7,196 legacy rows survive (27.28%)**, and 1,962 (27.27%) survive coherence
*and* plausibility together. The same code on 19,749 NHANES rows leaves
19,747 (99.99%).

The tolerance is the named constant `EXACT_IDENTITY_TOLERANCE`, and
`evaluate()` takes no tolerance argument at all, so it cannot be chosen per
run. It is also not load-bearing: across a 10× sweep from 1% to 10% the
legacy survivor count moves only 1,939 → 2,129 (26.95% → 29.59%). Those rows
are not near-misses that a looser bound would admit; they are unrelated to
their own derived values.

10.6% of rows have WBC >100 thousand/µL (hyperleukocytosis — under 0.1% of any
real population).

**The synthetic rows are interleaved, not appended.** MCHC identity error is
sharply bimodal (p70 = 0.15, p80 = 3.64) and bad indices run from 13 to 7191
throughout the file. This cannot be fixed by truncating or slicing — it needs
row-level filtering.

## Finding 3 — a single ÷1000 on WBC and PLATELETS, with no column rename

Not two sources disagreeing. All three root CSVs derive from one source file
(Finding 5), which stores **cells/µL throughout**. During derivation of
`cbc_with_anemia_risk.csv`, WBC and PLATELETS were divided by 1000 and the
column names were left unchanged.

- `cbc_with_anemia_risk.csv` — WBC and PLATELETS in **thousands/µL**
  (medians 10.99, 297.0)
- `cbc_with_neutropenia_risk.csv`, `cbc_with_thrombocytopenia_risk.csv` —
  **same column names**, **cells/µL** (medians 10990, 297000)

Verified column by column against the source: WBC and PLATELETS differ by a
ratio of exactly 1000.0000 on all 7,196 rows, and the other 14 shared columns
are byte-identical. The unit conversion is the only difference between the
source file and the anemia CSV.

A 1000× clash with no rename. The same defect as the runtime extraction
heuristic, already baked into the training data.

## Finding 4 — unit inference fails toward "normal"

Confirmed concretely: platelets of 900 cells/µL (profound thrombocytopenia)
is read as 900 thousand/µL and returns **Low risk**. Both 404000 cells/µL and
404 thousand/µL also return Low. Every error points critical → normal.

**Resolved 2026-09-21 — the explicit-unit boundary.** All 18 runtime and
training magnitude-inference sites are gone. A CBC value is now a
`units.Quantity` — a number and a `Unit` — from extraction through to
`.predict()`, and `Quantity.to()` is the only conversion in the codebase.

- `units.py` — `Unit`, `Quantity`, the alias table, the error family.
  Symbols are ASCII (`uL`), because unit strings reach the cp1252 console.
- `cbc.py` — the analyte registry, one declared canonical unit each, the
  `CBCPanel`, ANC derivation, and the three accepted request shapes.
- `model_contract.py` — `MODEL_CONTRACTS`, the declared per-model feature
  units, and `features_for`, the single place a Quantity becomes a float.
- `clinical_flags.py` — CTCAE v5.0 grading over canonical values.
- `prediction_log.py` — one JSON record per prediction *and* per refusal.

Canonical units: RBC `million/uL`; WBC, ANC, platelets `cells/uL`; Hb and
MCHC `g/dL`; HCT and neutrophil % `%`; MCV `fL`; MCH `pg`. cells/uL for the
counts keeps 900 cells/µL and 900 thousand/µL distinguishable after
conversion (900 against 900000) instead of collapsing them onto one float,
which is the mechanism of this finding.

**A missing unit is a refusal, not a default.** `MissingUnitError` → HTTP
422, distinct from the 400 for an absent field. An unrecognised unit, a unit
of the wrong dimension, and a value that is physiologically impossible once
converted are all refused the same way. Nothing infers.

Storage renamed rather than reinterpreted: `wbc` → `wbc_cells_per_ul`,
`input_rbc` → `input_rbc_million_per_ul`, and so on, plus
`reported_quantities` holding what the source actually said. Old rows now
fail to load instead of reading 1000× wrong — a 1000× change without a
rename is Finding 3 itself. See `MIGRATION_UNITS.md`; there is no automatic
migration, because the unit of a historical row was never recorded and any
script would have to infer it.

Regression coverage in `tests/test_units.py`, `tests/test_api_units.py`,
`tests/test_prediction_path.py` and `tests/test_report_fixtures.py`. The
whole suite is 116 tests, 3 of which skip without the Tesseract binary. The
load-bearing one asserts that 900 cells/µL and 900 thousand/µL return
*different* risks through the same estimator — under magnitude inference
they produced an identical feature value, so it could not have passed.

The prediction tests install stand-in estimators implementing the threshold
rules of Finding 1, because `models/` is empty and stays so until the NHANES
artefacts land. They exercise the boundary, not the pickles.

## Finding 5 — provenance: the source file, and its withdrawal

`Clinically Validated CBC Dataset of 7,196 Samples/BDCBC7196_Hematology_Dataset.csv`
(7,196 rows × 16 columns, no nulls) is the upstream source of all three root
CSVs:

- `cbc_with_neutropenia_risk.csv` and `cbc_with_thrombocytopenia_risk.csv` are
  row-identical to it on all 16 shared columns, plus their derived labels.
- `cbc_with_anemia_risk.csv` differs only by the exact ÷1000 in Finding 3.

**Nothing in the codebase reads it.** The only CSV loads are the three literal
filenames at `train_risk_prediction_models.py:43-45`. No `glob`, no `listdir`,
no constructed paths anywhere in the Python. The folder was an orphaned raw
download sitting next to its own derivatives.

**Traced upstream:** Mendeley Data, DOI `10.17632/rnfzzy4wz6`, titled
"Clinically Validated CBC Dataset of 7,196 Samples for Hematological Disorder
Analysis". Found via a third-party repo
(`piyalsahaofficial/Hemoglobin-Independent-Anemia-Severity-Prediction`) that
cites it with an access date of 2026-01-12. No README, licence file or source
note ever accompanied the copy in this repo.

**The record has since been withdrawn.** Both the DOI landing page and the
`/1` version URL return:

> Version 1 of the dataset has been removed
> The reason given for this removal was : As per author's request.

Checked 2026-09-21. No reason beyond "author's request" is published, and the
licence, collection methodology, contributing institution and any validation
procedure went down with the record. No mirror or independent citation of the
data itself could be found.

**Provenance could not be established.** We have a DOI and a title and nothing
else. Treat "Clinically Validated" as an unverified claim from a withdrawn
record, not a property of the data: four identity checks fail (Finding 2), a
fifth fails (Finding 6), and the extremes are clamped (Finding 6). Do not
repeat the phrase, and do not cite this dataset as clinically validated in the
writeup or anywhere else.

**Quarantined 2026-09-21 — all four files.** The source file and all three
derived CSVs now live outside the repo at
`C:\Users\tanii\Downloads\chemoguard-withdrawn-dataset\`, retained for audit:

| File | Bytes |
|---|---|
| `Clinically Validated CBC Dataset of 7,196 Samples/BDCBC7196_Hematology_Dataset.csv` | 543,337 |
| `cbc_with_anemia_risk.csv` | 653,977 |
| `cbc_with_neutropenia_risk.csv` | 766,147 |
| `cbc_with_thrombocytopenia_risk.csv` | 672,104 |

Each was SHA-256 matched and re-read at the destination *before* the original
was deleted — 7,196 rows, zero nulls, all 16 shared columns reconciled against
the source (allowing the ÷1000 on the anemia file). No CSV remains anywhere in
the repo.

Removing the source alone achieved nothing: the three derived CSVs carry the
same 7,196 rows, so the data was still in-tree. All four had to go together.

**Nothing trains on these again.** They are unlicensed, their upstream record
is withdrawn, roughly 90% of rows fail joint coherence (Finding 6), and the
labels are threshold rules over the inputs (Finding 1). They are kept solely
so the audit trail is reproducible. Do not copy them back, do not point a
loader at the quarantine path, and do not commit them. No CSV was ever tracked
by git or entered history; `.gitignore` now names all four as a guard against
someone re-downloading them.

**Expected breakage:** `train_risk_prediction_models.py:43-45` now reads three
files that no longer exist and will raise `FileNotFoundError`. This is
intended and should not be patched — that loader gets rewritten against NHANES
under the work order in Direction.

**Models quarantined 2026-09-21.** All fifteen files from `models/` — three
estimators, three label encoders, their timestamped duplicates, and three
metadata `.txt` files — now sit at
`C:\Users\tanii\Downloads\chemoguard-withdrawn-dataset\models\`, alongside
the CSVs. Each was SHA-256 matched and re-unpickled at the destination
before the original was deleted; all twelve `.pkl` files load under the
pinned scikit-learn 1.3.2.

They went because they still worked. The pickles loaded, `/health` reported
`models_loaded: true`, and the app served predictions — so it looked
functional while returning arithmetic over a withdrawn dataset. A broken
artefact announces itself; a working one trained on rule-derived labels does
not. The estimators corroborate Finding 1 on their face: `n_features_in_` is
3, 4 and 3, exactly the feature lists at
`train_risk_prediction_models.py:53`, `:76` and `:104`, and every label is a
threshold rule over those same columns.

`models/` is now empty, and **absent models are the expected state** until
the NHANES replacements land. The app was changed to match:

- `app.py` no longer calls `sys.exit(1)` when loading fails. It starts,
  serves, and prints a notice. Previously `python app.py` would have refused
  to start at all after the quarantine.
- `/health` returns `status: "degraded"`, `models_loaded: false`, and a
  `missing_models` list with an explanation, instead of reporting
  `"healthy"`. It still returns HTTP 200 — the service is up; read the field,
  not the status code.
- `/predict` already returned 503 when models were unavailable, which is the
  right behaviour and was left alone.

`MODELS_README.md` carries a withdrawal notice at the top: its ">99%
accuracy" figures are artefacts of rule-derived labels, and its "trained on
7,196 samples" refers to the withdrawn Mendeley record.

## Finding 6 — a fifth failed identity, and clamped extremes

**PCT is derived, and does not reconcile.** Plateletcrit is
`PCT% = platelets[10^9/L] × MPV[fL] / 10000`, computable from two columns in
the same file:

| Identity | corr(reported, derived) | MAE |
|---|---|---|
| PCT = PLATELETS×MPV/10,000,000 | 0.0863 | 0.926 |

The weakest of the five — worse than MCHC. The split is sharp: relative error
is 0.82% at p50 and 89.06% at p70, so about half the rows satisfy the identity
almost exactly and the rest bear no relation to it. Same interleaving as
Finding 2.

The **derived** values are physiologically sane throughout (p50 0.268, p75
0.333, against a normal plateletcrit of ~0.15–0.35%). The **reported** column
is not: p50 0.31 but p75 2.59 and max 3.00, with 43.9% of rows above the
physiological ceiling and 30 rows pinned at exactly 3.000. The synthesis is in
the reported PCT column, not in the platelet or MPV inputs it should follow
from.

**Clamped extremes — evidence of synthesis independent of the formulas.**
Several columns terminate on round numbers that read as generator bounds
rather than observed values:

| Column | min | max |
|---|---|---|
| Hb | 2.100 | **25.000** |
| MCV | **50.000** | **130.000** |
| MCH | **10.010** | **50.000** |
| MCHC | 20.116 | 39.999 |
| RDW | **11.000** | 29.943 |
| Age | **20** | **69** |
| PCT | 0.014 | **3.000** |

Ceilings landing exactly on 25 / 50 / 130 / 3.000, and MCHC stopping at
39.999, are the signature of a uniform draw over a bounded interval. Age
spanning exactly 20–69 with no paediatric and no geriatric rows is not a
clinical sample. None of this depends on an identity holding, so it is
separate evidence from Finding 2.

**Joint filtering is far more severe than Finding 2 implies.** Requiring all
five identities within 5% leaves **709 of 7,196 rows (9.9%)** — 158 (2.2%) at
1%, 1,524 (21.2%) at 10%. Per-identity pass rates at 5%: HCT 41.9%, MCV 76.0%,
MCH 63.0%, MCHC 77.1%, PCT 51.4%.

**Superseded 2026-09-21.** Those five-identity counts are arithmetically
correct and reproduce exactly — but the filter that produced them is the
wrong filter, for two independent reasons, and the figure should not be
quoted. See *Coherence: the staged filter* below. The caveat this entry
already carried about HCT≈3×Hb turns out to be most of the story: it alone
costs 84% of the survivors.

## Finding 7 — unit conventions inferred at fit time, never persisted

`train_risk_prediction_models.py` decides its unit convention by inspecting
whatever data it is handed, at three sites:

- `train_risk_prediction_models.py:59-66` — anemia: PLATELETS, WBC
- `train_risk_prediction_models.py:82-93` — neutropenia: WBC, PLATELETS, ANC
- `train_risk_prediction_models.py:109-115` — thrombocytopenia: WBC, PLATELETS

Each is the same shape:

```python
if anemia_X['PLATELETS'].mean() > 1000:
    anemia_X['PLATELETS'] = anemia_X['PLATELETS'] / 1000
```

**This is not a train/label mismatch.** Scaling a feature by a constant leaves
label consistency intact — the tree learns the boundary at 1.5 instead of
1500. Training is internally coherent and the reported accuracies are
unaffected by it.

**The defect is that the transformation is data-dependent and unrecorded.**
Whether the division happens is a property of the training set's composition,
not of the column's units. Retrain on a mostly-neutropenic cohort — precisely
the population this tool is aimed at — and the mean falls below the threshold,
no division occurs, and the model learns its boundary at 1500 instead of 1.5.
The artefact is then persisted with no record of which scale it expects:
`save_models()` writes the estimator, the encoder, and a metadata file listing
feature *names* only (`train_risk_prediction_models.py:293`). The scale is
nowhere in the saved state.

Inference then applies whatever convention it applies. When the two disagree
the predictions are wrong by 1000× at the decision boundary, no exception is
raised, and nothing downstream can detect it. Same root cause as Finding 4 —
magnitude taken as evidence of unit — but on the training side, where it is
frozen into a persisted artefact instead of recomputed per request.

For the rebuild:

- The unit convention is part of the model contract. Declare it, do not infer
  it, and persist it with the estimator.
- Retraining on a different cohort must not be able to silently change what a
  persisted model expects.
- This is why the explicit-unit boundary in Direction comes *before* NHANES
  ingestion: the loader has to be written against declared units, or this
  same failure re-enters through the new data path.

**Resolved 2026-09-21, all three points.** The seven `.mean() > 1000` sites
are replaced by `SOURCE_UNITS` — each dataset states the unit of each of its
columns — and `to_contract_units`, which converts to whatever
`model_contract.MODEL_CONTRACTS` declares for that estimator. A dataset with
no declared units raises rather than training. `save_models()` now writes the
feature contract, units included, not just feature names, and `load_models()`
runs `check_estimator`, refusing any artefact whose `n_features_in_`
disagrees with its declared contract. `/health` publishes both the contracts
and a SHA-256 of each loaded artefact, so a mismatch is visible from outside.

The NHANES loader inherits this: it declares its units in `SOURCE_UNITS` or
it does not run.

## NHANES ingestion — landed 2026-09-21

Real data, written against the explicit-unit boundary of Findings 4 and 7.

| Module | What it owns |
|---|---|
| `scripts/fetch_nhanes.py` | CLI: download, cache, join, summarise. Decides nothing. |
| `scripts/coherence_report.py` | The side-by-side run, and the tolerance sweep. |
| `chemoguard/data/nhanes.py` | Cycle definitions, the column map, the unit of every column. |
| `chemoguard/data/analytes.py` | MPV and plateletcrit; the one place a column is rescaled. |
| `chemoguard/data/legacy_audit.py` | Audit-only reader for the quarantined copies. |
| `chemoguard/data/coherence.py` | The staged filter. |

**Cycles and counts.** 2021–August 2023 (`CBC_L` + `DEMO_L`) gives 7,593 rows;
2017–March 2020 pre-pandemic (`P_CBC` + `P_DEMO`) gives 12,156; **19,749
together**, joined to demographics on `SEQN`. 2,750 rows across the two files
carried no CBC result at all — interviewed, never phlebotomised — and are
dropped before any stage, because there is no measurement there to filter.

The 2017–2020 file is not a two-year cycle. NHANES 2019–2020 was cut short by
COVID-19, so CDC combined it with 2017–2018 into one pre-pandemic release with
a `P_` prefix rather than a cycle suffix, and it names its sample weight
`WTMECPRP` where the other names `WTMEC2YR`. Both are declared per cycle in
`CYCLES`, not assumed.

**Units come from the codebook, not from the values.** NHANES states the unit
in the variable label itself — `LBXWBCSI - White blood cell count (1000
cells/uL)` — and each label is quoted verbatim in the `documented_as` field
beside the unit it justifies, so the claim is checkable without rerunning
anything. `python scripts/fetch_nhanes.py --show-units` prints the map.

The map is an explicit dictionary, never pattern matching. `LBXMC` is mean
cell haemoglobin *concentration* (g/dL) and `LBXMCHSI` is mean cell
haemoglobin (pg): one character apart, different dimensions, and exactly the
pair a substring rule silently swaps.

**Two things the loader records rather than papers over.**

- **NHANES does not report plateletcrit.** `ABSENT_FROM_NHANES = ("pct",)`.
  The PCT identity is therefore *not evaluable* on NHANES, which is not the
  same as passing, and the report prints "not reported" rather than skipping
  the check. Any NHANES-vs-legacy comparison is over two identities on one
  side and three on the other; quote it that way. `test_coherence.py` pins
  this — a filter that silently drops a missing identity hands NHANES an
  easier test and makes the whole contrast meaningless.
- **`LBDNENO` is segmented neutrophils only.** NHANES derives it as
  `LBXWBCSI × LBXNEPCT/100`, rounded to one decimal, and does not report
  bands separately. It is a slight underestimate of a clinical ANC. Mapped to
  `anc` because it is the best available, with the discrepancy written down
  at the mapping rather than discovered later.

**For the labelling step that comes next:** NHANES measures CBC from age 1,
and the joined age range is 1–80 years. WHO anaemia cut-offs are age-specific
below 15 and differ again in pregnancy, so `RIDEXPRG` is joined now rather
than rejoined later. This is the opposite problem to the withdrawn dataset,
whose age span was exactly 20–69 (Finding 6).

## Coherence: the staged filter

`chemoguard/data/coherence.py`. Four stages, each with its own reported
survivor count. There is no headline number, because the previous single
figure came from mixing two methods.

**Stage 1 — exact analyser identities, jointly, at 2%.** `MCV = HCT/RBC×10`,
`MCH = Hb/RBC×10`, `PCT = PLATELETS×MPV/10000`. Only these three filter.

**Stage 2 — MCHC, checked and reported, excluded from the filter.** It is
algebraically implied by the two identities already in it:

```
MCH / MCV × 100  =  (Hb/RBC × 10) / (HCT/RBC × 10) × 100  =  Hb/HCT × 100  =  MCHC
```

RBC cancels. Counting MCHC again double-counts the same evidence. Measured,
not just asserted: among legacy rows passing MCV and MCH, **99.15% pass MCHC**
against a marginal rate of 74.96%; on NHANES the conditional rate is 100.00%.
Adding it to the joint filter costs 29 legacy rows for no new information.

**Stage 3 — HCT ≈ 3×Hb, reported at a 15% band, excluded from the filter.**
It is the MCHC identity with MCHC pinned at 33.3: `HCT = Hb/MCHC × 100`, so
`HCT = 3×Hb` exactly when MCHC = 33.3 g/dL and is wrong by however much real
MCHC differs. A bedside rule of thumb, not an analyser identity. At a band
wide enough to be fair to it, 99.90% of NHANES rows pass, so it earns a report
line and not a filter.

Held at the 2% meant for exact identities it would cost **84% of the legacy
survivors** (1,963 → 313) — because MCHC is not 33.3, which is a property of
being alive rather than of being synthetic. Even at its own 15% band it
still removes 210 coherent rows (1,963 → 1,753), which is why it is excluded
outright rather than merely widened.

**Stage 4 — plausibility bounds, separate stage, separate count.** A
different question: an identity asks whether a row is self-consistent, a
bound asks whether a value could be a measurement at all. Bounds are the ones
`cbc.ANALYTES` already declares — the same ones the API enforces on a live
request — extended for MPV and plateletcrit in `chemoguard/data/analytes.py`.

### Results, side by side

2% exact-identity tolerance. All four quarantined files produce identical
numbers, which is the check that their unit declarations are right — including
the ÷1000 on `cbc_with_anemia_risk.csv` that is Finding 3.

| Stage | NHANES (19,749) | Legacy (7,196) |
|---|---|---|
| loaded | 19,749 (100%) | 7,196 (100%) |
| plausibility | 19,749 (100.00%) | 4,711 (65.47%) |
| joint exact identities | **19,747 (99.99%)** | **1,963 (27.28%)** |
| MCHC *(reported only)* | 19,749 (100.00%) | 5,394 (74.96%) |
| HCT≈3×Hb @15% *(reported only)* | 19,730 (99.90%) | 5,642 (78.40%) |
| coherent **and** plausible | 19,747 (99.99%) | 1,962 (27.27%) |

Per-identity pass rates:

| Identity | NHANES | Legacy |
|---|---|---|
| MCV | 99.99% | 75.21% |
| MCH | 100.00% | 61.33% |
| PCT | *not reported* | 51.13% |
| MCHC | 100.00% | 74.96% |
| HCT≈3×Hb @15% | 99.90% | 78.40% |

Like for like — the two identities NHANES can be scored on — the contrast is
**99.99% against 60.8%** (4,378 of 7,196 legacy rows pass MCV and MCH
jointly). The 27.28% figure is over three identities and is the right number
for the legacy data on its own terms; it is not comparable to NHANES.

**The shape of the legacy curve is the finding, not the level.** Real
analyser data saturates at 100% and stops. The legacy pass rate saturates at
75% / 61% / 51% and is flat in the tolerance — 26.95% at 1% to 29.59% at 10%.
A fixed fraction of rows satisfies each identity to the last decimal and the
rest bear no relation to it. That is interleaved synthesis, and it is why no
tolerance choice rescues the dataset.

**Two NHANES rows fail**, both in 2017–2020, and both look like reporting
floors rather than measurements: MCV pinned at exactly 35.4 (the column
minimum) with MCHC at exactly 25.2 (also the column minimum), while their MCH
identity passes to within 0.2%. The filter catching 2 rows in 19,749 — and
catching the right ones — is the evidence that it is calibrated, not lax.

### Why the old numbers disagreed, exactly

Both prior figures are withdrawn. The reconciliation, since Finding 6 asked
for one:

- **Finding 6's 158 / 709 / 1,524 reproduce exactly**, and the method is what
  was wrong with them: all five identities jointly, including the redundant
  MCHC and the rule-of-thumb HCT check at a tolerance meant for exact
  identities. Two of the five were never independent constraints.
- **The one deliberate methodological change** is the denominator.
  `_relative_error` divides by the *derived* value — the reference — where
  Finding 6 divided by the reported one. Dividing by the reported value
  scales the error by the quantity under test, so a row that overstates a
  column is judged more leniently than one that understates it equally.
  Switching the denominator back reproduces Finding 6's counts to the row;
  keeping it on the derived value gives 157 / 683 / 1,392 for that same
  five-identity filter.
- **Finding 2's "3,000–3,800" could not be reproduced by any defensible
  method** and has been removed rather than corrected. Per-identity pass
  counts at 5% span 2,931 (HCT rule) to 5,554 (MCHC); no joint combination
  lands in that range. It appears to have been a per-identity figure quoted
  as though it were a joint one.

### Rules this establishes

- **Tolerances are module constants with a measured justification.**
  `EXACT_IDENTITY_TOLERANCE = 0.02` is set by quantisation, not taste: NHANES
  reports Hb/HCT/MCV/MCH/MCHC to one decimal and RBC to two, so at the low
  end of each column the MCH identity cannot do better than
  `0.05/5.4 + 0.005/2.32 + 0.05/10.2 = 1.63%`. The observed maximum over
  19,749 rows is 1.41%, inside that bound. 2% clears both and rejects
  nothing on rounding alone.
- **`evaluate()` takes no tolerance argument.** A per-run tolerance is what
  made the old number unreproducible; there is no override to pass.
- **An absent column is never a pass.** Reported as not evaluable, and the
  joint filter names which identities it actually applied.
- **Never quote a single survivor count.** Quote the stage.

## Labels and features — landed 2026-09-21

`chemoguard/labels.py`, `chemoguard/features.py`, `chemoguard/leakage.py`,
`scripts/build_labels.py`. Run the whole thing with
`python scripts/build_labels.py`.

**Thresholds are WHO 2024, not 2011.** WHO replaced the 2011 table in March
2024 (*Haemoglobin cutoffs to define anaemia in individuals and populations*,
ISBN 978-92-4-008854-2, Table 3). The **adult bands did not change** — men
15–65 and non-pregnant women 15–65 are exactly the 2011 values. Three things
around them did:

- **6–59 months was split.** 6–23 months has its own lower cutoff (105 g/L,
  down from 110); 24–59 months keeps 110.
- **Pregnancy is trimester-specific**: 110 / 105 / 110 g/L.
- **The adult rows are scoped to ages 15–65.** WHO names haemoglobin in
  over-65s as an open research gap. There is no WHO cutoff for that group.

Thresholds are declared **in g/L, the unit WHO prints them in**, and
converted by `Quantity.to`. Hand-converting a clinical table is a
transcription error waiting to happen.

**Three stated exclusions**, each counted rather than inferred:

| Excluded | Rows | Why |
|---|---|---|
| Age > 65 | 3,707 (18.8%) | Outside the range WHO scopes its own cutoff to. `allow_over_65=True` labels them and marks them `extrapolated`. |
| Pregnant | 113 (0.57%) | 2024 cutoffs are trimester-specific; NHANES publishes `RIDEXPRG` without a trimester, and the choice moves the boundary by 5 g/L — half a severity band. |
| Pregnancy unascertained | 56 | Same reason. |

Plus two recorded assumptions: 3,577 women aged 15–65 have no NHANES
pregnancy record and are treated as not pregnant; and haemoglobin is **not**
adjusted for elevation or smoking, both of which WHO 2024 now recommends.
Neither adjustment is available from the public release as joined — elevation
is restricted-use, smoking is in `SMQ`. Both would *raise* the threshold, so
unadjusted labels under-call anaemia. Every error points from abnormal toward
normal, which is the direction Finding 4 is about.

Result on 19,749 rows: **15,873 labelled** — none 14,555 (91.70%), mild 932
(5.87%), moderate 357 (2.25%), severe 29 (0.18%). Morphology from MCV:
normocytic 17,180, microcytic 2,358, macrocytic 211.

**`green_king` is not the Green & King index.** The published 1989 formula is
`MCV² × RDW / (Hb × 100)`. What is implemented is `RDW × MCV / RBC`, as
specified in the work order — an Hb-free variant over the same three red-cell
inputs. The published form reads haemoglobin and so could not enter this
matrix at all. Flagged because a reviewer reading the name will assume the
1989 formula.

## Finding 8 — the anaemia hold-out cannot work on a CBC panel

Holding haemoglobin out of the anaemia task is not possible while the rest of
the panel is present. This is Finding 1 wearing new clothes: the label is a
threshold rule over something the feature matrix still contains.

Every coherence identity is a route back:

```
MCH  = Hb/RBC × 10    ->  Hb  = MCH × RBC / 10
MCV  = HCT/RBC × 10   ->  HCT = MCV × RBC / 10
MCHC = Hb/HCT × 100   ->  Hb  = MCHC × MCV × RBC / 1000
```

**Fixing the data made this worse, not better.** On the withdrawn dataset
only ~27% of rows satisfied these identities, so reconstruction would have
been noise on three-quarters of the file. On NHANES 19,747 of 19,749 satisfy
them to within 2%, so the reconstruction is essentially exact everywhere.

Measured at three depths of hold-out. Majority-class baseline is **91.82%**,
so the entire available headroom on this task is 8.18 points:

| Profile | Closed-form Hb | Learned R² | Within 2% | Label agreement | Over baseline |
|---|---|---|---|---|---|
| `name-excluded` (17 cols) | **100.00%** | 0.9933 | 98.11% | 98.97% | **+7.16** |
| `reconstruction-safe` (15) | none | 0.9521 | 63.34% | 97.23% | +5.41 |
| `red-cell-free` (11) | none | 0.5130 | 22.68% | 92.47% | +0.65 |

- **Dropping the column name achieves nothing.** `MCH × RBC / 10` recovers
  haemoglobin on 100.00% of rows to within 2%, median error 0.199%. A learned
  model gets MAE 0.059 g/dL and reproduces the WHO severity label 98.97% of
  the time — **87% of all the headroom there is**, purely by arithmetic.
- **Dropping MCH and MCHC is not enough either.** MCHC varies so little
  between people that red cell mass alone predicts haemoglobin: `MCV × RBC`
  still recovers *haematocrit* exactly (99.99%), and the label still comes
  back 97.23% of the time.
- **Breaking it means removing red cell mass entirely** — RBC, MCV, and the
  derived indices that reconstruct MCV from their own definition
  (`mentzer_index × RBC = MCV`). At that point the task is gone: +0.65 points
  over baseline, with mild recall 15.2%, moderate 33.9%, severe 11.1%.

**The same defect applies to the morphology label.** Microcytic/normocytic/
macrocytic is a threshold rule over MCV, and MCV is a feature. Predicting it
is arithmetic. It is still worth *computing* — as a stratifier, or as an
input to a real task — but it is not a target.

**What this means for the work order.** "WHO anaemia severity with Hb/HCT
held out" is not a well-posed ML task on a CBC panel, and no feature
selection rescues it. Computing the label is correct and useful; training a
model to predict it is Finding 1 again. The defensible replacement is a
target the CBC does **not** contain — most directly **iron deficiency**,
which is what `mentzer_index` and Green & King exist to discriminate in the
first place. NHANES publishes ferritin and soluble transferrin receptor for
both loaded cycles (`FERTIN_L` / `TFR_L`, `P_FERTIN` / `P_TFR`), joinable on
the same `SEQN`. Ferritin is not recoverable from a CBC by any identity.

`chemoguard/leakage.py` is the audit: `closed_form` walks the declared
identities, `empirical` fits a model on a train split to catch routes nobody
enumerated, and `label_recovery` asks the only question that matters —
whether the reconstruction reproduces the label, reported against the
majority baseline and with per-class recall, per the hard rule below.

**Rule this establishes:** *a hold-out is a claim about algebra, not about
column names.* Any future held-out target ships with a `leakage.audit` run,
and `RECONSTRUCTION_TOLERANCE` is pinned to the coherence filter's tolerance
so a leak cannot pass a test the data itself would fail.

## Iron deficiency — the replacement target, landed 2026-09-21

`chemoguard/iron.py`, `scripts/build_iron_target.py`. Taken because Finding 8
killed the anaemia target. Ferritin is measured on a separate assay from a
separate aliquot, no CBC identity produces it, and it is the question the
Mentzer and Green & King indices were invented to answer.

**Thresholds are WHO 2020**, *Serum ferritin concentrations for the
assessment of iron status in individuals and populations: technical brief*,
Table 1, reproducing the 2020 guideline
(https://apps.who.int/iris/handle/10665/331505):

| Age | Healthy | With inflammation |
|---|---|---|
| 0–23 months, 24–59 months | <12 µg/L | <30 µg/L |
| 5–<10 y, 10–<20 y, 20–59 y | <15 µg/L | <70 µg/L |
| **60+ years** | **<15 µg/L** | **<70 µg/L** |

**The over-65 exclusion does not carry over.** WHO's anaemia table stops at
65 and calls older adults a research gap; its ferritin table has an explicit
"Older persons (60+ years)" row at the same cut-off. Over-65s are labelled
here, not excluded.

### The NHANES ferritin subsample is the real constraint

Ferritin is **not** run on the examined sample, and the eligibility rule
differs between the two loaded releases. Quoted from the codebooks:

| Cycle | Eligible sample | Ferritin ∩ CBC | Over-65s |
|---|---|---|---|
| 2021–2023 | "1–5 years old and 12–49 years old **females**" | 1,945 (25.6%) | **0** |
| 2017–2020 | "1–5 years old and **12 years and older**" | 10,546 (86.8%) | 1,899 |

The 2021–2023 contribution is 93% female with nobody over 49. Pooling without
saying so builds a cohort whose demographic mix is an artefact of which cycle
a row came from. `nhanes.FERRITIN_ELIGIBILITY_WARNING` travels with the
dataset and `scripts/build_iron_target.py` prints the composition per cycle.
sTfR is worse: a *different*, smaller subsample again, 5,561 rows in total.

### Inflammation

Ferritin is an acute-phase reactant, so a fixed threshold under-calls
deficiency in inflamed people. WHO gives four remedies; this uses the first,
**raising the cut-off for inflamed individuals**, with inflammation defined
as **CRP > 5 mg/L** (WHO's own worked example). The other three are
unavailable or worse here: the arithmetic and BRINDA-regression corrections
both need AGP, which NHANES does not measure, and excluding the inflamed
would bias the cohort toward the healthy in a way a model would inherit
silently. CRP is available on 18,856 of 19,749 rows.

**This choice is large and must be quoted with the prevalence.** It nearly
doubles the measured rate: **7.27% unadjusted against 13.53% adjusted**.
WHO says as much — "the application of different adjustment approaches will
result in a high degree of variability in the estimated prevalence". Both
numbers are in `IronLabelReport.prevalence`.

Rows with no CRP are **excluded, not assumed uninflamed**: the healthy
cut-off is the lower one, so assuming it would under-call deficiency, and
every error would point abnormal → normal. That costs 288 rows.

**Ferritin alone is the primary definition.** sTfR is a reported sensitivity
analysis only: WHO sets no sTfR threshold because the assays are not
standardised against a common reference, published cut-offs span 2.48–6.70
mg/L by method, and requiring it costs 55% of the cohort.
`STFR_ID_THRESHOLD` is 6.70 mg/L — the Roche immunoturbidimetric figure,
matching the assay NHANES states it runs — and is flagged as not a WHO value.

### Cohort, class balance and headroom (pooled — see *Training* for the real cohort)

The figures in this subsection are for the **pooled** cohort, which is now
a sensitivity analysis only. The training cohort is 2017–2020 alone; see
*Training the iron classifier* below.

19,749 rows in, **12,037 labelled**. Exclusions: no ferritin 7,255
(the eligibility subsample), no CRP 288, pregnant 113, pregnancy
unascertained 56. 2,320 labelled rows are inflamed.

| | n | share |
|---|---|---|
| deficient | 1,629 | **13.53%** |
| replete | 10,408 | 86.47% |
| **majority-class baseline** | | **86.47%** |

**Leak audit — the target is clean.** Same table as Finding 8:

| | anaemia (`name-excluded`) | iron deficiency |
|---|---|---|
| closed-form route | **100.00%** | **none exists** |
| learned R² | 0.9933 | **0.1585** |
| within 2% | 98.11% | **2.29%** |
| label agreement | 98.97% | 89.87% |
| majority baseline | 91.82% | 86.88% |
| over baseline | +7.16 | +2.99 |

R² of 0.16 and a within-2% rate of 2.29% — barely above chance — is what a
target outside the panel's arithmetic looks like. `leakage.FERRITIN_PATHS` is
empty **by finding, not by omission**, and a test pins it so that adding a
route later fails loudly.

**Read the +2.99 correctly.** It comes from the audit's
regression-then-threshold path, which is a lower bound, not the ceiling: it
optimises ferritin in µg/L and only then thresholds. A direct classifier does
better, and accuracy is the wrong metric on a 13.5% positive class anyway:

| Model | Accuracy | over baseline | ROC AUC | Recall (deficient) |
|---|---|---|---|---|
| HistGradientBoosting | 89.95% | +3.49 | **0.8708** | 44.17% |
| depth-3 tree | 88.68% | +2.21 | 0.8098 | 31.90% |

**AUC 0.87 is the result**; the accuracy column is nearly all baseline. The
depth-3 leakage test passes — 1.27 points and 0.061 AUC behind the ensemble —
but it passes narrowly on accuracy, so quote the AUC gap, which is the
clearer separation. Recall on the positive class is poor at the default
threshold and will need calibration, not more features.

### Feature renamed

`green_king` was **not** the Green & King index. It is now:

- **`rdw_weighted_mentzer`** — `RDW × MCV / RBC`, the formula previously
  misnamed. Hb-free, so it survives an anaemia hold-out. No published
  provenance; kept for that property alone.
- **`green_king`** — the published 1989 index, `MCV² × RDW / (Hb × 100)`. It
  reads haemoglobin, so it is available on the iron target and barred from
  the anaemia one. That asymmetry is Finding 8 in one line.

## Training the iron classifier — 2026-09-21

`chemoguard/training.py`, `scripts/train_iron.py`. Stratified 5-fold × 5
seeds = **25 fits**, every metric mean ± sd across folds.
HistGradientBoostingClassifier, default hyperparameters — untuned, because
the questions here are about the target and the cohort, not about squeezing
the estimator.

**Cohort: 2017–2020 only.** Not a sample-size call. NHANES ran ferritin on
"1–5 years old and 12–49 years old females" in 2021–2023 and on "1–5 years
old and 12 years and older" in 2017–2020, so the pooled cohort's sex and age
distribution is partly a function of which release a row came from — and sex
and age are *features of this model*, so the confound lands on the inputs
rather than staying in the sampling frame where weighting could reach it.
`TRAINING_CYCLES = ("2017-2020",)`. The pooled fit runs under
`--cycles pooled` and is reported as a sensitivity analysis: it gives AUC
0.8666 ± 0.0080 against 0.8658 ± 0.0139, i.e. the confound buys nothing.

**Both label rules are trained and reported.** `adjust_for_inflammation` is
keyword-only with no default. Neither setting is authoritative: WHO reports
high variability across adjustment methods and BRINDA is unavailable without
AGP, so quoting one alone would overstate what this project knows. CRP is
required either way, so both run on exactly the same 10,181 rows and the
difference between them measures the adjustment and nothing else.

| | adjusted | unadjusted |
|---|---|---|
| labelled | 10,181 | 10,181 |
| deficient | 1,183 (**11.62%**) | 624 (**6.13%**) |
| majority baseline | 88.38% | 93.87% |
| **ROC AUC** | **0.8658 ± 0.0139** | **0.9210 ± 0.0098** |
| accuracy | 0.9102 ± 0.0038 (+2.64 pts) | 0.9532 ± 0.0018 (+1.45 pts) |
| Brier | 0.0705 ± 0.0034 | 0.0377 ± 0.0017 |
| average precision | 0.5991 (no-skill 0.1162) | 0.5996 (no-skill 0.0613) |
| deficient P / R / F1 | 0.697 / 0.405 / 0.512 | 0.699 / 0.418 / 0.522 |
| depth-3 AUC gap | 0.0630 | 0.0575 |

**The unadjusted label is the easier task, not the better one.** Its AUC is
five points higher because it is a pure ferritin threshold, where the
adjusted label folds in inflammation status that a CBC sees only weakly
through the white-cell channel. Do not read 0.92 > 0.87 as "the unadjusted
rule works better" — they are different questions.

Confusion matrix, adjusted, summed over 25 folds (n = 50,905):

| | predicted replete | predicted deficient |
|---|---|---|
| **actual replete** | 43,938 | 1,052 |
| **actual deficient** | 3,519 | 2,396 |

**Depth-3 leakage test passes** on both rules: 0.8658 vs 0.8028 (gap 0.0630)
and 0.9210 vs 0.8635 (gap 0.0575), against a 0.02 margin. The ensemble is
doing more than thresholding, which is exactly what the anaemia label could
not say.

**Calibration is decent and biased one way.** Brier 0.0705 adjusted. The
reliability curve is close to the diagonal below 0.3 and over-predicts above
it — the 0.6–0.8 bins run about 0.045 high. Over-confident at the top end,
which for a screening use would mean the high-probability group is less
enriched than the score implies.

### Subgroup breakdown, with intervals

Out-of-fold, threshold 0.50, adjusted rule. AUC carries a Hanley & McNeil
(1982) 95% interval — **whose width is set by the number of cases, not the
number of rows**:

| Group | n | cases | positive rate | AUC | 95% CI | recall |
|---|---|---|---|---|---|---|
| female | 5,121 | 974 | 0.190 | 0.8490 | [0.833, 0.865] | 0.466 |
| male | 5,060 | 209 | 0.041 | 0.8171 | [0.782, 0.853] | **0.124** |
| age 1–4 | 683 | **37** | 0.054 | 0.7233 | **[0.629, 0.818]** | **0.027** |
| age 5–19 | 1,769 | 276 | 0.156 | 0.8467 | [0.817, 0.876] | 0.431 |
| age 20–59 | 4,862 | 700 | 0.144 | 0.8982 | [0.883, 0.914] | 0.484 |
| age 60+ | 2,867 | 170 | 0.059 | 0.7911 | [0.750, 0.832] | 0.124 |

Which groups are **not** distinguishable from each other (overlapping
intervals):

| Group | overlaps |
|---|---|
| 1–4 | 5–19, 60+, male |
| 5–19 | 1–4, 60+, female, male |
| 20–59 | **separated from all others** |
| 60+ | 1–4, 5–19, male |

Comparing everything against the best-powered band alone would mislead: the
20–59 interval is 0.031 wide, so *everything* separates from it. The
pairwise view is the honest one.

**Correction, 2026-09-21.** This entry previously read "under-5s are
genuinely weak… AUC 0.7233 is materially below every other band". That
overstated it. The under-5 interval is **[0.629, 0.818] — 6.1× wider than
the 20–59 band's** — and overlaps 5–19, 60+ and male. The finding is
*suggestive and underpowered*, not established. 37 cases cannot support the
stronger claim.

What survives:

- **Male and 60+ recall collapse is a threshold artefact, not a
  discrimination failure.** AUC 0.82 and 0.79 against 0.85 for women, with
  overlapping intervals. What differs is prevalence (4.1% and 5.9% against
  19.0%), so a fixed 0.50 cut almost never fires. A per-group operating
  point would largely close this.
- **The under-5 band is underpowered and should be reported separately**,
  not because it is proven worse but because 37 cases cannot show either
  way. Collect more or scope the model above it; do not quote 0.7233 as a
  point estimate without its interval.

### The paediatric MCV hypothesis — tested, not supported

The proposed explanation for the weak under-5 band was that the features
encode adult MCV scaling. Two things wrong with that as first stated, both
recorded because the second is a real negative result:

1. **The premise was imprecise.** The iron matrix feeds *raw continuous*
   MCV; there are no cut-points in the features to be wrong. Only the
   morphology label bands MCV, and morphology is not an iron feature. The
   defensible version is that a tree trained on a mostly-adult cohort places
   its MCV splits where adult data is dense.
2. **Tested directly, and it does not hold.** `mcv_vs_age_lln` — MCV minus
   the age-specific lower limit of normal (`70 + age` below 10, else 80;
   Koerper & Dallman, *J Pediatr* 1976, PMID 956999) — was added as a
   feature and the whole evaluation rerun on the identical cohort, label,
   estimator and seeds.

| Band | without `mcv_vs_age_lln` | with |
|---|---|---|
| OVERALL | 0.8658 ± 0.0139 | 0.8652 ± 0.0135 |
| 1–4 | 0.7233 [0.629, 0.818] | 0.7114 [0.616, 0.807] |
| 5–19 | 0.8467 | 0.8468 |
| 20–59 | 0.8982 | 0.8974 |
| 60+ | 0.7911 | 0.7928 |

The under-5 AUC moves **−0.0119** against an interval half-width of 0.095.
**The limitation is sample size, not age scaling.** Do not spend more effort
on age-normalised red-cell features for this cohort; spend it on cases.

`labels.mcv_lower_limit` and `morphology_for(mcv, age_years=...)` stay,
because the morphology *label* genuinely was mis-scaled for children and
that was worth fixing on its own. Passing no age keeps the adult behaviour,
so nothing changed silently.

### Figures

`scripts/build_figures.py` writes `reports/figures/` — PNG at 150 dpi plus
the CSV each was drawn from, so a number in the write-up traces to a row
without rerunning anything.

| Figure | What it shows |
|---|---|
| `coherence_tolerance_sweep` | The saturation curve. Strongest single piece of audit evidence. |
| `reliability_curve` | Calibration; close below 0.3, over-confident above. |
| `precision_recall_curve` | AP 0.5991 against a 0.1162 no-skill floor. |
| `subgroup_auc` | AUC with Hanley-McNeil intervals. |
| `confusion_matrix` | Threshold 0.50, summed over 25 folds. |

Palette is a validated two-slot categorical set (blue `#2a78d6`, orange
`#eb6834`): all-pairs CVD ΔE 24.7, normal-vision ΔE 33.6. Charts with two
series carry a legend, so identity is never colour alone.

**matplotlib is pinned to 3.8.4**, and the pin is load-bearing: 3.9+
resolves numpy to 2.x, which breaks the numpy 1.x ABI scikit-learn 1.3.2 is
built against, and pandas then fails at import with "numpy.dtype size
changed". Installing it unpinned was tried on 2026-09-21 and did exactly
that.

### No operating point is chosen

`training.threshold_table` reports candidates; nothing selects one. Adjusted
rule, out-of-fold:

| target recall | threshold | recall | precision | FPR | share flagged |
|---|---|---|---|---|---|
| 50% | 0.3582 | 0.500 | 0.621 | 0.040 | 9.4% |
| 60% | 0.2291 | 0.600 | 0.515 | 0.074 | 13.5% |
| 70% | 0.1379 | 0.701 | 0.423 | 0.126 | 19.3% |
| 80% | 0.0709 | 0.801 | 0.324 | 0.220 | 28.7% |
| 90% | 0.0288 | 0.900 | 0.225 | 0.408 | 46.5% |
| 95% | 0.0151 | 0.950 | 0.178 | 0.577 | 62.0% |

Choosing a row trades missed iron deficiency against over-referral. That is
a judgement about relative clinical harm, and **this project is not
positioned to make it** — it is a student exercise on public survey data,
with no clinical validation, no intended-use statement and no care pathway
behind it. The table is the deliverable; the choice is not.

## Finding 9 — the extraction benchmark, and what it measured

Landed 2026-09-22. `chemoguard/extraction_bench/`,
`scripts/build_extraction_corpus.py`, `scripts/run_extraction_bench.py`,
`docs/extraction-benchmark.md`. 162 tests across
`tests/test_extraction_bench.py` (the harness and the scorer),
`tests/test_number_parsing.py` (M1's separator rule and M2's SI units) and
`tests/test_report_fixtures.py` (the real OCR path, including M4's mode).

**Scoped as a characterisation of the regex path plus real OCR, not a
backend comparison.**
The GPT-4o-mini arm is deferred by decision: the adapter, its response cache
and its parsing are complete in `backends.py` and guarded by two independent
switches, because every uncached image is a billable request against the key
in `.env`. **The Tesseract arm ran 2026-09-22**, once the binary was
installed; see *The OCR arm* below. Two arms therefore carry numbers: the
**pattern layer**, the production regex set over the exact text the generator
drew with no OCR in the loop — which is a *ceiling*, not a performance figure
— and the **OCR arm**, the real production path. The difference between them
is glyph reading and nothing else, which is the measurement the whole design
was for.

### The corpus

814 images, 8,140 fields. Deterministic, gitignored, regenerable in 29s.

| Stratum | Images | Fields | extractable | no_unit | absent |
|---|---|---|---|---|---|
| A — the four real images, hand-labelled | 4 | 40 | 29 | 4 | 7 |
| B — notation cross, pristine renders | 108 | 1080 | 720 | 360 | 0 |
| C — degradation over a fixed notation subset | 702 | 7020 | 4554 | 2412 | 54 |

Stratum B is fully crossed over unit scale (cells/uL, K/uL, SI) × unit
placement (column, in the label, absent) × decimal mark (point, comma) ×
reference ranges (present, absent) × 3 panels, so every axis level carries
36–54 images and no per-level comparison is confounded. Stratum C pins the
decimal mark and reference ranges and varies 26 quality variants over all
nine scale-by-placement cells.

`no_unit` fields are the point of the design: the page prints a value and no
unit, **refusing is correct**, and any number emitted is a guess. They can
only score TN or FP, never TP. A metric that scored them otherwise would
punish the behaviour Findings 3, 4 and 7 exist to establish.

### Four defects, each confirmed against the module

**M1 — the value parse had no decimal-mark policy. Fixed 2026-09-22.**
`_NUMBER` was `(?P<value>\d[\d,]*\.?\d*)` finished with
`float(text.replace(",", ""))`: commas stripped, points honoured — the US
convention hard-coded. `6.500` became 6.5 and `0,9` became 9. Recall across
the two decimal styles was **0.867 → 0.233** and precision **1.000 → 0.724**.

Most failed safe on the `cbc.ANALYTES` bounds. **32 did not, and 16 of those
understated a CTCAE grade.** A WBC printed `0,8` — 800 cells/uL, grade 4
leukopenia — read as 8,000 and graded 0. An ANC printed `0,2` read as 2,000
and graded 0. Finding 4's signature through a new route: not magnitude
inference, a separator strip, and pointing the same way — critical to normal.

The fix rests on one fact: **a thousands separator is always followed by
exactly three digits.** So the token shape settles most cases with no
context: no separator is an integer; a separator followed by one, two, or
four-plus digits can only be a decimal mark; two different separators mean
the last is the decimal mark; one separator repeated can only be grouping,
because a number has one decimal mark. Only *separator plus exactly three
digits* is genuinely ambiguous.

That alone resolves all four understating cases deterministically — `0,9`,
`0,8`, `0,2` and `2,0` each have one digit after the separator. A
**document-level second pass** then recovers the rest: a page printing `13.0`
anywhere has shown `.` is its decimal mark and therefore `,` groups its
thousands throughout, so `6,500` resolves from the page's own evidence rather
than from a convention we picked. Contradictory evidence yields no inference
rather than a majority vote. What nothing settles is refused with the printed
token attached — `cbc.AmbiguousNumberError`, the same stance this codebase
takes on a missing unit. `tests/test_number_parsing.py`.

**M2 — the SI unit alternation could not match any SI unit. Fixed
2026-09-22.** `_U_COUNT` built `r'10\^?[369]\s*/\s*[l{0}]+'.format(_MU)` with
`_MU = r'[uµμ]'`. Interpolating a character class inside another gives
`[l[uµμ]]+`, which Python parses as the class `[l[uµμ]` plus a **required
literal `]`**. So `10^9/L]` matched and `10^9/L` did not. Two further defects
in the same expression: `[369]` is one digit, so `10^12/L` — the SI unit for a
red cell count — could not match even with the brackets fixed; and `10\^?`
made the caret optional, so `109/L` matched and was read as an SI unit.

Every count reported in SI lost its unit. It failed safe, and the loss was
**entirely** in the pattern layer: `units.parse_unit` had accepted `10^9/L`
and `10^12/L` all along. SI recall was 0.450 against 0.617 for cells/uL.

The fix splits the micro characters into a bare list so no class is nested,
allows `(?:12|0|3|6|9)` as the exponent, and **requires** an exponent marker —
a caret, `e`, `*`, or a run of superscript digits, which are self-marking. One
case is deliberately given up: OCR flattening a superscript turns `10⁹/L` into
`109/L`, now a refusal. That is the right way round — `109` is a possible
printed value, so the old match was never safe, and a refusal is recoverable
where a wrong scale is not.

**M3 — three gaps stack on `test_4.png`. Open.** Its rows read `RBC Count`,
`WBC Count`, `Platelet Count`, with units `million cells/mcL`, `cells/mcL`,
`/mcL`. Three of four fields are lost at the *first* of three independent
gaps: no `"<abbrev> Count"` label form in `_ANALYTE_LABELS`; then no `mcL` in
`_U_COUNT`; then no `mcL` in the alias table, which is now closed. Fixing gap
1 alone turns them into refusals rather than values. **These three fields are
the only remaining losses in the whole pass.**

**M4 — the page-segmentation mode read almost nothing off a table. Fixed
2026-09-22.** Found the moment Tesseract was installed, and it would have
made every OCR number in this benchmark meaningless. Both OCR call sites
passed `--oem 3 --psm 6`. **psm 6 means "assume a single uniform block of
text"**, which a ruled table is not: Tesseract runs the rows together across
the grid lines. On a pristine, high-contrast generated report at OCR
confidence 90 it returned the two title lines perfectly and, for the entire
ten-row table, `Cr ee` / `rucr ||| ars`.

Preprocessing was ruled out before the mode was. Denoising, Otsu thresholding
and 2× upscaling each change the result by nothing under psm 6, and every
other mode reads 8/8 values with or without them. The mode was the whole
cause.

Replacement chosen by measurement over 24 images spanning the clean control
and the worst level of every degradation family, scored end to end:

| psm | Meaning | Precision | Recall |
|---|---|---|---|
| **3** | fully automatic (**chosen**) | 0.987 | 0.419 |
| 4 | single column, variable sizes | 0.974 | 0.425 |
| 11 | sparse text | 1.000 | 0.229 |
| 12 | sparse text with OSD | 0.977 | 0.235 |
| 6 | single uniform block (**as shipped**) | 1.000 | 0.022 |

psm 3 over 4 on precision and on silent errors, the trade this application
makes everywhere else. Note that the sparse-text modes read *more characters*
and score *worse*: they do not preserve reading order, and the patterns need a
row's label, value and unit on one line. Glyph accuracy is not the objective;
field recovery is.

`PAGE_SEGMENTATION_MODE` is now a named constant carrying that table, and both
call sites read one `_ocr_config()` — they were two separate
`r'--oem 3 --psm 6'` literals, which is how a mode gets corrected in the image
path and left wrong in the PDF path.

Worth recording how close this came to being invisible. The arm would have
run, produced a number, and that number would have been attributed to glyph
reading — which is precisely the inference the benchmark was asked to make. A
one-line configuration default would have been published as the difficulty of
reading laboratory reports.

**Also fixed: the real-OCR test had never run.** `test_report_fixtures.py`
gated its Tesseract test on `pytesseract.get_tesseract_version()` without
setting `tesseract_cmd`, so it checked `PATH` only. `.env` configures the
binary through `TESSERACT_CMD` and the production code reads it — so on a
machine set up the documented way the test skipped silently and had never once
executed. It now honours the variable, runs, and passes.

### Three runs, and what each fix moved

Identical corpus, identical ground truth; only the production code changed.

| | 1. As found | 2. After the alias fix | 3. After M1 + M2 |
|---|---|---|---|
| Precision | 0.948 | 0.948 | **1.000** |
| Recall | 0.622 | 0.622 | **0.997** |
| TP / FP / FN | 578 / 32 / 351 | 578 / 32 / 351 | **926 / 0 / 3** |
| Silent wrong values | 32 | 32 | **0** |
| of which understate severity | 16 | 16 | **0** |
| Alias spellings accepted (of 48) | 28 | 43 | 43 |
| Fields whose outcome changed | — | **0** | +348 recovered |

**The alias fix moved nothing.** Fifteen more spellings parsed and not one
extraction field changed outcome; `alias_gap` was charged zero times in both
runs. A field only reaches `parse_unit` if a pattern matched its label *and*
captured a unit string, so every field the gaps would have touched was already
lost upstream. The alias table was never the binding constraint.

That is the case for stage attribution over a headline number. Recall 0.622
with a list of unrecognised unit spellings beside it reads as a units problem,
and the obvious fix is the wrong one. The fix was right to make — the gaps are
real and on the checked-in images — but it recovers nothing alone.

**M1 and M2 moved everything left.** Both are refusal-preserving: neither
invents a reading. But every number in that table is a **ceiling**, measured
on perfect text. The OCR arm reaches 0.750 on the fields it shares with the
ceiling — see *The OCR arm* below, where a lost decimal point puts the silent
errors back.

`G/L` and `T/L` are left unregistered deliberately. `_normalise` lower-cases,
so `G/L` (a count, 10^9/L) collapses onto `g/L` (a mass concentration).
Separating them needs a case-sensitive exception for one pair of units, and a
wrongly case-folded haemoglobin and a wrongly case-folded white-cell count are
different kinds of harm. Today the dimension check in `Quantity.to` refuses
`G/L` for a count analyte, so it fails safe — at the cost of an error message
naming the wrong reason. A judgement for a person, not for a benchmark.

### The OCR arm — run 2026-09-22, and the gap is glyph reading

Tesseract 5.5.3 installed. Three arms over the same corpus and truth. ~13 min
of compute per OCR arm for 814 images.

| | Pattern (ceiling) | Tesseract psm 3 | Tesseract psm 6 (as shipped) |
|---|---|---|---|
| Stratum A — real images | 1.000 / 0.897 | 1.000 / **0.172** | 1.000 / 0.138 |
| Stratum B — notation cross | 1.000 / 1.000 | 0.995 / 0.767 | n/a / **0.000** |
| Stratum C — degradation | 1.000 / 1.000 | 0.985 / 0.506 | 1.000 / 0.004 |
| Pooled, all 8,140 fields | — | 0.987 / 0.540 | — |
| Silent wrong values | 0 | **38** | 0 |
| of which understate severity | 0 | **13** | 0 |

Precision / recall; the psm 3 column is after the `pL` fix below. Over the
1,390 fields both the ceiling and the OCR arm scored, the gap is recall 0.997
→ 0.750: **229 fields the ceiling read correctly and OCR did not.** Three were
lost by both and belong to M3. None went the other way. Nothing differs
between those two arms but where the text came from, so the gap is glyph
reading and nothing else.

**Stratum A is the worst, at 0.172.** The only non-synthetic images in the
corpus are the hardest thing in it — they are 511-pixel-wide JPEGs, and
resolution is the binding constraint. Generated pristine renders are not a
proxy for a photographed report.

#### One glyph explained most of it: `µ` reads as `p`

From the OCR text of a pristine render at confidence 91:

    Absolute Neutrophil Count (ANC) 3,900 cells/pL 1,500 - 8,000
                                          ^^^^^^^^ the page prints cells/µL

`cells/pL` did not match `_U_COUNT`, so the unit was dropped and the field
refused. Every count analyte prints a microlitre denominator and no other
analyte does, which was the entire per-parameter pattern — the counts scored
0.221–0.305 against 0.670–0.779 for everything else.

**Fixed 2026-09-22.** `pL` is accepted in the microlitre denominator and
rewritten to `uL` at the OCR boundary, *not* registered in `units.py`: a
picolitre is a real unit and `cells/pL` is a misreading rather than a
spelling, and a corruption admitted to the alias table would then be accepted
from the keyboard and from the REST API, where there is no OCR to blame. The
rewrite is scoped to a denominator following a slash, because a blanket
`p`-for-`u` would turn picograms into micrograms and corrupt every MCH. There
is no ambiguity cost: a picolitre is a millionth of a microlitre, so no
laboratory reports a cell count per picolitre.

Measured on the full 814:

| | Before | After |
|---|---|---|
| Stratum B | 0.994 / 0.689 | 0.995 / **0.767** |
| Stratum C | 0.987 / 0.460 | 0.985 / **0.506** |
| Pooled | 0.988 / 0.489 | 0.987 / **0.540** |
| Count-analyte recall | 0.228 | **0.353** |
| — ANC | 0.101 | **0.287** |
| `unit_dropped` losses | 877 | 695 |
| Silent wrong values | 31 | **38** |
| — understating severity | 9 | **13** |

**268 fields recovered, and silent wrong values up by 7.** That second half is
not a side effect to bury. The recovered fields were previously refused for a
*unit* problem that was masking a *digit* problem underneath; raising recall
on a pipeline with no digit check necessarily raises the count of plausible
wrong answers. See *Standing limitation*.

#### Against the upscaling lever — substitutes, not complements

Same stratified 40-image sample, end to end:

| | Recall | Count-analyte recall | Silent |
|---|---|---|---|
| 1× , no `pL` rule | 0.587 | 0.266 | 3 |
| **1× + `pL` rule** | **0.674** | **0.481** | 3 |
| 2× , no `pL` rule | 0.697 | 0.571 | 1 |
| 2× + `pL` rule | 0.697 | 0.571 | 1 |

2× with the rule is identical to 2× without it: upscaling already cuts the
`/pL` misreads from 53 to 7 and leaves the rule nothing to do. **Applying both
buys nothing over either.**

The pattern rule takes most of the gain for no image processing, no runtime
cost and no proximity to the 3× cliff, where recall collapses to 0.084 and
silent errors jump to 52. Upscaling still edges ahead, so the rule is **not
strictly better** — it is cheaper and safer for most of it. Taking the last
~0.09 of count-analyte recall means accepting a preprocessing parameter with a
cliff one step past its optimum, which is an operational-risk judgement, not a
measurement. **Not applied.**

Next glyph in the same sample: `µ` read as `y`, 24 occurrences of `/yL` against
53 of `/pL`. Not handled, deliberately — the `p` rule was scoped to one glyph
so its effect could be attributed, and a second in the same change would have
made both unmeasurable.

#### Degradation is cliff-shaped, not gradual

| Family | Worst level still above half the control | First below | Floor |
|---|---|---|---|
| jpeg | q20 (0.611) | never | 0.611 |
| skew | 4° (0.417) | 5° (0.000) | 0.000 |
| blur | σ0.5 (0.844) | σ1.0 (0.000) | 0.000 |
| contrast | moderate (0.778) | severe (0.222) | 0.222 |
| shadow | diagonal (0.667) | left (0.000) | 0.000 |
| crop | last two rows | never | — |

Pristine control 0.778. JPEG to quality 20 — visibly wrecked to a human eye —
costs little. Half a pixel of blur past σ0.5 costs everything. Anyone tuning
this pipeline on the assumption of graceful degradation would tune the wrong
axis.

Two anomalies, named rather than smoothed over. **Blur σ0.5 scores above the
pristine control** (0.844 vs 0.778): a half-pixel blur anti-aliases the micro
sign and helps — the same resolution story as the upscaling probe. And the
**left-lit shadow is fatal while diagonal and corner are not**, because the
left of the page is where the row labels are; darken those and no pattern
matches, so the row is never sighted.

### The benchmark had a ground-truth bug, and it looked like a finding

Recorded because it is the third time in this repo that a symptom has pointed
at the wrong cause, and because of what it nearly published.

The first OCR run reported nine fields where the backend appeared to **invent
a unit** on a page that printed none — under a crop, which is exactly where a
model under duress might be expected to guess. That reads as a dangerous
hallucination and would have been reported as one.

It was not. `test1.jpg` and the generated column layouts both print
`Neutrophils %` in the row label *and* `%` in the units column — real reports
duplicate it. `corpus.truth_from_render` asked whether the unit *column* had
been clipped, not whether the unit was still readable **anywhere**, so a crop
that took the column marked the field `no_unit`. The unit was still there, in
the label. Tesseract read it correctly and was scored as a false positive for
doing so.

All nine were mine. Fixed by testing for the unit's survival rather than its
column's, with a regression test naming the case.

The general lesson, which is the same one as the `debug=True` and
`UnicodeEncodeError` entries under *Fixes that look done and aren't*: **a
benchmark's own defects present as findings about the thing under test**, and
they present as the *interesting* kind. The safeguard that caught this was
per-field attribution — nine identical failures, all on one analyte, all
under one crop variant, is not what a hallucination looks like. A pooled
accuracy would have hidden it completely.

### Ground truth records meaning, not parseability

A field's label says what the printed unit *denotes* — `cells/mcL` denotes
`cells/uL`, because mcL is a microlitre — independent of what the alias table
accepts. That is the property that made three-way comparison possible at all:
## Standing limitation — the unit boundary does not guarantee correct digits

*Summarised in `CLAUDE.md`, which is where a fresh session should read it.
This is the full text, including the worked example and the reason each
individual guard passes it.*

**This is a limitation of the product, not a finding about a benchmark run,
and it is the first critical→normal defect in this project that cannot be
fixed in code.** Findings 3, 4 and 7 were all code defects with code fixes:
magnitude inference, a `÷1000` with no column rename, unit conventions
inferred at fit time. This one is not.

`units.py` and `cbc.py` guarantee that a value arrives with a **known scale**.
They do not, and cannot, guarantee that the digits are the ones printed on the
page.

A dropped decimal point is the clearest case. OCR reads `0.8` as `08`:

    White Blood Cells (K/uL) 08        ← the page printed "0.8"

`parse_number("08")` is 8.0. True 800 cells/uL is CTCAE grade 4 leukopenia;
read as 8,000 cells/uL it grades **0**. Every existing guard passes it, and
each one passes it for a good reason:

- the **unit** is read correctly, so the explicit-unit boundary is satisfied —
  there is nothing missing or unrecognised to refuse;
- there is **no separator at all**, so the decimal-mark rule (Finding 9 / M1)
  has nothing to act on — `08` is an unambiguous integer to any parser;
- 8,000 cells/uL is a **perfectly normal WBC**, so the `cbc.ANALYTES`
  plausibility bounds accept it. Those bounds are deliberately wide enough to
  admit real pathology — hyperleukocytosis reaches 500,000 — and a bound tight
  enough to catch this would reject genuine readings.

The direction is the one that matters: a lost decimal point reads *high*, and
for a cytopenia high means normal. Measured on the 2026-09-22 OCR run: 38
silent wrong values, 13 of them understating a CTCAE grade, several of those
turning a grade 3 or 4 cytopenia into a grade 0.

**What would actually catch it**, none of which exists here:

1. a **second independent read** that has to agree with the first (two OCR
   engines, or OCR against a vision model) — disagreement becomes a refusal;
2. a **confidence gate** that refuses a field whose per-character OCR
   confidence is below a threshold, rather than accepting whatever came back;
3. **operator confirmation** of the extracted figures against the image before
   a prediction is made.

Until one of those exists, the honest statement of what this application
offers is: *it will not silently change the scale of a value, and it may
silently misread one.* Any intended-use statement, any label, and any
conversation with a reviewer has to say the second half as well as the first.

The three mitigations are ranked by cost, not by preference; (3) is the only
one that needs no new machinery and is the one an assistive tool should
arguably have had from the start. Choosing between them is a clinical-safety
judgement about workflow, not an engineering call — see *Contacts* in the
governance policy rather than settling it in a commit.

the same manifest scored every run.
## Fixes that look done and aren't

Three times now the recorded symptom has not matched the real cause. Check
the mechanism, not the message.

- **`debug=True` was not the only switch.** Replacing `app.run(debug=True)`
  with an env flag does not disable the debugger on its own. `app.py` loads
  `config['development']`, and `DevelopmentConfig` sets `DEBUG = True`, so
  the Werkzeug console stays reachable through `app.config` under
  `flask run` — a path that never touches the `app.run()` call. The fix is
  only real because `app.config['DEBUG']` is now assigned from the same
  `FLASK_DEBUG` flag immediately after `from_object()`. One source of truth.
  A security fix that leaves a second enabling path is not a fix.
- **A console encoding error wearing a model-loading error's face.**
  `load_models()` printed a `✓` from *inside* its `try`, so on the Windows
  cp1252 console the `UnicodeEncodeError` from `print` was caught and
  re-raised as `Error loading anemia model`. Two of three models silently
  never loaded, and the message pointed at joblib. Status prints belong
  outside the try that guards the actual work, and stay ASCII.
- **"Unloadable pickles" was two different failures.** See the sklearn entry
  under Known issues: one model raises, two load, and the one that loads is
  the one to worry about.


## Write down why something works, not just that it does

A companion to *Fixes that look done and aren't*. That section is about a
symptom pointing at the wrong cause; this one is about a property that was
never chosen holding anyway, and what happens next.

Four mutating endpoints once resisted CSRF as a pure side effect: they parsed
with a bare `request.get_json()`, so Flask returned 415 on any non-JSON
content type, which forced a preflight a cross-origin form could not satisfy.
Nobody designed that. It was written down as accidental, in an entry titled
*"Incidental CSRF protection — preserve deliberately, do not rely on it"*,
which named the exact way it would end:

> It disappears the moment someone adds `force=True`, a `request.form`
> fallback, or permissive CORS.

**`routes.py:172` now reads `request.get_json(silent=True)`.** That is the
predicted keyword, in one of the four named endpoints, added within days. Under
the old arrangement it would have removed `/api/predict`'s 415 and the
accidental preflight with it, silently, in a change about JSON parsing that
nobody would have reviewed as a security change. It was safe only because
`CSRFProtect` had landed first — the protection it removed had already been
replaced by protection someone chose.

The near miss is the point, not the outcome. Nothing in the code said that
`get_json()`'s strictness was load-bearing; the property lived in a Markdown
file and in nobody's head. Had the two changes landed in the other order, the
regression would have been invisible: no test would have failed, no behaviour
would have changed for any legitimate caller, and the endpoint would simply
have become reachable cross-origin.

**The rule:** a property that holds by accident is not documented by noting
that it holds. Either make it deliberate — a test, an assertion, an explicit
configuration — or write down *why* it holds and what would end it, next to the
code that provides it rather than in a findings file. This entry got the second
half right and still only survived on timing.

The same reasoning is why `/api/predict` was recorded under *Accepted design
decisions* in `CLAUDE.md` rather than quietly left undecorated: an unauthenticated
route that is deliberate and one that is an oversight look identical in a diff.


## Direction — the work order as it stood before the split

`CLAUDE.md` now carries only the open items. This is the section as it read on
2026-09-24, with the completed steps and the reasoning that closed each one,
kept because the *order* was argued for and the argument is the reusable part.

The one sentence most worth not losing:

> Units come before data ingestion deliberately. Finding 3 shows implicit units
> are already a source of silent corruption; the NHANES loader should be written
> against an explicit-unit boundary rather than retrofitted afterwards.

As it read:

## Direction

Keep the app layer — auth, DB schema, both extraction backends, PDF export,
REST API all work. Replace the dataset and the modelling layer.

**Work order: ~~security~~ → ~~explicit units~~ → ~~NHANES ingestion~~ →
~~labels~~ → ~~pick a target that isn't leaked~~ → ~~training~~ →
extraction benchmark.** Units landed 2026-09-21; see Findings 4 and 7.
Ingestion landed 2026-09-21; see *NHANES ingestion* and *Coherence: the
staged filter*. Labels landed 2026-09-21, and Finding 8 retired the anaemia
target they were built for. The target is iron deficiency from ferritin,
audited clean. Training landed 2026-09-21: **AUC 0.8658 ± 0.0139** on
2017–2020, depth-3 leakage test passed, no operating point selected. See
*Training the iron classifier*.

Open before anything ships, in order:

1. **Per-subgroup operating points, or a documented single-group scope.** A
   fixed 0.50 threshold gives 12% recall in men and 3% in under-5s, and that
   is a threshold artefact on comparable ranking — the fixable one.
2. **The under-5 band needs cases, not features.** 37 cases give an interval
   6× wider than the adult band's, and the age-normalised MCV feature moved
   it −0.012. Report the band separately or scope the model above it.
3. ~~Extraction benchmark (Tesseract vs GPT-4o-mini), still untouched.~~
   Landed 2026-09-22. Pattern-layer ceiling **and** the real OCR arm; the
   GPT-4o-mini arm stays deferred by decision, adapter built and guarded. See
   *Finding 9* and `docs/extraction-benchmark.md`. Four defects found: M1, M2
   and M4 fixed, M3 open in *Known issues*. The headline is that the ceiling
   is now 0.997 and OCR reaches 0.750 on shared fields, so **the image is the
   binding constraint, not the patterns** — and that glyph errors reintroduce
   silent wrong values the unit boundary cannot catch. Next on it: decide
   whether to take the 2× upscale (measured on 40 images, needs a full run),
   and the deferred axes — table layout, non-English labels, `value_cell`
   placement.

Units come before data ingestion deliberately. Finding 3 shows implicit units
are already a source of silent corruption; the NHANES loader should be written
against an explicit-unit boundary rather than retrofitted afterwards.

- ~~Real data: NHANES CBC, joined to demographics on `SEQN`.~~ Done —
  19,749 rows across 2021–2023 and 2017–2020. Load with
  `chemoguard.data.nhanes.load_nhanes()`.
- ~~New tasks: WHO sex-adjusted anemia severity with `Hb`/`HCT` held out;
  morphological subtype from MCV bands~~ — both are threshold rules over
  columns that stay in the matrix, and neither survives Finding 8. Compute
  them; do not model them. Joint multi-output cytopenia has the same defect
  for its anaemia component.
- ~~Candidate replacement target: iron deficiency~~ — taken 2026-09-21 and
  audited clean. See *Iron deficiency* above. The binding constraint is not
  the target, it is that NHANES measures ferritin on a per-cycle eligibility
  subsample.
- Next, in order: calibrate the decision threshold (recall on the deficient
  class is 44% at the default and the class is 13.5% of the cohort); decide
  whether to pool the two cycles or restrict to 2017–2020, which is the only
  one with a representative ferritin sample; then the extraction benchmark.
- ~~Second ML contribution: benchmark the two extraction backends (Tesseract
  vs GPT-4o-mini vision) per field on real report layouts.~~ Landed
  2026-09-22, scoped to the regex path. Next on it, in order: install
  Tesseract and run the OCR arm (the corpus and scorer are ready and the
  arm is one command); decide the decimal-comma policy in *Known issues*;
  then the deferred axes — table layout, non-English labels, and the
  `value_cell` unit placement, which is implemented and unmeasured.


## Hard rules, Environment and Conventions — as they read before the split

These three sections were **condensed** in `CLAUDE.md` rather than moved, so
unlike everything else in this file their live version is the shorter one.
Kept here verbatim so the split deleted nothing.

Two deliberate changes, called out so they are not read as drift:

- The Environment section used to close with "all three pickles load without
  warnings, and `import app` succeeds with all three models loaded". That is no
  longer true and was **not** carried over: the pickles were quarantined on
  2026-09-21 and `models/` is empty. The verification it describes was real on
  the date it was made.
- Conventions read "Python 3.x"; `CLAUDE.md` now says 3.12, which is what the
  Environment section had already established as both floor and ceiling.

As they read:

## Hard rules

- **No bare numbers past the extraction boundary.** A value without a unit is
  a bug. No magnitude-based inference anywhere — `if value > 1000: value /=
  1000` is the exact pattern that produces Finding 4.

  **This rule is about scale, and scale only.** It does not and cannot make a
  value correct — see *Standing limitation* above, where a dropped decimal
  point produces a plausible in-range number that satisfies every guard in
  this list. Do not cite the explicit-unit boundary as evidence that extracted
  figures are right.
- **Every model change ships with the depth-3 leakage test, scored on AUC.**
  If a 3-level tree comes within **0.02 AUC** of the ensemble, the label is a
  rule, not a target. `training.LEAKAGE_AUC_MARGIN`.

  **Changed from accuracy 2026-09-21.** The rule used to read "within 1% of
  the ensemble" on accuracy, which is the wrong scale on an imbalanced
  target: at an 88% majority class the two models were 1.27 accuracy points
  apart — inside fold-to-fold noise — while being 0.063 apart in AUC, which
  is not. Accuracy is compressed against the base rate; AUC is not. The
  margin sits above the measured fold-to-fold AUC sd (~0.014 over 25 folds)
  and well below the gap a learnable target shows.
- **Never report bare accuracy.** Always with the majority-class baseline
  first, plus per-class recall and calibration. The baselines are high (69%,
  92%, 91%) — a model that doesn't clear them by a wide margin has learned
  little.
- **Never commit** `.env`, `venv/`, `uploads/`, `*.pkl`, NHANES caches, or
  the four quarantined CBC CSVs (Finding 5).
- No real patient data in the repo. NHANES and MIMIC are de-identified
  research datasets; anything else is out of bounds.

## Environment

The bundled `venv/` was a macOS venv (`/Users/sulamba/...`, `bin/` not
`Scripts/`, 280 MB) and has been deleted.

`requirements.txt` is now fully pinned and **targets Python 3.12**, which is
both the floor and the ceiling: the previous `Werkzeug==2.2.3` pin calls
`ast.Str`, removed in 3.12, while `scikit-learn==1.3.2` ships wheels only up
to cp312. Flask and Werkzeug were moved to 3.x to clear the first constraint.

scikit-learn is pinned exactly at 1.3.2 because that is the version that
wrote the pickles. That pin protects nothing of value and is free to move
once the NHANES models land — those artefacts were trained on the withdrawn
dataset with rule-derived labels (Findings 1 and 5) and are being deleted.
It is kept only because it works on 3.12 and nothing forces a change today.

Verified on Python 3.12.10: clean install, `pip check` reports no broken
requirements, all three pickles load without warnings, and `import app`
succeeds with all three models loaded.

`chemoguard/data` needs pandas and nothing else. Its output is byte-identical
on the pinned pandas 2.1.4 and on 3.0.6, so the coherence numbers above do not
depend on which is installed. Note that `tests/conftest.py` imports `app` at
module scope, so the whole suite — including `test_coherence.py`, which only
needs pandas — fails at collection without Flask installed.

## Conventions

- Python 3.x, Flask, scikit-learn, pandas
- Tests in `tests/`, run with pytest
- Small reviewable commits; stop for review before large refactors

## Resolved issues — archive

Entries struck off *Known issues*, kept with the text they carried while open
so the record of what was wrong survives the fix. Each carries how it was
verified, because "it was fixed in commit X" is a claim about a commit
message and not about the repo.

**Swept 2026-09-24** against the working tree at `3c3dce5`. Verification for
this sweep was a static read of the live code, not a test run: the bundled
`venv/` was deleted (see *Environment*) and no interpreter on this machine
currently has the dependencies, so `tests/test_security.py` could not be
executed here. The code reads below are conclusive on their own — a decorator
is either on the route or it is not — but they are a weaker check than a
passing suite, and the distinction is recorded rather than glossed.

### Fixed in `c63afc1` — the debug server

Verified 2026-09-24: `app.py:49` reads
`app.config['DEBUG'] = _env_flag('FLASK_DEBUG')`, immediately after
`from_object()` at `app.py:45`. `_env_flag` (`app.py:35-41`) returns `False`
for an unset variable. `app.py:512` is now
`app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)`. Both halves
of the defect are closed — the `app.run()` argument *and* the
`DevelopmentConfig.DEBUG = True` path that the entry below warned was the
part a naive fix would miss. `config.py` still sets `DEBUG = True` on
`DevelopmentConfig`; that is now inert, because line 49 overrides it
unconditionally.

`host='0.0.0.0'` is unchanged and is not by itself the defect — binding all
interfaces without the Werkzeug console is an ordinary development server.

As it read while open:

- `app.py:391` — `debug=True, host='0.0.0.0'`. The Werkzeug console is remote
  code execution on all interfaces. **Line reference corrected 2026-09-21:**
  this entry previously read "`app.py:338` and `app_new.py:391`", as though
  both files were live. They were not. The old `app.py` had a `SyntaxError`
  at line 44 (a bare `elif` with an empty body) and could not be imported or
  run at all; `app_new.py` was the only working entry point, despite being
  the one with the provisional name. `start_app.bat`, `start_app.sh` and
  `FLASK_APP_README.md` all pointed at the dead file. Recorded because the
  misdirection cost real time: every instruction written against this repo
  assumed `app.py` was the application. The dead file has been deleted and
  `app_new.py` renamed to `app.py`; line 391 is the post-rename location.

### Fixed in `c63afc1` — `/upload`, both defects

Verified 2026-09-24: `@login_required` sits at `routes.py:38`, directly above
`def upload_file()` at `:39`. The `user_id = ... else 0` fallback is gone —
`routes.py:57` is a bare `user_id = current_user.id`. CSRF is closed by
`CSRFProtect(app)` at `app.py:52`, with `from flask_wtf.csrf import
CSRFProtect` at `app.py:15` and **no `csrf.exempt` anywhere in the tree**, so
the blueprints registered at `app.py:339-340` are covered.

Both defects were independent and both are closed by different changes, which
is what the entry said would be needed.

As it read while open:

- `routes.py:23` — **`/upload` has two independent defects on one route.**
  1. *No auth check.* The route is undecorated. `routes.py:42` has a
     `user_id = current_user.id if current_user.is_authenticated else 0`
     fallback and the file is written at `routes.py:49` before any identity
     is established. Note the earlier description of this entry was wrong:
     `current_user.id` at `routes.py:74` sits inside
     `if current_user.is_authenticated:` (line 71), so an anonymous request
     does **not** error. It writes the file, runs a **billable OpenAI call**
     (`routes.py:57`), skips the DB block, and returns 200 with the extracted
     values. Only the error path cleans up (60-64), so the file stays on disk
     forever. It is a fully functional unauthenticated endpoint granting
     arbitrary file write plus free use of the OpenAI key — worse than
     "writes then errors", not milder.
  2. *No CSRF protection.* `/upload` takes `multipart/form-data`, which is a
     CORS-simple content type and triggers no preflight. An attacker page can
     make a logged-in victim's browser upload an attacker-chosen file:
     writing to disk, burning the OpenAI key, and creating a `MedicalReport`
     row under the victim's `user_id`.

  The two are independent. Adding `@login_required` does not fix the CSRF
  exposure, and adding CSRF does not fix the missing auth.

### Fixed in `c63afc1` — Flask-WTF declared but never imported

Verified 2026-09-24: imported at `app.py:15`, initialised at `app.py:52`.

As it read while open:

- Flask-WTF is in `requirements.txt` with no CSRF import anywhere.

### Superseded by real CSRF protection — the incidental-protection entry

This entry described protection nobody had chosen, and asked that it be
preserved until something deliberate replaced it. `CSRFProtect` is that
replacement, so the entry is retired: all six mutating endpoints are now
protected by token, including the two the entry named as exposed
(`/api/upload` and `/api/auth/logout`).

**And the thing it predicted has since happened.** `routes.py:172` now reads
`request.get_json(silent=True)`. Under the old arrangement that single
keyword would have removed `/api/predict`'s 415 and with it the accidental
preflight — exactly the "it disappears the moment someone adds `silent=True`"
case. It was added safely only because `CSRFProtect` had already landed.
Worth keeping as the clearest example in this repo of a property that held by
accident, was written down as accidental, and was then relied on and quietly
broken within days.

As it read while open:

- **Incidental CSRF protection — preserve deliberately, do not rely on it.**
  Four of the six mutating endpoints (`/api/auth/register`, `/api/auth/login`,
  `PUT /api/auth/profile`, `/api/predict`) resist CSRF today only as a side
  effect: they parse with a bare `request.get_json()` — no `force=True`, no
  `silent=True`, and no `request.form` fallback anywhere in the codebase — so
  Flask returns 415 on any non-JSON content type, which forces a preflight
  that a cross-origin form cannot satisfy. Nobody chose this. It disappears
  the moment someone adds `force=True`, a `request.form` fallback, or
  permissive CORS. The other two are exposed: `/api/upload` (multipart) and
  `/api/auth/logout` (parses no body at all).

### Moot — the pickle/sklearn compatibility table

The three artefacts this table describes were quarantined on 2026-09-21 to
`C:\Users\tanii\Downloads\chemoguard-withdrawn-dataset\models\` (Finding 5).
Verified 2026-09-24: `models/` is empty. There is no pickle in the repo to
load under any scikit-learn version, so the table describes files that are no
longer here.

**It is moot, not wrong, and the reasoning outlives the artefacts.** The
`scikit-learn==1.3.2` pin in `requirements.txt` exists only because that is
the version that wrote those pickles; with the pickles gone the pin protects
nothing and is free to move (see *Environment*). Keep the finding the table
carried — a `ModuleNotFoundError` stops the app while an
`InconsistentVersionWarning` does not, so the artefact that *loads* is the
dangerous one. The NHANES replacements must not reintroduce the same shape,
which is why `load_models()` runs `check_estimator` against a declared
contract (Finding 7).

As it read while open:

- **Pickle/sklearn compatibility — corrected 2026-09-21.** This entry used to
  read "Pickled models unloadable on newer sklearn". Only two of the three
  are. Verified directly against `models/`, whose artefacts carry a 1.3.2
  version stamp:

  | Model | Estimator | sklearn 1.9.1 | pinned 1.3.2 |
  |---|---|---|---|
  | anemia | RandomForest | **loads, with `InconsistentVersionWarning`** | loads clean |
  | neutropenia | GradientBoosting | `ModuleNotFoundError: sklearn.ensemble._gb_losses` | loads clean |
  | thrombocytopenia | GradientBoosting | same | loads clean |

  The RandomForest case is the dangerous one. A `ModuleNotFoundError` stops
  the app; a warning does not. The anemia model appears to work while
  scikit-learn itself states its results may be invalid — silent-wrong, the
  same shape as Findings 4 and 7. "It loaded" is not "it is correct".

### Moot — the dead scaler code path

Verified 2026-09-24: the reader side still exists — `app.py:96-125` builds a
`models/{model_type}_risk_scaler_latest.pkl` path and loads it if present,
else stores `None`; `predict_risk.py:47-60` does the same. No scaler file was
ever written, and since the quarantine `models/` is empty, so both branches
resolve to `None` on every run.

Moot on two counts rather than one. `predict_risk.py` is itself dead: nothing
imports it. `app.py:200` defines its own `predict_risk` function and that is
what `routes.py:181` imports, so the module's copy of the scaler path is
unreachable code inside an unreachable module.

The writer side (`train_risk_prediction_models.py:304`, `:344`, `:354`,
`:384-389`) writes a scaler only for Logistic Regression, which was never the
selected model — that is why no file was ever produced. That script is being
rewritten against NHANES rather than patched, so the loose end closes with
the rewrite. Until then this is dead code, not a defect.

As it read while open:

- Dead scaler code path — no scaler files were ever written.

### Resolved 2026-09-21 — missing models handled two different ways

- ~~Missing models are handled two different ways~~ — resolved 2026-09-21.
  `initialize_models()` warned and continued on import while the `__main__`
  block called `sys.exit(1)`, so `flask run` started without models and
  `python app.py` did not. Both paths now start; see the quarantine note in
  Finding 5.
