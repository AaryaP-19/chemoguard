# ChemoGuard — project context

Flask + scikit-learn app that extracts values from CBC (complete blood count)
reports and reports haematological risk. Assistive tool only — never present
output as a validated diagnosis.

The ML core is being rebuilt. The findings below were independently verified
against the repo and datasets; treat them as established.

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

Around 3,000–3,800 of 7,196 rows survive coherence and plausibility filtering,
depending on tolerance. **The tolerance must be a named constant in
`coherence.py`, not chosen per run** — otherwise the number isn't reproducible.

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
MCH 63.0%, MCHC 77.1%, PCT 51.4%. Finding 2's "3,000–3,800 rows survive" was
computed over four identities and evidently not jointly. **Reconcile the two
before quoting either number.** Caveat: HCT≈3×Hb is a clinical rule of thumb,
not an exact identity, so a 5% bound on it is strict and drags the joint count
down.

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

---

## Direction

Keep the app layer — auth, DB schema, both extraction backends, PDF export,
REST API all work. Replace the dataset and the modelling layer.

**Work order: security → explicit units → NHANES ingestion → labels →
training → extraction benchmark.**

Units come before data ingestion deliberately. Finding 3 shows implicit units
are already a source of silent corruption; the NHANES loader should be written
against an explicit-unit boundary rather than retrofitted afterwards.

- Real data: NHANES CBC (public, no application; cycle 2021–2023 = `CBC_L`,
  plus 2017–2020), joined to demographics on `SEQN`. Files are XPT —
  `pandas.read_sas`.
- New tasks: WHO sex-adjusted anemia severity with `Hb`/`HCT` held out;
  morphological subtype from MCV bands; joint multi-output cytopenia.
- Second ML contribution: benchmark the two extraction backends (Tesseract vs
  GPT-4o-mini vision) per field on real report layouts.

## Hard rules

- **No bare numbers past the extraction boundary.** A value without a unit is
  a bug. No magnitude-based inference anywhere — `if value > 1000: value /=
  1000` is the exact pattern that produces Finding 4.
- **Every model change ships with the depth-3 leakage test.** If a 3-level
  tree comes within 1% of the ensemble, the label is a rule, not a target.
- **Never report bare accuracy.** Always with the majority-class baseline
  first, plus per-class recall and calibration. The baselines are high (69%,
  92%, 91%) — a model that doesn't clear them by a wide margin has learned
  little.
- **Never commit** `.env`, `venv/`, `uploads/`, `*.pkl`, NHANES caches, or
  the four quarantined CBC CSVs (Finding 5).
- No real patient data in the repo. NHANES and MIMIC are de-identified
  research datasets; anything else is out of bounds.

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

## Known issues not yet fixed

- `train_risk_prediction_models.py:43-45` reads three CSVs that are now
  quarantined (Finding 5), so the script raises `FileNotFoundError`.
  Expected — it gets rewritten against NHANES, not patched.
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
- `routes.py:147` — `/predict` is also unauthenticated. `app.py:357` —
  `/reload_models` is an unauthenticated POST.
- Flask-WTF is in `requirements.txt` with no CSRF import anywhere.
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
- Dead scaler code path — no scaler files were ever written.
- Missing models are handled two different ways: `initialize_models()` at
  `app.py:86` warns and continues on import, but the `__main__` block calls
  `load_models()` and `sys.exit(1)` on failure. So `flask run` starts without
  models and `python app.py` does not.

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

## Conventions

- Python 3.x, Flask, scikit-learn, pandas
- Tests in `tests/`, run with pytest
- Small reviewable commits; stop for review before large refactors
