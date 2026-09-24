# ChemoGuard — project context

Flask + scikit-learn app that extracts values from CBC (complete blood count)
reports and reports haematological risk. Assistive tool only — never present
output as a validated diagnosis.

**The full record is [`docs/findings.md`](docs/findings.md)** — evidence tables,
resolution notes, superseded numbers, the relabelling explanations. Each finding
below stands in for several pages there. Read it before quoting a number or
reopening a question. All findings were independently verified against the repo
and datasets.

---

## Standing limitation — the unit boundary does not guarantee correct digits

**A limitation of the product, not of a benchmark run, and the first
critical→normal defect here that cannot be fixed in code.** Findings 3, 4 and 7
were code defects with code fixes; this one is not.

`units.py` and `cbc.py` guarantee a value arrives with a **known scale**. They
cannot guarantee the digits are the ones printed on the page. OCR reads `0.8` as
`08`; `parse_number("08")` is 8.0; true 800 cells/uL is CTCAE grade 4
leukopenia, and 8,000 cells/uL grades **0**. Every guard passes it for a good
reason — the unit *is* read correctly, there is no separator for the
decimal-mark rule to act on, and 8,000 cells/uL is a normal WBC well inside
bounds that must stay wide enough to admit real pathology. The direction is what
matters: a lost decimal point reads *high*, and for a cytopenia high means
normal. On the 2026-09-22 OCR run: 38 silent wrong values, 13 understating a
CTCAE grade.

What would catch it, none of which exists here: (1) a second independent read
that must agree, (2) a per-character OCR confidence gate, (3) operator
confirmation against the image before prediction. Ranked by cost, not
preference; (3) needs no new machinery. Choosing is a clinical-safety judgement
about workflow, not an engineering call.

**The honest statement of what this offers:** *it will not silently change the
scale of a value, and it may silently misread one.* Any intended-use statement,
any label, and any conversation with a reviewer has to say the second half too.
→ [full text](docs/findings.md#standing-limitation--the-unit-boundary-does-not-guarantee-correct-digits)

---

## The findings, one paragraph each

**1 — labels are threshold rules over the training features.** Each legacy label
reconstructs from its own features on 100.00% of 7,196 rows, and a depth-3 tree
matches or beats the 100-tree Random Forest on every task, against majority
baselines of 69/92/91%. 100.00% is arithmetic, not learning.
→ [detail](docs/findings.md#finding-1--the-labels-are-threshold-rules-over-the-training-features)

**2 — roughly half the legacy dataset is synthetic.** Identities that hold in
any real CBC do not (HCT≈3×Hb r=0.408, MCHC r=0.049), and synthetic rows are
**interleaved, not appended**, so only row-level filtering helps. 1,963 of 7,196
survive the current filter (27.28%) against 19,747/19,749 on NHANES.
→ [detail](docs/findings.md#finding-2--roughly-half-the-dataset-is-synthetic)

**3 — a single ÷1000 on WBC and PLATELETS, no column rename.** Not two sources
disagreeing: all three root CSVs derive from one file storing cells/µL, and two
columns were divided by 1000 with the names left alone.
→ [detail](docs/findings.md#finding-3--a-single-1000-on-wbc-and-platelets-with-no-column-rename)

**4 — unit inference failed toward "normal". Resolved 2026-09-21.** 900 cells/µL
read as 900 thousand/µL and returned Low risk. All 18 magnitude-inference sites
are gone; a value is a `units.Quantity` from extraction to `.predict()`, and **a
missing unit is a refusal, not a default** (422).
→ [detail](docs/findings.md#finding-4--unit-inference-fails-toward-normal)

**5 — provenance could not be established; everything quarantined.** The
Mendeley record (DOI `10.17632/rnfzzy4wz6`) is **withdrawn at the author's
request**, taking licence, methodology and any validation with it. Do not repeat
"Clinically Validated"; do not cite it as such. All four CSVs and all fifteen
files from `models/` sit at `C:\Users\tanii\Downloads\chemoguard-withdrawn-dataset\`,
SHA-256 matched before deletion, kept only so the audit reproduces. **Absent
models are the expected state** until the NHANES artefacts land.
→ [detail](docs/findings.md#finding-5--provenance-the-source-file-and-its-withdrawal)

**6 — a fifth failed identity, and clamped extremes.** Plateletcrit does not
reconcile (r=0.086), and columns terminate on round numbers reading as generator
bounds (Hb max 25, MCV 50–130, PCT max 3.000, Age exactly 20–69) — evidence
independent of any identity. Its survivor counts are **superseded**.
→ [detail](docs/findings.md#finding-6--a-fifth-failed-identity-and-clamped-extremes)

**7 — unit conventions inferred at fit time, never persisted. Resolved
2026-09-21.** Seven `if col.mean() > 1000` sites set the convention from the
cohort's composition and saved artefacts with no record of the scale. Now
`SOURCE_UNITS` + `to_contract_units`; undeclared units raise, `load_models()`
runs `check_estimator`, `/health` publishes contracts and artefact hashes.
→ [detail](docs/findings.md#finding-7--unit-conventions-inferred-at-fit-time-never-persisted)

**8 — the anaemia hold-out cannot work on a CBC panel.** `MCH × RBC / 10`
recovers Hb on 100.00% of rows and reproduces the WHO label 98.97% of the time —
87% of available headroom, by arithmetic. **Fixing the data made this worse**,
since on NHANES the reconstruction is near-exact. Breaking it means removing red
cell mass entirely, and then the task is gone (+0.65 over baseline). Compute the
label; do not model it. Same for MCV morphology.
→ [detail](docs/findings.md#finding-8--the-anaemia-hold-out-cannot-work-on-a-cbc-panel)

**9 — the extraction benchmark.** 814 images, 8,140 fields; the regex path plus
real OCR, not a backend comparison. **M1** (no decimal-mark policy — `0,8` read
as 8,000, 16 fields understating a CTCAE grade), **M2** (a nested character class
made SI units unmatchable) and **M4** (`--psm 6` read almost nothing off a ruled
table, and would have made every OCR number meaningless) are fixed; **M3** is
open below. Ceiling is now 1.000/0.997 and OCR 0.750 on shared fields — **the
image is the binding constraint, not the patterns.** One glyph (`µ` as `p`)
explained most of the gap. The benchmark's own ground-truth bug presented as a
hallucination finding; per-field attribution caught it, pooled accuracy would not.
→ [detail](docs/findings.md#finding-9--the-extraction-benchmark-and-what-it-measured)

**The replacement data and target.** NHANES 2021–2023 + 2017–2020, 19,749 rows on
`SEQN`, units from the codebook and never from the values. Anaemia died with
Finding 8; the target is **iron deficiency from ferritin** (WHO 2020, CRP
>5 mg/L for inflammation), audited clean — no closed-form route, learned R²
0.1585 against 0.9933 for anaemia. Trained on **2017–2020 only**, because
ferritin eligibility differs per cycle and sex and age are features here. **ROC
AUC 0.8658 ± 0.0139**, depth-3 leakage test passed, calibration decent and
over-confident above 0.3. **No operating point is chosen** —
`training.threshold_table` lists candidates; the choice is a clinical judgement
this project is not positioned to make.
→ [NHANES](docs/findings.md#nhanes-ingestion--landed-2026-09-21) ·
[coherence](docs/findings.md#coherence-the-staged-filter) ·
[labels](docs/findings.md#labels-and-features--landed-2026-09-21) ·
[iron](docs/findings.md#iron-deficiency--the-replacement-target-landed-2026-09-21) ·
[training](docs/findings.md#training-the-iron-classifier--2026-09-21)

**Fixes that look done and aren't.** Three times the recorded symptom has not
matched the real cause: `debug=True` was not the only switch, a console encoding
error wore a model-loading error's face, and "unloadable pickles" was two
different failures. Check the mechanism, not the message. Its companion —
**write down why something works, not just that it does** — is the near miss
where CSRF protection that held by accident was nearly removed by a keyword,
days after being documented as accidental.
→ [detail](docs/findings.md#fixes-that-look-done-and-arent) ·
[companion](docs/findings.md#write-down-why-something-works-not-just-that-it-does)

---

## Direction

Keep the app layer — auth, DB schema, both extraction backends, PDF export, REST
API all work. Replace the dataset and the modelling layer.

**Work order: ~~security~~ → ~~explicit units~~ → ~~NHANES ingestion~~ →
~~labels~~ → ~~pick a target that isn't leaked~~ → ~~training~~ →
~~extraction benchmark~~.** All landed. Open before anything ships, in order:

1. **Per-subgroup operating points, or a documented single-group scope.** A fixed
   0.50 threshold gives 12% recall in men and 3% in under-5s against 47% in
   women — a threshold artefact on comparable ranking (AUC 0.82/0.79 against
   0.85, intervals overlapping), so it is the fixable one.
2. **The under-5 band needs cases, not features.** 37 cases give an interval 6×
   the adult band's, and the age-normalised MCV feature moved it −0.012 (tested;
   does not hold). Report separately or scope the model above it, and never
   quote 0.7233 without its interval.
3. **Extraction, next.** Decide on the 2× upscale (measured on 40 images, needs a
   full run; it and the `pL` rule are substitutes, not complements, and it sits
   one step from a cliff). Then the deferred axes: table layout, non-English
   labels, `value_cell` placement — implemented and unmeasured. The GPT-4o-mini
   arm stays **deferred by decision**, adapter built and double-guarded, because
   every uncached image is billable against the key in `.env`.
4. **The `G/L` / `g/L` case-folding question.** Unregistered deliberately:
   `_normalise` lower-cases, so separating them needs a case-sensitive exception,
   and a mis-folded haemoglobin and a mis-folded white-cell count are different
   kinds of harm. Fails safe today via the dimension check, at the cost of an
   error naming the wrong reason. A judgement for a person.

Standing data constraint: NHANES runs ferritin on a per-cycle eligibility
subsample. That, not the target, limits cohort size and composition.

---

## Hard rules

- **No bare numbers past the extraction boundary.** A value without a unit is a
  bug. No magnitude-based inference anywhere — `if value > 1000: value /= 1000`
  is the exact pattern that produces Finding 4. **This rule is about scale, and
  scale only.** It cannot make a value correct — see *Standing limitation*, where
  a dropped decimal point produces a plausible in-range number satisfying every
  guard in this list. Do not cite the explicit-unit boundary as evidence that
  extracted figures are right.
- **Every model change ships with the depth-3 leakage test, scored on AUC.** A
  3-level tree within **0.02 AUC** of the ensemble means the label is a rule, not
  a target (`training.LEAKAGE_AUC_MARGIN`). AUC and not accuracy since
  2026-09-21: at an 88% majority class the two were 1.27 accuracy points apart —
  inside fold noise — and 0.063 apart in AUC, which is not. Accuracy is
  compressed against the base rate; AUC is not. The margin sits above the
  measured fold-to-fold AUC sd (~0.014 over 25 folds) and well below the gap a
  learnable target shows.
- **A hold-out is a claim about algebra, not about column names.** Any held-out
  target ships with a `leakage.audit` run, and `RECONSTRUCTION_TOLERANCE` is
  pinned to the coherence tolerance so a leak cannot pass a test the data itself
  would fail. Finding 8.
- **Never report bare accuracy.** Majority-class baseline first, plus per-class
  recall and calibration. The baselines are high — a model that doesn't clear
  them by a wide margin has learned little.
- **Never quote a single survivor count — quote the stage.** Tolerances are
  module constants with a measured justification, and `evaluate()` takes no
  tolerance argument; a per-run tolerance is what made the old numbers
  unreproducible. An absent column is never a pass.
- **Never commit** `.env`, `venv/`, `uploads/`, `*.pkl`, NHANES caches, or the
  four quarantined CBC CSVs (Finding 5).
- No real patient data in the repo. NHANES and MIMIC are de-identified research
  datasets; anything else is out of bounds.

---

## Known issues not yet fixed

Swept against the working tree 2026-09-24. Entries struck off since the last
sweep are archived in
[`docs/findings.md`](docs/findings.md#resolved-issues--archive) with the text
they carried while open — the debug server, both `/upload` defects, the missing
CSRF import, the incidental-CSRF entry, the sklearn pickle table and the dead
scaler path. `/reload_models` was fixed in the same pass; see *Accepted design
decisions* below for `/predict`, which was closed as a decision rather than a
fix.

- **`train_risk_prediction_models.py:140-142` reads three quarantined CSVs**
  (Finding 5), so the script raises `FileNotFoundError`. *(Was `:43-45`; the file
  grew when `SOURCE_UNITS` replaced the inference sites.)* Expected — it gets
  rewritten against NHANES, not patched.
- **`app.py:72` uses the legacy `Query.get()`** — cosmetic, **not** being fixed
  now. `User.query.get(int(user_id))` in the Flask-Login user loader raises
  SQLAlchemy's `LegacyAPIWarning`; `db.session.get(User, int(user_id))` is the
  2.0 spelling. It is the **only one of the suite's 17 warnings that comes from
  our code** — the rest are in sqlalchemy, reportlab and flask_login. Behaviour
  is identical and nothing is deprecated out from under us on the pinned
  versions, so this is noise, not risk. Recorded so the warning is recognised
  rather than re-investigated, and so it does not hide a future warning that
  does matter.
- **`ocr_extraction._ANALYTE_LABELS` — no `"<abbrev> Count"` row form.** Finding
  9 / M3. `RBC Count`, `WBC Count` and `Platelet Count` match no pattern at all,
  so three of `test_4.png`'s four fields are never sighted. Note the order of the
  gaps: fixing this alone turns them into refusals rather than values, because
  `mcL` is missing from `_U_COUNT` too. These three are the only remaining losses
  in the whole benchmark pass.

**Moot, kept visible so they are not rediscovered.** The **pickle/sklearn
compatibility table** describes quarantined artefacts and an empty `models/` —
but keep its reasoning: a `ModuleNotFoundError` stops the app while an
`InconsistentVersionWarning` does not, so the artefact that *loads* is the
dangerous one, and the NHANES replacements must not reintroduce that shape. The
**dead scaler path** has no file to load and is doubly moot — nothing imports
`predict_risk.py` at all, since `app.py:200` defines its own `predict_risk` and
that is what `routes.py:181` uses. Both close with the training rewrite.

**Line references drift.** Those above were re-checked 2026-09-24; the ones in
`docs/findings.md` describe the state when each finding was written and were
**not** re-swept. Treat a line number there as a pointer to a mechanism, not an
address.

---

## Accepted design decisions

Not open issues. Recorded because an undecorated route looks identical to an
oversight, and the next sweep should not reopen it.

**Anonymous prediction is intended.** `/api/predict` (`routes.py:163`) and the
legacy `/predict` (`app.py:344`) are unauthenticated by choice, decided
2026-09-24. Both are reachable without a session and both work. The reasoning:

- The anonymous path is *designed*, not defaulted. `routes.py:177` sets
  `user_id = current_user.id if current_user.is_authenticated else None`, and
  the docstring reads "works with or without auth" — a considered `None`, not
  the `user_id = ... else 0` fallback that made `/upload` a genuine hole.
- It is CSRF-protected by the app-wide `CSRFProtect`.
- It **writes no file and makes no billable call.** That is what separates it
  from `/upload`, which did both. The cost of an anonymous request is compute.
- With no session it also writes no `Prediction` row, so it cannot pollute
  another user's history.

**Revisit if any of those change** — in particular if the endpoint gains a
write of any kind, starts calling a paid API, or if rate limiting becomes a
concern under real traffic. The decision is about today's behaviour, not about
the route being inherently safe to leave open.

Prediction endpoints still refuse with 503 while `models/` is empty, so this
decision has not yet been exercised against a working model.

---

## Environment

The bundled `venv/` was a macOS venv (`/Users/sulamba/...`, 280 MB) and has been
deleted. There is no checked-in environment, so build one before running
anything:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

**378 passed, 17 warnings, 0 skipped** on Python 3.12.10, 2026-09-24 —
`pip check` clean, all 8 security tests passing. `.venv/` is gitignored.
Nothing skipped because Tesseract is installed here; three OCR tests skip
without it, and `.env` points at the binary via `TESSERACT_CMD`.

Of the 17 warnings **exactly one is ours** — `app.py:72`, in *Known issues*.
The other three sites are in dependencies: `sqlalchemy/sql/schema.py`,
`reportlab/lib/rl_safe_eval.py` and `flask_login/login_manager.py`.

`requirements.txt` is fully pinned and **targets Python 3.12**, both floor and
ceiling: the old `Werkzeug==2.2.3` calls `ast.Str`, removed in 3.12, while
`scikit-learn==1.3.2` ships wheels only to cp312. Flask and Werkzeug moved to 3.x
to clear the first. **matplotlib is pinned to 3.8.4 and that pin is
load-bearing**: 3.9+ pulls numpy 2.x, breaking the numpy 1.x ABI sklearn 1.3.2 is
built against, and pandas then fails at import. scikit-learn is pinned at 1.3.2
because that is the version that wrote the quarantined pickles — **the pin now
protects nothing** and is free to move once the NHANES models land.

`chemoguard/data` needs pandas and nothing else, and its output is identical on
pandas 2.1.4 and 3.0.6 — but `tests/conftest.py` imports `app` at module scope,
so the whole suite, `test_coherence.py` included, fails at collection without
Flask. Install everything; there is no pandas-only subset to run.

---

## Conventions

- Python 3.12, Flask, scikit-learn, pandas
- Tests in `tests/`, run with pytest
- Small reviewable commits; stop for review before large refactors
- New evidence, superseded numbers and resolution notes go in
  [`docs/findings.md`](docs/findings.md). Keep this file at working size — if an
  entry here needs more than a paragraph, it belongs there with a pointer.
