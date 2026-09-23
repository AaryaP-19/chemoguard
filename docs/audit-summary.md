# ChemoGuard: what the audit found, and what replaced it

ChemoGuard reads values off complete blood count (CBC) reports and returns a
haematological risk assessment. It is an assistive tool, and nothing in it has
been clinically validated. This note records what the original version claimed,
why almost none of it survived, and what replaced it.

## The original claim

Three risk models — anaemia, neutropenia and thrombocytopenia — trained on a
7,196-row CBC dataset, reporting **99.93% accuracy**. Fifteen artefacts sat in
`models/`, the pickles loaded, `/health` reported `models_loaded: true`, and the
app served predictions. It looked finished.

## The dataset

The 7,196 rows × 16 columns turned out to be largely fabricated, on four
independent lines of evidence.

**Five arithmetic identities that hold in any real analyser output fail here.**
An analyser computes most of its panel from a few measured channels, so the
derived columns are algebraically pinned to their inputs. Correlating each
reported column against the value derived from its own inputs gives: HCT ≈ 3×Hb,
0.408; MCV = HCT/RBC×10, 0.206; MCH = Hb/RBC×10, 0.160; MCHC = Hb/HCT×100,
0.049; and plateletcrit, PCT = PLATELETS×MPV/10,000,000, 0.0863 with a mean
absolute error of 0.926. These are not noisy measurements — they are numbers
unrelated to the numbers that define them. The failures are also interleaved
rather than appended, so truncating the file fixes nothing: MCHC identity error
is sharply bimodal (0.15 at p70, 3.64 at p80) and offending row indices run from
13 to 7191 throughout.

**Several columns terminate on generator bounds rather than observed values.**
Haemoglobin stops at exactly 25.000; MCV spans exactly 50.000–130.000; MCH
10.010–50.000; MCHC stops at 39.999; RDW starts at exactly 11.000; PCT stops at
exactly 3.000. Age spans exactly 20–69, with no paediatric and no geriatric
rows, which is not a clinical sample. Separately, 10.6% of rows carry WBC above
100 thousand/µL — hyperleukocytosis, under 0.1% of any real population. None of
this depends on an identity holding, so it is evidence independent of the
correlations above.

**The reported plateletcrit contradicts its own inputs, and the inputs are the
sane ones.** PCT derived from platelets and MPV is physiologically ordinary
throughout (median 0.268, p75 0.333, against a normal plateletcrit of roughly
0.15–0.35%). The reported column is not: median 0.31 but p75 2.59 and max 3.00,
with 43.9% of rows above the physiological ceiling and 30 rows pinned at exactly
3.000. Relative error against the derived value is 0.82% at the median and
89.06% at p70. The synthesis sits in the reported column, not in the platelet
and MPV measurements it should follow from.

**The saved artefacts corroborate the training script.** The three estimators
report `n_features_in_` of 3, 4 and 3 — exactly the hardcoded feature lists at
`train_risk_prediction_models.py:53`, `:76` and `:104`. Nothing else reaches
`.fit()`, which matters below: each label is a threshold rule over columns
sitting in its own feature matrix.

## Provenance

The upstream source is Mendeley Data, DOI `10.17632/rnfzzy4wz6`, found via a
third-party repository citing it with an access date of 2026-01-12. No README,
licence file or source note accompanied the copy in this repo. Checked on
2026-09-21, both the DOI landing page and the version URL return: *"Version 1 of
the dataset has been removed. The reason given for this removal was: As per
author's request."* No further reason is published, and the licence, collection
methodology, contributing institution and any validation procedure went down
with the record. No mirror or independent citation of the data could be found.
Provenance could not be established — there is a DOI, a title, and nothing else.
The validation asserted in that title is an unverified claim from a withdrawn
record, not a property of the data. All four files (the source and its three
derivatives) are quarantined outside the repo, SHA-256 matched and re-read at
the destination before the originals were deleted, so the audit trail
reproduces.

## Why the models were arithmetic rather than learned

Each published label reconstructs exactly — on **100.00% of 7,196 rows**, not
approximately — from the features the model was given. `anemia_risk` counts how
many of RBC<3.5, WBC<4.0 and PLATELETS<150 hold; `neutropenia_risk` thresholds
ANC at 1500 and 1000; `thrombocytopenia_risk` thresholds platelets at 150 and
75. In every case the thresholded columns are the model's own inputs.

The scores show it. Against majority-class baselines of 69.38%, 92.29% and
90.90%, a depth-3 decision tree matches or beats the 100-tree Random Forest on
every task: anaemia 99.93% for the tree against 99.72% for the forest,
neutropenia 99.93% against 100.00%, thrombocytopenia 100.00% for both. When
three levels of splits reproduce an ensemble, a rule is being recovered, not a
signal learned — and the headline 99.93% sits about seven points above always
answering "Low". The anaemia model also never reads haemoglobin: its High-risk
rows average Hb 11.82 and reach 24.8 (polycythaemia), while its Low-risk rows
include Hb 5.1, a transfusion threshold.

## Moving to NHANES

The replacement data is NHANES, the CDC's national health survey: 19,749 rows
across the 2021–August 2023 cycle (7,593) and the 2017–March 2020 pre-pandemic
release (12,156), joined to demographics on `SEQN`, with units taken from the
codebook rather than inferred from the values.

One coherence filter run over both datasets gives the sharpest contrast in the
audit. Applying three exact analyser identities jointly at a 2% tolerance,
**19,747 of 19,749 NHANES rows (99.99%)** pass coherence and plausibility
together, against **1,962 of 7,196 (27.27%)** for the legacy data. Legacy
per-identity rates saturate at 75.21% (MCV), 61.33% (MCH) and 51.13% (PCT); the
NHANES equivalents are 99.99% and 100.00%, with PCT not reported by NHANES at
all and therefore scored as not evaluable rather than as a pass.

The tolerance sweep is the argument that this is not a threshold artefact — see
[coherence_tolerance_sweep.png](../reports/figures/coherence_tolerance_sweep.png)
and the CSV beside it. Real analyser data saturates at 100% and stops; the
legacy pass rate is nearly flat in the tolerance, moving only 1,939 → 2,129
(26.95% → 29.59%) across a tenfold sweep from 1% to 10%. The failing rows are
not near-misses a looser bound would admit, so no tolerance choice rescues the
dataset. The two NHANES rows that do fail look like reporting floors rather than
measurements — MCV pinned at the column minimum of exactly 35.4 and MCHC at its
minimum of exactly 25.2, while their MCH identity passes to within 0.2%. A
filter that catches 2 rows in 19,749 and catches the right ones is calibrated,
not lax.

## The anaemia hold-out was leaked, and fixing the data exposed it

The planned replacement task was WHO anaemia severity with haemoglobin held out.
That is not well posed on a CBC panel, because every coherence identity is a
route back: `Hb = MCH × RBC / 10`, `Hb = MCHC × MCV × RBC / 1000`, and so on.

**Repairing the data made this worse, not better.** On the withdrawn dataset
only about 27% of rows satisfied those identities, so reconstruction would have
been noise on three-quarters of the file — the fabrication was masking the leak.
On NHANES, where 19,747 of 19,749 rows satisfy them to within 2%, the
reconstruction is essentially exact everywhere.

Against a 91.82% majority baseline — 8.18 points of headroom in total — merely
dropping the column name recovers haemoglobin on 100.00% of rows to within 2% by
closed form, with a learned R² of 0.9933 and the WHO severity label reproduced
98.97% of the time. That is 87% of all available headroom, purely by arithmetic.
Dropping MCH and MCHC as well still leaves R² 0.9521 and 97.23% label agreement,
because MCV × RBC recovers haematocrit exactly. Only removing red cell mass
entirely closes the route, and then the task is gone: +0.65 points over
baseline, with severe-class recall of 11.1%. A hold-out is a claim about
algebra, not about column names.

## The ferritin pivot

The defensible target is one the CBC does not contain. NHANES publishes serum
ferritin, run on a separate assay from a separate aliquot, and it is the
question the Mentzer and Green & King indices were invented to answer. Iron
deficiency is defined by WHO 2020 thresholds, with the cut-off raised for
inflamed individuals (CRP > 5 mg/L); that adjustment nearly doubles measured
prevalence — 7.27% unadjusted against 13.53% adjusted — so both rules are
trained and reported and neither is treated as authoritative.

The leak audit is clean, and the contrast with the anaemia target is the point:
no closed-form route exists, learned R² is 0.1585 against 0.9933, and the
within-2% reconstruction rate is 2.29% against 98.11%. The enumerated route list
is empty by finding rather than omission, with a test pinning it so a route
added later fails loudly.

Training runs on 2017–2020 alone, because NHANES ran ferritin on an eligibility
subsample that differs between releases — "1–5 years old and 12–49 years old
females" in 2021–2023 against "1–5 years old and 12 years and older" in
2017–2020. Pooling would make the cohort's age and sex mix partly a function of
which release a row came from, and age and sex are features of this model. On
10,181 rows with a deficient rate of 11.62% against an 88.38% majority-class
baseline, a HistGradientBoosting classifier over 5 folds × 5 seeds gives **ROC
AUC 0.8658 ± 0.0139**, accuracy 0.9102 ± 0.0038 (+2.64 points on baseline),
Brier 0.0705 ± 0.0034, and average precision 0.5991 against a no-skill floor of
0.1162 ([precision_recall_curve.png](../reports/figures/precision_recall_curve.png)).
A depth-3 tree trails by 0.0630 AUC against a 0.02 margin, so the ensemble is
doing more than thresholding — which is exactly what the anaemia label could not
say. Calibration is close to the diagonal below 0.3 and over-predicts above it,
the 0.6–0.8 bins running about 0.045 high
([reliability_curve.png](../reports/figures/reliability_curve.png)). Summed over
25 folds, the confusion matrix is 43,938 replete called replete, 1,052 replete
called deficient, 3,519 deficient called replete and 2,396 deficient called
deficient ([confusion_matrix.png](../reports/figures/confusion_matrix.png)).

Subgroup AUCs carry Hanley & McNeil 95% intervals, whose width is set by the
number of cases rather than the number of rows
([subgroup_auc.png](../reports/figures/subgroup_auc.png)): female 0.8490
[0.833, 0.865] on 974 cases, male 0.8171 [0.782, 0.853] on 209, ages 20–59
0.8982 [0.883, 0.914] on 700, 60+ 0.7911 [0.750, 0.832] on 170, ages 5–19
0.8467 [0.817, 0.876] on 276, and ages 1–4 0.7233 [0.629, 0.818] on 37. Only the
20–59 band separates from all the others. Two conclusions follow, and they are
different in kind. Recall at a fixed 0.50 threshold collapses to 0.124 in both
men and the 60+ band, but on comparable ranking with overlapping intervals: what
differs is prevalence (4.1% and 5.9% against 19.0% in women), so a fixed cut
almost never fires, and a per-group operating point would largely close it. The
under-5 band is a separate matter — recall 0.027, and an interval 6.1× wider
than the 20–59 band's, overlapping three other groups. That band is underpowered
at 37 cases, not demonstrably worse, and 0.7233 should not be quoted as a point
estimate without its interval.

**No operating point is chosen, deliberately.** The candidate table spans 50%
target recall (threshold 0.3582, precision 0.621, 9.4% of the cohort flagged) to
90% (threshold 0.0288, precision 0.225, 46.5% flagged). Picking a row trades
missed iron deficiency against over-referral, which is a judgement about
relative clinical harm. This is a student exercise on public survey data, with
no clinical validation, no intended-use statement and no care pathway behind it.
The table is the deliverable; the choice is not.

## What the extraction path actually reads

The models are one half of the application; the other half is the step that
reads numbers off a photographed report before any model sees them. That step
had never been measured. It is now, against a purpose-built corpus of 814
images and 8,140 hand- and generator-labelled fields.

The corpus had to be built rather than collected, and the reason is itself a
finding about the earlier state of the project. The four report images in the
repository are near-uniform — clean, high-contrast, English, single-column —
and both extraction backends score near-perfectly on the kind of thing they
represent, so measuring against them would have measured nothing. The
generated strata vary the axes that matter: unit scale (cells/µL, K/µL, SI
10⁹/L), where the unit is printed (a dedicated column, inside the row label,
absent entirely), decimal point against decimal comma, and reference ranges
present or absent — fully crossed, 36 cells × 3 clinical panels. A third
stratum applies 26 image-quality variants over nine of those cells. Fields
where the page prints no unit are labelled as such and scored as **correct to
refuse**, so the metric cannot reward a backend for guessing.

The design decision that made the results interpretable was running a
**pattern layer** arm: the production regex set fed the exact text the
generator drew, with no OCR in the loop. That arm is a ceiling rather than a
performance figure, and the two arms differ in nothing but where the text came
from — same patterns, same alias table, same canonicalisation, same ground
truth — so the difference between them is glyph reading and admits no other
explanation.

## Four defects, three of them invisible without the benchmark

The value parse had no decimal-mark policy. `float(text.replace(",", ""))`
strips commas and honours points, which is the US convention hard-coded:
`6.500` read as 6.5 and `0,9` read as 9. Recall across the two decimal styles
was 0.867 against 0.233. Thirty-two wrong quantities reached the panel, and
**sixteen of them graded a cytopenia less severe than it is** — a WBC printed
`0,8`, meaning 800 cells/µL and CTCAE grade 4 leukopenia, read as 8,000 and
graded 0. The replacement rests on one fact, that a thousands separator is
always followed by exactly three digits, so a separator followed by one, two
or four-plus digits can only be a decimal mark; the genuinely ambiguous
exactly-three case is then settled by evidence from the page itself, and
refused where the page settles nothing.

The SI unit alternation could not match any SI unit. Interpolating one
character class inside another produced `[l[uµμ]]+`, which Python parses as a
class followed by a *required literal* `]`, so the pattern matched `10^9/L]`
and never `10^9/L`. Two further defects sat in the same expression. Every
count reported in SI lost its unit, and the loss was entirely in the pattern
layer: `units.parse_unit` had accepted `10^9/L` and `10^12/L` all along.

The alias table was extended — mcL, superscript and E-notation exponents, a
trailing result flag — and **not one extraction field changed outcome.** That
null result is the most useful thing the stage attribution produced. A recall
of 0.622 beside a list of unrecognised unit spellings reads as a units
problem, and the obvious fix would have been the wrong one: a field only
reaches the alias table if a pattern first matched its label *and* captured a
unit string, and every field the gaps would have touched was already lost
upstream.

The fourth defect appeared the moment Tesseract was installed, and would have
invalidated every OCR number in the exercise. Both call sites passed
`--psm 6`, "assume a single uniform block of text", which a ruled table is
not. On a pristine render at OCR confidence 91 it returned the two title lines
perfectly and, for the entire ten-row table, `Cr ee` / `rucr ||| ars`.
Preprocessing was ruled out before the mode was — denoising, thresholding and
upscaling each change the result by nothing under psm 6 — and the replacement
was chosen by measurement across five modes rather than by reading the manual:
psm 3 at recall 0.419 against psm 6 at 0.022. The sparse-text modes read *more
characters* and score *worse*, because they do not preserve reading order and
the patterns need a row's label, value and unit on one line. Had the arm
simply been run, a one-line configuration default would have been published as
the difficulty of reading laboratory reports.

## The ceiling, the floor, and one glyph between them

With the patterns fixed, the ceiling is precision 1.000 and recall 0.997: on
perfect text the extraction logic is essentially complete, and the three
remaining failures are one real image whose rows are labelled `RBC Count`
rather than `RBC`. The OCR arm on the same 1,390 fields reaches precision
0.996 and recall 0.750 — **229 fields the ceiling read correctly and OCR did
not.** Over the full corpus, including the degraded strata the ceiling cannot
score, the OCR arm is 0.987 / 0.540.

Most of that gap was one glyph. Tesseract reads the micro sign as `p`,
returning `cells/pL` where the page prints `cells/µL`, and the per-parameter
table split accordingly: the four count analytes all print a microlitre
denominator and scored 0.221–0.305, while the analytes measured in %, fL, pg
and g/dL scored 0.670–0.779. Accepting `pL` in the denominator and rewriting
it at the OCR boundary — not in the alias table, since a picolitre is a real
unit and a misreading is not a spelling — recovered 268 fields and took
count-analyte recall from 0.228 to 0.353, with ANC nearly tripling from 0.101
to 0.287. Upscaling the image 2× before OCR addresses the same glyph and
reaches slightly further, but the two are substitutes rather than
complements, and upscaling has a cliff one step past its optimum: at 3× recall
collapses to 0.084 and silent wrong values jump from 3 to 52.

Image quality does not degrade extraction gracefully. It falls off a cliff,
and the cliffs are not where intuition puts them. JPEG compression is
survivable to quality 20, a setting that visibly wrecks the image — recall
0.611 against a pristine 0.778. Half a pixel of additional Gaussian blur takes
everything: σ0.5 scores 0.844, *above* the pristine control because the blur
anti-aliases the micro sign, and σ1.0 scores 0.000. Skew degrades steadily to
4° and dies at 5°. A left-lit shadow is fatal at 0.000 while a diagonal one
costs little at 0.667, because the left of the page is where the row labels
are and a pattern that cannot match a label never sights the row at all. See
[extraction_degradation_curves.png](../reports/figures/extraction_degradation_curves.png).

## The limitation that has no code fix

Every defect above was a code defect with a code fix. The finding that matters
most is not.

Raising recall raised the count of silent wrong values from 31 to 38, and the
subset that understates a CTCAE grade from 9 to 13. That is not a regression
to be netted off against the recall gain — it is the arithmetic of a pipeline
with no digit check. The fields recovered were previously refused for a *unit*
problem which was masking a *digit* problem underneath, and letting a field
through means letting through whatever it says.

The mechanism is a dropped decimal point. OCR reads `0.8` as `08`, which
parses as 8.0. True 800 cells/µL is grade 4 leukopenia; read as 8,000 it
grades 0. Every guard in the application passes it, and each for a good
reason: the unit is read correctly, so the explicit-unit boundary is
satisfied; there is no separator at all, so the decimal-mark rule has nothing
to act on; and 8,000 cells/µL is a perfectly normal white cell count, so the
plausibility bounds accept it — bounds tight enough to catch this would reject
genuine pathology.

The explicit-unit boundary, which is the single largest piece of remediation
in this project, guarantees that a value arrives with a **known scale**. It
does not guarantee that the digits are the ones on the page, and it was never
capable of that. Catching these requires a second independent read that has to
agree with the first, a confidence gate that refuses low-confidence fields, or
an operator confirming the extracted figures against the image. The
application has none of the three.

So the honest statement of what this extraction path offers is: *it will not
silently change the scale of a value, and it may silently misread one.* Any
intended-use statement has to carry the second half as well as the first.
