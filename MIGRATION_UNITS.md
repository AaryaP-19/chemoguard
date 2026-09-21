# Migration: explicit units

The schema change that comes with the unit boundary. Read this before
pointing the application at a database that already has rows in it.

## What changed

Every stored laboratory value moved to a column named for the unit it holds.

| Table | Was | Now |
|---|---|---|
| `medical_reports` | `hb` | `hb_g_per_dl` |
| | `rbc` | `rbc_million_per_ul` |
| | `wbc` | `wbc_cells_per_ul` |
| | `anc` | `anc_cells_per_ul` |
| | `platelets` | `platelets_cells_per_ul` |
| | `hematocrit` | `hematocrit_percent` |
| | `mcv` | `mcv_fl` |
| | `mch` | `mch_pg` |
| | `mchc` | `mchc_g_per_dl` |
| | *(new)* | `reported_quantities` JSON |
| `predictions` | `input_rbc` | `input_rbc_million_per_ul` |
| | `input_wbc` | `input_wbc_cells_per_ul` |
| | `input_platelets` | `input_platelets_cells_per_ul` |
| | `input_anc` | `input_anc_cells_per_ul` |
| | *(new)* | `input_reported_quantities` JSON |
| | *(new)* | `model_feature_units` JSON |
| | *(new)* | `request_id` |

`neutrophils_percent` keeps its name: it already stated its unit.

## Why a rename and not a reinterpretation

The obvious migration is to keep `wbc` and change what it means. Don't.

The old columns hold whatever the extraction path happened to produce. The
GPT prompt asked the vision model to convert to thousands/µL and mostly it
did; the Tesseract path applied a range check that only passed values
already in thousands; the manual form accepted either and let the server
guess. So the old `wbc` column holds a mixture, and no per-row marker
distinguishes the cases.

Redefining the column to mean cells/µL would read every historical row as
1000x its real value, silently, with no error anywhere. That is CLAUDE.md
Finding 3 exactly -- a 1000x unit change with no column rename -- and it
took a column-by-column audit against the upstream file to find.

Renaming makes the old rows fail loudly instead. `Prediction.input_rbc`
no longer exists, so anything still reading it raises `AttributeError` at
the point of use rather than returning a plausible wrong number.

## What to do with existing rows

**Nothing automatic.** There is no safe programmatic conversion, because the
unit of a historical row was never recorded. Any migration script would have
to infer it from the magnitude, which is the defect being removed.

Pick one, deliberately:

1. **Drop and recreate** (recommended for development and for any database
   whose contents are test data). `python init_db.py` against a fresh
   database. The repository has no Alembic setup and the app calls
   `db.create_all()`, which creates missing tables but does not alter
   existing ones -- so an old database will *not* pick up the new columns
   on its own and will fail on first write.

2. **Migrate with a stated assumption.** If historical rows must be kept,
   write a one-off script that names the assumption in a comment and
   records it per row in `reported_quantities` with
   `"unit_source": "assumed-at-migration"`. Then the assumption is visible
   in the data and a later reader can discount those rows. Do not let the
   script infer per row.

3. **Archive and start clean.** Copy the old tables to `*_pre_units` and
   create the new schema empty. The old values remain available for audit
   without any claim about their scale.

## Checking a database is on the new schema

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'predictions' AND column_name LIKE 'input_%';
```

If that returns `input_rbc` rather than `input_rbc_million_per_ul`, the
database predates this change.

## API changes that go with it

`/predict` and `/api/predict` no longer accept a bare number. A request body
of `{"wbc": 5800}` returns **422** with `reason: "missing_or_unknown_unit"`.
Three shapes are accepted, all documented in `cbc.panel_from_mapping`:

```json
{"wbc": {"value": 5800, "unit": "cells/uL"}}
{"wbc": 5800, "wbc_unit": "cells/uL"}
{"wbc": 5800, "units": {"wbc": "cells/uL"}}
```

Responses now return `input_values` as an object per analyte, carrying the
canonical value, the value as reported, and where the unit came from --
rather than a bare number.

`GET /api/reports` and `GET /api/predictions` changed the same way: the flat
`rbc` / `input_rbc` keys are replaced by a `values` / `input_values` object.

## Status codes

| Code | Meaning |
|---|---|
| 400 | A required analyte is absent from the request. |
| 422 | A value is present and we will not interpret it: no unit, an unrecognised unit, a unit of the wrong dimension, or a value that is impossible once converted. |
| 503 | No model is available, or a loaded estimator disagrees with its declared feature contract. |
