# Report fixtures

Plain-text transcriptions of the four report images checked in at the
repository root. Each `.txt` is what the page says; the corresponding
`.jpg` or `.png` is the page.

They exist because the Tesseract binary is not installed everywhere the
suite runs, and because a transcription isolates what these tests are
about -- whether a unit printed on a report survives the journey to the
model -- from whether OCR read the glyphs correctly. `test_report_fixtures.py`
runs the real Tesseract path as well, and skips it when the binary is
absent.

None of the four images carries a patient identifier: no name, no MRN, no
date of birth, no accession number. They are synthetic demonstration
layouts, which is why they can sit in the repository at all (CLAUDE.md,
"No real patient data in the repo").

| File | What it exercises |
|---|---|
| `test1.txt` | Uniform K/uL and M/uL, all values normal. The reference-range column repeats the result, which is one reason this is clearly a synthetic layout. |
| `test_2.txt` | Genuinely mixed units on one page: RBC in millions/uL, WBC and platelets in cells/uL. Platelets 85,000 cells/uL is real thrombocytopenia (CTCAE grade 1) and reads as a comfortable 85,000 thousand/uL to anything that infers unit from magnitude. |
| `test_3.txt` | Mixed units again, all values normal, the unit inside the value cell rather than in a column, and four rows — platelets, HCT, MCV, MCH — that print **no unit anywhere**. The MCHC row is below the bottom edge of the image. **Corrected 2026-09-22:** this row previously read "a platelet row whose unit appears only in the reference-range column". It is not there either; that range reads `150,000 - 450,000` with no unit. The row is genuinely unit-less, which is why `test_report_fixtures.py` asserts it is refused. |
| `test_4.txt` | A two-column screenshot carrying four analytes, units inside the value cell and spelled with `mcL` rather than the micro sign: `3.2 million cells/mcL`, `5,100 cells/mcL`, `210,000 /mcL`. Added 2026-09-22 for the extraction benchmark (CLAUDE.md Finding 9), which uses it as stratum A4. Its rows are labelled `RBC Count` / `WBC Count` / `Platelet Count`, a form `ocr_extraction._ANALYTE_LABELS` does not match, so three of its four fields are lost before the units are ever reached. |
