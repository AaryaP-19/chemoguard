# Report fixtures

Plain-text transcriptions of the three report images checked in at the
repository root. Each `.txt` is what the page says; the corresponding
`.jpg` is the page.

They exist because the Tesseract binary is not installed everywhere the
suite runs, and because a transcription isolates what these tests are
about -- whether a unit printed on a report survives the journey to the
model -- from whether OCR read the glyphs correctly. `test_report_fixtures.py`
runs the real Tesseract path as well, and skips it when the binary is
absent.

None of the three images carries a patient identifier: no name, no MRN, no
date of birth, no accession number. They are synthetic demonstration
layouts, which is why they can sit in the repository at all (CLAUDE.md,
"No real patient data in the repo").

| File | What it exercises |
|---|---|
| `test1.txt` | Uniform K/uL and M/uL, all values normal. The reference-range column repeats the result, which is one reason this is clearly a synthetic layout. |
| `test_2.txt` | Genuinely mixed units on one page: RBC in millions/uL, WBC and platelets in cells/uL. Platelets 85,000 cells/uL is real thrombocytopenia (CTCAE grade 1) and reads as a comfortable 85,000 thousand/uL to anything that infers unit from magnitude. |
| `test_3.txt` | Mixed units again, all values normal, and a platelet row whose unit appears only in the reference-range column. |
