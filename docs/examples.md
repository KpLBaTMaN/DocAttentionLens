# Example datasets

A source clone includes two small example datasets, so the application has something to
investigate without the full benchmark exports. Both come from one annual report in the
doclayout-mm-rag benchmark: NASDAQ_AROW_2019 (Arrow Financial Corporation, 2019).

| Source folder | Library label | Contents |
|---|---|---|
| data/doclayout_mm_rag_examples/ | Document layout example | The whole report: 8 pages, 100 layout elements, 29 questions. Files are byte-identical to the benchmark. |
| data/TableGroundQA-Examples-v1/ | Table grounding examples | Three of its tables (35, 85 and 150 cells), annotated cell by cell, with 26 lookup questions. |

Register either one from Library. Compose mounts both folders read-only next to the
benchmark exports. Nothing from PubTables-1M or the TableGroundQA pilot is committed.
The report and the project's annotations have separate terms: see
[data/LICENSE.md](../data/LICENSE.md).

## Table annotation

The table export uses the TableGroundQA export format, so the same adapter reads it and its
files show how such an export is built. The tables were annotated from scratch:

- Rows, column spans, header roles, cell texts and questions are AI generated, in
  data/TableGroundQA-Examples-v1/authoring/tables.json.
- scripts/build_examples.py requires every cell text to match the report's PDF text layer,
  ignoring whitespace, and derives all geometry from the matched words. Row and column edges
  lie midway between neighbouring text, so logical cells tile each table.
- Human review is still pending, and each table is marked `needs_review`.

See the export's [README](../data/TableGroundQA-Examples-v1/README.md) for the three tables
and the file layout.

### Checks against the pilot and the benchmark (29 October 2026)

- **Format:** question and answer-link records carry every field the pilot's do. Value
  types use the pilot's vocabulary (integer, decimal, text). The fields the pilot has and
  these lack describe PubTables provenance only: archives, XML offsets, PMC IDs and the
  XML's one-pixel crop frame. Page 6 holds two tables, so table bindings sit in
  `table_bindings` rather than at the page level.
- **Cells:** read independently from the PDF, the words inside each logical box equal the
  cell text for all 270 cells. Every row and column slot is covered once, and no word in a
  table lies outside a cell.
- **Benchmark:** the benchmark's OCR of the same three table elements contains all 267
  non-blank cell texts in reading order. The benchmark's own questions on these tables
  agree with the cells: 11 numeric answers match, and its true/false question and series
  names are consistent.
- **Questions:** all 26 answers equal their cell text and the PDF text at the answer box.
  The question names the column's year, and the row path ends in the answer's row.

## Rebuild

    ./.venv/Scripts/python.exe scripts/build_examples.py --tables

This regenerates the table export from authoring/tables.json and the committed report. It
needs no benchmark data, and the same input gives the same bytes. To change an annotation,
edit authoring/tables.json and rebuild. Unchanged files keep their hashes, and the dataset
version changes. Without `--tables`, the script first copies the report again from
data/doclayout_mm_rag_bench/.

.gitattributes stops Git converting line endings in both folders, so checked-out files keep
the bytes listed in SHA256SUMS.txt on every platform. library/tests/test_examples.py and
tests/integration/test_examples_api.py run against the committed examples and need no
benchmark data.

## Adapter support added for the examples (29 October 2026)

- read_dataset takes the registered source name, which enters the dataset identity.
  The benchmark keeps its name, so its identities and hashes are unchanged.
- read_table_dataset reads `domain`, `coordinate_convention` and `geometry_validation`
  from the manifest. They default to the pilot's values.
- A page record may bind several tables in `table_bindings`, and a document may hold
  several pages. Pages are registered once. A document's `annotation_version` is its
  table's hash when it has one table, and otherwise the hash of its tables' hashes in
  manifest order.
- A table case supplies only the question's support page (its table's page) unless the
  request names pages. Each pilot document holds just that page, so pilot cases are
  unchanged.
- The case input note follows `input.mode`: `table_crop` for the pilot, `full_page` for
  these examples.

The pilot and benchmark snapshots were byte-identical before and after these changes.
