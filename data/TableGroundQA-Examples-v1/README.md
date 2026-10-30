# TableGroundQA-Examples-v1

Three tables from the example annual report (NASDAQ_AROW_2019, in
[../doclayout_mm_rag_examples](../doclayout_mm_rag_examples/README.md)), annotated cell by cell
from scratch, with 26 lookup questions. Each table has a different size and shape:

| Table | Page | Rows × columns | Cells | Questions | Structure |
|---|---|---|---|---|---|
| Five-year cumulative total return, 2014-2019 | 6 | 5 × 7 | 35 | 6 | One header row; four series |
| Fifteen-year cumulative total return, 2004-2019 | 6 | 5 × 17 | 85 | 8 | Wide; row labels wrap over two or three lines |
| Consolidated income and selected ratios, 2015-2019 | 4 | 29 × 6 | 150 | 12 | Two header rows with a spanning header, four section rows, dollar signs and percentages |

Page 6 holds two tables, so its page record binds each table separately (`table_bindings`).
Register the export in Library as **Table grounding examples**.

## How it was made

`authoring/tables.json` is the source. Every row, column span, header role, cell text and
question in it is AI generated from the page images.
`scripts/build_examples.py` then:

- requires each authored cell text to equal the PDF text layer of the report, ignoring
  whitespace. A single wrong character stops the build.
- takes each cell's native box from its matched words, and splits rows and columns midway
  between neighbouring text. The logical cells tile the table.
- cuts the table crop from the page image, so the crop-to-page transform is a pure
  translation (registration correlation 1.0).
- writes the export below, with checksums. Rebuilding gives identical bytes.

The overlays in `checks/overlays/` show the logical cells (blue), word boxes (green),
shaded header cells and the answer cells of the questions (red).
Independent human review is pending, so each table's quality status is `needs_review`.
One advisory note records that the unit header "(in thousands, except per-share amounts)"
runs past its column edge.

## Files

```text
authoring/tables.json            AI-generated rows, cells and questions (the source)
manifest.json                    export identity, domain, frames and table inventory
documents/<document>/            the report's identity and source PDF hash
pages/<page>/page.jpg, page.json the benchmark page image and its table bindings
pages/table_crops/               table crops cut from the page images
structure/<table>/               table snapshot (crop pixels), PDF annotation (points)
                                 and page grounding (page pixels)
qa/questions.jsonl               26 questions with answer cells and header paths
qa/questions/<table>.jsonl       the same records split by table (do not count them twice)
qa/page_grounding.jsonl          each question's answer box on its page
checks/SHA256SUMS.txt            every file the adapter reads, except this README
checks/overlays/                 visual checks
```

Coordinates are 300 DPI page-image pixels with a top-left origin. PDF points times 300/72
gives page pixels; crop pixels plus the crop origin gives page pixels. Questions were
written against the full page (`input.mode` is `full_page`), and a case supplies only the
table's own page.
