# Data licenses

The root [LICENSE](../LICENSE) covers the code. It does not cover this folder. The two
committed examples combine the project's own annotations with a third-party report, so the
terms depend on the file.

## Project annotations: CC BY 4.0

Copyright (c) 2026 the DocAttentionLens authors. Licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

This covers the questions, answer links, layout elements, table structure and metadata in:

- `doclayout_mm_rag_examples/`: `manifest.json`, `annual_reports/documents.json`,
  `annual_reports/annotations/` and `annual_reports/layouts/`
- `TableGroundQA-Examples-v1/`: `manifest.json`, `authoring/`, `documents/`, `structure/`,
  `qa/`, the `page.json` records in `pages/` and `checks/SHA256SUMS.txt`

Text quoted from the report inside these files, such as table cells and answers, remains
part of the report and is covered by the section below.

## Arrow Financial report: not licensed here

**NASDAQ_AROW_2019** is the 2019 annual report of Arrow Financial Corporation. It remains
the copyright of its owner, and this repository grants no license to it. This covers:

- `doclayout_mm_rag_examples/annual_reports/`: `pdfs/`, `document_pages/` and `ocr_texts/`
- `TableGroundQA-Examples-v1/`: page images and table crops in `pages/`, and
  `checks/overlays/`

The report is included as a research example, so a source clone has a document to
investigate. If you hold rights in it and want it removed, open an issue on this repository.
