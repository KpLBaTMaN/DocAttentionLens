# doclayout_mm_rag_examples

One annual report from the doclayout-mm-rag benchmark, committed so a fresh clone has a
document to investigate: **NASDAQ_AROW_2019** (Arrow Financial Corporation, 2019 annual
report). It has 8 pages, 100 layout elements (4 of them tables) and 29 questions.

The document record, page images, PDF, layout, OCR and question files are byte-identical
to `annual_reports` document 30 of `data/doclayout_mm_rag_bench/`. The folder uses the
benchmark's layout, so the same adapter reads it:

```text
manifest.json                       counts for this one document
annual_reports/
  documents.json                    the benchmark's record for document 30
  annotations/pdf_30.json           questions, answers and support element IDs
  layouts/pdf_30.json               elements: type, category, box, level, reading order
  ocr_texts/pdf_30.json             OCR text per element
  document_pages/pdf_30/page_N.jpg  300 DPI page images
  pdfs/pdf_30.pdf                   the source PDF
```

Register it in Library as **Document layout example**. Three of its tables are annotated
cell by cell in [../TableGroundQA-Examples-v1](../TableGroundQA-Examples-v1/README.md).
`scripts/build_examples.py` makes this copy from the benchmark; see
[docs/examples.md](../../docs/examples.md).
