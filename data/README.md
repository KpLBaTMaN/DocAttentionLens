# Local datasets

Place the document-layout benchmark export in data/doclayout_mm_rag_bench/.
The optional table-grounding export belongs in data/TableGroundQA-Pilot-v1/
and must declare that same export_id in its manifest.

Compose mounts each root read-only. Register available exports through Library.
Uploaded PDFs and images are stored in the artifact volume, rather than here.

Benchmark images, PDFs, and annotations remain local and are excluded from Git and
Docker builds. Dataset integration tests for them require the relevant exports.

Two small examples are committed, so a source clone has something to register:

- doclayout_mm_rag_examples/: one annual report from the benchmark, with its layout
  elements and questions.
- TableGroundQA-Examples-v1/: three tables of different sizes from that report,
  annotated cell by cell, with lookup questions.

See docs/examples.md for how they were made and rebuilt, and LICENSE.md for the terms that
apply to each file. The code license does not cover this folder.
