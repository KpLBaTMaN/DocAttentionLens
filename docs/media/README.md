# README demo

`demo.gif` is an edited walkthrough of the running DocAttentionLens application,
using a saved ColQwen2.5 v0.2 capture. `demo-poster.png` provides a static view.

The investigation uses the Middlefield Banc Corp. annual-report example from the
DocLayout-MM-RAG benchmark: the question asks for net income in 2011, and the
selected support region is table 51 on page 6 of `NASDAQ_MBCN_2019`.

The walkthrough selects a question token, inspects its document region, and
explores the region's layer–head profile. Frames come from the real interface;
captions and reading pauses are added during editing. Analysis is recomputed from
the saved capture without a new model run.

These captures belong to the local demonstration deployment. They are not
bundled with a fresh installation. See [data setup](../../data/README.md) and
[running the application](../getting-started.md) to prepare an investigation.
