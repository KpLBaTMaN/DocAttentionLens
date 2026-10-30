# How the attention views work

DocAttentionLens links captured model attention to the document it came from:
page overlays, annotated regions, table structure, and layers and heads.

## From model positions to document regions

Image positions have footprints on the processor-consumed page image. Geometric
overlap assigns those footprints to annotated regions. Overlapping regions share
ownership; unannotated image content, flagged geometry, and non-image targets are
accounted for separately rather than silently discarded.

The displayed measurements distinguish:

- **Mass:** a region's allocation from the complete source attention row.
- **Image share:** that allocation divided by the attention reaching the supplied
  page images, including image background.
- **Enrichment:** image share divided by the region's share of image-position
  ownership. A value of 1 means attention in proportion to area.

Display opacity and colour scaling change the presentation, not these quantities.

## Linked inspection

Choose a captured source token and an analysis method, then select a document
region. The layer–head profile shows raw attention to that region at each
captured layer and head. Choosing a profile cell updates its readout; **Show on
page** switches the page to that layer and head when the current method uses a
different aggregation.

Table views add cell measurements and row and column summaries. Small cells can
share the same image-position footprint, so the interface reports spatial
resolution limits alongside the measurements.

Saved investigations retain capture references, selections, regions, methods and
parameters. Reopening an investigation recomputes measurements from saved
attention without repeating model inference.

## What the signal means

Diagnostic attention describes captured allocations; it does not establish causal
influence or answer correctness. Retriever diagnostic readings that combine a
page and question also differ from retrieval-time MaxSim similarity, which encodes
pages and queries independently. The interface labels the selected signal and its
normalisation.

Explore the running application's [Methods catalogue](http://localhost:8080/toolbox)
for each method's inputs, calculation, parameters and references. The corresponding
[method implementations](../library/src/docattentionlens/methods/) are the source
of truth for the calculations. Availability depends on what the selected capture
contains.
