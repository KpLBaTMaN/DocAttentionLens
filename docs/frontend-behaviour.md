# Frontend behaviour notes

Detailed behaviour and design notes for the frontend, moved out of the source so the code
carries one-line summaries. Sections follow the files under `frontend/src/`; each heading
names the component, function, type or CSS rule the note describes.

## app/HeaderSlot.tsx

### `HeaderSlot`

Renders its children inside the top bar's context area. The element may appear after the first render;
children render once it exists. Without a provider the element is located by id (and watched for if absent).
`inline` renders the children in place when there is no top bar, for isolated component tests.

## features/attention-tool/fullAttentionEstimate.ts

### `UNKNOWN_PAGE_POSITIONS`

Storage estimate for "Save full attention matrices".

The capture keeps, per decoder layer, every query head's causal lower triangle of the prefill
attention as raw BF16: layers × heads × n(n+1)/2 × 2 bytes, where n is the number of prefill
positions. It also keeps the decode row of every answer token (float32), not only the first
`captured_steps` that ordinary captures keep. Step k is the row of position n+k−1 over n+k causal
targets, so the rows add layers × heads × Σ_{k=0}^{T−1} (n+k) × 4 bytes. T is the model's
`full_attention_captured_steps`, the most answer tokens a capture keeps; an answer may stop sooner,
so the estimate is an upper bound ("up to about X GB"). Ordinary captures keep the first
`captured_steps` rows too; they are a few percent of the total and stay inside the bound.

n is estimated before the run:
- Image positions per page follow the Qwen-VL smart resize: the page is scaled, keeping its aspect
  ratio, to a multiple of one merged cell (patch size × spatial merge) per side, with at most
  `image_positions_per_page` merged cells (and at least 4). With the page's pixel size unknown, a
  page counts as `UNKNOWN_PAGE_POSITIONS` (494, an A4 page at the current resize).
- Text positions: about 20 fixed template and special tokens, 7 more per extra page, and one
  token per 4 characters of system instruction and question.

Saved Qwen2.5-VL, Qwen2-VL and Qwen3-VL captures (16 October 2026) match this: a 2480 × 3509
page has 494 image positions (475 for 2475 × 3225); one page adds 7 special and 41 template
tokens with the default instruction, two pages 9 and 46; the question takes 3.1–4.6 characters
per token. The estimate is slightly high (about 8 positions on one page), never far low.

### `fullAttentionSteps`

Most answer tokens a full-attention capture keeps attention rows for: the catalogue's
`full_attention_captured_steps` (64 for the Qwen generators). A catalogue that predates the field falls back
to `max_new_tokens`, since a full-attention capture keeps every answer token. Undefined when neither is declared.

## features/attention-tool/ModelCapabilities.tsx

### `answerRowCoverage`

Which answer tokens keep their attention row: the first `captured_steps` on an ordinary capture, and on a
capture with "Save full attention matrices" every answer token (`full_attention_captured_steps`, up to the
output limit). Undefined for a model without answer rows (a retriever).

## features/document/attentionDisplay.ts

### `attentionAlpha`

Page patches and query tokens share one intensity mapping, so a token and a
patch with the same weight look the same at the same display maximum.
`floor` (default 0) is where the scale starts: the page minimum when the range is stretched.

### `AttentionMode`

Original hides attention; Heatmap replaces covered patches with opaque colour; Overlay tints the
page amber by weight; Blend lays the colour map over the whole page at the attention opacity, so
weak areas darken the page and strong areas glow.

## features/document/AttentionLayer.tsx

### `AttentionLayer`

Page attention in source coordinates. Overlay keeps the page visible and lets intensity carry
weight; Heatmap replaces covered patches with an opaque colour scale; Blend lays the colour scale
over every patch at the attention opacity. Smoothing interpolates one pixel per patch over a
regular patch grid. With a page-minimum floor the scale is stretched from the smallest shown value.

## features/document/AttentionPanel.tsx

### `AttentionSource`

What the calculation reads, at the top of Details › Method: for a generator, the source row (a prediction step
or a captured question token); for a retriever, the reading (encoding) and its source position, with shortcuts.
Without a capture it offers to choose or acquire one. The source line above the page chooses the same source.

### `MethodReadiness`

Whether the chosen method can run on this capture, in Details › Method above the method list: the catalogue's
and the prepared structure's errors, the method's readiness, the document-aware structure's provenance and,
while grouped, what is grouped.

## features/document/AttentionToolbar.tsx

### `AttentionToolbar`

Display-only attention controls for comparison maps; captured values and normalization never
change. Mode, opacity and scale stay in the bar with Patches and Smooth behind ⚙. The workspace's
own display choices are in Display settings, opened from the legend on the page.

## features/document/CanvasInstruments.tsx

### `MethodSentence`

The method sentence, on one line: what the page shows ("What “Tony” attends to") and the method that
computes it (a button opening the method picker). The exact row it reads (step, positions, layer,
head, column target, population or MaxSim winner) is its tooltip and screen-reader text, and is
stated in Details › Result; the layer scrubber shows layer and head.

### `CanvasLegend`

The colour legend at the canvas's bottom-left, on one line: the ramp with its floor and bound, the
image share ("Page gets 0.9% of attention"), ⓘ when that share is small (expands on click) and a
button naming the colour mode that opens Details › Display. The bound and the share never depend on
hover; the value under the pointer appears inline only while hovering. The full quantity and what
sets the bound are in Details › Display and, for screen readers, in `range` here.

## features/document/CaptureMenu.tsx

### `CaptureMenu`

The analysed role, model and capture: the model chip in the top bar. Opening it shows what the role
does, the full model identity and capture coverage, switches between the case's saved generator
captures, and starts a new capture (more evidence for this one, or a new configuration). The role is
fixed by the capture, so it is described here rather than switched.

## features/document/DetailsPanel.tsx

### `DetailsPanel`

The Details panel (right; a bottom sheet on phones): Method, Element, Result, Display, Run and Notes as tabs,
or the drawer's Result, Display, Run and Notes in the three-column layout (`tabs`).
Only the chosen tab is rendered. "Open as wide table" widens the panel over the page for large
result tables; Escape or the same button returns it to its column.

## features/document/DisplaySettingsPanel.tsx

### `DisplaySettingsTab`

Details › Display, in two views switched at its top. Settings: every display choice in one scrolling
column beside the page, grouped as Look (style, colour mode, opacity), Colour and scale (colour map,
scale, range), Detail (smoothing, patch grid) and Page (input view, element colouring, document
overlays). Scale: the readout of what each colour means, where the row mass goes and the row totals.
Nothing covers the page, so every change previews live; settings save with the investigation. Display
only: captured values and normalization never change.

## features/document/elementEnrichment.ts

### `pageEnrichment`

Area-normalised allocation of every ownership bucket on one page, with the element profile's denominators.
The measured image positions J are the row's image targets below its target count and, for a causal source,
within its reach; |J| counts every supplied page, as the row's image mass does. For bucket b on this page,
area share s_b = Σ_{j∈J} W_jb / |J|, image share p_b = m_b / I, and Enrichment E_b = p_b / s_b.

## features/document/ElementProfile.tsx

### `ElementProfile`

Details › Element: every captured layer and query head's attention to the selected element from the shown
source, as one grid on a common scale with layer means, the strongest heads and a readout of the focused
cell. Choosing a cell sets the layer scrubber (`linked`), so the page shows that head; otherwise a cell only
updates the readout and Show on page switches the page to element aggregation at that layer and head.

## features/document/ElementSummary.tsx

### `SourceReach`

What one source of a causal encoding can attend to: itself and earlier active positions, so on
the page the patches above it and to its left. `seen` of `total` page patches are reachable;
`onWholePage` switches to the last position, which reaches the whole page.

### `elementCoverage`

Element values on one page: each element's mass, the patches it owns there, and how many of them
the row measures (below its target count and, for a causal source, within its reach). The page
rail and the element summary read the same coverage, so they can never disagree.

### `ElementSummary`

Element attention mass for the displayed page, linked to the canvas through the shared
element selection. Missing structure (no mapped patches, unreviewed geometry, unsupplied
page) is stated as missing; only mapped elements carry a measured value, which may be zero.

## features/document/EncodingReadings.tsx

### `EncodingReadings`

What the retriever read: Question, Page, or the diagnostic Page + question pass, which may also
carry the case's reference answer. Attention captures save all three readings; one missing from an
older capture stays visible but disabled. The opt-in reference-answer pass has a link to capture it.
The prompt itself, with its attention, is the Prompt view of the token row above the page.

## features/document/Filmstrip.tsx

### `Filmstrip`

The page filmstrip: the capture's pages as thumbnails with their page number, supplied/omitted state
or retrieval rank and score, and, in the context lens, the shown row's attention mass on each page.
Shown only when there is more than one page to choose. Arrow keys move between pages, Enter or
Space chooses one; PageUp and PageDown change page from anywhere in the workspace.

## features/document/LayerScrubber.tsx

### `LayerRange`

Two endpoints on the layer track (rollout-style interval). The rules match
`RolloutLayerControls`: moving the start past the end pushes the end, moving the end
below the start pulls the start, and an invalid saved interval is reported, never
changed on load. `fixedStart` is full rollout: the interval always starts at layer 0.

### `LayerScrubber`

Layer and head selection under the page: a layer track (or a rollout interval) and head
cells. Keeps the accessible names of the panel controls it replaces ("Attention layer",
"Attention layer slider", "Attention head", "Attention head slider", "Start layer",
"End layer", "Rollout layer interval").

## features/document/MethodBar.tsx

### `MethodPanel`

The method panel of the three-column layout: what the method reads, its grouping, its settings and the method
list, opened under the bar from the method step or Browse methods (M). Escape, its close button, the method step
and a press outside it close it; presses in `keepOpen` (the source line and the layer scrubber, where the source,
layer and head are chosen) keep it, so a method's settings can be tuned as its result updates.

### `MethodBar`

The method bar under the source line: how the page's colour is computed, as labelled steps
("Attention: Mean across heads → Grouped into: Element totals"). Steps open Details › Method, or in the
three-column layout the method panel under the bar and Grouping's own menu; the arrows step through the
ready methods in place. It is part of the sticky source band, so it stays in view and the page fit accounts for it.

## features/document/methodChoices.ts

### `MethodChoice`

Method choices for the workspace method picker. These are the rules the attention
panel applies today, as pure functions: which descriptors each model role lists,
whether the open capture supports them and, if not, why.

## features/document/methodState.ts

### `openedView`

Clicking shows what a position attends to. Only a generator image patch, which has no
captured row, opens its incoming column. Every retriever position has its own row, so a
saved retriever column reopens as its target's row.

### `allHeadsState`

What the scrubber's "All" cell is for the shown method: `single` when it is a single-head method with a
head mean to switch to, `all` when that head mean is shown with every head explicitly selected (what the
All cell itself selects, so the saved parameters say which heads were averaged). Anything else,
including a head mean over a subset or over the default (empty) selection, has no All cell.

## features/document/PageElements.tsx

### `PageElements`

The page's element list (the ledger left of the page, or Details › Element), in reading order or by mass.
Whenever the shown row allocates to elements, each element carries its share of the row's image attention
(or, by choice, of the whole row, where the text tokens usually hold most of it), a bar scaled to the page's
largest element and its exact mass and both shares in the tooltip. Enrichment instead divides the image share
by the element's share of image positions, on a log bar about 1×, and marks elements owning less than one
image position, whose ratio is unstable. Elements a causal source cannot reach keep
their place, greyed, so the end of its reach is visible. The full table and patch contributions
are in the element summary (`onDetails`).

## features/document/perspective.ts

### `Perspective`

Canonical method perspectives describe their calculation requirements. The frontend
can additionally group a compatible General result using saved document ownership;
documentGrouping.ts resolves that view without changing the canonical descriptor.

## features/document/PromptTokens.tsx

### `PromptTokens`

The whole prompt as the model read it, in sequence order. A header names it and the source the page shows;
a bar shows how the source's attention divides between the prompt's parts; then one row per part: the
context (template runs and page images), the question and the answer. Tokens are shaded with TokenStrip's
attention heat. Clicking a token, or stepping with the arrows, chooses the source.

With `onOpen` the prompt can be hidden to its header, and with `onHeight` its rows are resized by dragging
the bottom edge (dragging it to the top hides it).

## features/document/ScaleKey.tsx

### `ScaleKey`

One key for every colour scale in view: each token strip and the page, with its range, what sets it
and where its maximum is, then how much of the row reaches the page. It sits at the bottom of the
right panel, and joins the colour bar under the page while that panel is hidden. In the panel it can
be folded to one range per line, which leaves the layer and head controls more room.

### `PageKey`

The page's colour scale in the canvas legend: the start value (0, or the page minimum when stretched),
the ramp in the shown style and the maximum on one line, with a marker and the value of the patch or
element under the pointer while hovering. Display only, like the ramp it explains.

## features/document/TokenExplorer.tsx

### `SourceLine`

The source line above the page: one horizontally scrolling row of the tokens that can be a source
(question → answer for a generator; the reading's question, reference answer or query for a retriever;
the query for MaxSim), then the page chip. Clicking a token, or a patch on the page, chooses what the page
shows. Template tokens appear only in the full prompt, the model's whole sequence in order, which the
Source tokens / Full prompt switch shows in place of the row. The prompt is given the switch and the ⋯
options to place in its own header. A generator patch's column replaces the row's meaning, so it is named
with a clear button.

## features/document/TokenStrip.tsx

### `TokenStrip`

One row of sequence tokens. Clicking a token selects it as the attention source;
when values are given, token heat uses the page heatmap's colour and intensity mapping.
The strip's range and maximum are stated in the colour scale key, beside the page's.

## features/document/useAttentionColumns.ts

### `useAttentionColumns`

"What attends to this": a generator image patch's weight in every captured answer and question
row (patches have no rows of their own), or one patch's similarity to every query token.
Retriever positions always have their own row, so the workspace never reads their columns.

## features/document/useMethodSelection.ts

### `useMethodSelection`

The method selection rules shared by the method picker on the canvas and the Details panel:
which methods each role lists, the selected method and perspective, element grouping, and the
perspective memory. Changing perspective re-analyses the same saved row; it never runs a model.

### `choose`

A choice from the method picker. Its perspective filter only filters; choosing a method from the
other perspective switches perspective the way the panel's perspective control did: element
aggregation (or the same calculation) re-groups the shown row, any other method is chosen with
the prediction or source, layer and head kept.

## features/document/useRetrieverAttention.ts

### `defaultSource`

Source shown when an encoding opens. In a causal encoding (ColQwen2) position 0 can
attend only to itself, so start from the last active position, which sees the whole
page and prompt. A Page + question reading starts on the last question word; with the
reference answer it starts on the last answer token. Bidirectional document encodings keep position 0.

## features/document/useWorkspaceShortcuts.ts

### `shortcutAction`

The workspace action for a key press, or undefined. Shortcuts never act while the user types in a
field (Ctrl/Cmd + S and K excepted), with Alt held, or while a dialog, menu or sheet is open
(`blocked`); Escape is left to whatever is open. Shift is allowed: it produces `{` `}` `?`.

### `useWorkspaceShortcuts`

One keyboard handler for the workspace. `handlers` may change on every render; an action without a
handler is ignored and leaves the key to the browser. `blocked` is the workspace's own open state
(picker, Display, palette, sheets); open dialogs and menus are detected as well. Ctrl/Cmd + S is
always kept from the browser's "Save page".

## features/document/Workspace.tsx

### `previousPerspective`

Switching to Document-aware opens Details on Element, where the element values are listed, unless the
switch was made in Details › Method, which stays open for browsing. The first perspective of a loaded method is not a switch.
The three-column layout lists the element values beside the page already.

### `ledgerLayout`

Wide windows put the page's elements left of the page and the selected element right of it; the method opens
under the method bar and Details is a drawer over the element. Table grounding cases (a table, not layout
elements) and narrower windows keep the tabbed Details panel.

### `fullAttention`

A full-attention capture keeps complete rows for every position below its square limit, so full-matrix
methods may read an image patch as their source (request-level `source_index`). Image-source methods
(vision → text, vision → vision) start from the first patch of the page; methods that read the
retained prediction and question rows return to the prediction step.

### `methodProperties`

The exact row the page shows: a column's target, a captured source population, the MaxSim winner, or the
source row with its positions, layer and head. It is the method sentence's tooltip and screen-reader text
(with its test ids) and is stated in Details › Result.

### `showIncoming`

A generator image patch precedes the question and has no captured row, so clicking it shows which
answer and question rows attend to it. Columns are raw attention: the click selects Raw attention
and position grouping, so page, tokens and result stay on one quantity.

### `reachSource`

A causal retriever encoding (ColQwen2): one source reaches itself and earlier active positions, so on
the page only the patches above it and to its left. The last position reaches the whole page.
Population views combine sources, so they state no single reach.

### `heatScale`

Token heat uses the page's colour and intensity mapping. Image contrast scales the page to image
weights, which would saturate every token, so tokens then use their own maximum; full-row and fixed
scales are shared exactly with the page (a column's full scale is its largest source weight).

### `uncapturedAnswer`

A generator answer token is the prediction of one step; steps whose token is not in the positions are added.
Every answer token has a step on a full-attention capture that kept every decode row (schema 0.8.0). Ordinary
captures, and full-attention captures made before that, keep the first prediction steps only; later answer
tokens say why they cannot be chosen.

### `openDetails`

Opens Details (a bottom sheet on phones), optionally on a tab, and moves focus into it. With
`sheet: false` a closed phone sheet stays closed; the tab is still chosen. In the three-column layout
Method opens the method panel and Element shows the element inspector.

### `Element attention profile (Details › Element)`

Every captured layer and head's attention to the selected element from the shown source: its own analysis
of the same saved row, so the page method, its axes and parameters are unchanged.

### `Details tabs`

The Element tab is the selected element, the page's element list and the element summary. The three-column
layout shows the list as the ledger left of the page and the rest as the element inspector right of it.

## features/document/WorkspaceControls.tsx

### `useMenuDismiss`

Dismissal for a <details> menu: a press outside it, Escape, or keyboard focus moving out.
A press on text inside the menu also moves focus out (to <main>); closing the menu during
that press crashed Chrome and Edge 155, so focus changes are ignored while a press is down.

## features/marking/markingModel.ts

### `xyCut`

Reading order by recursive XY-cut. Each step cuts once, at the widest empty band across the boxes (a row band
wins a tie), and orders each side the same way: a full-width heading separates first and columns then read one
by one, even where their paragraph gaps line up. Boxes that no band separates read top to bottom, then left to right.

## features/tables/TableComparisons.tsx

### `TableComparisons`

The saved captures a table-aware comparison reads: the question-swap baseline for Lookup-path
allocation, or the partner question for a matched-question contrast. The chosen IDs are sent with
the request, so the saved analysis records exactly which captures it used.

## features/tables/tableGeometry.ts

### `TableResolution`

How finely a planned input resolves a table: an estimate before the run, from the page size and the
model's resize rule. A cell is *isolable* when at least one merged image position lies at least half
inside it; other cells share every position with their neighbours, so cell-level values blur.

## features/tables/TableHeatmap.tsx

### `TableHeatmap`

Table-aware result as the logical table: one block per cell (merged cells keep their span), coloured
by mass, enrichment or resolution, with table row and column marginals and the page-region budget.
Hatched cells have no image position of their own. Grounding outlines are evaluation-only.

## features/tables/TableOverlay.tsx

### `TableOverlay`

Table structure drawn over the original page, in page pixels: the table and its caption-and-notes
area, the logical cell grid, and (evaluation only) the question's answer cell and lookup path. This is
orientation; no attention value is drawn here.

## features/toolbox/methodFacets.ts

### `facetOptions`

Values offered by one facet with the number of methods each would show,
counted against the search and the other facets. Statuses are always listed so
an empty "Planned" reads as none rather than missing.

## geometry/transform.ts

### Module overview

Display geometry for page overlays. Source coordinates are page image pixels
with a top-left origin (see ElementPage.coordinate_convention). One transform
maps source pixels to the visible image rectangle; boxes, markers and hit
testing all use it, so they stay aligned through resize, zoom and pan.
This is presentation only: it never changes source geometry or ownership.

## styles/analytical.css

### File overview

 Attention Tool workspace.
Layout: the app top bar carries the case, lens tabs, model chip and Save; under it the source line, then the
page canvas with floating instruments (method sentence, Details toggle, legend, zoom pill) and the layer
scrubber. The filmstrip (left, only with more than one page) and Details (right, tabbed) are optional
columns; on phones they are bottom sheets opened from the bottom bar. At 1200 px and wider the workspace has
three columns: the element ledger left of the page and the element inspector right of it (the Details drawer
covers the inspector while open); the method panel opens under the method bar.

### `.docattentionlens-app .docattentionlens-analysis .docattentionlens-canvas`

 The canvas box follows the fit mode. Fit page keeps the page's aspect ratio within the first-screen window;
Fit width uses the column width and the window scrolls; manual zoom pans inside the first-screen window.
PageCanvas fits its single source transform to this box.

## styles/library.css

### `.docattentionlens-app .docattentionlens-library`

 Library area (/artifacts): side list of sections, flat section headers and
record rows that lead with the name. Scoped to the page root so these rules
win over the shared application rules.

## styles/prompt-tokens.css

### `.docattentionlens-prompt-tokens`

 The whole prompt in sequence order (PromptTokens): a header naming it and the source, a bar of its parts, then
one row per part (context, question, answer). Heat is the chip background and amber means attention only, so
parts are named by row labels and chip underlines in their own hues, and the source is outlined in the page's
source blue. Missing parts are dashed and dimmed.

## styles/foundation.css

### File overview

 DocAttentionLens visual foundation.
Scope the application with .docattentionlens-app. Themes are explicit user preferences.
Scientific color scales and source-page colors are separate from UI accents.

### `.docattentionlens-app :is(.docattentionlens-button, .docattentionlens-quiet, .docattentionlens-icon-button)`

 Buttons: .docattentionlens-button is Secondary (outlined); add .docattentionlens-primary for the one Primary action in
a view. .docattentionlens-quiet is text only and .docattentionlens-icon-button is a 28 px icon-only control that must carry
an aria-label. Labels stay in sentence case and the sans face.
