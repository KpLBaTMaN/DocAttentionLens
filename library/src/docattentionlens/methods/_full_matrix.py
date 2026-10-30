"""Generator captures with saved full attention, analysed as a causal encoding truncated at one source.

A full-attention generator capture keeps every prefill row (`generation/full_attention.py`) and the
retained prediction-step rows. For a selected source s, positions 0..s form a complete causal square:
prefill rows below n come from the saved triangle and rows n..s from prediction steps 1..s-n+1.
Square-matrix methods then run exactly as on a causal retriever encoding
(`retrieval/attention.row_context`). Nothing after the source is read or synthesized, no row is
renormalized, and any axis or identity mismatch is an error.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from ..generation import full_attention as fa
from .contracts import Context

# Methods whose descriptors need complete squares or arbitrary (image-patch) source rows.
FULL_MATRIX_REQUIREMENTS = frozenset({"full_square_attention_matrices", "captured_image_sources"})
# A (heads, s+1, s+1) float32 square is large for long multimodal prompts; bound what one calculation keeps.
CACHED_LAYERS = 4
MISSING = "This capture has no saved full attention matrices; run a new generator capture with 'Save full attention matrices'"
IMAGE_SCOPE = ("Image patches precede the question, so an image-patch row attends only to earlier template "
               "tokens and patches, never to the question or the answer.")


def full_attention(capture):
    return getattr(capture, "full_attention", None)


def applies(descriptor, capture) -> bool:
    """True when this method on this capture must use the full-matrix view."""
    return full_attention(capture) is not None and bool(FULL_MATRIX_REQUIREMENTS & set(descriptor.requires))


def checked(capture):
    """The capture's full-attention declaration, after checking it against the capture's own axes."""
    full = full_attention(capture)
    if full is None:
        raise ValueError(MISSING)
    n = full.position_count
    if full.layout != fa.LAYOUT:
        raise ValueError(f"Unsupported full-attention layout {full.layout!r}")
    if (full.layer_arrays != [fa.array_name(layer) for layer in range(capture.layer_count)]
            or full.query_head_count != capture.query_head_count or n < 1
            or len(capture.context.input_token_ids) != n):
        raise ValueError("Full attention differs from the capture's layer, head or prefill axes")
    if full.square_limit != n + max(0, len(capture.steps) - 1):
        raise ValueError("Full-attention square limit differs from the retained prediction steps")
    # Step k is the row of position n+k-1 over n+k targets; the decode rows depend on this alignment.
    for k, step in enumerate(capture.steps):
        if step.step != k or step.source_index != n + k - 1 or step.target_count != n + k:
            raise ValueError("Prediction steps are not aligned with the full-attention prefill")
    return full


def select(capture, step=0, source_index=None):
    """Selected source s, its predicted (next) position and the retained row array for s, if any.

    A prediction step selects its consumed position n+k-1; an explicit source selects any position
    with a complete saved row (question tokens, template text or image patches).
    """
    full = checked(capture)
    if source_index is None:
        selected = next((s for s in capture.steps if s.step == step), None)
        if selected is None:
            raise ValueError("Selection is outside captured prediction steps")
        source, predicted, array = selected.source_index, selected.predicted_index, selected.array
    else:
        source, predicted = source_index, source_index + 1
        retained = next((r for r in [*capture.steps, *getattr(capture, "source_rows", [])] if r.source_index == source), None)
        array = retained.array if retained else None
    if not 0 <= source < full.square_limit:
        raise ValueError(f"Source position {source} has no saved full-attention row: this capture keeps complete rows "
                         f"for positions 0–{full.square_limit - 1} (prefill plus retained prediction steps); "
                         "later rows were never captured")
    return source, predicted, array


def view(capture, source):
    """Causal encoding record of positions 0..source, mirroring a retriever encoding row context."""
    full = checked(capture)
    n = full.position_count
    recorded = list(capture.context.input_attention_mask)
    if len(recorded) != n or any(v not in (0, 1) for v in recorded):
        raise ValueError("Captured input mask differs from the full-attention prefill")
    # Generated positions are appended unmasked by decoding; nothing beyond the source is included.
    mask = (recorded + [1] * max(0, source + 1 - n))[:source + 1]
    positions = list(capture.positions[:source + 1])
    if len(positions) != source + 1 or any(p.index != i for i, p in enumerate(positions)):
        raise ValueError("Position identities do not cover the selected source")
    if not mask[source]:
        raise ValueError("The selected source is a masked input position")
    return SimpleNamespace(capture_id=capture.capture_id, encoding_attention=True, causal=True,
        layer_count=capture.layer_count, query_head_count=capture.query_head_count,
        attention_mask=mask, positions=positions,
        steps=[SimpleNamespace(step=0, source_index=source, predicted_index=source + 1, target_count=source + 1)])


def decode_steps(capture, source):
    """Prediction steps whose rows complete the square up to the source (steps 1..source-n+1)."""
    n = checked(capture).position_count
    return list(capture.steps[1:source - n + 2]) if source >= n else []


class Reader:
    """Layer squares and source rows of one selected source; `read(name)` returns a verified array."""

    def __init__(self, capture, source, read):
        full = checked(capture)
        self.capture, self.source, self.read = capture, source, read
        self.n, self.heads = full.position_count, capture.query_head_count
        self.arrays = full.layer_arrays
        self.steps = decode_steps(capture, source)
        # Decode arrays are small (layers, heads, n+k) rows; each is read and verified once.
        self._step = lru_cache(maxsize=None)(lambda name: np.asarray(self.read(name)))
        self.layer_matrix = lru_cache(maxsize=CACHED_LAYERS)(self._square)

    def _layer(self, layer):
        if not 0 <= layer < self.capture.layer_count:
            raise ValueError("Layer is outside captured coverage")
        return layer

    def packed(self, layer):
        packed = np.asarray(self.read(self.arrays[self._layer(layer)]))
        if packed.dtype != np.uint16 or packed.shape != (self.heads, fa.triangle_length(self.n)):
            raise ValueError("Full attention triangle differs from its declared axes")
        return packed

    def decode_rows(self, layer):
        rows = []
        for step in self.steps:
            values = self._step(step.array)
            if values.shape != (self.capture.layer_count, self.heads, step.target_count) or values.dtype != np.float32:
                raise ValueError("Retained decode row differs from the full-attention axes")
            rows.append(values[self._layer(layer)])
        return rows

    def _square(self, layer):
        """Complete causal (heads, s+1, s+1) square for one layer."""
        return fa.square(self.packed(layer), self.n, self.source, self.decode_rows(layer))

    def source_row(self, layer):
        """The source's own (heads, s+1) row, exactly as in the square, without building the square."""
        if self.source >= self.n:
            return np.asarray(self.decode_rows(layer)[self.source - self.n], dtype=np.float32)
        # Rows are stored in order, so row s occupies entries s(s+1)/2 .. s(s+1)/2 + s.
        start = fa.triangle_length(self.source)
        return fa.from_bits(self.packed(layer)[:, start:start + self.source + 1])


def context(capture, source, read, layers, document=None):
    """Method context: the truncated record, the source row for each used layer and a square reader.

    Unused layers stay zero, as in the retriever path; a method reading one fails its row-sum check.
    """
    record = view(capture, source)
    reader = Reader(capture, source, read)
    values = np.zeros((capture.layer_count, capture.query_head_count, source + 1), dtype=np.float32)
    for layer in sorted(set(layers)):
        values[reader._layer(layer)] = reader.source_row(layer)
    return Context(record, values, document, read_array=reader.layer_matrix), record


def prepare(capture, definition, parameters, source_index=None):
    """Select the source and parse parameters against the truncated view.

    Returns (source, predicted index, retained array or None, view record, execution parameters,
    saved parameters). Execution always uses the view's single step 0; the saved parameters keep
    the requested prediction step, so analyses of different steps have different identities.
    """
    from .registry import prepare_parameters
    descriptor = definition.descriptor
    requested = definition.parameters.model_validate(parameters)
    step = 0 if source_index is not None else requested.step
    source, predicted, array = select(capture, step, source_index)
    record = view(capture, source)
    parsed = prepare_parameters(descriptor.method_id, descriptor.variant_id, {**parameters, "step": 0}, record)
    saved = parsed.model_dump(mode="json") | {"step": step}
    return source, predicted, array, record, parsed, saved


def used_layers(parameters):
    """Layers whose source rows populate `values`: the selected layer and any listed layers."""
    return {getattr(parameters, "layer", 0), *getattr(parameters, "layers", [])}


def identity(capture, source):
    """Hashes of every full-attention array and of the decode rows used for this source."""
    full = checked(capture)
    refs = {r.name: r.sha256 for r in capture.arrays}
    needed = [*full.layer_arrays, *(s.array for s in decode_steps(capture, source))]
    if any(name not in refs for name in needed):
        raise ValueError("Capture is missing a full-attention array identity")
    return {"layout": full.layout, "position_count": full.position_count, "square_limit": full.square_limit,
            "source_index": source, "arrays": {name: refs[name] for name in full.layer_arrays},
            "decode_arrays": {s.array: refs[s.array] for s in decode_steps(capture, source)}}


def notices(record, source, n):
    rows = (f"positions 0–{n - 1} from the saved prefill triangle and {n}–{source} from retained prediction steps"
            if source >= n else f"positions 0–{source} from the saved prefill triangle")
    return [f"Analysed as a causal encoding truncated at source position {source}: rows of {rows}. "
            "Later positions are outside this view; nothing is renormalized.",
            *([IMAGE_SCOPE] if record.positions[source].role == "image" else [])]
