"""Element attention profile: each mapped element's mass at every captured layer and query head."""
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ..generation.contracts import GENERATORS, MAX_NEW_TOKENS
from .contracts import Calculation, ElementProfile, ProfileElement
from ._document import REQUIRES, define, document, ownership_weights, table
from ._shared import selected_rows

# Bounds: the selected source's complete layer × head × target payload, and the returned profile.
VALUE_BUDGET = 4_000_000
PROFILE_BUDGET = 300_000
DIGITS = 7


class Parameters(BaseModel):
    """Only the source: the profile covers every captured layer and head, so it never varies with them."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    step: int = Field(default=0, ge=0, le=MAX_NEW_TOKENS - 1, strict=True)


def rounded(values):
    return [[float(f"{v:.{DIGITS}g}") for v in row] for row in values]


def deepstack_layers(record):
    model = (getattr(record, "provenance", None) or {}).get("model_id")
    profile = next((p for p in GENERATORS.values() if p.model_id == model), None)
    # DeepStack adds vision features after decoder layers 0..n-1, so layers 1..n read them.
    return list(range(1, len(profile.deepstack_visual_indexes) + 1)) if profile else []


def compute(context, parameters):
    record = context.record
    if context.values.size > VALUE_BUDGET:
        raise ValueError("The element profile exceeds the bounded analysis budget")
    known = {e.element_id: e for e in document(context).elements}
    rows, mask = selected_rows(context, parameters, heads=list(range(record.query_head_count)),
                               layers=list(range(record.layer_count)))
    positions = record.positions[:len(mask)]
    owned = set()
    for j, position in enumerate(positions):
        for bucket in position.ownership:
            if not bucket.startswith("element_"):
                continue
            if bucket not in known or known[bucket].page_id != position.page_id:
                raise ValueError("Ownership refers to an element outside the verified annotation context")
            if mask[j]:
                owned.add(bucket)
    if not owned:
        raise ValueError("No annotated element is mapped to this selection")
    pages = list(getattr(record.context, "selected_page_ids", []) or [])
    order = lambda i: (pages.index(known[i].page_id) if known[i].page_id in pages else len(pages),
                       known[i].reading_order if known[i].reading_order is not None else float("inf"), i)
    ids = sorted(owned, key=order)
    if record.layer_count * record.query_head_count * len(ids) > PROFILE_BUDGET:
        raise ValueError(f"The element profile is bounded to {PROFILE_BUDGET:,} layer × head × element values; "
                         f"this capture has {len(ids)} mapped elements")
    weights = ownership_weights(record, mask, ids)
    mass = rows @ weights                                   # layers × heads × elements
    image = np.array([bool(mask[j]) and p.role == "image" for j, p in enumerate(positions)])
    image_count = int(image.sum())
    image_mass = rows[..., image].sum(axis=-1)
    area = weights[image].sum(axis=0) / image_count if image_count else None
    elements = [ProfileElement(element_id=i, page_id=known[i].page_id, label=f"{known[i].element_type} · {known[i].native_id}",
                               reading_order=known[i].reading_order, area_share=None if area is None else float(area[k]),
                               mass=rounded(mass[..., k])) for k, i in enumerate(ids)]
    unmeasured = sorted((i for i, e in known.items() if e.page_id in pages and i not in owned), key=order)
    layers = deepstack_layers(record)
    profile = ElementProfile(layer_count=record.layer_count, head_count=record.query_head_count,
        kv_head_count=getattr(record, "kv_head_count", None), deepstack_layers=layers, image_positions=image_count,
        image_mass=rounded(image_mass), elements=elements, unmeasured_element_ids=unmeasured, significant_digits=DIGITS)
    peaks = []
    for k in range(len(ids)):
        layer, head = np.unravel_index(int(np.argmax(mass[..., k])), mass.shape[:2])
        peaks.append([float(mass[layer, head, k]), float(layer), float(head), float(mass[..., k].mean()), elements[k].area_share])
    notices = ["Each cell is one captured layer and query head allocating the selected source row to this element. "
               "It describes attention, not causal influence."]
    if unmeasured:
        notices.append(f"{len(unmeasured)} annotated element{'s' if len(unmeasured) != 1 else ''} on supplied pages "
                       "have no mapped valid position: unmeasured, not zero.")
    if layers:
        notices.append(f"Decoder layers {', '.join(map(str, layers))}: image keys and values already include DeepStack vision features.")
    return Calculation(element_profile=profile, notices=notices,
        table=table("Element peaks", ["Peak mass", "Peak layer", "Peak head", "Mean over layers and heads", "Mapped-cell area share"],
                    [e.label for e in elements], peaks, "mass; layer; head; mass; fraction", ids, [e.page_id for e in elements]),
        normalization=f"original captured domain; no renormalization; profile values rounded to {DIGITS} significant digits")


DEFINITION = define("attention.element_profile", "Element attention profile",
    "M(l,h,e) = sum_j A(s,l,h,j) w(e,j) for every captured layer l and query head h of one source row s; "
    "within-image share M/I(l,h); area-relative ratio (M/I)/area share(e).",
    Parameters, compute, requires=REQUIRES, displays=["element_profile"],
    limitations=["Opened from Details › Element for the selected element; the page method is unchanged.",
                 "Raw mass favours larger elements; the area-relative ratio uses mapped processor cells, not source-box area.",
                 "A layer mean is a separate allocation per layer, not propagation through layers."],
    updates={
    "question": "Which captured layers and query heads allocate the most attention from the selected source to a document element?",
    "summary": "Every captured layer and query head's attention from the shown source to each mapped element, as mass, "
               "within-image share and area-adjusted enrichment. Opened from Details › Element for the selected element.",
    "validation_evidence": ["library/tests/methods/test_document_methods.py: per-cell agreement with element allocation, "
                            "designated support and the head mean; area ratio; missing versus zero"]})
