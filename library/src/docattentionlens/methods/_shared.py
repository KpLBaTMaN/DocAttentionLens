"""Common input checks and descriptor construction; no method-specific formulas."""
import numpy as np
from .contracts import LiteratureRelation, MethodDefinition, MethodDescriptor, Statistic


def selected_step(record, step):
    selected = next((s for s in record.steps if s.step == step), None)
    if selected is None:
        raise ValueError("Selection is outside captured prediction steps")
    return selected


def validate_payload(record, values, step):
    selected = selected_step(record, step)
    values = np.asarray(values)
    if values.shape != (record.layer_count, record.query_head_count, selected.target_count) or values.dtype != np.float32 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Attention payload does not match captured axes")
    if not getattr(record, "encoding_attention", False) and (selected.source_index != selected.target_count-1 or selected.predicted_index != selected.source_index+1):
        raise ValueError("Prediction and consumed-position alignment differs")
    if len(record.positions) < selected.target_count or any(p.index != i for i, p in enumerate(record.positions[:selected.target_count])):
        raise ValueError("Target position identities are not contiguous")
    return selected


def valid_support(record, step):
    selected = selected_step(record, step)
    if getattr(record, "encoding_attention", False):
        mask = np.asarray(record.attention_mask, dtype=bool)
        if len(mask) != selected.target_count or not mask.any() or not mask[selected.source_index]:
            raise ValueError("Selected encoding source or valid-key domain is unavailable")
        if getattr(record, "causal", False):
            # A causal source never sees later keys: they are outside its support, not zero weights.
            mask = mask & (np.arange(len(mask)) <= selected.source_index)
        return mask
    if not record.coverage.get("all_causal_targets"):
        raise ValueError("Complete declared causal support is required")
    mask = np.ones(selected.target_count, dtype=bool)
    recorded = record.context.input_attention_mask
    if len(recorded) != len(record.context.input_token_ids) or any(v not in (0, 1) for v in recorded):
        raise ValueError("Captured input mask differs from its position domain")
    prefix = min(len(recorded), selected.target_count)
    mask[:prefix] = np.asarray(recorded[:prefix], dtype=bool)
    if not mask.any():
        raise ValueError("Captured row has no valid causal targets")
    return mask


def selection(indices, count):
    selected = list(range(count)) if not indices else list(indices)
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < count for i in selected):
        raise ValueError("Selected axes must be distinct captured indices")
    return selected


def validate_parameters(parameters, record):
    step = selected_step(record, parameters.step)
    if hasattr(parameters, "layer") and not 0 <= parameters.layer < record.layer_count:
        raise ValueError("Layer is outside captured coverage")
    if hasattr(parameters, "head") and not 0 <= parameters.head < record.query_head_count:
        raise ValueError("Head is outside captured coverage")
    for name, count in (("heads", record.query_head_count), ("layers", record.layer_count)):
        if hasattr(parameters, name):
            selection(getattr(parameters, name), count)
    # Retriever encodings are bidirectional, so the bound names valid targets, not causal support.
    if hasattr(parameters, "k") and parameters.k > (support := int(valid_support(record, parameters.step).sum())):
        raise ValueError(f"Top-k exceeds this source's {support} valid target positions")
    return step


def selected_rows(context, parameters, heads=None, layers=None):
    record, values = context.record, context.values
    validate_payload(record, values, parameters.step)
    validate_parameters(parameters, record)
    heads = selection(heads if heads is not None else [parameters.head], record.query_head_count)
    layers = selection(layers if layers is not None else [parameters.layer], record.layer_count)
    rows = values[np.ix_(layers, heads, range(values.shape[-1]))].astype(np.float64)
    mask = valid_support(record, parameters.step)
    if np.any(rows[..., ~mask] > 1e-8) or np.any(np.abs(rows.sum(axis=-1)-1) > .02):
        raise ValueError("Captured softmax normalization or masked support differs")
    return rows, mask


def probability_row(context, parameters):
    rows, mask = selected_rows(context, parameters)
    row = rows[0, 0, mask]
    # Explicitly disclosed in statistic results; never modify saved attention.
    return row / row.sum(), mask


def statistic(name, value, units="dimensionless"):
    return Statistic(name=name, value=None if value is None else float(value), units=units)


STAT_NORMALIZATION = "probability statistics divide valid captured weights by their selected-row sum; saved weights unchanged"


def parameter_schema(parameters):
    """Materialize default factories so visible controls match actual execution defaults."""
    schema=parameters.model_json_schema()
    for name,field in parameters.model_fields.items():
        if not field.is_required():
            schema["properties"][name].setdefault("default",field.get_default(call_default_factory=True))
    return schema


# Known-answer workspace rendering check for every method that declares a page overlay.
RENDERED = "frontend/tests/rendering/page-overlay-methods.spec.ts: the workspace page draws this method's own known-answer row at each patch's source box"


def definition(*, method_id, name, perspective, family, quantity, formula, parameters, calculate,
               units="dimensionless mass", variant_id="captured_prediction_v1", displays=None,
               summary=None, limitations=None, requires=None, normalization="original causal target domain; no renormalization",
               references=None, literature=None):
    """`references` are checked primary sources, listed before the design note; `literature`
    is a (kind, note) pair stating how the calculation relates to them."""
    descriptor = MethodDescriptor(method_id=method_id, variant_id=variant_id,
        descriptor_version="0.10.0", status="implemented", implementation_version="saved_prediction_methods_v1",
        perspective=perspective, family=family, name=name, summary=summary or formula, question=f"What does {name} show over the declared captured prediction support?",
        evidence_kind="attention_allocation" if quantity in ("attention_weights", "head_envelope", "element_attention_mass") else "attention_summary",
        quantity=quantity, definition=formula, source_references=[*(references or []), "design/proposals/attention-method-implementation-plan.md"],
        literature_relation=LiteratureRelation(kind=literature[0], note=literature[1]) if literature else None,
        requires=requires or ["attention_weights", "generator_prediction_row_capture", "complete_declared_support",
                             *(["validated_ownership_map"] if quantity in ("attention_weights", "head_envelope", "element_attention_mass") else [])],
        execution_requirement="saved_capture", target_requirement="None", parameters_schema=parameter_schema(parameters),
        output={"schema": "method_analysis", "units": units, "signed": False,
                "support": "valid causal targets of selected captured predictions", "normalization": normalization},
        displays=displays or ["page_overlay", "element_allocation"],
        comparison_constraints=["Same captured source, causal support, axes and method normalization."],
        limitations=["One retained prediction or question-token source; no arbitrary prefill or vision-source attention.",
                     "Descriptive allocation or statistics do not establish output importance or causation.", *(limitations or [])],
        validation_evidence=["library/tests/methods: mathematical fixtures; saved-capture API/UI checks are scoped separately",
            *([RENDERED] if "page_overlay" in (displays or ["page_overlay"]) else [])])
    return MethodDefinition(descriptor, parameters, calculate, True)
