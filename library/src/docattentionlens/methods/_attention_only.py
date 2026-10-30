"""Input adapters for attention-only methods; method formulas remain in their owners."""
from dataclasses import replace
import numpy as np
from .contracts import ResultTable
from ._shared import definition

MATRIX_REQUIRES = ["attention_weights", "full_square_attention_matrices", "complete_declared_support"]


def define(identifier, name, family, quantity, formula, parameters, calculate, *, requires=None,
           units="dimensionless", displays=None, limitations=None, normalization="declared captured support",
           references=None, literature=None):
    result = definition(method_id=identifier, name=name, perspective="general", family=family,
        quantity=quantity, formula=formula, parameters=parameters, calculate=calculate,
        variant_id="captured_attention_v1", requires=requires or MATRIX_REQUIRES, units=units,
        displays=displays or ["scalar_statistic"], normalization=normalization, literature=literature)
    descriptor = result.descriptor.model_copy(update={
        "descriptor_version":"0.1.0", "implementation_version":"attention_only_methods_v1",
        "source_references":[*(references or []), "design/attention-only-methods.md"],
        "output":result.descriptor.output.model_copy(update={"support":"explicit captured source and active-key axes"}),
        "limitations":["Descriptive attention does not establish output importance or causation.", *(limitations or [])]})
    if quantity in {"whole_query_image_allocation","modality_block_attention","raw_attention_edges"}:
        descriptor=descriptor.model_copy(update={"evidence_kind":"attention_allocation"})
    return replace(result, descriptor=descriptor)


def text(position):
    """Textual block includes prompt, special and generated positions; it is non-image."""
    return position.role != "image"


def matrix(context, layer):
    record = context.record
    if not getattr(record, "encoding_attention", False) or context.read_array is None:
        raise ValueError("This method needs complete saved attention matrices (a retriever encoding, or a generator capture with 'Save full attention matrices'); retained prediction rows are insufficient")
    if not 0 <= layer < record.layer_count:
        raise ValueError("Layer is outside captured coverage")
    n = len(record.positions)
    value = np.asarray(context.read_array(layer))
    active = np.asarray(record.attention_mask, dtype=bool)
    if value.shape != (record.query_head_count,n,n) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("Encoding matrix differs from the declared attention axes")
    rows = value[:,active]
    if np.any(rows[...,~active] > 1e-8) or np.any(np.abs(rows.sum(-1)-1) > .02):
        raise ValueError("Encoding matrix has invalid normalization or masked support")
    if getattr(record,"causal",False) and np.any(value[:,np.triu(np.ones((n,n),bool),1)] > 1e-8):
        raise ValueError("Encoding matrix includes future causal keys")
    return value


def population(context, layer, head, group="all"):
    """Return rows, explicit row masks and source identities. Never infer support from nonzero weights."""
    record = context.source_record or context.record
    if not 0 <= layer < record.layer_count or not 0 <= head < record.query_head_count:
        raise ValueError("Layer or head is outside captured coverage")
    if getattr(record,"encoding_attention",False):
        active = np.asarray(record.attention_mask,bool)
        indices = np.flatnonzero(active).tolist()
        payload = matrix(context,layer)[head]
        masks = np.broadcast_to(active,(len(indices),len(active))).copy()
        if getattr(record,"causal",False):
            masks &= np.arange(len(active))[None,:] <= np.asarray(indices)[:,None]
        rows = payload[indices].astype(np.float64)
    else:
        if context.read_source is None:
            raise ValueError("This method needs a verified reader for multiple captured source rows")
        indices = sorted({s.source_index for s in record.steps} | {s.source_index for s in record.source_rows})
        rows = np.zeros((len(indices),len(record.positions)),dtype=np.float64)
        masks = np.zeros_like(rows,dtype=bool)
        for j,index in enumerate(indices):
            value, mask = context.read_source(index)
            row = np.asarray(value)[layer,head]
            if row.ndim!=1 or len(row)!=len(mask) or not np.isfinite(row).all() or np.any(row<0) or np.any(row[~mask]>1e-8) or abs(row.sum()-1)>.02:
                raise ValueError("A retained source row has invalid normalization or support")
            rows[j,:len(row)] = row
            masks[j,:len(row)] = mask
    keep = [j for j,i in enumerate(indices) if group=="all" or
        (group=="question" and record.positions[i].role=="query") or
        (group=="text" and text(record.positions[i])) or
        (group=="image" and record.positions[i].role=="image")]
    if not keep:
        raise ValueError(f"No captured {group} source rows are available")
    return rows[keep], masks[keep], [indices[j] for j in keep]


def source_masks(record):
    active = np.asarray(record.attention_mask,bool)
    masks = np.broadcast_to(active,(len(active),len(active))).copy()
    if getattr(record,"causal",False): masks &= np.tri(len(active),dtype=bool)
    return masks


def table(title, columns, labels, values, units="dimensionless"):
    return ResultTable(title=title, columns=columns, labels=labels, values=values, units=units)

