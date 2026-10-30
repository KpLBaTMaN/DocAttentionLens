"""Verified document selections and shared bookkeeping; method formulas stay in their owners."""
import numpy as np
from .contracts import ResultTable, ResearchPlot, ResearchPoint
from ._shared import selected_rows, selected_step, definition
from .element_allocation import allocate_row

REQUIRES = ["attention_weights", "generator_prediction_row_capture", "complete_declared_support",
            "validated_ownership_map", "mapped_elements", "element_type_annotations"]


def document(context):
    if context.document is None:
        raise ValueError("Verified case annotations are required")
    ids = [e.element_id for e in context.document.elements]
    if len(ids) != len(set(ids)):
        raise ValueError("Document element identities are not unique")
    return context.document


def elements(context):
    return {e.element_id: e for e in document(context).elements}


def row(context, parameters):
    rows, mask = selected_rows(context, parameters)
    selected = selected_step(context.record, parameters.step)
    return allocate_row(context.record, rows[0, 0], selected, parameters.step, parameters.layer, parameters.head), mask


def masses(context, parameters):
    result, mask = row(context, parameters)
    known = elements(context)
    measured = {}
    for allocation in result.allocations:
        if allocation.element_id:
            if allocation.element_id not in known or known[allocation.element_id].page_id != allocation.page_id:
                raise ValueError("Ownership refers to an element outside the verified annotation context")
            measured[allocation.element_id] = allocation.mass
    if not measured:
        raise ValueError("No annotated element is mapped to this selection")
    return result, measured, known, mask


def ownership_weights(record, mask, element_ids):
    """Fractional ownership of each valid target position by each listed element: positions × elements.

    Masked targets carry no weight; callers verify the element identities. A layer × head × element
    mass is then `rows @ weights`, and a summed element set uses the row sum of its columns.
    """
    column = {element_id: i for i, element_id in enumerate(element_ids)}
    weights = np.zeros((len(mask), len(element_ids)))
    for j, position in enumerate(record.positions[:len(mask)]):
        if not mask[j]:
            continue
        for bucket, fraction in position.ownership.items():
            if bucket in column:
                weights[j, column[bucket]] += fraction
    return weights


def table(title, columns, labels, values, units="dimensionless mass", ids=None, pages=None):
    return ResultTable(title=title, columns=columns, labels=[str(x) for x in labels],
                       values=values, units=units, element_ids=ids or [], page_ids=pages or [])


def plot(kind, title, x_label, y_label, points):
    return ResearchPlot(kind=kind, title=title, x_label=x_label, y_label=y_label,
                        points=[p if isinstance(p, ResearchPoint) else ResearchPoint(**p) for p in points])


def define(identifier, name, formula, parameters, compute, *, requires=None, units="dimensionless mass",
           variant="document_saved_rows_v1", displays=None, limitations=None, signed=False, updates=None,
           normalization="original captured domain; method-specific conditional quantities are labelled separately"):
    """`signed` declares a signed output; `updates` replaces further descriptor fields (e.g. evidence kind)."""
    result=definition(method_id=identifier, name=name, perspective="document_aware", family="document_structure",
        quantity=identifier.split(".")[-1], formula=formula, parameters=parameters, calculate=compute,
        requires=requires or REQUIRES, units=units, variant_id=variant,
        displays=displays or ["table"], limitations=limitations,normalization=normalization)
    output=result.descriptor.output.model_copy(update={"support":"verified document elements/categories on declared captured sources and explicit selected samples; residuals remain separate",
        **({"signed":True} if signed else {})})
    return result.__class__(result.descriptor.model_copy(update={"implementation_version":"document_saved_rows_v1",
        "output":output,**(updates or {})}),parameters,compute,True)


def sample_contexts(context):
    samples = context.samples or (context,)
    ids = [c.record.capture_id for c in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample captures must be distinct")
    identities = [(c.record.provenance.get("model_id"), c.record.provenance.get("model_revision"),
                   c.record.layer_count, c.record.query_head_count) for c in samples]
    if len(set(identities)) != 1:
        raise ValueError("Corpus profiles require the same model, revision and layer/head axes")
    for sample in samples:
        document(sample)
    return samples


def aligned_masses(context, parameters):
    if context.comparison is None:
        raise ValueError("Select a comparison capture")
    other = context.comparison
    for key in ("model_id", "model_revision"):
        if context.record.provenance.get(key) != other.record.provenance.get(key):
            raise ValueError("Comparison model and revision must match")
    a, ma, ea, _ = masses(context, parameters)
    b, mb, eb, _ = masses(other, parameters)
    sa = selected_step(context.record, parameters.step)
    sb = selected_step(other.record, parameters.step)
    # Free-running sources are comparable only while their consumed histories agree.
    if (context.record.context.query != other.record.context.query
        or getattr(context.record.context,"system_prompt",None) != getattr(other.record.context,"system_prompt",None)
        or context.record.output_token_ids[:parameters.step] != other.record.output_token_ids[:parameters.step]
        or sa.consumed_token_id != sb.consumed_token_id):
        raise ValueError("Prediction sources diverge; select an earlier shared history or a matched replay")
    alignment = context.alignment or {i:i for i in ea if i in eb}
    if len(set(alignment.values())) != len(alignment):
        raise ValueError("Element lineage must be one-to-one")
    if any(i not in ea or j not in eb for i,j in alignment.items()):
        raise ValueError("Element lineage refers to absent annotations")
    common = sorted(i for i,j in alignment.items() if i in ma and j in mb)
    if not common:
        raise ValueError("Captures share no measured aligned elements")
    return a, b, common, ma, mb, alignment
