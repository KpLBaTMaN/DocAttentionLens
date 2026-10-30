"""Table structure ownership computed at analysis time; method formulas stay in their owners.

Captures are unchanged. Every image position keeps its box in page pixels, and four partitions
are computed from those boxes (policy `table_cell_partition_v1`): logical cells, native row
bands, native column bands and page regions. For position j with box b_j and region r,
w(r, j) = area(b_j ∩ r) / area(b_j). Table annotation hashes enter the analysis identity, so a
reviewed table revision applies to saved captures without another model run.
"""
from dataclasses import dataclass
from typing import Literal
import numpy as np
from pydantic import Field
from .contracts import (RowParameters, ResultTable, ResearchPlot, ResearchPoint, TableAllocation, TableBandMass,
                        TableCellMass, TableRegionMass, TableResolution)
from ._shared import selected_rows, selected_step, definition
from .element_allocation import allocate_row

REQUIRES = ["attention_weights", "generator_prediction_row_capture", "complete_declared_support", "image_mask",
            "table_structure_annotations"]
GROUNDING_REQUIRES = [*REQUIRES, "table_grounding_annotations"]
HEADER_ROLES = {"row_header", "column_header"}
# A cell is isolable when one position lies at least this share inside it (as in the setup estimate).
ISOLABLE_SHARE = .5
REGION_LABELS = {"table": "Table cells", "table_gap": "Inside the table, in no cell", "table_wrap": "Caption and notes",
                 "rest_of_page": "Rest of the page (unannotated)", "other_page": "Other pages"}
NON_IMAGE_LABELS = {"query": "Question tokens", "prompt": "Prompt template", "special": "Special tokens",
                    "generated": "Generated tokens", "document_text": "Document text tokens"}


class Parameters(RowParameters):
    head_reduction: Literal["selected_head", "head_mean"] = Field(
        default="selected_head", title="Heads",
        description="The selected head, or the equal-weight mean of every query head at the selected layer.")
    layer_reduction: Literal["selected_layer", "rollout"] = Field(
        default="selected_layer", title="Layers",
        description="The selected layer's attention row, or partial attention rollout (Abnar and Zuidema) from the start "
                    "layer through the selected layer: every head averaged, residual identity mixed in, rows renormalised. "
                    "Rollout needs complete saved layer matrices (retriever encodings or full-attention captures).")
    start_layer: int = Field(default=0, ge=0, lt=128, strict=True, title="Rollout start layer")
    residual_coefficient: float = Field(default=.5, ge=0, le=1, title="Rollout residual coefficient")


class GroundingParameters(Parameters):
    include_flagged_binding: bool = Field(
        default=False, title="Include an answer/cell mismatch",
        description="Grounding analyses refuse questions whose saved answer differs from the text in their bound PDF cell.")


def inputs(context):
    if context.table is None:
        raise ValueError("Verified table annotations are required")
    return context.table


def grounding(context, parameters):
    table = inputs(context)
    ground = table.grounding
    if ground is None:
        raise ValueError("This case has no table grounding labels")
    if ground.table_id != table.table.table_id:
        raise ValueError("Grounding labels refer to a different table")
    if ground.binding_status == "needs_source_qa_review" and not getattr(parameters, "include_flagged_binding", False):
        raise ValueError("Answer/cell mismatch: the saved answer differs from the text in its PDF cell. "
                         "Choose 'Include an answer/cell mismatch' to analyse it anyway.")
    return ground


def _boxes(items):
    return np.array([[b.x1, b.y1, b.x2, b.y2] for b in items], dtype=np.float64).reshape(-1, 4)


def overlap(a, b):
    """Pairwise intersection areas of boxes a (N, 4) and b (M, 4)."""
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return w*h


def _area(boxes):
    return (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])


@dataclass(frozen=True)
class Partition:
    """Ownership of each target position in every partition; rows are target positions."""
    count: int
    on_page: np.ndarray
    cells: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    regions: dict
    non_image: dict
    cell_area: np.ndarray
    row_area: np.ndarray
    column_area: np.ndarray
    region_area: dict
    resolution: TableResolution
    isolable: np.ndarray
    largest_share: np.ndarray
    touching: np.ndarray


def partition(record, table_inputs, count):
    table, page = table_inputs.table, table_inputs.page
    positions = record.positions[:count]
    if any(p.index != i for i, p in enumerate(positions)):
        raise ValueError("Target position identities are not contiguous")
    on_page = np.array([p.role == "image" and p.page_id == table.page_id and p.bbox is not None for p in positions])
    if not on_page.any():
        raise ValueError("No captured image position lies on the table's page")
    boxes = np.zeros((count, 4))
    boxes[on_page] = np.array([p.bbox for p, keep in zip(positions, on_page) if keep], dtype=np.float64)
    area = _area(boxes)
    if np.any(area[on_page] <= 0):
        raise ValueError("A captured image position has no area")
    safe = np.where(on_page, area, 1.0)[:, None]
    share = lambda regions: np.where(on_page[:, None], overlap(boxes, regions)/safe, 0.0)
    t, w = table.bbox, table.wrap_bbox
    # Projected cell and band edges can overhang the table box by a fraction of a pixel (transform
    # rounding); the logical cells partition the table, so they are clipped to it.
    clip = lambda boxes: np.column_stack([np.maximum(boxes[:, 0], t.x1), np.maximum(boxes[:, 1], t.y1),
                                          np.minimum(boxes[:, 2], t.x2), np.minimum(boxes[:, 3], t.y2)])
    cells = share(clip(_boxes([c.bbox for c in table.cells])))
    rows = share(clip(_boxes([r.bbox for r in table.rows])))
    columns = share(clip(_boxes([c.bbox for c in table.columns])))
    inner = [max(t.x1, w.x1), max(t.y1, w.y1), min(t.x2, w.x2), min(t.y2, w.y2)]
    table_share = share(_boxes([t]))[:, 0]
    wrap_share = share(_boxes([w]))[:, 0]
    both = share(np.array([inner]))[:, 0] if inner[0] < inner[2] and inner[1] < inner[3] else np.zeros(count)
    tolerance = 1e-6
    gap = table_share-cells.sum(axis=1)
    # Sub-pixel slivers between projected cells (below the conservation tolerance) are rounding of an
    # exact tiling, not a measured gap; they stay with the table region.
    gap = np.where(np.abs(gap) < tolerance, 0.0, gap)
    if np.any(gap < -tolerance):
        raise ValueError("Table cells overlap, so they do not partition the table")
    wrap_only = wrap_share-both
    rest = 1-table_share-wrap_only
    if np.any(rest < -tolerance):
        raise ValueError("Table regions exceed a position's area")
    regions = {"table": table_share-np.clip(gap, 0, None), "table_gap": np.clip(gap, 0, None)*on_page,
               "table_wrap": np.clip(wrap_only, 0, None)*on_page, "rest_of_page": np.clip(rest, 0, None)*on_page}
    regions["other_page"] = np.array([float(p.role == "image" and not keep) for p, keep in zip(positions, on_page)])
    non_image = {}
    for i, p in enumerate(positions):
        if p.role != "image":
            non_image.setdefault(p.role, np.zeros(count))[i] = 1.0
    total = sum(regions.values())+sum(non_image.values(), np.zeros(count))
    if np.any(np.abs(total-1) > 1e-6):
        raise ValueError("Table ownership does not conserve each position")
    page_area = float(page.width*page.height)
    largest = np.where(on_page[:, None], cells, 0).max(axis=0) if cells.size else np.zeros(0)
    isolable = largest >= ISOLABLE_SHARE-1e-9
    touching = (cells > 0).sum(axis=0)
    on_table = int(((table_share > 0) & on_page).sum())
    page_boxes = boxes[on_page]
    position_height = float(np.median(page_boxes[:, 3]-page_boxes[:, 1]))
    position_width = float(np.median(page_boxes[:, 2]-page_boxes[:, 0]))
    row_heights = np.array([r.bbox.y2-r.bbox.y1 for r in table.rows]) if table.rows else np.array([position_height])
    column_widths = np.array([c.bbox.x2-c.bbox.x1 for c in table.columns]) if table.columns else np.array([position_width])
    resolution = TableResolution(image_positions=int(on_page.sum()), positions_on_table=on_table, cell_count=len(table.cells),
                                 isolable_cells=int(isolable.sum()), rows_per_position=position_height/float(np.median(row_heights)),
                                 columns_per_position=position_width/float(np.median(column_widths)))
    region_area = {"table": (t.x2-t.x1)*(t.y2-t.y1)/page_area, "table_wrap": max(0.0, (w.x2-w.x1)*(w.y2-w.y1)-(
        (inner[2]-inner[0])*(inner[3]-inner[1]) if inner[0] < inner[2] and inner[1] < inner[3] else 0))/page_area}
    region_area["rest_of_page"] = max(0.0, 1-region_area["table"]-region_area["table_wrap"])
    return Partition(count, on_page, cells, rows, columns, regions, non_image,
                     _area(_boxes([c.bbox for c in table.cells]))/page_area, _area(_boxes([r.bbox for r in table.rows]))/page_area,
                     _area(_boxes([c.bbox for c in table.columns]))/page_area, region_area, resolution, isolable, largest, touching)


def rollout_selected(parameters):
    return getattr(parameters, "layer_reduction", "selected_layer") == "rollout"


def vector(context, parameters, layer=None):
    """The selected attention row, the equal-weight head mean at the selected layer, or the partial
    rollout row ending at the selected layer. An explicit `layer` always reads that raw layer."""
    if layer is None and rollout_selected(parameters):
        from types import SimpleNamespace
        from ._propagation import rollout
        from ._shared import valid_support
        if not parameters.start_layer <= parameters.layer:
            raise ValueError("The rollout start layer must not follow its end layer")
        mask = valid_support(context.record, parameters.step)
        weights = rollout(context, SimpleNamespace(heads=[], head_reduction="mean",
                                                   residual_coefficient=parameters.residual_coefficient),
                          parameters.start_layer, parameters.layer)
        if np.any(weights[~mask] > 1e-8) or abs(float(weights.sum())-1) > 1e-6:
            raise ValueError("Rollout row is not a distribution over the declared support")
        return weights, mask
    layers = [parameters.layer if layer is None else layer]
    if parameters.head_reduction == "head_mean":
        rows, mask = selected_rows(context, parameters, heads=list(range(context.record.query_head_count)), layers=layers)
        return rows[0].mean(axis=0), mask
    rows, mask = selected_rows(context, parameters, layers=layers)
    return rows[0, 0], mask


def selection_label(parameters):
    if rollout_selected(parameters):
        return (f"step {parameters.step} · rollout layers {parameters.start_layer}–{parameters.layer} · head mean · "
                f"residual {parameters.residual_coefficient:g}")
    heads = "head mean" if parameters.head_reduction == "head_mean" else f"head {parameters.head}"
    return f"step {parameters.step} · layer {parameters.layer} · {heads}"


@dataclass(frozen=True)
class Masses:
    """Masses of one row in every partition. A cell, band or region with no valid overlapping
    position is unmeasured (NaN here, None in results), never a measured zero."""
    total: float
    image: float
    cells: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    regions: dict
    non_image: dict


def masses(part, weights, mask):
    weights = np.where(mask, np.asarray(weights, dtype=np.float64), 0.0)
    def measured(matrix):
        values = weights @ matrix
        reached = ((matrix > 0) & mask[:, None]).any(axis=0)
        return np.where(reached, values, np.nan)
    return Masses(float(weights.sum()), float(weights[part.on_page].sum()), measured(part.cells), measured(part.rows),
                  measured(part.columns), {k: float(weights @ v) for k, v in part.regions.items()},
                  {k: float(weights @ v) for k, v in part.non_image.items()})


def _value(x):
    return None if x is None or not np.isfinite(x) else float(x)


def enrichment(mass, area_share, image_mass):
    """Share of image attention over share of page area; 1 is uniform allocation over the page."""
    if mass is None or not np.isfinite(mass) or image_mass <= 0 or area_share <= 0:
        return None
    return float(mass/image_mass/area_share)


def allocation(table_inputs, part, m, label):
    table = table_inputs.table
    cells = [TableCellMass(cell_id=c.native_id, row_indices=c.row_indices, column_indices=c.column_indices,
                           mass=_value(m.cells[i]), area_share=float(part.cell_area[i]),
                           enrichment=enrichment(m.cells[i], part.cell_area[i], m.image), isolable=bool(part.isolable[i]),
                           touching_positions=int(part.touching[i]), largest_position_share=float(part.largest_share[i]))
             for i, c in enumerate(table.cells)]
    band = lambda axis, values, areas: [TableBandMass(band_id=a.native_id, index=a.index, mass=_value(values[i]), area_share=float(areas[i]),
                                                      enrichment=enrichment(values[i], areas[i], m.image)) for i, a in enumerate(axis)]
    regions = [TableRegionMass(region=k, label=REGION_LABELS[k], mass=v, area_share=part.region_area.get(k),
                               enrichment=enrichment(v, part.region_area.get(k, 0), m.image))
               for k, v in m.regions.items() if k != "other_page" or v > 0]
    regions += [TableRegionMass(region=f"non_image:{k}", label=NON_IMAGE_LABELS.get(k, k.replace("_", " ")), mass=v) for k, v in m.non_image.items()]
    return TableAllocation(table_id=table.table_id, native_table_id=table.native_id, selection=label, total_mass=m.total,
                           image_mass=m.image, table_mass=m.regions["table"], cells=cells,
                           rows=band(table.rows, m.rows, part.row_area), columns=band(table.columns, m.columns, part.column_area),
                           regions=regions, resolution=part.resolution)


def measure(context, parameters):
    """One row aggregated into the table: (TableInputs, Partition, Masses, page AttentionRow, mask)."""
    table_inputs = inputs(context)
    weights, mask = vector(context, parameters)
    selected = selected_step(context.record, parameters.step)
    part = partition(context.record, table_inputs, selected.target_count)
    if rollout_selected(parameters):
        row = allocate_row(context.record, weights, selected, parameters.step, parameters.layer, -1,
                           quantity="partial_rollout_connectivity")
    else:
        row = allocate_row(context.record, weights, selected, parameters.step, parameters.layer,
                           -1 if parameters.head_reduction == "head_mean" else parameters.head)
    return table_inputs, part, masses(part, weights, mask), row, mask


def cell_index(table):
    return {c.native_id: i for i, c in enumerate(table.cells)}


def cell_label(cell):
    text = cell.text or cell.pdf_text
    where = f"r{'–'.join(str(i+1) for i in cell.row_indices)} c{'–'.join(str(i+1) for i in cell.column_indices)}"
    return f"{where} · {text[:40]}" if text else f"{where} · no text"


def header_cells(table):
    return {c.native_id for c in table.cells if any(r.role in HEADER_ROLES for r in c.roles) or c.native_flags}


def resolution_notice(part):
    r = part.resolution
    return (f"Resolution: {r.isolable_cells} of {r.cell_count} cells have an image position of their own "
            f"({r.positions_on_table} positions on the table; {r.rows_per_position:.1f} table rows per position). "
            "A shared cell's mass cannot be separated from its neighbours'.")


EVALUATION_NOTICE = "Grounding labels are evaluation-only and never reach the model."


def result_table(title, columns, labels, values, units="dimensionless mass", cell_ids=None):
    return ResultTable(title=title, columns=columns, labels=[str(x) for x in labels], values=values, units=units, cell_ids=cell_ids or [])


def plot(kind, title, x_label, y_label, points):
    return ResearchPlot(kind=kind, title=title, x_label=x_label, y_label=y_label,
                        points=[p if isinstance(p, ResearchPoint) else ResearchPoint(**p) for p in points])


def define(identifier, name, formula, parameters, compute, *, requires=None, units="dimensionless mass", displays=None,
           limitations=None, summary=None, normalization="original captured domain; image- and table-conditional shares are labelled"):
    result = definition(method_id=identifier, name=name, perspective="table_aware", family="table_structure",
                        quantity=identifier.split(".")[-1], formula=formula, parameters=parameters, calculate=compute,
                        requires=requires or REQUIRES, units=units, variant_id="table_cell_partition_v1", summary=summary,
                        displays=displays or ["page_overlay", "table"], normalization=normalization,
                        limitations=["Ownership is computed from captured position boxes and the projected logical cell boxes; "
                                     "a position overlapping several cells contributes fractionally to each, so detail inside it is not recovered.",
                                     "Grounding labels are evaluation-only and never reach the model.",
                                     *(limitations or [])])
    descriptor = result.descriptor.model_copy(update={"implementation_version": "table_saved_rows_v1",
        "source_references": ["design/proposals/table-aware-methods.md", "design/proposals/table-grounding-integration.md"],
        "validation_evidence": ["library/tests/methods/test_table_methods.py: known-answer table partitions, conservation and grounding"],
        "output": result.descriptor.output.model_copy(update={"support": "table cells, row and column bands, page regions and non-image buckets on declared captured targets"})})
    return result.__class__(descriptor, parameters, compute, True)


def same_page_geometry(a, b):
    """Two captures read the same page at the same resolution when their image positions match."""
    boxes = lambda r: [(p.page_id, tuple(round(v, 6) for v in p.bbox)) for p in r.positions if p.role == "image" and p.bbox]
    return boxes(a) == boxes(b)


PATH_PARTS = (("answer", "Answer cell"), ("row_path", "Row-header path"), ("column_path", "Column-header path"),
              ("support", "Support cells"), ("answer_row", "Rest of the answer row"), ("answer_column", "Rest of the answer column"),
              ("other_header", "Other header cells"), ("other_body", "Other body cells"))


def path_groups(table, ground):
    """Disjoint cell groups of the lookup path, assigned in PATH_PARTS order (cell indices)."""
    index = cell_index(table)
    missing = [i for i in [*ground.answer_cell_ids, *ground.row_header_path, *ground.column_header_path, *ground.support_cell_ids] if i not in index]
    if missing:
        raise ValueError("Grounding refers to cells absent from the table revision: " + ", ".join(missing[:5]))
    rows = {a.native_id: a.index for a in table.rows}
    columns = {a.native_id: a.index for a in table.columns}
    answer_rows = {rows[i] for i in ground.answer_row_ids if i in rows}
    answer_columns = {columns[i] for i in ground.answer_column_ids if i in columns}
    headers = header_cells(table)
    taken, groups = set(), {}
    def claim(name, ids):
        chosen = [index[i] for i in ids if index[i] not in taken]
        taken.update(chosen); groups[name] = chosen
    claim("answer", ground.answer_cell_ids)
    claim("row_path", ground.row_header_path)
    claim("column_path", ground.column_header_path)
    claim("support", ground.support_cell_ids)
    claim("answer_row", [c.native_id for c in table.cells if answer_rows & set(c.row_indices)])
    claim("answer_column", [c.native_id for c in table.cells if answer_columns & set(c.column_indices)])
    claim("other_header", [c.native_id for c in table.cells if c.native_id in headers])
    claim("other_body", [c.native_id for c in table.cells])
    return groups


def lookup_cells(table, ground):
    """The cells a lookup must bind: the answer cell and both header paths."""
    index = cell_index(table)
    return {index[i] for i in [*ground.answer_cell_ids, *ground.row_header_path, *ground.column_header_path] if i in index}


def group_mass(values, indices):
    """Sum over measured cells; None when no cell of the group is measured."""
    chosen = [values[i] for i in indices if np.isfinite(values[i])]
    return float(sum(chosen)) if chosen else None


def rank(values, index):
    """1-based rank of one measured entry by descending value; ties broken by order."""
    if not np.isfinite(values[index]):
        return None
    return 1+sum(1 for i, v in enumerate(values) if np.isfinite(v) and (v > values[index] or (v == values[index] and i < index)))


def matched_record(base, other, base_table, other_table):
    """Two table captures are comparable when they read the same table revision on the same page geometry
    with the same model; their questions may differ."""
    if other_table is None:
        raise ValueError("A compared capture has no verified table annotations")
    if (other_table.table.table_id, other_table.table.table_sha256) != (base_table.table.table_id, base_table.table.table_sha256):
        raise ValueError("Compared captures must read the same table revision")
    for key in ("model_id", "model_revision"):
        if base.provenance.get(key) != other.provenance.get(key):
            raise ValueError("Compared captures must use the same model and revision")
    if (base.layer_count, base.query_head_count) != (other.layer_count, other.query_head_count):
        raise ValueError("Compared captures must share layer and head axes")
    if not same_page_geometry(base, other):
        raise ValueError("Compared captures must read the page at the same resolution (identical image positions)")
