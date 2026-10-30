"""Spatial attention distance: canonical definition and calculation ownership.

The mean attention distance of Dosovitskiy et al. (2021, appendix D.7) and Raghu et
al. (2021): distances between the query patch and every attended patch, weighted by
attention. Here it is computed for one mapped source row, conditional on the mass
that reaches mapped image patches; non-image and unresolved mass is excluded and
reported. Only sources with page geometry qualify (image patches of a document
encoding); text prediction sources have no position on the page.
"""
from typing import Literal
import numpy as np
from .contracts import Calculation, MethodDefinition, MethodDescriptor, RowParameters, ResearchPlot, ResearchPoint
from ._shared import parameter_schema, selected_rows, selected_step, statistic
from .element_allocation import allocate_row


class DistanceParameters(RowParameters):
    units: Literal["source_pixels", "page_diagonal_fraction"] = "page_diagonal_fraction"


DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.spatial_distance',
 'variant_id': 'image_conditional_distance_v1',
 'descriptor_version': '0.2.0',
 'status': 'implemented',
 'implementation_version': 'mapped_patch_distance_v1',
 'perspective': 'general',
 'family': 'spatial_attention',
 'question': 'How far from its source does the image-conditioned row attend?',
 'evidence_kind': 'attention_summary',
 'quantity': 'weighted_distance',
 'definition': 'Mean source-to-target Euclidean distance weighted by image-conditional attention; '
               'excluded mass reported. Distances join mapped patch-box centres in source-page '
               'coordinates: Σ_t w_t·‖c_s − c_t‖ / Σ_t w_t over valid image targets t.',
 'source_references': ['definition:weighted Euclidean distance',
                       'https://arxiv.org/abs/2010.11929',
                       'https://arxiv.org/abs/2108.08810'],
 'literature_relation': {'kind': 'adaptation',
                         'note': 'Adapts mean attention distance (Dosovitskiy et al. 2021; Raghu et al. 2021): one '
                                 'selected image source, image-mass conditioning, page-coordinate patch centres and '
                                 'optional page-diagonal units, rather than averages over query patches and images.'},
 'requires': ['attention_weights', 'validated_source_target_geometry', 'image_mask'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': parameter_schema(DistanceParameters),
 'output': {'schema': 'spatial_statistic',
            'units': 'declared distance units',
            'signed': False,
            'support': 'mapped image target positions',
            'normalization': 'conditional on eligible image mass'},
 'displays': ['page_overlay', 'layer_series'],
 'comparison_constraints': ['Same coordinate units, geometric support and conditioning.'],
 'limitations': ['Non-image/unresolved mass is excluded and must be reported.',
                 'Requires a source with page geometry: an image-patch source of a document encoding. '
                 'Text and prediction sources have no page position.',
                 'Patch-box centres approximate patch positions; distance is in source-page coordinates, '
                 'not the resized model input.'],
 'validation_evidence': ['library/tests/methods/test_planned_methods.py: hand-computed weighted '
                         'distances, unit conversion, excluded-mass and missing-geometry fixtures',
                         'frontend/tests/rendering/page-overlay-methods.spec.ts: the workspace page '
                         "draws this method's own known-answer row"],
 'name': 'Spatial attention distance',
 'summary': 'Summarise the distance from a mapped source position to attended image regions, with '
            'excluded non-image attention reported separately.'})


def centre(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=np.float64)


def image_geometry(record, count, source):
    """Image targets with boxes, their centres and the page extent spanned by mapped patches."""
    positions = record.positions[:count]
    origin = positions[source]
    if getattr(origin, "role", None) != "image" or not getattr(origin, "bbox", None):
        raise ValueError("Spatial distance needs a mapped image-patch source; the selected source has no page geometry")
    targets = [p.index for p in positions if p.role == "image" and p.bbox and p.page_id == origin.page_id]
    boxes = np.array([positions[i].bbox for i in targets], dtype=np.float64)
    extent = np.hypot(boxes[:, 2].max()-boxes[:, 0].min(), boxes[:, 3].max()-boxes[:, 1].min())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Mapped image patches do not span a page area")
    distances = np.linalg.norm(np.array([centre(b) for b in boxes]) - centre(origin.bbox), axis=1)
    return np.array(targets), distances, float(extent)


def weighted_distance(weights, targets, distances, mask):
    eligible = mask[targets]
    mass = float(weights[targets][eligible].sum())
    if mass <= 0:
        return None, 0.0
    return float((weights[targets][eligible]*distances[eligible]).sum()/mass), mass


def calculate(context, parameters):
    record = context.record
    selected = selected_step(record, parameters.step)
    rows, mask = selected_rows(context, parameters)
    weights = rows[0, 0]
    targets, distances, extent = image_geometry(record, selected.target_count, selected.source_index)
    scale = 1.0 if parameters.units == "source_pixels" else 1/extent
    units = "source pixels" if parameters.units == "source_pixels" else "fraction of page diagonal"
    mean, image_mass = weighted_distance(weights, targets, distances*scale, mask)
    if mean is None:
        raise ValueError("The selected row assigns no attention to mapped image patches")
    total = float(weights[mask].sum())
    points = []
    if context.read_array is not None:
        for layer in range(record.layer_count):
            matrix = np.asarray(context.read_array(layer), dtype=np.float64)
            by_head = [weighted_distance(matrix[h, selected.source_index], targets, distances*scale, mask)[0] for h in range(record.query_head_count)]
            if by_head[parameters.head] is not None:
                points.append(ResearchPoint(label=f"Layer {layer} · head {parameters.head}", x=layer, y=by_head[parameters.head], group=f"head {parameters.head}"))
            measured = [d for d in by_head if d is not None]
            if measured:
                points.append(ResearchPoint(label=f"Layer {layer} · mean of {len(measured)} heads", x=layer, y=float(np.mean(measured)), group="head mean"))
    row = allocate_row(record, weights, selected, parameters.step, parameters.layer, parameters.head)
    return Calculation(row=row,
        statistics=[statistic("Mean attention distance", mean, units),
                    statistic("Image mass in the distance", image_mass),
                    statistic("Excluded non-image or unresolved mass", total-image_mass),
                    statistic("Source self share (distance 0)", float(weights[selected.source_index])),
                    statistic("Mapped image targets", int(mask[targets].sum()), "patches"),
                    statistic("Page extent diagonal", extent, "source pixels")],
        plots=[ResearchPlot(kind="line", title="Mean attention distance by layer", x_label="Layer", y_label=f"Distance ({units})", points=points)] if points else [],
        normalization="image-conditional: weights divided by their image mass; excluded mass reported separately; saved weights unchanged",
        notices=[f"{1-image_mass/total:.1%} of the row's valid mass falls outside mapped image patches and is excluded from the distance." if total > 0 else "The row has no valid mass."])


DEFINITION = MethodDefinition(DESCRIPTOR, DistanceParameters, calculate, True)
