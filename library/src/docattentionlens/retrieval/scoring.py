"""FP32 MaxSim and exact rectangular ownership with explicit residuals."""


def maxsim(query, document, query_mask=None, document_mask=None):
    from ..methods.retrieval_score_decomposition import maxsim as calculate
    return calculate(query, document, query_mask, document_mask)


def ownership(box, elements):
    """Partition a patch by annotation edges; share each cell equally among covers.

    Flagged raw rectangles never become element ownership: their positive source
    intersection is allocated to unreviewed_geometry. Empty cells are background.
    This policy is independent of the frontend's clipped display rectangles.
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Patch has no area")
    covers = []
    for element in elements:
        raw = element.bbox
        rect = (max(x1, raw.x1), max(y1, raw.y1), min(x2, raw.x2), min(y2, raw.y2))
        if rect[2] > rect[0] and rect[3] > rect[1]:
            covers.append((rect, "unreviewed_geometry" if element.geometry_flags else element.element_id))
    xs = sorted({x1, x2, *(v for rect, _ in covers for v in (rect[0], rect[2]))})
    ys = sorted({y1, y2, *(v for rect, _ in covers for v in (rect[1], rect[3]))})
    result = {}
    area = (x2-x1)*(y2-y1)
    for a, b in zip(xs, xs[1:]):
        for c, d in zip(ys, ys[1:]):
            labels = [label for rect, label in covers if rect[0] <= (a+b)/2 < rect[2] and rect[1] <= (c+d)/2 < rect[3]]
            labels = labels or ["background"]
            fraction = (b-a)*(d-c)/area/len(labels)
            for label in labels:
                result[label] = result.get(label, 0.0) + fraction
    if abs(sum(result.values())-1) > 1e-8:
        raise ValueError("Ownership failed conservation")
    return result


def contributions(winners, maxima, mappings):
    from ..methods.retrieval_score_decomposition import contributions as calculate
    return calculate(winners, maxima, mappings)
