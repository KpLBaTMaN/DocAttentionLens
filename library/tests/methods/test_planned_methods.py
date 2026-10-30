"""Formerly planned methods against independently written references.

Mathematical fixtures only, never model evidence. Rollout is checked against an
explicit full-matrix product (the method carries one row vector), distance against
a per-target loop over box centres, and profile similarity against cosines of the
head preference catalogue's own profile table.
"""
import math
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.methods.registry import execute, get_method
from docattentionlens.retrieval.attention import calculate as encoding_calculate
from docattentionlens.retrieval.contracts import EncodingAttention, Position, RetrievalAttentionRequest
from test_document_methods import document_fixture, run as document_run
from test_saved_rows import fixture as prediction_fixture

SEEDS = [0, 1, 2, 3]


# Encoding fixtures --------------------------------------------------------------

def matrices(rng, layers, heads, n, mask, causal=False):
    values = rng.dirichlet(np.full(n, .6), size=(layers, heads, n))
    values[..., ~mask] = 0
    if causal:
        values = np.tril(values)
    sums = values.sum(-1, keepdims=True)
    values = np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)
    values[:, :, ~mask, :] = 0  # inactive sources have no row
    return values.astype(np.float32)


def query_encoding(seed, n=8, layers=3, heads=2, causal=False):
    rng = np.random.default_rng(seed)
    mask = np.ones(n, dtype=bool)
    mask[[2, 5]] = False
    arrays = matrices(rng, layers, heads, n, mask, causal)
    encoding = EncodingAttention(encoding_id="query", pass_kind="query", layer_count=layers, query_head_count=heads,
                                 position_count=n, attention_mask=mask.astype(int).tolist(),
                                 layer_arrays=[f"l{i}" for i in range(layers)], causal=causal)
    roles = ["bos", "query_text", "augmentation", "query_text", "query_text", "augmentation", "query_text", "query_text"]
    record = NS(capture_id="numeric_fixture", attention_encodings=[encoding], pages=[],
                query_positions=[Position(index=i, token_id=i, piece=str(i), role=roles[i], eligible=bool(mask[i])) for i in range(n)])
    return record, {f"l{i}": arrays[i] for i in range(layers)}, mask


def document_encoding(seed, grid=3, prompt=2, layers=2, heads=2, width=300.0, height=150.0):
    """A grid of image patches over a width × height page, followed by prompt tokens."""
    rng = np.random.default_rng(seed)
    n = grid*grid + prompt
    mask = np.ones(n, dtype=bool)
    arrays = matrices(rng, layers, heads, n, mask)
    positions, ownership = [], []
    for i in range(n):
        if i < grid*grid:
            r, c = divmod(i, grid)
            box = [c*width/grid, r*height/grid, (c+1)*width/grid, (r+1)*height/grid]
            positions.append(Position(index=i, token_id=i, piece="<img>", role="visual_patch", eligible=True, bbox=box))
            ownership.append({"element_a" if c < grid-1 else "background": 1.0})
        else:
            positions.append(Position(index=i, token_id=i, piece="p", role="visual_prompt", eligible=True))
            ownership.append({"non_page": 1.0})
    encoding = EncodingAttention(encoding_id="document", pass_kind="document", page_id="page_a", layer_count=layers,
                                 query_head_count=heads, position_count=n, attention_mask=[1]*n,
                                 layer_arrays=[f"l{i}" for i in range(layers)])
    record = NS(capture_id="numeric_fixture", attention_encodings=[encoding], query_positions=[],
                pages=[NS(page_id="page_a", positions=positions, ownership=ownership)])
    return record, {f"l{i}": arrays[i] for i in range(layers)}, positions


def encoded(record, arrays, encoding_id, method, source, **parameters):
    return encoding_calculate(record, RetrievalAttentionRequest(encoding_id=encoding_id, source_index=source, method_id=method,
                                                                parameters=parameters), arrays.__getitem__)


# Attention rollout ---------------------------------------------------------------

def reference_rollout(arrays, layers, mask, residual, source):
    """Abnar & Zuidema eq. 1 with an explicit square product over the valid key domain."""
    domain = np.flatnonzero(mask)
    composed = []
    for layer in range(layers):
        a = arrays[f"l{layer}"].astype(np.float64).mean(axis=0)[np.ix_(domain, domain)]
        mixed = residual*np.eye(len(domain)) + (1-residual)*a
        composed.append(mixed/mixed.sum(axis=1, keepdims=True))
    product = composed[0] if len(composed) == 1 else np.linalg.multi_dot(composed[::-1])
    row = np.zeros(len(mask))
    row[domain] = product[list(domain).index(source)]
    return row


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("causal", [False, True])
def test_rollout_matches_explicit_square_product(seed, causal):
    record, arrays, mask = query_encoding(seed, causal=causal)
    for source in np.flatnonzero(mask):
        for end, residual in ((0, .5), (2, .5), (2, .2), (1, 0.0)):
            result = encoded(record, arrays, "query", "attention.rollout", int(source), layer=end, residual_coefficient=residual)
            expected = reference_rollout(arrays, end+1, mask, residual, int(source))
            assert result.row.values == pytest.approx(expected.tolist(), abs=1e-9)
            assert result.row.quantity == "rollout_connectivity"
            assert sum(result.row.values) == pytest.approx(1, abs=1e-9)
            assert all(v == 0 for v in np.asarray(result.row.values)[~mask])
            if causal:
                assert all(v == 0 for v in result.row.values[int(source)+1:])


def test_rollout_identity_residual_keeps_the_source_and_records_its_axes():
    record, arrays, mask = query_encoding(0)
    result = encoded(record, arrays, "query", "attention.rollout", 3, layer=2, residual_coefficient=1.0)
    assert result.row.values == pytest.approx(np.eye(len(mask))[3].tolist())
    stats = {s.name: s.value for s in result.statistics}
    assert stats["Layers composed"] == 3 and stats["Heads averaged"] == 2 and stats["Source self share"] == pytest.approx(1)


def test_rollout_refuses_selected_prediction_rows():
    method = get_method("attention.rollout")
    with pytest.raises(ValueError, match="complete square attention matrices"):
        execute(method.method_id, method.variant_id, {"layer": 1}, prediction_fixture())


# Spatial attention distance ------------------------------------------------------

def reference_distance(weights, positions, source, scale):
    sx, sy = (positions[source].bbox[0]+positions[source].bbox[2])/2, (positions[source].bbox[1]+positions[source].bbox[3])/2
    numerator = mass = 0.0
    for p, w in zip(positions, weights):
        if p.bbox is None:
            continue
        cx, cy = (p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2
        numerator += w*math.sqrt((cx-sx)**2+(cy-sy)**2)
        mass += w
    return numerator/mass*scale, mass


@pytest.mark.parametrize("seed", SEEDS)
def test_distance_matches_box_centre_loop_in_both_units(seed):
    record, arrays, positions = document_encoding(seed)
    diagonal = math.hypot(300, 150)
    for source in range(9):
        for layer, head in ((0, 0), (1, 1)):
            weights = arrays[f"l{layer}"][head, source].astype(np.float64)
            for units, scale in (("source_pixels", 1.0), ("page_diagonal_fraction", 1/diagonal)):
                result = encoded(record, arrays, "document", "attention.spatial_distance", source, layer=layer, head=head, units=units)
                stats = {s.name: s.value for s in result.statistics}
                expected, mass = reference_distance(weights, positions, source, scale)
                assert stats["Mean attention distance"] == pytest.approx(expected, rel=1e-9)
                assert stats["Image mass in the distance"] == pytest.approx(mass, abs=1e-9)
                assert stats["Excluded non-image or unresolved mass"] == pytest.approx(weights[9:].sum(), abs=1e-6)
                assert stats["Page extent diagonal"] == pytest.approx(diagonal)
                assert result.row.values == pytest.approx(weights.tolist(), abs=1e-7)


def test_distance_layer_series_and_missing_geometry():
    record, arrays, positions = document_encoding(1)
    result = encoded(record, arrays, "document", "attention.spatial_distance", 4, layer=0, head=1, units="source_pixels")
    series = result.plots[0].points
    assert {p.group for p in series} == {"head 1", "head mean"} and len(series) == 4
    by_head = [reference_distance(arrays["l1"][h, 4].astype(np.float64), positions, 4, 1.0)[0] for h in range(2)]
    assert next(p.y for p in series if p.x == 1 and p.group == "head mean") == pytest.approx(np.mean(by_head))
    with pytest.raises(ValueError, match="image-patch source"):
        encoded(record, arrays, "document", "attention.spatial_distance", 10, layer=0, head=0)


def test_distance_is_unavailable_for_prediction_rows_without_source_geometry():
    method = get_method("attention.spatial_distance")
    assert "validated_source_target_geometry" in method.requires
    with pytest.raises(ValueError, match="image-patch source"):
        execute(method.method_id, method.variant_id, {"layer": 0, "head": 0}, prediction_fixture())


# Head profile similarity ---------------------------------------------------------

def cosine(p, q):
    return sum(a*b for a, b in zip(p, q)) / math.sqrt(sum(a*a for a in p)*sum(b*b for b in q))


def test_profile_similarity_is_the_cosine_of_the_declared_catalogue_profiles():
    a = document_fixture(); b = document_fixture("generation_"+"b"*32, "case_b")
    sample = replace(a, samples=(a, b))
    catalogue = document_run("attention.head_specialisation", sample, layers=[0, 1], heads=[0, 1, 2])
    profiles = {(int(v[0]), int(v[1])): v[2:-1] for v in catalogue.table.values}
    result = document_run("heads.profile_similarity", sample, layers=[0, 1], heads=[0, 1, 2])
    keys = [(layer, head) for layer in (0, 1) for head in (0, 1, 2)]
    assert result.table.labels == [f"L{l} · H{h}" for l, h in keys]
    for i, x in enumerate(keys):
        for j, y in enumerate(keys):
            assert result.table.values[i][j] == pytest.approx(cosine(profiles[x], profiles[y]), abs=1e-12)
        assert result.table.values[i][i] == pytest.approx(1)
    for row, key in zip(result.tables[0].values, keys, strict=True):
        assert row == pytest.approx(profiles[key], abs=1e-12)
    assert len(result.plots[0].points) == len(keys)**2


def test_undefined_profiles_are_unavailable_and_one_head_is_refused():
    a = document_fixture(); b = document_fixture("generation_"+"b"*32, "case_b")
    values = b.values.copy()
    values[0, 1] = [0, 0, .5, .5]  # no image mass: this head's profile is undefined on one sample row
    b = replace(b, values=values)
    sample = replace(a, samples=(a, b))
    result = document_run("heads.profile_similarity", sample, layers=[0], heads=[0, 1, 2])
    assert result.table.values[1] == [None, None, None]
    assert [row[1] for row in result.table.values] == [None, None, None]
    assert result.table.values[0][2] is not None
    assert any("L0 · H1" in notice for notice in result.notices)
    with pytest.raises(ValueError, match="at least two heads"):
        document_run("heads.profile_similarity", sample, layers=[0], heads=[2])


# Integrated gradients -------------------------------------------------------------

from docattentionlens.methods.integrated_gradients import integrate_with_check, riemann_integrate, token_totals


def test_midpoint_rule_is_exact_for_quadratics_and_complete_for_non_separable_functions():
    rng = np.random.default_rng(5)
    x, b, a = rng.normal(size=(3, 7)), rng.normal(size=(3, 7)), rng.normal(size=(3, 7))
    # F = Σ a x²: the path integrand is linear in α, so the midpoint rule is exact.
    exact = riemann_integrate(lambda p: 2*a*p, x, b, 2)
    assert exact == pytest.approx(a*(x**2 - b**2), abs=1e-12)
    # F = (w·x)² + Σ x³: completeness Σ IG = F(x) − F(b) up to the O(1/m²) midpoint error.
    w = rng.normal(size=(3, 7))
    f = lambda p: float((w*p).sum()**2 + (p**3).sum())
    gradient = lambda p: 2*(w*p).sum()*w + 3*p**2
    for steps, tolerance in ((16, 5e-2), (256, 2e-4)):
        attribution = riemann_integrate(gradient, x, b, steps)
        assert attribution.sum() == pytest.approx(f(x) - f(b), abs=tolerance)
    # The two interleaved half-step rules average to the full rule; their gap shrinks as m grows.
    attribution, halves = integrate_with_check(gradient, x, b, 64)
    assert sum(halves)/2 == pytest.approx(attribution.sum(), abs=1e-9)
    coarse = abs(integrate_with_check(gradient, x, b, 8)[1][0] - integrate_with_check(gradient, x, b, 8)[1][1])
    assert abs(halves[0] - halves[1]) < coarse
    assert integrate_with_check(gradient, x, b, 7)[1] is None
    # Zero input-minus-baseline receives zero attribution (the paper's sensitivity property).
    same = riemann_integrate(gradient, x, x.copy(), 8)
    assert np.all(same == 0)


def test_integration_guards_and_token_sums():
    x = np.ones((4, 6))
    with pytest.raises(ValueError, match="two Riemann steps"):
        riemann_integrate(lambda p: p, x, x*0, 1)
    with pytest.raises(ValueError, match="identical shape"):
        riemann_integrate(lambda p: p, x, np.zeros((4, 5)), 4)
    with pytest.raises(ValueError, match="not finite"):
        riemann_integrate(lambda p: p*np.nan, x, x*0, 4)
    with pytest.raises(RuntimeError, match="interrupted"):
        riemann_integrate(lambda p: p, x, x*0, 4, interrupted=lambda: True)
    tokens = np.arange(12, dtype=float).reshape(3, 4)
    assert token_totals(tokens).tolist() == [6, 22, 38]
    with pytest.raises(ValueError, match="embedding vector per visual token"):
        token_totals(tokens.ravel())


def ig_run(context, **parameters):
    method = get_method("attribution.integrated_gradients")
    return execute(method.method_id, method.variant_id, parameters, context)


def ig_context(values=(.3, -.1, None, None), step=0, target=-.2, reference=-1.4):
    context = document_fixture()
    study = NS(method_id="attribution.integrated_gradients", kind="result", parameters={"step": step, "steps": 32, "baseline": "black"},
               outcomes=[NS(capture_id=context.record.capture_id, attribution=list(values), target_token_id=7,
                            target_log_probability=target, baseline_log_probability=reference, half_rule_totals=[.5, -.1])])
    return replace(context, experiment=study)


def test_attribution_result_reports_completeness_elements_and_signed_page_grid():
    result = ig_run(ig_context(), steps=32)
    stats = {s.name: s.value for s in result.statistics}
    assert stats["Sum of patch attributions"] == pytest.approx(.2)
    assert stats["F(x) − F(x′)"] == pytest.approx(1.2)
    assert stats["Completeness gap"] == pytest.approx(.2 - 1.2)
    assert stats["Relative completeness gap"] == pytest.approx(1/1.2)
    assert stats["Positive attribution"] == pytest.approx(.3) and stats["Negative attribution"] == pytest.approx(-.1)
    assert stats["Interleaved half-rule disagreement"] == pytest.approx(.6)
    assert any("interleaved half-step rules disagree" in notice for notice in result.notices)
    elements = dict(zip(result.table.labels, (row[0] for row in result.table.values)))
    assert elements == pytest.approx({"element_a": .15, "element_b": .15 - .1})
    assert [p.value for p in result.plots[0].points] == pytest.approx([.3, -.1])
    assert any("Completeness gap is" in notice for notice in result.notices)


def test_non_greedy_attribution_target_is_never_silent():
    context = ig_context()
    assert not any("not the float32 forward's greedy" in n for n in ig_run(context, steps=32).notices)
    context.experiment.outcomes[0].target_is_greedy = True
    assert not any("greedy prediction" in n for n in ig_run(context, steps=32).notices)
    context.experiment.outcomes[0].target_is_greedy = False
    notice = next(n for n in ig_run(context, steps=32).notices if "greedy prediction" in n)
    assert "-0.2 nats" in notice and "probability 0.819" in notice


def test_attribution_requires_its_experiment_and_exact_image_support():
    with pytest.raises(ValueError, match="integrated-gradients experiment"):
        ig_run(document_fixture())
    with pytest.raises(ValueError, match="consumed image positions"):
        ig_run(ig_context(values=(.3, -.1, .5, None)))
    with pytest.raises(ValueError, match="input positions"):
        ig_run(ig_context(values=(.3, -.1, None)))
    method = get_method("attribution.integrated_gradients")
    assert method.evidence_kind == "output_attribution" and "experiment_outcomes" in method.requires
    assert set(method.parameters_schema["properties"]) == {"step", "steps", "baseline"}
