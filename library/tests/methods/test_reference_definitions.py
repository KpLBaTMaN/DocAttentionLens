"""Method outputs against independently written reference definitions.

Mathematical fixtures only, never model evidence. Each reference uses a different
formula from its method module where one exists (Gini from the mean absolute
difference, Jensen-Shannon from the entropy identity, ownership from point
sampling), so a shared mistake is unlikely to pass both.
"""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.generation.contracts import GenerationPosition
from docattentionlens.methods.contracts import Context
from docattentionlens.methods.registry import execute, get_method, list_methods
from docattentionlens.retrieval.attention import calculate as encoding_calculate, received_column
from docattentionlens.retrieval.contracts import EncodingAttention, Position, RetrievalAttentionRequest, RetrievalReceivedRequest
from docattentionlens.retrieval.scoring import contributions, maxsim, ownership

ROLES = ["image", "query", "prompt", "special", "generated"]
SEEDS = [0, 1, 2, 3]


# Independent references -----------------------------------------------------

def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(2))


def gini(p):
    return float(np.abs(p[:, None] - p[None, :]).sum() / (2 * len(p) ** 2 * p.mean()))


def inverse_simpson(p):
    # Exponential of the order-2 Renyi entropy.
    return math.exp(-math.log(float((p ** 2).sum())))


def js_similarity(p, q):
    return 1 - (entropy_bits((p + q) / 2) - (entropy_bits(p) + entropy_bits(q)) / 2)


def ledger(weights, positions):
    totals = {}
    for weight, position in zip(weights, positions):
        for bucket, fraction in position.ownership.items():
            totals[bucket, position.page_id] = totals.get((bucket, position.page_id), 0.0) + float(weight) * fraction
    return totals


# Fixtures ---------------------------------------------------------------------

def capture(seed, n=23, layers=3, heads=4):
    """A causal prediction row over `n` targets, with every role and random ownership."""
    rng = np.random.default_rng(seed)
    roles = ROLES + [ROLES[i] for i in rng.integers(0, 4, n - len(ROLES))]
    positions = []
    for i, role in enumerate(roles):
        if role == "image":
            fractions = rng.dirichlet(np.ones(3))
            owned = dict(zip(["element_a", "element_b", "background"], fractions.tolist()))
            owned["background"] = 1 - owned["element_a"] - owned["element_b"]
            positions.append(GenerationPosition(index=i, token_id=i, piece=str(i), role=role, page_id="page_a", bbox=[i, 0, i + 1, 1], ownership=owned))
        else:
            positions.append(GenerationPosition(index=i, token_id=i, piece=str(i), role=role, ownership={"non_image:" + role: 1.0}))
    record = NS(capture_id="generation_" + "c" * 32, layer_count=layers, query_head_count=heads, positions=positions,
                steps=[NS(step=0, target_count=n, source_index=n - 1, predicted_index=n, array="step_0")],
                context=NS(input_attention_mask=[1] * n, input_token_ids=list(range(n))),
                coverage={"all_causal_targets": True, "quantity": "post_softmax", "ownership": "reviewed"})
    # Peaked and diffuse rows; float32 storage as in real captures.
    values = rng.dirichlet(np.full(n, .3), size=(layers, heads)).astype(np.float32)
    values[0, 0] = 0
    values[0, 0, n - 1] = 1
    return Context(record, values)


def run(method_id, context, **parameters):
    return execute(method_id, get_method(method_id).variant_id, {"step": 0, **parameters}, context)


def stats(result):
    return {s.name: s.value for s in result.statistics}


def check_row(row, weights, positions):
    assert row.values == pytest.approx(weights, abs=1e-9)
    assert row.total_mass == pytest.approx(float(np.sum(weights)), abs=1e-9)
    assert row.image_mass == pytest.approx(float(sum(w for w, p in zip(weights, positions) if p.role == "image")), abs=1e-9)
    assert {(a.bucket, a.page_id): a.mass for a in row.allocations} == pytest.approx(ledger(weights, positions), abs=1e-9)
    assert row.display_max == pytest.approx(float(np.max(weights)))


# Row reductions ---------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_head_reductions_and_comparisons_match_direct_numpy(seed):
    context = capture(seed)
    rows = context.values.astype(np.float64)
    positions = context.record.positions
    heads = [3, 0, 2]
    for layer in range(context.record.layer_count):
        selected = rows[layer, heads]
        check_row(run("attention.row", context, layer=layer, head=2).row, rows[layer, 2], positions)
        check_row(run("attention.element_allocation", context, layer=layer, head=1).row, rows[layer, 1], positions)
        check_row(run("attention.mean_heads", context, layer=layer, heads=heads).row, selected.mean(0), positions)
        check_row(run("attention.max_heads", context, layer=layer, heads=heads).row, selected.max(0), positions)
        check_row(run("attention.min_heads", context, layer=layer, heads=heads).row, selected.min(0), positions)
        weights = [.2, 3., 0.]
        check_row(run("attention.weighted_heads", context, layer=layer, heads=heads, weights=weights).row,
                  np.average(selected, axis=0, weights=weights), positions)
        comparison = run("attention.head_by_head", context, layer=layer, heads=heads).comparisons
        assert [c.row.head for c in comparison] == heads
        for c, h in zip(comparison, heads):
            check_row(c.row, rows[layer, h], positions)
    depth = run("attention.across_layers", context, head=1, layers=[2, 0]).comparisons
    assert [c.row.layer for c in depth] == [2, 0]
    for c, layer in zip(depth, [2, 0]):
        check_row(c.row, rows[layer, 1], positions)


@pytest.mark.parametrize("seed", SEEDS)
def test_envelopes_bracket_the_mean_and_are_labelled_as_non_probabilities(seed):
    context = capture(seed)
    mean = np.array(run("attention.mean_heads", context, layer=1).row.values)
    maximum = run("attention.max_heads", context, layer=1).row
    minimum = run("attention.min_heads", context, layer=1).row
    assert np.all(np.array(minimum.values) <= mean + 1e-12) and np.all(mean <= np.array(maximum.values) + 1e-12)
    assert maximum.total_mass >= 1 - 1e-6 >= minimum.total_mass
    assert (maximum.quantity, minimum.quantity) == ("head_maximum_envelope", "head_minimum_envelope")


def test_weighted_heads_ignore_weight_scale_and_reduce_to_one_head():
    context = capture(5)
    base = run("attention.weighted_heads", context, layer=2, heads=[0, 1, 3], weights=[1, 2, 5]).row.values
    scaled = run("attention.weighted_heads", context, layer=2, heads=[0, 1, 3], weights=[1e-3, 2e-3, 5e-3]).row.values
    assert scaled == pytest.approx(base, abs=1e-12)
    single = run("attention.weighted_heads", context, layer=2, heads=[0, 1, 3], weights=[0, 7, 0]).row.values
    assert single == pytest.approx(run("attention.row", context, layer=2, head=1).row.values, abs=1e-12)


# Statistics -------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_statistics_match_independent_definitions(seed):
    context = capture(seed)
    rng = np.random.default_rng(seed + 100)
    positions = context.record.positions
    for layer in range(context.record.layer_count):
        for head in range(context.record.query_head_count):
            a = context.values[layer, head].astype(np.float64)
            p = a / a.sum()
            entropy = stats(run("attention.entropy", context, layer=layer, head=head))
            assert entropy["Shannon entropy"] == pytest.approx(entropy_bits(p), abs=1e-9)
            assert entropy["Normalized entropy"] == pytest.approx(entropy_bits(p) / math.log2(len(p)), abs=1e-9)
            concentration = stats(run("attention.concentration", context, layer=layer, head=head))
            assert concentration["Maximum probability"] == pytest.approx(p.max(), abs=1e-12)
            assert concentration["Gini concentration"] == pytest.approx(gini(p), abs=1e-9)
            assert concentration["Inverse Simpson effective support"] == pytest.approx(inverse_simpson(p), rel=1e-9)
            k = int(rng.integers(1, len(a) + 1))
            topk = stats(run("attention.topk_mass", context, layer=layer, head=head, k=k))
            assert topk[f"Top-{k} attention mass"] == pytest.approx(np.partition(a, len(a) - k)[len(a) - k:].sum(), abs=1e-9)
            roles = ["special", "image"]
            sinks = stats(run("attention.sinks", context, layer=layer, head=head, roles=roles))
            assert sinks["Declared-group attention mass"] == pytest.approx(sum(w for w, q in zip(a, positions) if q.role in roles), abs=1e-9)
            table = run("attention.modality_ratio", context, layer=layer, head=head).table
            for role, (mass, count, share, enrichment) in zip(table.labels, table.values):
                members = [i for i, q in enumerate(positions) if q.role == role]
                assert (mass, count, share) == pytest.approx((a[members].sum(), len(members), len(members) / len(a)), abs=1e-9)
                assert enrichment == pytest.approx((a[members].sum() / a.sum()) / (len(members) / len(a)), abs=1e-9)
        heads = [0, 1, 2, 3]
        rows = context.values[layer].astype(np.float64)
        distributions = rows / rows.sum(1, keepdims=True)
        matrix = run("heads.distribution_similarity", context, layer=layer, heads=heads).table.values
        expected = [[js_similarity(x, y) for y in distributions] for x in distributions]
        assert np.asarray(matrix) == pytest.approx(np.clip(expected, 0, 1), abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_distribution_statistics_do_not_depend_on_target_order(seed):
    context = capture(seed)
    order = np.random.default_rng(seed).permutation(context.values.shape[-1])
    shuffled = Context(context.record, np.ascontiguousarray(context.values[..., order]))
    for method in ("attention.entropy", "attention.concentration"):
        assert stats(run(method, shuffled, layer=1, head=2)) == pytest.approx(stats(run(method, context, layer=1, head=2)), abs=1e-9)
    assert stats(run("attention.topk_mass", shuffled, layer=1, head=2, k=5)) == pytest.approx(stats(run("attention.topk_mass", context, layer=1, head=2, k=5)), abs=1e-9)


def test_closed_form_values_from_the_definitions():
    n = 16
    record = capture(0, n=n).record
    values = np.zeros((3, 4, n), dtype=np.float32)
    values[:, 0] = 1 / n                      # uniform
    values[:, 1, n - 1] = 1                   # one-hot
    values[:, 2, [3, 7]] = .5                 # two equal targets, disjoint from the one-hot head
    values[:, 3, :8] = 1 / 8
    context = Context(record, values)
    uniform = {**stats(run("attention.entropy", context, head=0)), **stats(run("attention.concentration", context, head=0))}
    assert uniform["Shannon entropy"] == pytest.approx(4) and uniform["Normalized entropy"] == pytest.approx(1)
    assert uniform["Gini concentration"] == pytest.approx(0, abs=1e-9) and uniform["Inverse Simpson effective support"] == pytest.approx(n)
    point = {**stats(run("attention.entropy", context, head=1)), **stats(run("attention.concentration", context, head=1))}
    assert point["Shannon entropy"] == 0 and point["Gini concentration"] == pytest.approx((n - 1) / n)
    assert point["Inverse Simpson effective support"] == pytest.approx(1)
    assert stats(run("attention.entropy", context, head=2))["Shannon entropy"] == pytest.approx(1)
    matrix = np.asarray(run("heads.distribution_similarity", context, heads=[1, 2, 0]).table.values)
    assert matrix[0, 1] == pytest.approx(0, abs=1e-9)          # disjoint supports: JSD is one bit
    assert np.diag(matrix) == pytest.approx(1)
    assert matrix == pytest.approx(matrix.T)
    table = run("attention.modality_ratio", context, head=0).table
    assert all(row[3] == pytest.approx(1) for row in table.values)  # uniform attention is never enriched


def test_every_implemented_prediction_method_has_a_reference_here():
    referenced = {"attention.row", "attention.element_allocation", "attention.mean_heads", "attention.max_heads",
                  "attention.min_heads", "attention.weighted_heads", "attention.entropy", "attention.concentration",
                  "attention.topk_mass", "attention.sinks", "attention.modality_ratio", "attention.head_by_head",
                  "attention.across_layers", "heads.distribution_similarity"}
    # Document references live in test_document_methods.py; this audit owns General rows.
    referenced.add("attention.modality_budget_depth")
    # Integrated gradients has its own references in test_planned_methods.py (integration rule and completeness).
    referenced.add("attribution.integrated_gradients")
    referenced.discard("attention.element_allocation")
    implemented = {m.method_id for m in list_methods() if m.status != "planned" and m.perspective=="general" and "generator_prediction_row_capture" in m.requires}
    assert implemented == referenced, "Add an independent reference test for each new row method"


# Retriever encodings: bidirectional rows with masked keys ---------------------

def encoding_fixture(seed, n=9, layers=2, heads=3):
    rng = np.random.default_rng(seed)
    mask = np.ones(n, dtype=int)
    mask[[2, 6]] = 0
    matrices = rng.dirichlet(np.full(n, .5), size=(layers, heads, n)).astype(np.float64)
    matrices[..., mask == 0] = 0
    matrices = (matrices / matrices.sum(-1, keepdims=True)).astype(np.float32)
    encoding = EncodingAttention(encoding_id="query", pass_kind="query", layer_count=layers, query_head_count=heads,
                                 position_count=n, attention_mask=mask.tolist(), layer_arrays=[f"l{i}" for i in range(layers)])
    roles = ["prefix_or_special", "query_text", "augmentation", "query_text", "query_text", "augmentation", "augmentation", "query_text", "prefix_or_special"]
    record = NS(capture_id="numeric_fixture", attention_encodings=[encoding], pages=[],
                query_positions=[Position(index=i, token_id=i, piece=str(i), role=roles[i], eligible=bool(mask[i])) for i in range(n)])
    return record, {f"l{i}": matrices[i] for i in range(layers)}, mask.astype(bool)


@pytest.mark.parametrize("seed", SEEDS)
def test_encoding_statistics_use_only_active_keys_and_allow_future_keys(seed):
    record, arrays, active = encoding_fixture(seed)

    def encoded(method, source, **parameters):
        return encoding_calculate(record, RetrievalAttentionRequest(encoding_id="query", source_index=source, method_id=method,
                                                                    parameters=parameters), arrays.__getitem__)
    for source in np.flatnonzero(active):
        a = arrays["l1"][2, source].astype(np.float64)
        if source < len(a) - 1:
            assert a[source + 1:].sum() > 0  # no causal truncation in an encoder
        assert encoded("attention.row", int(source), layer=1, head=2).row.values == pytest.approx(a, abs=1e-9)
        p = a[active] / a[active].sum()
        entropy = stats(encoded("attention.entropy", int(source), layer=1, head=2))
        assert entropy["Shannon entropy"] == pytest.approx(entropy_bits(p), abs=1e-9)
        assert entropy["Valid targets"] == active.sum()
        assert entropy["Normalized entropy"] == pytest.approx(entropy_bits(p) / math.log2(active.sum()), abs=1e-9)
        assert stats(encoded("attention.concentration", int(source), layer=1, head=2))["Gini concentration"] == pytest.approx(gini(p), abs=1e-9)
        with pytest.raises(ValueError, match="Top-k"):
            encoded("attention.topk_mass", int(source), layer=1, head=2, k=int(active.sum()) + 1)
    with pytest.raises(ValueError):
        encoded("attention.row", 2, layer=0, head=0)  # a masked source has no row


@pytest.mark.parametrize("seed", SEEDS)
def test_encoding_received_column_is_the_matrix_column_over_active_sources(seed):
    record, arrays, active = encoding_fixture(seed)
    for target in np.flatnonzero(active):
        column = received_column(record, RetrievalReceivedRequest(encoding_id="query", target_index=int(target), layer=1, head=0), arrays.__getitem__)
        assert [s.source_index for s in column.sources] == np.flatnonzero(active).tolist()
        assert [s.value for s in column.sources] == pytest.approx(arrays["l1"][0, active, target].tolist(), abs=1e-9)


# Retrieval scoring and ownership -----------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_maxsim_matches_brute_force_late_interaction(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(7, 5)).astype(np.float32)
    d = rng.normal(size=(11, 5)).astype(np.float32)
    d[4] = d[9] = rng.normal(size=5).astype(np.float32) * 6  # exact tie: the first eligible position wins
    q[1] = d[4]
    query_mask = rng.random(7) > .2
    query_mask[1] = True
    document_mask = rng.random(11) > .2
    document_mask[[4, 9]] = True
    score, winners, maxima, _ = maxsim(q, d, query_mask.tolist(), document_mask.tolist())
    expected_score, expected_winners = 0.0, []
    for i in np.flatnonzero(query_mask):
        best = max(np.flatnonzero(document_mask), key=lambda j: (float(q[i] @ d[j]), -j))
        expected_winners.append(int(best))
        expected_score += float(q[i] @ d[best])
    assert winners == expected_winners and score == pytest.approx(expected_score, rel=1e-5)
    assert winners[list(np.flatnonzero(query_mask)).index(1)] == 4
    owned = [dict(zip(["element_a", "background"], [f, 1 - f])) for f in rng.random(11)]
    totals = contributions(winners, maxima, owned)
    assert sum(totals.values()) == pytest.approx(sum(maxima))


def element(identifier, box, flags=()):
    return NS(element_id=identifier, bbox=NS(x1=box[0], y1=box[1], x2=box[2], y2=box[3]), geometry_flags=list(flags))


@pytest.mark.parametrize("seed", range(12))
def test_ownership_matches_point_sampling_on_aligned_layouts(seed):
    """Element edges lie on a 0.5 grid, so cell-centre sampling at that resolution is exact."""
    rng = np.random.default_rng(seed)
    elements = []
    for k in range(int(rng.integers(1, 6))):
        x1, y1 = rng.integers(-6, 18, 2) / 2
        width, height = rng.integers(0, 14, 2) / 2
        elements.append(element(f"element_{k}", [x1, y1, x1 + width, y1 + height], ["out_of_bounds"] if rng.random() < .25 else []))
    mapping = ownership([0, 0, 10, 10], elements)
    sampled = {}
    for y in np.arange(.25, 10, .5):
        for x in np.arange(.25, 10, .5):
            labels = ["unreviewed_geometry" if e.geometry_flags else e.element_id for e in elements
                      if e.bbox.x2 > e.bbox.x1 and e.bbox.y2 > e.bbox.y1 and e.bbox.x1 <= x < e.bbox.x2 and e.bbox.y1 <= y < e.bbox.y2] or ["background"]
            for label in labels:
                sampled[label] = sampled.get(label, 0.0) + 1 / len(labels) / 400
    assert mapping == pytest.approx(sampled, abs=1e-9)
    assert sum(mapping.values()) == pytest.approx(1)
