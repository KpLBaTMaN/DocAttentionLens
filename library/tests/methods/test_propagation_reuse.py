"""Reused layer reductions and flow networks must reproduce the previous per-call results exactly."""
from collections import deque
import numpy as np
import pytest
from docattentionlens.methods import _propagation
from docattentionlens.methods._propagation import reduced, rollout, rollouts
from docattentionlens.methods._max_flow import FlowNetwork, maximum_flow
from docattentionlens.methods.attention_flow import strongest
from docattentionlens.methods.contracts import Context
from docattentionlens.methods.rollout_evolution import Parameters as EvolutionParameters
from docattentionlens.retrieval.attention import row_context
from test_planned_methods import document_encoding, encoded, query_encoding
from test_saved_rows import run as saved_run


def context_of(record, arrays, encoding_id, source):
    _, row_record = row_context(record, encoding_id, source)
    return Context(row_record, None, read_array=lambda layer: arrays[f"l{layer}"])


def previous_rollout(context, p, start, end):
    """The rollout loop before layer reuse: every call reduces every layer again."""
    source = context.record.steps[0].source_index
    row = np.zeros(len(context.record.positions)); row[source] = 1
    for layer in range(end, start-1, -1):
        row = row @ reduced(context, p, layer)
    return row


def counting(monkeypatch):
    calls = []
    original = _propagation.matrix
    monkeypatch.setattr(_propagation, "matrix", lambda context, layer: calls.append(layer) or original(context, layer))
    return calls


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("settings", [{}, {"heads": [1], "head_reduction": "max", "residual_coefficient": .2}])
def test_rollouts_reduce_each_layer_once_and_equal_previous_rollouts_bit_for_bit(monkeypatch, causal, settings):
    record, arrays, _ = query_encoding(3, layers=5, causal=causal)
    context = context_of(record, arrays, "query", 7)
    p = EvolutionParameters(**settings)
    ends = [4, 1, 3, 2]
    expected = [previous_rollout(context, p, 1, end) for end in ends]
    calls = counting(monkeypatch)
    rows = rollouts(context, p, 1, ends)
    assert sorted(calls) == [1, 2, 3, 4]
    assert all(np.array_equal(a, b) for a, b in zip(rows, expected))
    cache = {}
    assert np.array_equal(rollout(context, p, 1, 3, cache), expected[2])
    assert sorted(cache) == [1, 2, 3]
    assert np.array_equal(rollout(context, p, 1, 4, cache), expected[0])
    assert sorted(calls[4:]) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="interval"):
        rollouts(context, p, 2, [1])


def test_rollout_evolution_reuses_layers_and_plots_image_and_self_share(monkeypatch):
    record, arrays, positions = document_encoding(4, grid=3, prompt=2, layers=4)
    calls = counting(monkeypatch)
    result = encoded(record, arrays, "document", "attention.rollout_evolution", 10, start_layer=1, layers=[3, 1, 2])
    assert sorted(calls) == [1, 2, 3]
    context = context_of(record, arrays, "document", 10)
    p = EvolutionParameters(start_layer=1)
    image = [i for i, position in enumerate(positions) if position.role == "visual_patch"]
    (plot,) = result.plots
    assert plot.kind == "line" and plot.x_label == "End layer"
    by_group = {group: [point for point in plot.points if point.group == group] for group in ("Image share", "Source self share")}
    for end, frame, image_point, self_point in zip([3, 1, 2], result.comparisons, *by_group.values()):
        expected = previous_rollout(context, p, 1, end)
        assert frame.row.values == expected.tolist()
        assert image_point.x == self_point.x == end
        assert image_point.y == pytest.approx(expected[image].sum(), abs=1e-12)
        assert self_point.y == expected[10]
    assert len(plot.points) == 6


def test_rollout_evolution_without_image_positions_plots_only_self_share():
    record, arrays, _ = query_encoding(1)
    result = encoded(record, arrays, "query", "attention.rollout_evolution", 7, layers=[0, 2])
    assert {point.group for point in result.plots[0].points} == {"Source self share"}
    assert [point.y for point in result.plots[0].points] == [frame.row.values[7] for frame in result.comparisons]
    assert any("no image positions" in notice for notice in result.notices)


def previous_maximum_flow(edges, source, target):
    """The Dinic solver before network reuse: a fresh graph for every target."""
    vertices = {source, target}
    for a, b, _ in edges: vertices.update((a, b))
    ordered = sorted(vertices); index = {v: i for i, v in enumerate(ordered)}
    adjacency = [[] for _ in ordered]
    for a, b, value in edges:
        u, v = index[a], index[b]
        forward = [v, float(value), len(adjacency[v])]
        reverse = [u, 0.0, len(adjacency[u])]
        adjacency[u].append(forward); adjacency[v].append(reverse)
    s, t = index[source], index[target]; total = 0.0
    while True:
        levels = [-1]*len(ordered); levels[s] = 0; queue = deque([s])
        while queue:
            u = queue.popleft()
            for v, capacity, _ in adjacency[u]:
                if capacity > 1e-12 and levels[v] < 0: levels[v] = levels[u]+1; queue.append(v)
        if levels[t] < 0: return total
        pointers = [0]*len(ordered)
        def push(u, value):
            if u == t: return value
            while pointers[u] < len(adjacency[u]):
                edge = adjacency[u][pointers[u]]; v, capacity, reverse = edge
                if capacity > 1e-12 and levels[v] == levels[u]+1:
                    sent = push(v, min(value, capacity))
                    if sent > 1e-12:
                        edge[1] -= sent; adjacency[v][reverse][1] += sent
                        return sent
                pointers[u] += 1
            return 0.0
        while (sent := push(s, float("inf"))) > 1e-12: total += sent


@pytest.mark.parametrize("seed", range(6))
def test_reused_flow_network_equals_fresh_graph_solves_for_every_target(seed):
    rng = np.random.default_rng(seed)
    width, depth = 7, 4
    edges = []
    reachable = {0}
    for layer in range(depth, 0, -1):
        following = set()
        for position in sorted(reachable):
            targets = rng.choice(width, size=rng.integers(1, 5), replace=False)
            for target in targets:
                capacity = float(rng.choice([.25, .5])) if rng.random() < .3 else float(rng.random())
                edges.append((f"L{layer}:P{position}", f"L{layer-1}:P{int(target)}", capacity))
                following.add(int(target))
        reachable = following
    network = FlowNetwork(edges)
    targets = [f"L0:P{i}" for i in range(width)] + ["L0:P99"]
    for _ in range(2):  # repeated solves must not see earlier residual capacities
        for target in targets:
            assert network.maximum_flow(f"L{depth}:P0", target) == previous_maximum_flow(edges, f"L{depth}:P0", target)
    assert maximum_flow(edges, f"L{depth}:P0", "L0:P99") == 0


def previous_selection(weights, active, coverage, limit):
    order = sorted([int(t) for t in active if weights[t] > 0], key=lambda t: (-weights[t], t))
    selected = []; capacity = 0
    for target in order:
        selected.append(target); capacity += float(weights[target])
        if capacity >= coverage-1e-12 or len(selected) >= limit: break
    return selected, [float(weights[t]) for t in selected], capacity


@pytest.mark.parametrize("seed", range(8))
def test_vectorised_edge_retention_equals_the_previous_loop(seed):
    rng = np.random.default_rng(seed)
    n = 40
    weights = np.round(rng.dirichlet(np.full(n, .3)), 3).astype(np.float64)  # rounding creates ties and zeros
    active = np.flatnonzero(rng.random(n) < .8)
    for coverage in (.3, .95, 1.0):
        for limit in (1, 4, 32, 128):
            assert strongest(weights, active, coverage, limit) == previous_selection(weights, active, coverage, limit)
    assert strongest(np.zeros(n), active, .95, 32) == ([], [], 0.0)


def test_flow_reduces_each_layer_once_and_ranks_by_the_same_rollout(monkeypatch):
    record, arrays, _ = document_encoding(2, grid=3, prompt=2, layers=4)
    calls = counting(monkeypatch)
    result = encoded(record, arrays, "document", "attention.flow", 10, layer=3, start_layer=1, max_targets=4)
    assert sorted(calls) == [1, 2, 3]
    context = context_of(record, arrays, "document", 10)
    from docattentionlens.methods.attention_flow import Parameters as FlowParameters
    expected = previous_rollout(context, FlowParameters(layer=3, start_layer=1), 1, 3)
    assert [row[2] for row in result.table.values] == [float(expected[int(row[0])]) for row in result.table.values]


def test_head_similarity_grid_matches_its_table():
    result = saved_run("heads.distribution_similarity", heads=[2, 0, 1])
    (plot,) = result.plots
    assert plot.kind == "grid" and plot.x_label == "Compared head" and plot.y_label == "Head"
    values = result.table.values
    assert len(plot.points) == 9
    for point in plot.points:
        i, j = round(point.y*2), round(point.x*2)
        assert point.label == f"{result.table.labels[i]} vs {result.table.labels[j]}"
        assert point.value == values[i][j]
    assert plot.points[5].label == "Head 0 vs Head 1" and plot.points[5].value == 0  # disjoint support


def test_rollout_evolution_default_ends_keep_the_bound_first_and_last_layer():
    from docattentionlens.methods.rollout_evolution import default_ends, MAX_END_LAYERS
    assert default_ends(0, 18) == list(range(18))
    ends = default_ends(0, 36)
    assert len(ends) == MAX_END_LAYERS and ends[0] == 0 and ends[-1] == 35 and ends == sorted(set(ends))
    ends = default_ends(3, 36)
    assert ends[0] == 3 and ends[-1] == 35 and len(ends) == MAX_END_LAYERS


def test_rollout_evolution_all_layers_default_resolves_to_bounded_ends():
    from types import SimpleNamespace
    from docattentionlens.methods.registry import prepare_parameters
    record = SimpleNamespace(layer_count=36, query_head_count=4, steps=[SimpleNamespace(step=0, source_index=3, target_count=4)], positions=[])
    parsed = prepare_parameters("attention.rollout_evolution", "captured_attention_v1", {"step": 0}, record)
    assert len(parsed.layers) == 32 and parsed.layers[0] == 0 and parsed.layers[-1] == 35
