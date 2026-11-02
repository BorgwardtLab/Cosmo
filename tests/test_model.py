"""Small-graph tests for CosmoModel forward shapes and attribution completeness."""

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_sum

from cosmic import (
    CosmoModel,
    KernelPointCosmo,
    KernelPointXCosmo,
    NeuralFieldCosmo,
    PointTransformerCosmo,
)
from cosmic.model import default_kernel_points
from cosmic.utilities import index_select_columns


def _grid_graph(n_side=4, dim=2, in_channels=3, seed=0):
    torch.manual_seed(seed)
    xs = torch.linspace(0.0, 1.0, n_side)
    if dim == 2:
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    else:
        yy, xx, zz = torch.meshgrid(xs, xs, xs, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    n = coords.shape[0]
    node_features = torch.randn(n, in_channels)
    node2inst = torch.zeros(n, dtype=torch.long)
    edge_index = gnn.knn_graph(coords, k=4, batch=node2inst)
    return node_features, coords, edge_index, node2inst


def _assert_complete(scores, logits, node2inst, pool_to, atol=1e-5):
    graph_scores = scatter_sum(scores, node2inst, dim=0)
    if pool_to == "graph":
        target = logits
    else:
        target = scatter_sum(logits, node2inst, dim=0)
    err = (graph_scores - target).abs().max().item()
    assert torch.allclose(graph_scores, target, atol=atol, rtol=1e-4), (
        f"attribution completeness failed: max |sum(scores) - logits| = {err}"
    )
    return err


def test_forward_graph_and_node_shapes():
    feats, coords, edges, batch = _grid_graph(dim=2)
    n, c_in = feats.shape
    model = CosmoModel(
        in_channels=c_in,
        out_channels=5,
        hidden_channels=8,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.5,
        pool="max",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits = model(feats, coords, edges, batch)
    assert logits.shape == (1, 5)

    node_model = CosmoModel(
        in_channels=c_in,
        out_channels=5,
        hidden_channels=8,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.5,
        pool="mean",
        pool_to="node",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    node_model.eval()
    node_logits = node_model(feats, coords, edges, batch)
    assert node_logits.shape == (n, 5)


def test_attribution_max_identity_2d():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=2, seed=1)
    model = CosmoModel(
        in_channels=2,
        out_channels=3,
        hidden_channels=6,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.5,
        pool="max",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    assert scores.shape == (feats.shape[0], 3)
    err = _assert_complete(scores, logits, batch, "graph")
    print(f"max-pool identity 2D completeness error: {err:.3e}")


def test_attribution_sum_linear_head_2d():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=4, seed=2)
    model = CosmoModel(
        in_channels=4,
        out_channels=3,
        hidden_channels=7,
        num_layers=3,
        dim=2,
        filter="neural_field",
        radius=0.6,
        pool="sum",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits, scores = model.attribute(feats, coords, edges, batch)
    assert scores.shape == (feats.shape[0], 3)
    err = _assert_complete(scores, logits, batch, "graph")
    print(f"sum-pool linear head 2D completeness error: {err:.3e}")


def test_attribution_mean_3d():
    feats, coords, edges, batch = _grid_graph(n_side=3, dim=3, in_channels=2, seed=3)
    model = CosmoModel(
        in_channels=2,
        out_channels=2,
        hidden_channels=5,
        num_layers=2,
        dim=3,
        filter="neural_field",
        radius=0.8,
        pool="mean",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    assert scores.shape == (feats.shape[0], 2)
    err = _assert_complete(scores, logits, batch, "graph", atol=1e-4)
    print(f"mean-pool 3D completeness error: {err:.3e}")


def test_kernel_point_matches_kpconv_objective():
    """Standard KPConv is a linear correlation over every kernel point."""
    torch.manual_seed(0)
    cin, cout, dim = 3, 2, 2
    mu = default_kernel_points(dim, radius=1.0)
    layer = KernelPointCosmo(cin, cout, mu)
    assert torch.allclose(layer.sigma, torch.tensor(1.0))
    n_ent = 4
    features = torch.randn(n_ent, cin)
    hood = torch.cat([mu, mu[:1] * 0.5, torch.full((1, dim), 4.0)], dim=0)
    n_msg = hood.shape[0]
    source = torch.arange(n_msg) % n_ent
    target = torch.arange(n_msg) % n_ent
    out = layer(source, target, features, hood)
    dist = torch.cdist(hood, layer.mu)
    h = (1.0 - dist / layer.sigma).clamp(min=0)
    messages = torch.einsum("mi,oki,mk->mo", features[source], layer.w, h)
    ref = scatter_sum(messages, target, dim=0, dim_size=n_ent)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)

    # A neighbor sitting on the last kernel point uses only that kernel's weights.
    last = mu[-1]
    far = (torch.cdist(last.unsqueeze(0), mu).squeeze(0) >= layer.sigma - 1e-5).tolist()
    assert far[-1] is False and all(far[:-1])
    src = torch.tensor([0])
    dst = torch.tensor([1])
    feat = features[:1]
    got = layer(src, dst, features, last.unsqueeze(0))
    expect = torch.zeros(n_ent, cout)
    expect[1] = (feat @ layer.w[:, -1, :].T).squeeze(0)
    assert torch.allclose(got, expect, atol=1e-6, rtol=1e-6)

    # Outside the kernel extent the standard filter is silent; KPConvX still fires.
    outside = torch.full((1, dim), 4.0)
    xlayer = KernelPointXCosmo(cin, cout, mu)
    with torch.no_grad():
        xlayer.w.copy_(layer.w)
    silent = layer(src, dst, features, outside)
    assert torch.allclose(silent, torch.zeros_like(silent))
    fired = xlayer(src, dst, features, outside)
    assert not torch.allclose(fired, torch.zeros_like(fired))

    # Midway between two kernel points the linear blend is not a hard assignment.
    mid = 0.5 * (mu[0] + mu[1])
    blended = layer(src, dst, features, mid.unsqueeze(0))
    hard = xlayer(src, dst, features, mid.unsqueeze(0))
    assert not torch.allclose(blended, hard, atol=1e-5, rtol=1e-5)

    feat_g = features.detach().requires_grad_(True)
    out_g = layer(source, target, feat_g, hood)
    out_g.square().mean().backward()
    assert feat_g.grad is not None and layer.w.grad is not None
    assert torch.isfinite(layer.w.grad).all()
    print("kernel-point linear correlation ok")


def test_kernel_point_bias_silu_after_neighborhood_sum():
    """Bias and SiLU run on the summed neighborhood, not on each message."""
    torch.manual_seed(0)
    cin, cout, dim = 3, 2, 2
    mu = default_kernel_points(dim, radius=1.0)
    features = torch.tensor(
        [[1.0, 0.0, 0.0], [-2.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    hood = mu[:2]
    source = torch.tensor([0, 1])
    target = torch.tensor([2, 2])
    for cls in (KernelPointCosmo, KernelPointXCosmo):
        plain = cls(cin, cout, mu)
        assert plain.bias is None and plain.silu is False
        layer = cls(cin, cout, mu, bias=True, silu=True)
        with torch.no_grad():
            layer.w.copy_(plain.w)
            layer.bias.copy_(torch.tensor([0.3, -0.4]))
        out = layer(source, target, features, hood)
        separate = plain(source, torch.tensor([0, 1]), features, hood)
        summed = separate[0] + separate[1]
        expect_row = F.silu(summed + layer.bias)
        assert torch.allclose(out[2], expect_row, atol=1e-6, rtol=1e-6)
        assert torch.allclose(out[:2], F.silu(layer.bias).expand(2, -1), atol=1e-6)
        before = F.silu(separate[0] + layer.bias) + F.silu(separate[1] + layer.bias)
        assert not torch.allclose(before, expect_row, atol=1e-5, rtol=1e-5)
    print("kernel-point bias and SiLU follow the neighborhood sum")


def test_neighbor_gate_keeps_current_state():
    """A closed gate returns the current state; an open gate returns the aggregate."""
    torch.manual_seed(0)
    cin = cout = 4
    n_ent, n_msg, dim = 6, 8, 2
    features = torch.randn(n_ent, cin)
    hood = torch.randn(n_msg, dim)
    source = torch.randint(0, n_ent, (n_msg,))
    target = torch.randint(0, n_ent, (n_msg,))
    mu = default_kernel_points(dim, radius=1.0)
    makers = (
        lambda gate: KernelPointCosmo(cin, cout, mu, gate=gate),
        lambda gate: KernelPointXCosmo(cin, cout, mu, gate=gate),
        lambda gate: NeuralFieldCosmo(
            cin, cout, field_channels=8, field_layers=2, dim=dim, gate=gate
        ),
        lambda gate: PointTransformerCosmo(cin, cout, dim=dim, gate=gate),
    )
    for make in makers:
        plain = make(False)
        gated = make(True)
        assert plain.gate is False and plain.neighbor_gate is None
        gated.load_state_dict(plain.state_dict(), strict=False)
        aggregated = plain(source, target, features, hood)
        half = gated(source, target, features, hood)
        expect = 0.5 * aggregated + 0.5 * features
        assert torch.allclose(half, expect, atol=1e-5, rtol=1e-5)
        with torch.no_grad():
            gated.neighbor_gate.bias.fill_(-40)
        closed = gated(source, target, features, hood)
        assert torch.allclose(closed, features, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(closed, aggregated, atol=1e-3, rtol=1e-3)
        with torch.no_grad():
            gated.neighbor_gate.bias.fill_(40)
        opened = gated(source, target, features, hood)
        assert torch.allclose(opened, aggregated, atol=1e-4, rtol=1e-4)

    wide = KernelPointCosmo(3, 5, mu, gate=True)
    assert wide.state_proj is not None
    feat_w = torch.randn(n_ent, 3)
    out = wide(source, target, feat_w, hood)
    plain = KernelPointCosmo(3, 5, mu, gate=False)
    wide.load_state_dict(plain.state_dict(), strict=False)
    aggregated = plain(source, target, feat_w, hood)
    update = torch.sigmoid(wide.neighbor_gate(feat_w))
    state = wide.state_proj(feat_w)
    expect = update * aggregated + (1.0 - update) * state
    assert torch.allclose(wide(source, target, feat_w, hood), expect, atol=1e-5)
    assert out.shape == (n_ent, 5)
    print("neighbor gate mixes the current state with the aggregate")


def test_attribution_kernel_point():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=3, seed=4)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="kernel_point",
        radius=0.5,
        pool="sum",
        pool_to="graph",
    )
    model.eval()
    assert isinstance(model.cosmo_layers[0], KernelPointCosmo)
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    err = _assert_complete(scores, logits, batch, "graph")
    print(f"kernel-point completeness error: {err:.3e}")


def test_attribution_kernel_point_x():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=3, seed=6)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="kernel_point_x",
        radius=0.5,
        pool="sum",
        pool_to="graph",
    )
    model.eval()
    assert isinstance(model.cosmo_layers[0], KernelPointXCosmo)
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    err = _assert_complete(scores, logits, batch, "graph")
    print(f"kernel-point-x completeness error: {err:.3e}")


def test_mixed_filters_and_node_pool_attribution():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=2, seed=5)
    model = CosmoModel(
        in_channels=2,
        out_channels=4,
        hidden_channels=6,
        num_layers=2,
        dim=2,
        filter=["neural_field", "neural_field"],
        radius=0.5,
        pool="sum",
        pool_to="node",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    assert logits.shape == (feats.shape[0], 4)
    assert scores.shape == (feats.shape[0], 4)
    err = _assert_complete(scores, logits, batch, "node")
    print(f"node-pool completeness error: {err:.3e}")


def test_attribution_batched_and_softmax():
    f1, c1, e1, b1 = _grid_graph(n_side=3, dim=2, in_channels=2, seed=8)
    f2, c2, e2, b2 = _grid_graph(n_side=3, dim=2, in_channels=2, seed=9)
    n1 = f1.shape[0]
    feats = torch.cat([f1, f2], dim=0)
    coords = torch.cat([c1, c2], dim=0)
    edges = torch.cat([e1, e2 + n1], dim=1)
    batch = torch.cat([b1, b2 + 1], dim=0)

    model = CosmoModel(
        in_channels=2,
        out_channels=3,
        hidden_channels=5,
        num_layers=2,
        dim=2,
        radius=0.7,
        pool="softmax",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    assert logits.shape == (2, 3)
    assert scores.shape == (feats.shape[0], 3)
    err = _assert_complete(scores, logits, batch, "graph")
    print(f"batched softmax completeness error: {err:.3e}")


def test_compose_coalesces_by_lifted_id():
    feats, coords, edges, batch = _grid_graph(n_side=6, dim=2, in_channels=3, seed=11)
    model = CosmoModel(
        in_channels=3,
        out_channels=4,
        hidden_channels=8,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.5,
        pool="sum",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    L = model._lift(feats, coords, edges, batch)
    lifted = model._cosmo_features(L)
    W, _ = model._linear_readout(lifted.shape[1], feats.device, feats.dtype)
    e0, f0 = model._initial_fields(L, lifted, W, feats.shape[0], 1)
    src, _dst = index_select_columns(L.source, L.target, e0, L.features.shape[0])
    e1, f1 = model._compose_fields(L, e0, f0)
    assert e1.shape[0] == torch.unique(e1).numel()
    assert e1.shape[0] == f1.shape[0]
    assert e1.shape[0] <= L.features.shape[0]
    assert src.shape[0] > e1.shape[0]
    logits, scores = model.attribute(feats, coords, edges, batch)
    err = _assert_complete(scores, logits, batch, "graph")
    print(
        f"coalesce: n_msg={src.shape[0]} unique={e1.shape[0]} "
        f"lifted={L.features.shape[0]} completeness={err:.3e}"
    )


def test_attribution_chunked_matches_unchunked():
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=3, seed=10)
    model = CosmoModel(
        in_channels=3,
        out_channels=4,
        hidden_channels=6,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.5,
        pool="sum",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    logits_full, scores_full = model.attribute(feats, coords, edges, batch)
    model.attribution_chunk_size = 7
    model.attribution_start_chunk = 5
    logits_chunk, scores_chunk = model.attribute(feats, coords, edges, batch)
    assert torch.allclose(logits_full, logits_chunk, atol=1e-5, rtol=1e-4)
    assert torch.allclose(scores_full, scores_chunk, atol=1e-5, rtol=1e-4)
    err = _assert_complete(scores_chunk, logits_chunk, batch, "graph")
    print(f"chunked attribution completeness error: {err:.3e}")


def test_attribution_cuda_peak_near_forward():
    if not torch.cuda.is_available():
        print("skip CUDA VRAM test")
        return
    device = torch.device("cuda")
    feats, coords, edges, batch = _grid_graph(n_side=16, dim=2, in_channels=4, seed=12)
    feats, coords, edges, batch = (
        feats.to(device),
        coords.to(device),
        edges.to(device),
        batch.to(device),
    )
    model = CosmoModel(
        in_channels=4,
        out_channels=8,
        hidden_channels=32,
        num_layers=2,
        dim=2,
        filter="neural_field",
        radius=0.2,
        pool="sum",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    ).to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = model(feats, coords, edges, batch)
    fwd_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logits, scores = model.attribute(feats, coords, edges, batch)
    attr_peak = torch.cuda.max_memory_allocated()
    err = _assert_complete(scores, logits, batch, "graph", atol=1e-3)
    extra = attr_peak - fwd_peak
    print(
        f"CUDA VRAM forward={fwd_peak/1e6:.1f} MB "
        f"attribute={attr_peak/1e6:.1f} MB extra={extra/1e6:.1f} MB "
        f"completeness={err:.3e}"
    )
    assert attr_peak <= fwd_peak * 1.25 + 32 * 1024 * 1024, (
        f"attribution peak {attr_peak} far above forward {fwd_peak}"
    )


def test_relu_blocks_exact_attribution():
    feats, coords, edges, batch = _grid_graph(dim=2, seed=6)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        activation="relu",
        pool="sum",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    try:
        model(feats, coords, edges, batch, attribution=True)
    except TypeError as exc:
        assert "identity activations" in str(exc)
        print("ReLU correctly rejected for exact attribution")
    else:
        raise AssertionError("expected TypeError for ReLU attribution")


def test_kernel_silu_blocks_exact_attribution():
    feats, coords, edges, batch = _grid_graph(dim=2, seed=6)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="kernel_point",
        kernel_silu=True,
        pool="sum",
        pool_to="graph",
    )
    model.eval()
    try:
        model(feats, coords, edges, batch, attribution=True)
    except TypeError as exc:
        assert "SiLU" in str(exc)
        print("kernel SiLU correctly rejected for exact attribution")
    else:
        raise AssertionError("expected TypeError for SiLU kernel attribution")


def test_neighbor_gate_blocks_exact_attribution():
    feats, coords, edges, batch = _grid_graph(dim=2, seed=8)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="neural_field",
        gate=True,
        pool="sum",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    )
    model.eval()
    assert model.cosmo_layers[0].gate is True
    try:
        model(feats, coords, edges, batch, attribution=True)
    except TypeError as exc:
        assert "neighbor gate" in str(exc)
        print("neighbor gate correctly rejected for exact attribution")
    else:
        raise AssertionError("expected TypeError for gated attribution")


def test_trainable_backward():
    feats, coords, edges, batch = _grid_graph(dim=2, seed=7)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=8,
        num_layers=2,
        dim=2,
        radius=0.5,
        pool="mean",
        pool_to="graph",
        head="linear",
        field_channels=8,
        field_layers=2,
    )
    logits = model(feats, coords, edges, batch)
    loss = logits.square().mean()
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad]
    assert any(g > 0 for g in grads)
    print(f"backward ok, loss={loss.item():.4f}")


if __name__ == "__main__":
    torch.set_printoptions(precision=5, sci_mode=True)
    tests = [
        test_forward_graph_and_node_shapes,
        test_attribution_max_identity_2d,
        test_attribution_sum_linear_head_2d,
        test_attribution_mean_3d,
        test_kernel_point_matches_kpconv_objective,
        test_kernel_point_bias_silu_after_neighborhood_sum,
        test_neighbor_gate_keeps_current_state,
        test_attribution_kernel_point,
        test_attribution_kernel_point_x,
        test_mixed_filters_and_node_pool_attribution,
        test_attribution_batched_and_softmax,
        test_compose_coalesces_by_lifted_id,
        test_attribution_chunked_matches_unchunked,
        test_attribution_cuda_peak_near_forward,
        test_relu_blocks_exact_attribution,
        test_kernel_silu_blocks_exact_attribution,
        test_neighbor_gate_blocks_exact_attribution,
        test_trainable_backward,
    ]
    for fn in tests:
        print(f"--- {fn.__name__}")
        fn()
    print("all tests passed")
