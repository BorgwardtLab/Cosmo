"""Small-graph tests for CosmoModel forward shapes and attribution completeness."""

import torch
import torch_geometric.nn as gnn
from torch_scatter import scatter_sum

from cosmic import CosmoModel, KernelPointCosmo


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
        test_attribution_kernel_point,
        test_mixed_filters_and_node_pool_attribution,
        test_attribution_batched_and_softmax,
        test_relu_blocks_exact_attribution,
        test_trainable_backward,
    ]
    for fn in tests:
        print(f"--- {fn.__name__}")
        fn()
    print("all tests passed")
