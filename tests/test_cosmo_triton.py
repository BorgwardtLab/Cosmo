"""Parity tests for Triton Cosmo layers against the reference implementations."""

import torch
from torch_scatter import scatter_sum

from cosmic import CosmoModel
from cosmic.cosmo import KernelPointCosmo, KernelPointXCosmo, NeuralFieldCosmo
from cosmic.cosmo_triton import (
    KernelPointCosmoTriton,
    KernelPointXCosmoTriton,
    NeuralFieldCosmoTriton,
)
from cosmic.model import default_kernel_points
from cosmic.triton_ops import TRITON_AVAILABLE, _kp_backward_torch, _kp_x_backward_torch


def _require_cuda():
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        raise RuntimeError("CUDA + Triton required for Triton Cosmo tests")


def _lifted_batch(n_ent=48, n_msg=80, cin=8, dim=3, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(n_ent, cin, generator=g)
    hood = torch.randn(n_msg, dim, generator=g)
    source = torch.randint(0, n_ent, (n_msg,), generator=g)
    target = torch.randint(0, n_ent, (n_msg,), generator=g)
    return (
        source.to(device),
        target.to(device),
        features.to(device),
        hood.to(device),
    )


def _max_err(a, b):
    return (a - b).abs().max().item()


def _assert_kp_parity(
    ref_cls, trt_cls, cin, cout, dim, batch_kwargs, atol=1e-4, layer_kwargs=None
):
    source, target, features, hood = _lifted_batch(cin=cin, dim=dim, **batch_kwargs)
    mu = default_kernel_points(dim, radius=1.0)
    kwargs = {} if layer_kwargs is None else layer_kwargs
    ref = ref_cls(cin, cout, mu, **kwargs).to(features.device)
    trt = trt_cls(cin, cout, mu, **kwargs).to(features.device)
    trt.load_state_dict(ref.state_dict())

    feat_ref = features.detach().requires_grad_(True)
    feat_trt = features.detach().requires_grad_(True)
    out_ref = ref(source, target, feat_ref, hood)
    out_trt = trt(source, target, feat_trt, hood)
    err_fwd = _max_err(out_ref, out_trt)
    assert torch.allclose(out_ref, out_trt, atol=atol, rtol=atol), err_fwd

    out_ref.square().mean().backward()
    out_trt.square().mean().backward()
    err_feat = _max_err(feat_ref.grad, feat_trt.grad)
    err_w = _max_err(ref.w.grad, trt.w.grad)
    assert torch.allclose(feat_ref.grad, feat_trt.grad, atol=atol, rtol=atol), err_feat
    assert torch.allclose(ref.w.grad, trt.w.grad, atol=atol, rtol=atol), err_w
    if ref.bias is not None:
        err_b = _max_err(ref.bias.grad, trt.bias.grad)
        assert torch.allclose(ref.bias.grad, trt.bias.grad, atol=atol, rtol=atol), err_b
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), trt.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=atol, rtol=atol), n1
    return err_fwd, err_feat, err_w


def test_kernel_backward_chunks_match_one_gather():
    torch.manual_seed(0)
    n, m, cin, cout, k = 12, 30, 5, 4, 3
    features = torch.randn(n, cin)
    source = torch.randint(0, n, (m,))
    weight = torch.randn(cout, k, cin)
    grad_msg = torch.randn(m, cout)
    influence = torch.rand(m, k)
    full = _kp_backward_torch(grad_msg, features, source, weight, influence, row_chunk=m)
    chunked = _kp_backward_torch(grad_msg, features, source, weight, influence, row_chunk=7)
    assert torch.allclose(full[0], chunked[0])
    assert torch.allclose(full[1], chunked[1])
    nearest = torch.randint(0, k, (m,))
    full_x = _kp_x_backward_torch(grad_msg, features, source, weight, nearest, row_chunk=m)
    chunked_x = _kp_x_backward_torch(
        grad_msg, features, source, weight, nearest, row_chunk=4
    )
    assert torch.allclose(full_x[0], chunked_x[0])
    assert torch.allclose(full_x[1], chunked_x[1])


def test_kernel_point_forward_backward_parity():
    _require_cuda()
    err_fwd, err_feat, err_w = _assert_kp_parity(
        KernelPointCosmo,
        KernelPointCosmoTriton,
        cin=8,
        cout=6,
        dim=3,
        batch_kwargs={"device": "cuda"},
    )
    print(f"KP parity  fwd={err_fwd:.3e}  dfeat={err_feat:.3e}  dw={err_w:.3e}")


def test_kernel_point_x_forward_backward_parity():
    _require_cuda()
    err_fwd, err_feat, err_w = _assert_kp_parity(
        KernelPointXCosmo,
        KernelPointXCosmoTriton,
        cin=8,
        cout=6,
        dim=3,
        batch_kwargs={"device": "cuda", "seed": 2},
    )
    print(f"KPX parity  fwd={err_fwd:.3e}  dfeat={err_feat:.3e}  dw={err_w:.3e}")


def test_kernel_point_bias_silu_parity():
    _require_cuda()
    for ref_cls, trt_cls, label in (
        (KernelPointCosmo, KernelPointCosmoTriton, "KP"),
        (KernelPointXCosmo, KernelPointXCosmoTriton, "KPX"),
    ):
        err_fwd, err_feat, err_w = _assert_kp_parity(
            ref_cls,
            trt_cls,
            cin=8,
            cout=6,
            dim=3,
            batch_kwargs={"device": "cuda", "seed": 4},
            layer_kwargs={"bias": True, "silu": True},
        )
        print(
            f"{label} bias+SiLU parity  fwd={err_fwd:.3e}  "
            f"dfeat={err_feat:.3e}  dw={err_w:.3e}"
        )


def test_neighbor_gate_triton_parity():
    _require_cuda()
    for ref_cls, trt_cls, label in (
        (KernelPointCosmo, KernelPointCosmoTriton, "KP"),
        (KernelPointXCosmo, KernelPointXCosmoTriton, "KPX"),
    ):
        err_fwd, err_feat, err_w = _assert_kp_parity(
            ref_cls,
            trt_cls,
            cin=8,
            cout=5,
            dim=3,
            batch_kwargs={"device": "cuda", "seed": 5},
            layer_kwargs={"gate": True},
        )
        print(
            f"{label} gate parity  fwd={err_fwd:.3e}  "
            f"dfeat={err_feat:.3e}  dw={err_w:.3e}"
        )
    cin, cout, dim = 7, 5, 3
    source, target, features, hood = _lifted_batch(
        n_ent=40, n_msg=64, cin=cin, dim=dim, seed=6, device="cuda"
    )
    ref = NeuralFieldCosmo(
        cin, cout, field_channels=16, field_layers=2, dim=dim, gate=True
    ).cuda()
    trt = NeuralFieldCosmoTriton(
        cin, cout, field_channels=16, field_layers=2, dim=dim, gate=True
    ).cuda()
    trt.load_state_dict(ref.state_dict())
    feat_ref = features.detach().requires_grad_(True)
    feat_trt = features.detach().requires_grad_(True)
    out_ref = ref(source, target, feat_ref, hood)
    out_trt = trt(source, target, feat_trt, hood)
    assert torch.allclose(out_ref, out_trt, atol=2e-4, rtol=2e-4)
    out_ref.square().mean().backward()
    out_trt.square().mean().backward()
    assert torch.allclose(feat_ref.grad, feat_trt.grad, atol=2e-4, rtol=2e-4)
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), trt.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=2e-4, rtol=2e-4), n1
    print("NF gate parity ok")


def test_neural_field_forward_backward_parity():
    _require_cuda()
    device = "cuda"
    cin, cout, dim = 7, 5, 3
    source, target, features, hood = _lifted_batch(
        n_ent=40, n_msg=64, cin=cin, dim=dim, seed=1, device=device
    )
    ref = NeuralFieldCosmo(
        cin, cout, field_channels=16, field_layers=2, dim=dim, radius=0.8
    ).to(device)
    trt = NeuralFieldCosmoTriton(
        cin, cout, field_channels=16, field_layers=2, dim=dim, radius=0.8
    ).to(device)
    trt.load_state_dict(ref.state_dict())

    feat_ref = features.detach().requires_grad_(True)
    feat_trt = features.detach().requires_grad_(True)
    hood_ref = hood.detach().requires_grad_(True)
    hood_trt = hood.detach().requires_grad_(True)
    out_ref = ref(source, target, feat_ref, hood_ref)
    out_trt = trt(source, target, feat_trt, hood_trt)
    err_fwd = _max_err(out_ref, out_trt)
    assert torch.allclose(out_ref, out_trt, atol=2e-4, rtol=2e-4), err_fwd

    out_ref.square().mean().backward()
    out_trt.square().mean().backward()
    err_feat = _max_err(feat_ref.grad, feat_trt.grad)
    err_hood = _max_err(hood_ref.grad, hood_trt.grad)
    assert torch.allclose(feat_ref.grad, feat_trt.grad, atol=2e-4, rtol=2e-4), err_feat
    assert torch.allclose(hood_ref.grad, hood_trt.grad, atol=2e-4, rtol=2e-4), err_hood
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), trt.named_parameters()):
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
            continue
        err_p = _max_err(p1.grad, p2.grad)
        assert torch.allclose(p1.grad, p2.grad, atol=2e-4, rtol=2e-4), (n1, err_p)
    print(f"NF parity  fwd={err_fwd:.3e}  dfeat={err_feat:.3e}  dhood={err_hood:.3e}")


def test_larger_channel_parity():
    _require_cuda()
    source, target, features, hood = _lifted_batch(
        n_ent=256, n_msg=1024, cin=32, dim=3, seed=11
    )
    mu = default_kernel_points(3)
    for ref_cls, trt_cls in (
        (KernelPointCosmo, KernelPointCosmoTriton),
        (KernelPointXCosmo, KernelPointXCosmoTriton),
    ):
        ref = ref_cls(32, 32, mu).cuda()
        trt = trt_cls(32, 32, mu).cuda()
        trt.load_state_dict(ref.state_dict())
        feat_r = features.detach().requires_grad_(True)
        feat_t = features.detach().requires_grad_(True)
        out_r = ref(source, target, feat_r, hood)
        out_t = trt(source, target, feat_t, hood)
        out_r.square().mean().backward()
        out_t.square().mean().backward()
        assert torch.allclose(out_r, out_t, atol=1e-4, rtol=1e-4)
        assert torch.allclose(feat_r.grad, feat_t.grad, atol=1e-4, rtol=1e-4)

    ref = NeuralFieldCosmo(32, 24, field_channels=32, field_layers=2, dim=3).cuda()
    trt = NeuralFieldCosmoTriton(32, 24, field_channels=32, field_layers=2, dim=3).cuda()
    trt.load_state_dict(ref.state_dict())
    feat_r = features.detach().requires_grad_(True)
    feat_t = features.detach().requires_grad_(True)
    out_r = ref(source, target, feat_r, hood)
    out_t = trt(source, target, feat_t, hood)
    out_r.square().mean().backward()
    out_t.square().mean().backward()
    assert torch.allclose(out_r, out_t, atol=2e-4, rtol=2e-4)
    assert torch.allclose(feat_r.grad, feat_t.grad, atol=2e-4, rtol=2e-4)
    print("larger-channel parity ok")


def test_unequal_message_and_entity_counts():
    """Messages and lifted entities are different sizes after a real lift."""
    _require_cuda()
    source, target, features, hood = _lifted_batch(
        n_ent=20, n_msg=55, cin=4, dim=2, seed=3
    )
    mu = default_kernel_points(2, radius=1.0)
    for ref_cls, trt_cls in (
        (KernelPointCosmo, KernelPointCosmoTriton),
        (KernelPointXCosmo, KernelPointXCosmoTriton),
    ):
        ref = ref_cls(4, 3, mu).cuda()
        trt = trt_cls(4, 3, mu).cuda()
        trt.load_state_dict(ref.state_dict())
        out_ref = ref(source, target, features, hood)
        out_trt = trt(source, target, features, hood)
        assert out_ref.shape == (20, 3)
        assert torch.allclose(out_ref, out_trt, atol=1e-4, rtol=1e-4)
    print("KP unequal m/n parity ok")


def test_kernel_point_padded_tile_last_row():
    """BLOCK_IN/OUT exceed c_in/c_out; gather the last feature row."""
    _require_cuda()
    n_ent, n_msg, cin, cout = 17, 64, 5, 7
    source = torch.full((n_msg,), n_ent - 1, dtype=torch.long, device="cuda")
    target = torch.zeros(n_msg, dtype=torch.long, device="cuda")
    features = torch.randn(n_ent, cin, device="cuda")
    hood = torch.randn(n_msg, 3, device="cuda")
    mu = default_kernel_points(3)
    for ref_cls, trt_cls in (
        (KernelPointCosmo, KernelPointCosmoTriton),
        (KernelPointXCosmo, KernelPointXCosmoTriton),
    ):
        ref = ref_cls(cin, cout, mu).cuda()
        trt = trt_cls(cin, cout, mu).cuda()
        trt.load_state_dict(ref.state_dict())
        out_ref = ref(source, target, features, hood)
        out_trt = trt(source, target, features, hood)
        assert torch.allclose(out_ref, out_trt, atol=1e-4, rtol=1e-4)
    print("KP padded-tile last-row parity ok")


def test_empty_graph():
    _require_cuda()
    mu = default_kernel_points(3)
    kp = KernelPointCosmoTriton(4, 3, mu).cuda()
    kpx = KernelPointXCosmoTriton(4, 3, mu).cuda()
    nf = NeuralFieldCosmoTriton(4, 3, field_layers=2, dim=3).cuda()
    source = torch.zeros(0, dtype=torch.long, device="cuda")
    target = torch.zeros(0, dtype=torch.long, device="cuda")
    features = torch.zeros(0, 4, device="cuda")
    hood = torch.zeros(0, 3, device="cuda")
    assert kp(source, target, features, hood).shape == (0, 3)
    assert kpx(source, target, features, hood).shape == (0, 3)
    assert nf(source, target, features, hood).shape == (0, 3)
    print("empty graph ok")


def _assert_complete(scores, logits, node2inst, atol=1e-4):
    graph_scores = scatter_sum(scores, node2inst, dim=0)
    err = (graph_scores - logits).abs().max().item()
    assert torch.allclose(graph_scores, logits, atol=atol, rtol=1e-4), err
    return err


def _grid_graph(n_side=4, dim=2, in_channels=3, seed=0, device="cuda"):
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
    import torch_geometric.nn as gnn

    edge_index = gnn.knn_graph(coords, k=4, batch=node2inst)
    return (
        node_features.to(device),
        coords.to(device),
        edge_index.to(device),
        node2inst.to(device),
    )


def test_attribution_kernel_point_triton():
    _require_cuda()
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=3, seed=4)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="kernel_point_triton",
        radius=0.5,
        pool="sum",
        pool_to="graph",
    ).cuda()
    model.eval()
    assert isinstance(model.cosmo_layers[0], KernelPointCosmoTriton)
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    err = _assert_complete(scores, logits, batch)
    print(f"KP triton attribution completeness {err:.3e}")


def test_attribution_kernel_point_x_triton():
    _require_cuda()
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=3, seed=6)
    model = CosmoModel(
        in_channels=3,
        out_channels=2,
        hidden_channels=4,
        num_layers=2,
        dim=2,
        filter="kernel_point_x_triton",
        radius=0.5,
        pool="sum",
        pool_to="graph",
    ).cuda()
    model.eval()
    assert isinstance(model.cosmo_layers[0], KernelPointXCosmoTriton)
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    err = _assert_complete(scores, logits, batch)
    print(f"KPX triton attribution completeness {err:.3e}")


def test_attribution_neural_field_triton():
    _require_cuda()
    feats, coords, edges, batch = _grid_graph(dim=2, in_channels=2, seed=5)
    model = CosmoModel(
        in_channels=2,
        out_channels=3,
        hidden_channels=6,
        num_layers=2,
        dim=2,
        filter="neural_field_triton",
        radius=0.5,
        pool="max",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    ).cuda()
    model.eval()
    assert isinstance(model.cosmo_layers[0], NeuralFieldCosmoTriton)
    logits, scores = model(feats, coords, edges, batch, attribution=True)
    err = _assert_complete(scores, logits, batch)
    print(f"NF triton attribution completeness {err:.3e}")


if __name__ == "__main__":
    tests = [
        test_kernel_backward_chunks_match_one_gather,
        test_kernel_point_forward_backward_parity,
        test_kernel_point_x_forward_backward_parity,
        test_kernel_point_bias_silu_parity,
        test_neighbor_gate_triton_parity,
        test_neural_field_forward_backward_parity,
        test_larger_channel_parity,
        test_unequal_message_and_entity_counts,
        test_kernel_point_padded_tile_last_row,
        test_empty_graph,
        test_attribution_kernel_point_triton,
        test_attribution_kernel_point_x_triton,
        test_attribution_neural_field_triton,
    ]
    for fn in tests:
        print(f"--- {fn.__name__}")
        fn()
    print("all triton parity tests passed")
