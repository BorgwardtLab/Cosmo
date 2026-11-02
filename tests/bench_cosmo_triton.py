"""Timing and peak-memory benchmark: reference Cosmo layers vs Triton kernels."""

from __future__ import annotations

import statistics

import torch

from cosmic.cosmo import KernelPointCosmo, KernelPointXCosmo, NeuralFieldCosmo
from cosmic.cosmo_triton import (
    KernelPointCosmoTriton,
    KernelPointXCosmoTriton,
    NeuralFieldCosmoTriton,
)
from cosmic.model import default_kernel_points
from cosmic.triton_ops import TRITON_AVAILABLE


def _sync():
    torch.cuda.synchronize()


def _make_inputs(n_ent, n_msg, cin, dim, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn(n_ent, cin, generator=g)
    hood = torch.randn(n_msg, dim, generator=g)
    source = torch.randint(0, n_ent, (n_msg,), generator=g)
    target = torch.randint(0, n_ent, (n_msg,), generator=g)
    return source.to(device), target.to(device), features.to(device), hood.to(device)


def _clone_inputs(source, target, features, hood, requires_grad):
    feat = features.detach()
    hd = hood.detach()
    if requires_grad:
        feat = feat.requires_grad_(True)
        hd = hd.requires_grad_(True)
    return source, target, feat, hd


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), statistics.mean(samples)


def _peak_mib(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _sync()
    baseline = torch.cuda.memory_allocated()
    fn()
    _sync()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024**2), peak / (1024**2)


def _run_forward(layer, inputs):
    source, target, features, hood = inputs
    return layer(source, target, features, hood)


def _run_fwd_bwd(layer, packed):
    source, target, features, hood = packed
    feat = features.detach().requires_grad_(True)
    hd = hood.detach().requires_grad_(True)
    out = layer(source, target, feat, hd)
    out.square().mean().backward()
    return out


def _fmt(v, digits=3):
    if v is None:
        return "OOM"
    return f"{v:.{digits}f}"


def bench_pair(name, ref, trt, source, target, features, hood, warmup, iters):
    packed = (source, target, features, hood)
    results = {"name": name, "n_ent": features.shape[0], "n_msg": source.shape[0]}

    # Compile / warmup Triton before measuring memory.
    try:
        _run_fwd_bwd(trt, packed)
        _sync()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        results.update(
            ref_fwd=None,
            trt_fwd=None,
            ref_bwd=None,
            trt_bwd=None,
            ref_mem=None,
            trt_mem=None,
        )
        return results

    def measure(layer, key_fwd, key_bwd, key_mem):
        try:
            med_fwd, _ = _time_ms(
                lambda: _run_forward(layer, _clone_inputs(*packed, False)),
                warmup,
                iters,
            )
            med_bwd, _ = _time_ms(lambda: _run_fwd_bwd(layer, packed), warmup, iters)
            delta, _ = _peak_mib(lambda: _run_fwd_bwd(layer, packed))
            results[key_fwd] = med_fwd
            results[key_bwd] = med_bwd
            results[key_mem] = delta
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            results[key_fwd] = None
            results[key_bwd] = None
            results[key_mem] = None

    measure(ref, "ref_fwd", "ref_bwd", "ref_mem")
    measure(trt, "trt_fwd", "trt_bwd", "trt_mem")
    return results


def _print_table(rows):
    headers = [
        "config",
        "n_msg",
        "ref fwd ms",
        "trt fwd ms",
        "speedup",
        "ref bwd ms",
        "trt bwd ms",
        "speedup",
        "ref MiB",
        "trt MiB",
        "mem x",
    ]
    print("  ".join(f"{h:>12}" for h in headers))
    for r in rows:
        sf = (
            r["ref_fwd"] / r["trt_fwd"]
            if r["ref_fwd"] and r["trt_fwd"]
            else None
        )
        sb = (
            r["ref_bwd"] / r["trt_bwd"]
            if r["ref_bwd"] and r["trt_bwd"]
            else None
        )
        sm = (
            r["ref_mem"] / r["trt_mem"]
            if r["ref_mem"] and r["trt_mem"]
            else None
        )
        vals = [
            r["name"],
            str(r["n_msg"]),
            _fmt(r["ref_fwd"]),
            _fmt(r["trt_fwd"]),
            _fmt(sf, 2) if sf else "n/a",
            _fmt(r["ref_bwd"]),
            _fmt(r["trt_bwd"]),
            _fmt(sb, 2) if sb else "n/a",
            _fmt(r["ref_mem"], 1),
            _fmt(r["trt_mem"], 1),
            _fmt(sm, 2) if sm else "n/a",
        ]
        print("  ".join(f"{v:>12}" for v in vals))


def main():
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        raise SystemExit("CUDA + Triton required")
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    warmup, iters = 8, 20
    dim = 3
    rows = []

    kp_cfgs = [
        ("C=32", 4096, 16384, 32, 32),
        ("C=64", 16384, 65536, 64, 64),
        ("C=128", 32768, 131072, 128, 128),
    ]
    kp_kinds = (
        ("KP", KernelPointCosmo, KernelPointCosmoTriton),
        ("KPX", KernelPointXCosmo, KernelPointXCosmoTriton),
    )
    for kind, ref_cls, trt_cls in kp_kinds:
        for name, n_ent, n_msg, cin, cout in kp_cfgs:
            source, target, features, hood = _make_inputs(
                n_ent, n_msg, cin, dim, seed=0, device=device
            )
            mu = default_kernel_points(dim)
            ref = ref_cls(cin, cout, mu).to(device).eval()
            trt = trt_cls(cin, cout, mu).to(device).eval()
            trt.load_state_dict(ref.state_dict())
            rows.append(
                bench_pair(
                    f"{kind} {name}",
                    ref,
                    trt,
                    source,
                    target,
                    features,
                    hood,
                    warmup,
                    iters,
                )
            )
            del ref, trt, source, target, features, hood
            torch.cuda.empty_cache()

    nf_cfgs = [
        ("NF  C=32", 2048, 8192, 32, 32, 32),
        ("NF  C=64", 8192, 32768, 64, 64, 32),
        ("NF C=128", 16384, 65536, 128, 128, 32),
    ]
    for name, n_ent, n_msg, cin, cout, field in nf_cfgs:
        source, target, features, hood = _make_inputs(
            n_ent, n_msg, cin, dim, seed=1, device=device
        )
        ref = NeuralFieldCosmo(
            cin, cout, field_channels=field, field_layers=2, dim=dim
        ).to(device).eval()
        trt = NeuralFieldCosmoTriton(
            cin, cout, field_channels=field, field_layers=2, dim=dim
        ).to(device).eval()
        trt.load_state_dict(ref.state_dict())
        rows.append(
            bench_pair(name, ref, trt, source, target, features, hood, warmup, iters)
        )
        del ref, trt, source, target, features, hood
        torch.cuda.empty_cache()

    try:
        import torch_geometric.nn as gnn

        from cosmic.lift import Lift3D
    except ImportError:
        gnn = None
    if gnn is not None:
        for name, n, k, cin, kind in (
            ("KP lift256", 256, 8, 64, "kp"),
            ("KPX lift256", 256, 8, 64, "kpx"),
            ("NF lift256", 256, 8, 64, "nf"),
            ("KP lift1k", 1024, 12, 64, "kp"),
            ("KPX lift1k", 1024, 12, 64, "kpx"),
            ("NF lift1k", 1024, 12, 64, "nf"),
        ):
            torch.manual_seed(0)
            coords = torch.randn(n, 3, device=device)
            feats = torch.randn(n, cin, device=device)
            batch = torch.zeros(n, dtype=torch.long, device=device)
            edges = gnn.knn_graph(coords, k=k, batch=batch)
            L = Lift3D()(feats, coords, edges, batch)
            mu = default_kernel_points(3)
            if kind == "kp":
                ref = KernelPointCosmo(cin, cin, mu).to(device).eval()
                trt = KernelPointCosmoTriton(cin, cin, mu).to(device).eval()
            elif kind == "kpx":
                ref = KernelPointXCosmo(cin, cin, mu).to(device).eval()
                trt = KernelPointXCosmoTriton(cin, cin, mu).to(device).eval()
            else:
                ref = NeuralFieldCosmo(
                    cin, cin, field_channels=32, field_layers=2, dim=3
                ).to(device).eval()
                trt = NeuralFieldCosmoTriton(
                    cin, cin, field_channels=32, field_layers=2, dim=3
                ).to(device).eval()
            trt.load_state_dict(ref.state_dict())
            rows.append(
                bench_pair(
                    name,
                    ref,
                    trt,
                    L.source,
                    L.target,
                    L.features,
                    L.hood_coords,
                    warmup=4,
                    iters=10,
                )
            )
            del ref, trt, L
            torch.cuda.empty_cache()

    print()
    _print_table(rows)
    return rows


if __name__ == "__main__":
    main()
