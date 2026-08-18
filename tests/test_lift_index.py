"""Parity tests: dense lift indices match the old CPU sparse implementation."""

import torch

from cosmic.utilities import (
    coalesce_edges,
    compute_edge_index,
    compute_hyperedge_index,
    index_select_columns,
)


def _sparse_compute_edge_index(adj):
    a, b = adj.indices()
    n = adj.size()[0]
    mask = a != b
    a, b = a[mask], b[mask]
    edge_index = torch.arange(a.shape[0], device=adj.device)
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([a, b]), size=adj.size(), values=edge_index
    )
    index, degrees = torch.unique_consecutive(a, return_counts=True)
    degrees = torch.zeros(n, device=adj.device, dtype=degrees.dtype).index_put_(
        (index,), degrees
    )
    ij = edge_index.repeat_interleave(degrees[b])
    jk = torch.index_select(_adj, 0, b).coalesce().values()
    return a, b, ij, jk


def _sparse_compute_hyperedge_index(adj):
    a, b, ij, jk = _sparse_compute_edge_index(adj)
    mask = a[ij] != b[jk]
    ij, jk = ij[mask], jk[mask]
    hyperedge_index = torch.arange(ij.shape[0], device=adj.device)
    n = a.shape[0]
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([ij, jk]), size=(n, n), values=hyperedge_index
    )
    index, degrees = torch.unique_consecutive(ij, return_counts=True)
    degrees = torch.zeros(n, device=adj.device, dtype=degrees.dtype).index_put_(
        (index,), degrees
    )
    ijk = hyperedge_index.repeat_interleave(degrees[jk])
    jkl = torch.index_select(_adj, 0, jk).coalesce().values()
    return a, b, ij, jk, ijk, jkl


def _random_graph(n=12, e=40, seed=0):
    torch.manual_seed(seed)
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    src, dst = coalesce_edges(src, dst)
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([src, dst]),
        values=torch.ones(src.shape[0], dtype=torch.int64),
        size=(n, n),
    ).coalesce()
    return src, dst, n, adj


def test_compute_edge_index_matches_sparse():
    src, dst, n, adj = _random_graph()
    got = compute_edge_index(src, dst, n)
    ref = _sparse_compute_edge_index(adj)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_compute_hyperedge_index_matches_sparse():
    src, dst, n, adj = _random_graph(n=8, e=24, seed=1)
    got = compute_hyperedge_index(src, dst, n)
    ref = _sparse_compute_hyperedge_index(adj)
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


def test_index_select_columns_matches_sparse():
    torch.manual_seed(2)
    m = 10
    e = 25
    row = torch.randint(0, m, (e,))
    col = torch.randint(0, m, (e,))
    row, col = coalesce_edges(row, col)
    values = torch.arange(row.shape[0])
    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]), values=values, size=(m, m)
    ).coalesce()
    query = torch.tensor([3, 3, 7, 1, 3])
    selected = torch.index_select(adj, 1, query).coalesce()
    ref_src, ref_dst = selected.indices()
    src, dst = index_select_columns(row, col, query, m)
    assert torch.equal(src, ref_src)
    assert torch.equal(dst, ref_dst)


def test_empty_graph():
    src = torch.zeros(0, dtype=torch.long)
    dst = torch.zeros(0, dtype=torch.long)
    a, b, ij, jk = compute_edge_index(src, dst, 5)
    assert a.numel() == 0 and ij.numel() == 0 and jk.numel() == 0
    src, dst = index_select_columns(src, dst, torch.tensor([0, 1]), 5)
    assert src.numel() == 0


def test_gpu_matches_cpu_and_exits_cleanly():
    if not torch.cuda.is_available():
        return
    src, dst, n, adj = _random_graph(n=10, e=30, seed=4)
    cpu = compute_hyperedge_index(src, dst, n)
    gpu = compute_hyperedge_index(src.cuda(), dst.cuda(), n)
    for a, b in zip(cpu, gpu):
        assert torch.equal(a, b.cpu())

    from cosmic import CosmoModel
    import torch_geometric.nn as gnn

    torch.manual_seed(0)
    coords = torch.rand(16, 3, device="cuda")
    feats = torch.randn(16, 2, device="cuda")
    batch = torch.zeros(16, dtype=torch.long, device="cuda")
    edges = gnn.knn_graph(coords, k=4, batch=batch)
    model = CosmoModel(
        in_channels=2,
        out_channels=1,
        hidden_channels=8,
        num_layers=2,
        dim=3,
        radius=0.5,
        pool="mean",
        pool_to="graph",
        field_channels=8,
        field_layers=2,
    ).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        logits = model(feats, coords, edges, batch)
        logits.square().mean().backward()
        opt.step()
    logits, scores = model.attribute(feats, coords, edges, batch)
    assert logits.shape == (1, 1)
    assert scores.shape == (16, 1)


if __name__ == "__main__":
    test_compute_edge_index_matches_sparse()
    test_compute_hyperedge_index_matches_sparse()
    test_index_select_columns_matches_sparse()
    test_empty_graph()
    test_gpu_matches_cpu_and_exits_cleanly()
    print("lift index tests passed")
