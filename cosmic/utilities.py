import math

import torch
import torch.nn.functional as F


def compute_bases(coords, a, b, c=None):
    """
    Computes transformation matrices for change of basis along an edge or triangle.
    The new basis will be centered at the start-node of the edge or triangle, and rotated such that the end-node is at 0 degrees.
    """
    if c is None:  # 2D
        edge_vectors = coords[b] - coords[a]
        edge_vectors = F.normalize(edge_vectors, dim=1)
        orthogonals = torch.stack((-edge_vectors[:, 1], edge_vectors[:, 0]), dim=1)
        bases = torch.stack((edge_vectors, orthogonals), dim=1)
    else:  # 3D
        v1 = coords[b] - coords[a]
        v2 = coords[c] - coords[a]
        w1 = v1
        w2 = (
            v2
            - torch.sum(w1 * v2, dim=-1).view(-1, 1)
            / torch.sum(w1 * w1, dim=-1).view(-1, 1)
            * w1
        )

        colinear = torch.linalg.norm(w2, dim=1) == 0
        noise = torch.randn_like(v1) * 1e-8
        noise[~colinear] = 0.0
        w2 += noise

        w3 = torch.cross(w1, w2, dim=1)

        w1 = F.normalize(w1, dim=1)
        w2 = F.normalize(w2, dim=1)
        w3 = F.normalize(w3, dim=1)
        bases = torch.stack((w1, w2, w3), dim=1)
    return bases


def compute_edge_index(adj):
    """
    Computes edge pairs for the lifted graph in 2D. Essentially, this function computes the line graph of G given by the adjacency matrix `adj`.
    Nodes are named i->j->k, where j is the center node, i is a neighbor node, i->j is the incoming message edge (source) and j->k is the aggregating query edge (target).
    Neighbors i will later be centered on j and rotated to the j->k edge (using the bases above).
    Return values (a, b) denote the edge index of the original graph, and (ij, jk) denote the edge index of the lifted graph.
    The lifted edge index (ij, jk) describes the i->j->k message passing connections between edges and determine how messages are aggregated in the Cosmo layer.
    """

    a, b = adj.indices()
    n = adj.size()[0]
    mask = a != b  # remove self-loops
    a, b = a[mask], b[mask]
    edge_index = torch.arange(a.shape[0]).to(adj.device)
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([a, b]), size=adj.size(), values=edge_index
    )
    index, degrees = torch.unique_consecutive(a, return_counts=True)
    degrees = torch.zeros(n).type_as(degrees).index_put_((index,), degrees)
    ij = edge_index.repeat_interleave(degrees[b])
    jk = torch.index_select(_adj, 0, b).coalesce().values()
    # mask = a[ij] != b[jk]  # remove backlinks
    # ij, jk = ij[mask], jk[mask]
    return a, b, ij, jk


def compute_hyperedge_index(adj):
    """
    Computes triangle pairs for the lifted graph in 3D. Essentially, this function computes the line graph of the line graph of G given by the adjacency matrix `adj`.
    Nodes are named i->j->k->l, where j is the center node, i is a neighbor node, i->j->k is the incoming message triangle (source) and j->k->l is the aggregating query triangle (target).
    Neighbors i will later be centered on j and rotated to the j->k->l triangle (using the bases above).
    Return values (a, b) denote the edge index of the original graph, and (ij, jk) denote the edge index of the 2D lifted graph, and (ijk, jkl) denote the edge index of the 3D lifted graph.
    The lifted hyperedge index (ijk, jkl) describes the i->j->k->l message passing connections between triangles and determine how messages are aggregated in the Cosmo layer.
    """
    # first part is identical to the 2D case and computes the line graph of G
    a, b = adj.indices()
    n = adj.size()[0]
    mask = a != b  # remove self-loops
    a, b = a[mask], b[mask]
    edge_index = torch.arange(a.shape[0]).to(adj.device)
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([a, b]), size=(n, n), values=edge_index
    )
    index, degrees = torch.unique_consecutive(a, return_counts=True)
    degrees = torch.zeros(n).type_as(degrees).index_put_((index,), degrees)
    ij = edge_index.repeat_interleave(degrees[b])
    jk = torch.index_select(_adj, 0, b).coalesce().values()
    mask = a[ij] != b[jk]  # remove backlinks (= no valid triangle)
    ij, jk = ij[mask], jk[mask]

    # second part follows the same logic and computes the line graph of the line graph of G
    hyperedge_index = torch.arange(ij.shape[0]).to(adj.device)
    n = len(edge_index)
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([ij, jk]), size=(n, n), values=hyperedge_index
    )
    index, degrees = torch.unique_consecutive(ij, return_counts=True)
    degrees = torch.zeros(n).type_as(degrees).index_put_((index,), degrees)
    ijk = hyperedge_index.repeat_interleave(degrees[jk])
    jkl = torch.index_select(_adj, 0, jk).coalesce().values()
    return a, b, ij, jk, ijk, jkl


def transform_coords(coords, bases, i, j, basis_index):
    """
    Applies base transformations to input coordinates to map them to the local reference frame of the query edge or triangle.
    Each neighbor i of center node j will be transformed with base j->k or j->k->l given by `base_index`.
    """
    centered_coords = coords[i] - coords[j]
    hood_coords = torch.bmm(bases[basis_index], centered_coords.unsqueeze(-1)).squeeze(
        -1
    )
    return hood_coords


def filter_bases(bases, minimum_angle, coords, triangles):
    """
    Filters bases (triangles) that are formed by (nearly) colinear triplets of nodes.
    """
    if minimum_angle and minimum_angle > 0:
        x, y, z = triangles.T
        u = coords[y] - coords[x]
        v = coords[z] - coords[x]
        denom = (torch.linalg.norm(u, dim=1) * torch.linalg.norm(v, dim=1)).clamp(
            min=1e-12
        )
        cos_theta = (u * v).sum(dim=1) / denom
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angles_deg = torch.acos(cos_theta) * (180.0 / math.pi)
        valid = angles_deg >= minimum_angle
        bases[~valid] = torch.nan
    return bases


def scatter_sum(src, index, size):
    assert index.dtype == torch.long
    assert index.shape[0] == src.shape[0]
    out_shape = (size,) + src.shape[1:]
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src, index, size):
    sum_out = scatter_sum(src, index, size)
    count = scatter_sum(torch.ones_like(src), index, size)
    return sum_out / count.clamp_min(1)


def scatter_max(src, index, size):
    assert src.is_floating_point() and index.dtype == torch.long
    N = src.size(0)
    idx_exp = index.view(N, *([1] * (src.ndim - 1))).expand_as(src)
    vals = torch.full(
        (size,) + src.shape[1:],
        torch.finfo(src.dtype).min,
        dtype=src.dtype,
        device=src.device,
    )
    vals.scatter_reduce_(0, idx_exp, src, reduce="amax")
    pos = (
        torch.arange(N, device=src.device)
        .view(N, *([1] * (src.ndim - 1)))
        .expand_as(src)
    )
    pos_mask = torch.where(src == vals[index], pos, torch.full_like(pos, N))
    argmax = torch.full_like(vals, N, dtype=torch.long)
    argmax.scatter_reduce_(0, idx_exp, pos_mask, reduce="amin")
    argmax[argmax == N] = -1
    count = torch.zeros_like(vals, dtype=torch.long)
    count.scatter_reduce_(
        0, idx_exp, torch.ones_like(src, dtype=torch.long), reduce="sum"
    )
    vals[count == 0] = 0
    return vals, argmax


def scatter_softmax(src, index, size, eps=1e-12):
    assert src.is_floating_point()
    assert index.dtype == torch.long
    N = src.shape[0]
    idx_exp = index.view(N, *([1] * (src.ndim - 1))).expand_as(src)
    max_vals = torch.full(
        (size,) + src.shape[1:],
        torch.finfo(src.dtype).min,
        dtype=src.dtype,
        device=src.device,
    )
    max_vals.scatter_reduce_(0, idx_exp, src, reduce="amax")
    ex = (src - max_vals[index]).exp()
    sum_per_group = torch.zeros_like(max_vals)
    sum_per_group.scatter_reduce_(0, idx_exp, ex, reduce="sum")
    return ex / sum_per_group[index].clamp_min(eps)
