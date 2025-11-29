import math

import torch
import torch.nn.functional as F


def compute_bases(coords, a, b, c=None):
    """
    Computes transformation matrices for change of basis along an edge.
    The new basis will be centered at the start-node of the edge, and rotated such that the end-node is at 0 degrees.
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
    Computes edge tuples for computing on the group.
    Nodes are named i->j->k, where j is the center node, i is a neighbor node, and k is the querying node.
    Neighbors i will later be centered around j and rotated to the j->k edge (using the bases above).
    ij and jk are indices of edges in the graph (a,b).
    ij indexes all edges that end in j (b==j), for all j.
    jk enumerates all edges j->k, repeated for all neighbors i that connect to j (b==j).
    ij,jk returns a flattened list of all possible i->j->k edges.
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
    hyperedge_index = torch.arange(ij.shape[0]).to(adj.device)
    n = len(edge_index)
    _adj = torch.sparse_coo_tensor(
        indices=torch.stack([ij, jk]), size=(n, n), values=hyperedge_index
    )
    # degrees = torch.unique(ij, return_counts=True)[1]
    index, degrees = torch.unique_consecutive(ij, return_counts=True)
    degrees = torch.zeros(n).type_as(degrees).index_put_((index,), degrees)
    ijk = hyperedge_index.repeat_interleave(degrees[jk])
    jkl = torch.index_select(_adj, 0, jk).coalesce().values()
    # mask = a[ij[ijk]] != b[jk[ijk]]  # remove backlinks
    # ijk, jkl = ijk[mask], jkl[mask]
    return a, b, ij, jk, ijk, jkl


def transform_coords(coords, bases, i, j, basis_index):
    """
    Applies the according base transformation to the coordinates.
    Each neighbor i of center node j will be transformed with base j->k->l.
    """
    centered_coords = coords[i] - coords[j]
    hood_coords = torch.bmm(bases[basis_index], centered_coords.unsqueeze(-1)).squeeze(
        -1
    )
    return hood_coords


def filter_bases(bases, minimum_angle, coords, triangles):
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


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    assert index.dtype == torch.long
    dim = dim if dim >= 0 else src.dim() + dim
    assert index.shape == src.shape

    if out is None:
        out_shape = list(src.shape)
        if dim_size is not None:
            out_shape[dim] = dim_size
        else:
            assert index.numel() > 0 or src.numel() == 0
            out_shape[dim] = 0 if index.numel() == 0 else int(index.max().item()) + 1
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    if index.numel() > 0:
        assert int(index.min().item()) >= 0
        assert int(index.max().item()) < out.size(dim)
    return out.scatter_add(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    sum_out = scatter_sum(src, index, dim=dim, out=out, dim_size=dim_size)

    # Count of values per index for division
    count_dtype = sum_out.dtype if sum_out.is_floating_point() else torch.float32
    ones = torch.ones_like(src, dtype=count_dtype, device=src.device)
    count_out = scatter_sum(ones, index, dim=dim, out=None, dim_size=sum_out.size(dim))

    denom = count_out.clamp_min(1)
    sum_out = sum_out / denom
    return sum_out


def scatter_softmax(src, index, dim=-1, eps=1e-12):
    assert src.is_floating_point()
    assert index.dtype == torch.long
    dim = dim if dim >= 0 else src.dim() + dim
    assert index.shape == src.shape

    # Determine reduced (group) shape
    out_size_dim = 0 if index.numel() == 0 else int(index.max().item()) + 1
    out_shape = list(src.shape)
    out_shape[dim] = out_size_dim

    if out_size_dim == 0:
        return torch.zeros_like(src)

    # Compute per-group max for numerical stability (simple loop)
    finfo = torch.finfo(src.dtype)
    neg_inf = finfo.min
    group_max = torch.full(out_shape, neg_inf, dtype=src.dtype, device=src.device)
    assert hasattr(torch.Tensor, "scatter_reduce_")
    group_max.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)

    # Subtract gathered group max
    max_gather = group_max.gather(dim, index)
    shifted = src - max_gather

    exp_shifted = torch.exp(shifted)

    # Sum of exponentials per group
    sum_exp = torch.zeros(out_shape, dtype=src.dtype, device=src.device).scatter_add(
        dim, index, exp_shifted
    )
    denom = sum_exp.gather(dim, index).clamp_min(eps)

    out = exp_shifted / denom
    return out
