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
