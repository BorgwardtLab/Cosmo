from types import SimpleNamespace

import torch
from torch_scatter import scatter_max, scatter_mean, scatter_softmax, scatter_sum

from .utilities import *


class Lift2D:
    """
    Parameter-free module to lift a 2D geometric graph. Given node features, global coordinates, and a graph adjacency matrix it computes the lifted adjacency and coordinates of neighborhoods in the local reference frame of the edges, together with some helper variables. These build the input to a Cosmo layer.
    """

    @torch.compiler.disable
    def __call__(
        self,
        node_features,  # Node features of shape (n, in_channels)
        coords,  # Global coordinates of shape (n, 2)
        edge_index,  # Edge index of shape (2, m)
        node2inst,  # Node-to-instance mapping of shape (n,)
    ):
        n = coords.shape[0]
        src, dst = coalesce_edges(edge_index[0], edge_index[1])
        a, b, ij, jk = compute_edge_index(src, dst, n)
        bases = compute_bases(coords, a, b)
        i, j = a[ij], b[ij]
        hood_coords = transform_coords(coords, bases, i, j, jk)
        mask = ~torch.any(torch.isnan(hood_coords), dim=1)
        hood_coords = hood_coords[mask]
        ij, jk = ij[mask], jk[mask]
        i, j = i[mask], j[mask]
        edges = torch.stack([i, j], dim=1)
        edge2node = a
        edge2inst = node2inst[edge2node]
        edge_features = node_features[edge2node]
        return SimpleNamespace(
            source=ij,  # Lifted source edges (m,)
            target=jk,  # Lifted target edges (m,)
            coords=coords,  # Global coordinates of shape (n, 2)
            hood_coords=hood_coords,  # Local coordinates of shape (m, 2)
            features=edge_features,  # Edge features of shape (m, in_channels)
            bases=bases,  # Bases of shape (m, 2, 2)
            i=i,  # Node indices i (m,)
            j=j,  # Node indices j (m,)
            edges=edges,  # Tuples of i,j (m, 2)
            node2inst=node2inst,  # Node-to-instance mapping of shape (n,)
            lifted2node=edge2node,  # Edge-to-node mapping of shape (m,)
            lifted2inst=edge2inst,  # Edge-to-instance mapping of shape (m,)
        )


class Lift3D:
    """
    Parameter-free module to lift a 3D geometric graph. Given node features, global coordinates, and a graph adjacency matrix it computes the lifted adjacency and coordinates of neighborhoods in the local reference frame of the triangles, together with some helper variables. These build the input to a Cosmo layer.
    """

    @torch.compiler.disable
    def __call__(
        self,
        node_features,  # Node features of shape (n, in_channels)
        coords,  # Global coordinates of shape (n, 3)
        edge_index,  # Edge index of shape (2, m)
        node2inst,  # Node-to-instance mapping of shape (n,)
        minimum_angle=0.0,  # Minimum angle to filter nearly colinear triangles (default: 0.0)
    ):
        n = coords.shape[0]
        src, dst = coalesce_edges(edge_index[0], edge_index[1])
        num_edges = int(src.shape[0])
        a, b, ij, jk, ijk, jkl = compute_hyperedge_index(src, dst, n)
        x, y, z = a[ij], b[ij], b[jk]
        triangles = torch.stack([x, y, z], dim=1)
        bases = compute_bases(coords, x, y, z)
        filter_bases(bases, minimum_angle, coords, triangles)
        i, j, k, l = a[ij[ijk]], b[ij[ijk]], a[jk[jkl]], b[jk[jkl]]
        hood_coords = transform_coords(coords, bases, i, j, jkl)
        mask = ~torch.any(torch.isnan(hood_coords), dim=1)
        hood_coords = hood_coords[mask]
        ijk, jkl = ijk[mask], jkl[mask]
        i, j, k, l = i[mask], j[mask], k[mask], l[mask]
        tri2node = a[ij]
        tri2edge = ij
        tri2inst = node2inst[tri2node]
        tri_features = node_features[tri2node]
        return SimpleNamespace(
            source=ijk,  # Lifted source triangles (m,)
            target=jkl,  # Lifted target triangles (m,)
            coords=coords,  # Global coordinates of shape (n, 3)
            hood_coords=hood_coords,  # Local coordinates of shape (m, 3)
            features=tri_features,  # Triangle features of shape (m, in_channels)
            bases=bases,  # Bases of shape (m, 3, 3)
            i=i,  # Node indices i (m,)
            j=j,  # Node indices j (m,)
            triangles=triangles,  # Tuples of i,j,k (m, 3)
            node2inst=node2inst,  # Node-to-instance mapping of shape (n,)
            lifted2node=tri2node,  # Triangle-to-node mapping of shape (m,)
            lifted2edge=tri2edge,  # Triangle-to-edge mapping of shape (m,)
            lifted2inst=tri2inst,  # Triangle-to-instance mapping of shape (m,)
            num_edges=num_edges,  # Coalesced directed edges in the input graph
            edge_src=src,  # Coalesced edge sources (num_edges,)
            edge_dst=dst,  # Coalesced edge targets (num_edges,)
        )


class Lower:
    """
    Parameter-free module to lower a lifted geometric graph back to the input graph. Given edge/triangle features and the corresponding index it aggregates the features to the input graph.
    """

    def __init__(self, agg="mean"):
        assert agg in ["sum", "mean", "max", "softmax"]
        self.agg = agg

    def pool_triangles(self, features, index, size, return_index=False):
        """Aggregate lifted triangle (or 2D edge) features onto an index map."""
        return self(features, index, size, return_index=return_index)

    def pool_to_nodes(self, features, lift, num_nodes):
        return self.pool_triangles(features, lift.lifted2node, num_nodes)

    def pool_to_edges(self, features, lift):
        """Pool onto coalesced input-graph edges (``lift.lifted2edge``)."""
        size = int(getattr(lift, "num_edges", 0))
        if size <= 0:
            width = features.shape[-1] if features.numel() else 0
            return features.new_zeros(0, width)
        return self.pool_triangles(features, lift.lifted2edge, size)

    def pool_to_graph(self, features, lift, num_inst):
        return self.pool_triangles(features, lift.lifted2inst, num_inst)

    def __call__(self, features, index, size, return_index=False):
        if self.agg == "sum":
            return scatter_sum(features, index, dim_size=size, dim=0)
        elif self.agg == "mean":
            return scatter_mean(features, index, dim_size=size, dim=0)
        elif self.agg == "max":
            result = scatter_max(features, index, dim_size=size, dim=0)
            return result[0] if not return_index else result
        elif self.agg == "softmax":
            a = scatter_softmax(features, index, dim_size=size, dim=0)
            return scatter_sum(a * features, index, dim_size=size, dim=0)
