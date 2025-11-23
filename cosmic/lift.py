from types import SimpleNamespace

import torch

from .utilities import *


class Lift2D:

    @torch.compiler.disable
    def __call__(self, node_features, coords, edge_index, node2inst):
        n = coords.shape[0]
        adj = (
            torch.sparse_coo_tensor(
                indices=edge_index,
                size=(n, n),
                values=torch.ones_like(edge_index[0]),
            )
            .coalesce()
            .to(edge_index.device)
        )
        a, b, ij, jk = compute_edge_index(adj)
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
            adj=adj,
            ij=ij,
            jk=jk,
            coords=coords,
            hood_coords=hood_coords,
            edge_features=edge_features,
            bases=bases,
            i=i,
            j=j,
            edges=edges,
            node2inst=node2inst,
            edge2node=edge2node,
            edge2inst=edge2inst,
        )


class Lift3D:

    @torch.compiler.disable
    def __call__(self, node_features, coords, edge_index, node2inst, minimum_angle=0.0):
        n = coords.shape[0]
        adj = (
            torch.sparse_coo_tensor(
                indices=edge_index, size=(n, n), values=torch.ones_like(edge_index[0])
            )
            .coalesce()
            .to(edge_index.device)
        )
        a, b, ij, jk, ijk, jkl = compute_hyperedge_index(adj)
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
            adj=adj,
            ijk=ijk,
            jkl=jkl,
            coords=coords,
            hood_coords=hood_coords,
            tri_features=tri_features,
            bases=bases,
            i=i,
            j=j,
            triangles=triangles,
            node2inst=node2inst,
            tri2node=tri2node,
            tri2edge=tri2edge,
            tri2inst=tri2inst,
        )
