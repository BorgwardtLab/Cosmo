"""Experimental line-graph lift with per-residue backbone LRFs (optional Cosmo path).

Kept separate from ``lift.py`` so it can be dropped if the experiment fails.
Line-graph topology matches ``Lift2D`` (messages between directed kNN edges).
Entity features are PiFold-style edge vectors (e.g. RBF); ``hood_coords`` are
3D offsets of the incoming neighbor in the query center residue's N–Cα–C frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .utilities import coalesce_edges, compute_edge_index


def compute_residue_backbone_bases(backbone):
    """Right-handed orthonormal frames from N, Cα, C, O stored as ``(n, 4, 3)``.

    Atom order matches likelihood-carving: index 0=N, 1=Cα, 2=C, 3=O.
    Origin is Cα; x̂ along Cα→C, ẑ = x̂ × (Cα→N) normalized, ŷ = ẑ × x̂.
    Returns ``(n, 3, 3)`` row bases (same layout as ``compute_bases`` in 3D).
    """
    n = int(backbone.shape[0])
    device, dtype = backbone.device, backbone.dtype
    n_vec = backbone[:, 0] - backbone[:, 1]
    c_vec = backbone[:, 2] - backbone[:, 1]
    x = F.normalize(c_vec, dim=-1)
    z = torch.cross(x, n_vec, dim=-1)
    z_norm = torch.linalg.norm(z, dim=-1)
    bad = z_norm < 1e-6
    if bad.any():
        fallback = torch.zeros_like(z)
        fallback[:, 2] = 1.0
        z = torch.where(bad.unsqueeze(-1), fallback, z)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=1)


class LiftBackboneLineGraph:
    """Line graph on directed edges with backbone-defined 3D hood coordinates."""

    @torch.compiler.disable
    def __call__(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        minimum_angle=0.0,
        edge_geom=None,
        backbone=None,
    ):
        del node_features, minimum_angle  # entity features come from edge_geom
        if edge_geom is None:
            raise ValueError(
                "LiftBackboneLineGraph requires edge_geom (one row per coalesced edge)."
            )
        if backbone is None:
            raise ValueError(
                "LiftBackboneLineGraph requires backbone of shape (n, 4, 3)."
            )
        n = int(coords.shape[0])
        if int(backbone.shape[0]) != n:
            raise ValueError(
                f"backbone residues {backbone.shape[0]} != coords nodes {n}"
            )
        src, dst = coalesce_edges(edge_index[0], edge_index[1])
        if edge_geom.shape[0] != int(src.shape[0]):
            raise ValueError(
                f"edge_geom rows {edge_geom.shape[0]} != coalesced edges {src.shape[0]}"
            )
        # Line-graph entities match ``compute_edge_index`` (no self-loops), like Lift2D.
        loop_mask = src != dst
        if not bool(loop_mask.all()):
            src, dst = src[loop_mask], dst[loop_mask]
            edge_geom = edge_geom[loop_mask]
        a, b, ij, jk = compute_edge_index(src, dst, n)
        num_edges = int(a.shape[0])
        if edge_geom.shape[0] != num_edges:
            raise ValueError(
                f"edge_geom rows {edge_geom.shape[0]} != line-graph edges {num_edges}"
            )
        residue_bases = compute_residue_backbone_bases(backbone)

        i_node = a[ij]
        j_node = b[ij]
        centered = coords[i_node] - coords[j_node]
        basis = residue_bases.index_select(0, j_node)
        hood_coords = torch.einsum("nij,nj->ni", basis, centered)

        mask = torch.isfinite(hood_coords).all(dim=1)
        hood_coords = hood_coords[mask]
        ij, jk = ij[mask], jk[mask]

        edge_inst = node2inst.index_select(0, b)
        return SimpleNamespace(
            source=ij,
            target=jk,
            coords=coords,
            hood_coords=hood_coords,
            features=edge_geom,
            bases=residue_bases,
            node2inst=node2inst,
            lifted2node=b,  # dst residue per directed edge (readout target)
            lifted2edge=torch.arange(num_edges, device=coords.device, dtype=torch.long),
            lifted2inst=edge_inst,
            num_edges=num_edges,
            edge_src=src,
            edge_dst=dst,
        )
