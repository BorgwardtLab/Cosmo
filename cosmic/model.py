import torch
from torch import nn
from torch_scatter import scatter_max, scatter_softmax, scatter_sum

from .cosmo import (
    KernelPointCosmo,
    KernelPointXCosmo,
    MessageMLPCosmo,
    NeuralFieldCosmo,
    PointTransformerCosmo,
    scale_hood_coords,
)
from .cosmo_triton import (
    KernelPointCosmoTriton,
    KernelPointXCosmoTriton,
    NeuralFieldCosmoTriton,
)
from .lift import Lift2D, Lift3D, Lower
from .utilities import index_select_columns

# Attribution walks messages backward. Gathered filter matrices are (E, C, C);
# start from a slice of lifted ids, then coalesce each hop by source id so the
# expanded path tensor is never stored.
ATTRIBUTION_CHUNK = 65536
ATTRIBUTION_START_CHUNK = 1024


def _iter_chunks(n, size):
    size = max(int(size), 1)
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _scatter_add_rows(dest, row, values):
    dest.scatter_add_(0, row.view(-1, 1, 1).expand_as(values), values)
    return dest


def _coalesce_fields(edges, fields):
    """Sum field matrices that share a lifted-entity id. ``edges`` need not be sorted."""
    if edges.numel() <= 1:
        return edges, fields
    if bool(torch.all(edges[1:] > edges[:-1])):
        return edges, fields
    order = torch.argsort(edges, stable=True)
    edges = edges[order]
    fields = fields[order]
    unique, inv = torch.unique_consecutive(edges, return_inverse=True)
    if unique.shape[0] == edges.shape[0]:
        return unique, fields
    coalesced = fields.new_zeros(unique.shape[0], fields.shape[1], fields.shape[2])
    return unique, _scatter_add_rows(coalesced, inv, fields)


FILTERS = {
    "neural_field": NeuralFieldCosmo,
    "neuralfieldcosmo": NeuralFieldCosmo,
    "neural_field_triton": NeuralFieldCosmoTriton,
    "neuralfieldcosmotriton": NeuralFieldCosmoTriton,
    "kernel_point": KernelPointCosmo,
    "kernelpointcosmo": KernelPointCosmo,
    "kpconv": KernelPointCosmo,
    "kernel_point_triton": KernelPointCosmoTriton,
    "kernelpointcosmotriton": KernelPointCosmoTriton,
    "kpconv_triton": KernelPointCosmoTriton,
    "kernel_point_x": KernelPointXCosmo,
    "kernelpointxcosmo": KernelPointXCosmo,
    "kpconvx": KernelPointXCosmo,
    "kernel_point_x_triton": KernelPointXCosmoTriton,
    "kernelpointxcosmotriton": KernelPointXCosmoTriton,
    "kpconvx_triton": KernelPointXCosmoTriton,
    "point_transformer": PointTransformerCosmo,
    "pointtransformercosmo": PointTransformerCosmo,
    "message_mlp": MessageMLPCosmo,
    "messagemlpcosmo": MessageMLPCosmo,
}


def default_kernel_points(dim, radius=1.0):
    """Kernel points at the origin and along the coordinate axes."""
    axes = torch.eye(dim)
    pts = torch.cat([torch.zeros(1, dim), axes, -axes], dim=0)
    return pts * radius


def _as_filter_name(spec):
    if isinstance(spec, str):
        return spec.lower().replace(" ", "_")
    if isinstance(spec, type):
        return spec.__name__.lower()
    return type(spec).__name__.lower()


def _resolve_filter_class(spec):
    if isinstance(spec, nn.Module):
        return type(spec)
    if isinstance(spec, type):
        return spec
    name = _as_filter_name(spec)
    if name not in FILTERS:
        raise ValueError(
            f"Unknown Cosmo filter {spec!r}. Expected one of {sorted(set(FILTERS))}."
        )
    return FILTERS[name]


def _resolve_activation(activation):
    if activation is None or activation is nn.Identity:
        return None
    if isinstance(activation, nn.Identity):
        return None
    if isinstance(activation, str):
        key = activation.lower()
        if key in ("none", "identity"):
            return None
        table = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "leakyrelu": nn.LeakyReLU,
            "silu": nn.SiLU,
        }
        if key not in table:
            raise ValueError(f"Unknown activation {activation!r}.")
        return table[key]()
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()
    if isinstance(activation, nn.Module):
        return activation
    raise TypeError(f"Cannot interpret activation {activation!r}.")


def _channel_pairs(in_channels, hidden_channels, out_channels, num_layers, has_head):
    if isinstance(hidden_channels, (list, tuple)):
        outs = list(hidden_channels)
        if len(outs) != num_layers:
            raise ValueError(
                "hidden_channels must have length num_layers when given as a sequence."
            )
        if not has_head and outs[-1] != out_channels:
            raise ValueError(
                "Last hidden size must equal out_channels when head is identity."
            )
    else:
        if has_head:
            outs = [hidden_channels] * num_layers
        else:
            outs = [hidden_channels] * (num_layers - 1) + [out_channels]
    ins = [in_channels] + outs[:-1]
    return list(zip(ins, outs))


class CosmoModel(nn.Module):
    """
    Multi-layer Cosmo network with configurable filters, pooling, and readout.

    Cosmo layers are linear in the node features when no feature-space activation
    is used (NeuralFieldCosmo / KernelPointCosmo / KernelPointXCosmo). In that case a prediction can
    be decomposed into per-node, per-class scores that sum to the output logits
    by composing filter weight matrices across layers in aligned local frames
    (Kucera et al. 2026). Attribution materializes filter gathers in chunks of
    ``attribution_chunk_size`` / ``attribution_start_chunk`` so completeness
    stays exact at lower peak VRAM.
    """

    attribution_chunk_size = ATTRIBUTION_CHUNK
    attribution_start_chunk = ATTRIBUTION_START_CHUNK

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=32,
        num_layers=3,
        dim=3,
        filter="neural_field",
        radius=1.0,
        activation=None,
        pool="max",
        pool_to="graph",
        head=None,
        kernel_points=None,
        kernel_sigma=None,
        kernel_bias=False,
        kernel_silu=False,
        kernel_norm=False,
        kernel_residual=False,
        gate=False,
        message_attention=False,
        attention_pool=False,
        pool_edges=False,
        field_channels=32,
        field_layers=3,
        field_dropout=0.0,
        field_activation=nn.Tanh,
        minimum_angle=0.0,
        edge_geom_dim=0,
        edge_geom_lift=False,
        triangle_ffn=False,
        message_mlp_hidden=128,
        lift_mode="default",
    ):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        if pool not in ("sum", "mean", "max", "softmax"):
            raise ValueError("pool must be one of 'sum', 'mean', 'max', 'softmax'.")
        if pool_to not in ("node", "graph", "edge"):
            raise ValueError("pool_to must be 'node', 'edge', or 'graph'.")
        if pool_edges and pool_to == "graph":
            raise ValueError("pool_edges cannot be combined with pool_to='graph'.")
        if pool_edges and pool_to == "edge":
            raise ValueError("Use pool_to='edge' alone, not with pool_edges.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dim = dim
        self.radius = radius
        self.pool = pool
        self.pool_to = pool_to
        self.minimum_angle = minimum_angle
        self.kernel_sigma = kernel_sigma
        self.kernel_bias = bool(kernel_bias)
        self.kernel_silu = bool(kernel_silu)
        self.kernel_norm = bool(kernel_norm)
        self.kernel_residual = bool(kernel_residual)
        self.gate = bool(gate)
        self.message_attention = bool(message_attention)
        self.attention_pool = bool(attention_pool)
        self.pool_edges = bool(pool_edges)
        self.edge_geom_dim = int(edge_geom_dim or 0)
        self.edge_geom_lift = bool(edge_geom_lift)
        self.triangle_ffn = bool(triangle_ffn)
        self.message_mlp_hidden = int(message_mlp_hidden)
        self.lift_mode = str(lift_mode or "default").lower()
        self.register_buffer("_hood_radius", torch.tensor(float(radius)))

        if self.lift_mode == "backbone_line":
            from .lift_backbone_line import LiftBackboneLineGraph

            if dim != 3:
                raise ValueError("lift_mode='backbone_line' requires dim=3.")
            self.lift = LiftBackboneLineGraph()
        elif dim == 2:
            self.lift = Lift2D()
        else:
            self.lift = Lift3D()
        self.lower = Lower(agg=pool)

        has_head = head is not None and head != "identity" and head is not nn.Identity
        pairs = _channel_pairs(
            in_channels, hidden_channels, out_channels, num_layers, has_head
        )
        filter_specs = (
            list(filter) if isinstance(filter, (list, tuple)) else [filter] * num_layers
        )
        if len(filter_specs) != num_layers:
            raise ValueError("filter sequence must have length num_layers.")

        if kernel_points is None:
            kernel_points = default_kernel_points(dim, radius=radius)
        self.register_buffer("kernel_points", kernel_points.float())

        layers = []
        for spec, (cin, cout) in zip(filter_specs, pairs):
            if isinstance(spec, nn.Module):
                layers.append(spec)
                continue
            cls = _resolve_filter_class(spec)
            if issubclass(cls, NeuralFieldCosmo):
                layers.append(
                    cls(
                        in_channels=cin,
                        out_channels=cout,
                        field_channels=field_channels,
                        field_layers=field_layers,
                        field_dropout=field_dropout,
                        field_activation=field_activation,
                        radius=radius,
                        dim=dim,
                        gate=self.gate,
                        message_attention=self.message_attention,
                    )
                )
            elif issubclass(cls, KernelPointXCosmo):
                layers.append(
                    cls(
                        in_channels=cin,
                        out_channels=cout,
                        kernel_points=self.kernel_points,
                        bias=self.kernel_bias,
                        silu=self.kernel_silu,
                        norm=self.kernel_norm,
                        residual=self.kernel_residual,
                        gate=self.gate,
                        message_attention=self.message_attention,
                        edge_geom_dim=self.edge_geom_dim,
                    )
                )
            elif issubclass(cls, KernelPointCosmo):
                layers.append(
                    cls(
                        in_channels=cin,
                        out_channels=cout,
                        kernel_points=self.kernel_points,
                        sigma=self.kernel_sigma,
                        bias=self.kernel_bias,
                        silu=self.kernel_silu,
                        norm=self.kernel_norm,
                        residual=self.kernel_residual,
                        gate=self.gate,
                        message_attention=self.message_attention,
                        edge_geom_dim=self.edge_geom_dim,
                    )
                )
            elif issubclass(cls, PointTransformerCosmo):
                layers.append(
                    cls(
                        in_channels=cin,
                        out_channels=cout,
                        radius=radius,
                        dim=dim,
                        gate=self.gate,
                        message_attention=self.message_attention,
                    )
                )
            elif issubclass(cls, MessageMLPCosmo):
                layers.append(
                    cls(
                        in_channels=cin,
                        out_channels=cout,
                        dim=dim,
                        radius=radius,
                        mlp_hidden=self.message_mlp_hidden,
                        gate=self.gate,
                        edge_geom_dim=self.edge_geom_dim,
                    )
                )
            else:
                raise TypeError(f"Unsupported Cosmo filter class {cls}.")
        self.cosmo_layers = nn.ModuleList(layers)

        self.triangle_ffn_layers = None
        if self.triangle_ffn:
            self.triangle_ffn_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(cout), max(int(cout) * 2, 16)),
                        nn.SiLU(),
                        nn.Linear(max(int(cout) * 2, 16), int(cout)),
                    )
                    for _cin, cout in pairs
                ]
            )

        if isinstance(activation, (list, tuple)):
            if len(activation) != num_layers - 1:
                raise ValueError(
                    "activation sequence must have length num_layers - 1 "
                    "(applied between Cosmo layers)."
                )
            acts = [_resolve_activation(a) or nn.Identity() for a in activation]
        else:
            act = _resolve_activation(activation)
            acts = [
                (act if act is not None else nn.Identity())
                for _ in range(max(num_layers - 1, 0))
            ]
        self.activations = nn.ModuleList(acts)

        last_dim = pairs[-1][1]
        if head is None or head == "identity" or head is nn.Identity:
            self.head = nn.Identity()
            if last_dim != out_channels:
                raise ValueError(
                    "Identity head requires the last Cosmo layer to output "
                    f"out_channels={out_channels}, got {last_dim}."
                )
        elif head == "linear":
            self.head = nn.Linear(last_dim, out_channels)
        elif isinstance(head, nn.Module):
            self.head = head
        else:
            raise TypeError(
                "head must be None, 'identity', 'linear', or an nn.Module."
            )
        if self.attention_pool:
            self.pool_attn = nn.Linear(last_dim, 1)
            nn.init.zeros_(self.pool_attn.weight)
            nn.init.zeros_(self.pool_attn.bias)
        else:
            self.pool_attn = None

    def forward(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        attribution=False,
        edge_geom=None,
        backbone=None,
    ):
        """
        Args:
            node_features: (n, in_channels)
            coords: (n, dim)
            edge_index: (2, e)
            node2inst: (n,) graph/instance id for each node
            attribution: if True, also return per-node scores (n, out_channels)
                that sum to the graph-level logits.

        Returns:
            logits, or (logits, scores) when attribution=True.
            logits have shape (num_instances, out_channels) if pool_to='graph',
            else (n, out_channels).
        """
        if attribution:
            was_training = self.training
            self.eval()
            try:
                return self._forward(
                    node_features,
                    coords,
                    edge_index,
                    node2inst,
                    attribution=True,
                    edge_geom=edge_geom,
                    backbone=backbone,
                )
            finally:
                self.train(was_training)
        return self._forward(
            node_features,
            coords,
            edge_index,
            node2inst,
            attribution=False,
            edge_geom=edge_geom,
            backbone=backbone,
        )

    def attribute(self, node_features, coords, edge_index, node2inst):
        """Predict and return (logits, scores) with completeness attribution."""
        return self.forward(
            node_features, coords, edge_index, node2inst, attribution=True
        )

    def forward_lifted(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        edge_geom=None,
        backbone=None,
    ):
        """Triangle (lifted) features before Lower pooling, with ``lifted2node`` index."""
        num_nodes = node_features.shape[0]
        L = self._lift(
            node_features,
            coords,
            edge_index,
            node2inst,
            edge_geom=edge_geom,
            backbone=backbone,
        )
        geom = None if self.lift_mode == "backbone_line" else edge_geom
        lifted = self._cosmo_features(L, edge_geom=geom)
        return lifted, L.lifted2node, num_nodes

    def _lift(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        edge_geom=None,
        backbone=None,
    ):
        if self.lift_mode == "backbone_line":
            return self.lift(
                node_features,
                coords,
                edge_index,
                node2inst,
                minimum_angle=self.minimum_angle,
                edge_geom=edge_geom,
                backbone=backbone,
            )
        if self.dim == 3:
            return self.lift(
                node_features,
                coords,
                edge_index,
                node2inst,
                minimum_angle=self.minimum_angle,
            )
        return self.lift(node_features, coords, edge_index, node2inst)

    def _triangle_edge_geom(self, L, edge_geom):
        if edge_geom is None:
            return None
        if not self.edge_geom_lift and self.edge_geom_dim <= 0:
            return None
        if edge_geom.shape[0] == 0:
            return edge_geom.new_zeros(L.features.shape[0], edge_geom.shape[-1])
        return edge_geom.index_select(0, L.lifted2edge)

    def _hood_for_layer(self, layer, hood_coords):
        """KP / neural-field filters use Å offsets; MessageMLP uses ``hood / radius``."""
        if isinstance(layer, MessageMLPCosmo):
            return scale_hood_coords(hood_coords, self._hood_radius)
        return hood_coords

    def _cosmo_features(self, L, edge_geom=None):
        tri_geom = self._triangle_edge_geom(L, edge_geom)
        features = L.features
        if self.edge_geom_lift and tri_geom is not None:
            features = torch.cat(
                [features, tri_geom.to(dtype=features.dtype, device=features.device)],
                dim=-1,
            )
            tri_geom = None
        hood_raw = L.hood_coords
        for i, layer in enumerate(self.cosmo_layers):
            hood = self._hood_for_layer(layer, hood_raw)
            if getattr(layer, "edge_geom_proj", None) is not None or (
                isinstance(layer, MessageMLPCosmo) and layer.edge_geom_dim > 0
            ):
                features = layer(
                    L.source,
                    L.target,
                    features,
                    hood,
                    tri_edge_geom=tri_geom,
                )
            else:
                features = layer(L.source, L.target, features, hood)
            if self.triangle_ffn_layers is not None:
                features = features + self.triangle_ffn_layers[i](features)
            if i < len(self.cosmo_layers) - 1:
                features = self.activations[i](features)
        return features

    def _pool_index(self, L, num_nodes, num_inst):
        if self.pool_to == "node":
            return L.lifted2node, num_nodes
        if self.pool_to == "edge":
            return L.lifted2edge, int(getattr(L, "num_edges", 0))
        return L.lifted2inst, num_inst

    def _pool_lifted(self, lifted, index, size):
        if self.attention_pool:
            return self._attention_pool(lifted, index, size)
        return self.lower.pool_triangles(lifted, index, size)

    def _attention_pool(self, features, index, size):
        """Softmax over the lifted entities that belong to each node or graph."""
        width = features.shape[-1]
        if features.shape[0] == 0 or size == 0:
            return features.new_zeros(size, width)
        score = self.pool_attn(features).squeeze(-1)
        alpha = scatter_softmax(score, index, dim_size=size)
        return scatter_sum(alpha.unsqueeze(-1) * features, index, dim_size=size, dim=0)

    def _forward(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        attribution,
        edge_geom=None,
        backbone=None,
    ):
        if attribution and (
            self.pool_edges
            or edge_geom is not None
            or self.lift_mode == "backbone_line"
        ):
            raise TypeError(
                "Exact attribution does not support pool_edges or edge_geom."
            )
        num_nodes = node_features.shape[0]
        num_inst = int(node2inst.max().item()) + 1 if num_nodes > 0 else 0
        L = self._lift(
            node_features,
            coords,
            edge_index,
            node2inst,
            edge_geom=edge_geom,
            backbone=backbone,
        )
        geom = None if self.lift_mode == "backbone_line" else edge_geom
        lifted = self._cosmo_features(L, edge_geom=geom)
        index, size = self._pool_index(L, num_nodes, num_inst)
        pooled = self._pool_lifted(lifted, index, size)
        if self.pool_edges:
            edge_pooled = self.lower.pool_to_edges(lifted, L)
            logits = self.head(pooled)
            edge_logits = self.head(edge_pooled)
            if not attribution:
                return logits, edge_logits
        else:
            logits = self.head(pooled)
            if not attribution:
                return logits
        scores = self._attribute(
            L, node_features, lifted, node2inst, num_nodes, num_inst
        )
        return logits, scores

    def _linear_readout(self, hidden_dim, device, dtype):
        if isinstance(self.head, nn.Identity):
            W = torch.eye(self.out_channels, hidden_dim, device=device, dtype=dtype)
            b = torch.zeros(self.out_channels, device=device, dtype=dtype)
            return W, b
        if isinstance(self.head, nn.Linear):
            W = self.head.weight
            if self.head.bias is None:
                b = torch.zeros(self.out_channels, device=device, dtype=dtype)
            else:
                b = self.head.bias
            return W, b
        raise TypeError(
            "Exact attribution requires an identity or linear readout head."
        )

    def _assert_attributable(self):
        for act in self.activations:
            if not isinstance(act, nn.Identity):
                raise TypeError(
                    "Exact attribution requires identity activations between Cosmo "
                    "layers (the feature transform must stay linear)."
                )
        for layer in self.cosmo_layers:
            if isinstance(layer, PointTransformerCosmo):
                raise TypeError(
                    "Exact linear attribution is not defined for PointTransformerCosmo."
                )
            if not isinstance(layer, (NeuralFieldCosmo, KernelPointCosmo, KernelPointXCosmo)):
                raise TypeError(
                    f"Exact attribution is not implemented for {type(layer).__name__}."
                )
            if getattr(layer, "gate", False):
                raise TypeError(
                    "Exact attribution requires layers without a neighbor gate "
                    "(the feature transform must stay linear)."
                )
            if isinstance(layer, (KernelPointCosmo, KernelPointXCosmo)) and (
                layer.bias is not None
                or layer.silu
                or layer.norm is not None
                or layer.residual
                or layer.message_attention
            ):
                raise TypeError(
                    "Exact attribution requires kernel-point layers without bias, "
                    "SiLU, LayerNorm, a residual, or message attention "
                    "(the feature transform must stay linear)."
                )
        if self.attention_pool:
            raise TypeError(
                "Exact attribution requires the parameter-free triangle pool."
            )
        if not isinstance(self.head, (nn.Identity, nn.Linear)):
            raise TypeError(
                "Exact attribution requires an identity or linear readout head."
            )

    def _filter_weights(self, layer, hood_coords):
        if isinstance(layer, NeuralFieldCosmo):
            w = layer.neural_field(hood_coords / layer.radius).view(
                -1, int(layer.out_channels), int(layer.in_channels)
            )
            return w, "mean"
        if isinstance(layer, KernelPointXCosmo):
            nn_idx = torch.cdist(hood_coords, layer.mu).argmin(dim=1)
            w = layer.w[:, nn_idx].permute(1, 0, 2)
            return w, "sum"
        if isinstance(layer, KernelPointCosmo):
            h = layer.influences(hood_coords)
            w = torch.einsum("mk,oki->moi", h, layer.w)
            return w, "sum"
        raise TypeError(f"No filter weights for {type(layer).__name__}.")

    def _initial_fields(self, L, lifted, W, num_nodes, num_inst):
        """
        Fields of shape (s, out_channels, last_hidden) on `s` starting lifted
        entities, such that composing them back to the input reconstructs
        `W @ pooled_hidden` (without bias).
        """
        edges, payload, mode, W = self._prepare_starts(
            L, lifted, W, num_nodes, num_inst
        )
        return self._start_field_chunk(edges, payload, mode, W, 0, edges.shape[0])

    def _prepare_starts(self, L, lifted, W, num_nodes, num_inst):
        """
        Compact readout-side starting state. Heavy (s, C, C) tensors are built
        per start-chunk, not for the full lift.
        """
        m, hidden = lifted.shape
        device, dtype = lifted.device, lifted.dtype
        index, size = self._pool_index(L, num_nodes, num_inst)
        W = W.to(dtype)

        if self.pool == "sum":
            return torch.arange(m, device=device), None, "broadcast", W

        if self.pool == "mean":
            count = scatter_sum(
                torch.ones(m, device=device, dtype=dtype),
                index,
                dim=0,
                dim_size=size,
            ).clamp(min=1.0)
            scale = (1.0 / count[index]).unsqueeze(-1).unsqueeze(-1)
            return torch.arange(m, device=device), scale, "scale", W

        if self.pool == "softmax":
            alpha = scatter_softmax(lifted, index, dim_size=size, dim=0)
            return torch.arange(m, device=device), alpha, "softmax", W

        _, max_index = scatter_max(lifted, index, dim_size=size, dim=0)
        valid = max_index >= 0
        edges = max_index.reshape(-1)
        channel = torch.arange(hidden, device=device).unsqueeze(0).expand(size, -1)
        channel = channel.reshape(-1)
        keep = valid.reshape(-1) & (edges < m)
        edges = edges[keep]
        channel = channel[keep]
        s = edges.shape[0]
        fields = torch.zeros(s, W.shape[0], hidden, device=device, dtype=dtype)
        if s > 0:
            fields[torch.arange(s, device=device), :, channel] = W[:, channel].T
        return edges, fields, "explicit", W

    def _start_field_chunk(self, edges, payload, mode, W, start, end):
        e = edges[start:end]
        n = e.shape[0]
        if mode == "broadcast":
            return e, W.unsqueeze(0).expand(n, -1, -1)
        if mode == "scale":
            return e, W.unsqueeze(0) * payload[start:end]
        if mode == "softmax":
            return e, W.unsqueeze(0) * payload[start:end].unsqueeze(1)
        return e, payload[start:end]

    def _compose_fields(self, L, edges, fields):
        """
        Walk Cosmo layers backward, composing filter matrices in the local
        reference frame of each query edge/triangle (see legacy/plot.py).

        Messages are applied in slices and immediately scatter-added onto
        unique source lifted ids, so the expanded (n_msg, C, C) path is never
        materialized. Completeness is unchanged because composition is linear.
        """
        m = L.features.shape[0]
        chunk = self.attribution_chunk_size
        edges, fields = _coalesce_fields(edges, fields)

        for layer in reversed(list(self.cosmo_layers)):
            if edges.numel() == 0:
                break
            src, dst = index_select_columns(L.source, L.target, edges, m)
            n_msg = src.shape[0]
            if n_msg == 0:
                edges = src
                fields = fields[:0]
                break

            unique_src, inv = torch.unique_consecutive(src, return_inverse=True)
            deg = None
            composed = None
            for start, end in _iter_chunks(n_msg, chunk):
                src_c = src[start:end]
                dst_c = dst[start:end]
                R = L.bases[edges[dst_c]]
                hood = (
                    L.coords[L.lifted2node[src_c]]
                    - L.coords[L.lifted2node[edges[dst_c]]]
                )
                hood = torch.bmm(R, hood.unsqueeze(-1)).squeeze(-1)
                w, agg = self._filter_weights(layer, hood)
                if agg == "mean":
                    if deg is None:
                        deg = scatter_sum(
                            torch.ones(n_msg, device=dst.device, dtype=w.dtype),
                            dst,
                            dim=0,
                            dim_size=edges.shape[0],
                        ).clamp(min=1)
                    w = w / deg[dst_c].unsqueeze(-1).unsqueeze(-1)
                chunk_fields = torch.einsum("boi,bij->boj", fields[dst_c], w)
                if composed is None:
                    composed = fields.new_zeros(
                        unique_src.shape[0],
                        chunk_fields.shape[1],
                        chunk_fields.shape[2],
                    )
                _scatter_add_rows(composed, inv[start:end], chunk_fields)
            edges = unique_src
            fields = composed
        return edges, fields

    def _scores_from_fields(self, L, edges, fields, node_features, num_nodes):
        """Chunked ``scatter_sum`` + ``einsum`` of composed fields onto nodes."""
        n = num_nodes
        device, dtype = node_features.device, node_features.dtype
        scores = torch.zeros(n, self.out_channels, device=device, dtype=dtype)
        if edges.numel() == 0:
            return scores
        nodes = L.lifted2node[edges]
        for start, end in _iter_chunks(edges.shape[0], self.attribution_chunk_size):
            idx = nodes[start:end]
            contrib = torch.einsum(
                "eci,ei->ec", fields[start:end], node_features[idx]
            )
            scores.scatter_add_(0, idx.unsqueeze(1).expand_as(contrib), contrib)
        return scores

    @torch.no_grad()
    def _attribute(self, L, node_features, lifted, node2inst, num_nodes, num_inst):
        """
        Per-node scores (n, C) whose graph-wise sums equal the graph logits.

        Completeness: scatter_sum(scores, node2inst) == graph-level logits.
        For node-level models the graph logits are the sum of node logits.
        """
        self._assert_attributable()
        n = num_nodes
        device, dtype = node_features.device, node_features.dtype
        W, b = self._linear_readout(lifted.shape[1], device, dtype)
        scores = torch.zeros(n, self.out_channels, device=device, dtype=dtype)
        if lifted.shape[0] > 0:
            starts, payload, mode, W = self._prepare_starts(
                L, lifted, W, num_nodes, num_inst
            )
            start_chunk = self.attribution_start_chunk
            for lo, hi in _iter_chunks(starts.shape[0], start_chunk):
                edges, fields = self._start_field_chunk(
                    starts, payload, mode, W, lo, hi
                )
                edges, fields = self._compose_fields(L, edges, fields)
                scores += self._scores_from_fields(
                    L, edges, fields, node_features, num_nodes
                )
                del edges, fields

        ones = torch.ones(n, device=device, dtype=dtype)
        node_count = scatter_sum(ones, node2inst, dim=0, dim_size=num_inst).clamp(min=1.0)
        if self.pool_to == "graph":
            scores = scores + b.unsqueeze(0) / node_count[node2inst].unsqueeze(1)
        else:
            scores = scores + b.unsqueeze(0)
        return scores
