import torch
from torch import nn
from torch_scatter import scatter_max, scatter_softmax, scatter_sum

from .cosmo import KernelPointCosmo, NeuralFieldCosmo, PointTransformerCosmo
from .lift import Lift2D, Lift3D, Lower
from .utilities import index_select_columns

FILTERS = {
    "neural_field": NeuralFieldCosmo,
    "neuralfieldcosmo": NeuralFieldCosmo,
    "kernel_point": KernelPointCosmo,
    "kernelpointcosmo": KernelPointCosmo,
    "kpconv": KernelPointCosmo,
    "point_transformer": PointTransformerCosmo,
    "pointtransformercosmo": PointTransformerCosmo,
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
    is used (NeuralFieldCosmo / KernelPointCosmo). In that case a prediction can
    be decomposed into per-node, per-class scores that sum to the output logits
    by composing filter weight matrices across layers in aligned local frames
    (Kucera et al. 2026).
    """

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
        field_channels=32,
        field_layers=3,
        field_dropout=0.0,
        field_activation=nn.Tanh,
        minimum_angle=0.0,
    ):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        if pool not in ("sum", "mean", "max", "softmax"):
            raise ValueError("pool must be one of 'sum', 'mean', 'max', 'softmax'.")
        if pool_to not in ("node", "graph"):
            raise ValueError("pool_to must be 'node' or 'graph'.")
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

        self.lift = Lift2D() if dim == 2 else Lift3D()
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
            if cls is NeuralFieldCosmo:
                layers.append(
                    NeuralFieldCosmo(
                        in_channels=cin,
                        out_channels=cout,
                        field_channels=field_channels,
                        field_layers=field_layers,
                        field_dropout=field_dropout,
                        field_activation=field_activation,
                        radius=radius,
                        dim=dim,
                    )
                )
            elif cls is KernelPointCosmo:
                layers.append(
                    KernelPointCosmo(
                        in_channels=cin,
                        out_channels=cout,
                        kernel_points=self.kernel_points,
                    )
                )
            elif cls is PointTransformerCosmo:
                layers.append(
                    PointTransformerCosmo(
                        in_channels=cin,
                        out_channels=cout,
                        radius=radius,
                        dim=dim,
                    )
                )
            else:
                raise TypeError(f"Unsupported Cosmo filter class {cls}.")
        self.cosmo_layers = nn.ModuleList(layers)

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

    def forward(
        self,
        node_features,
        coords,
        edge_index,
        node2inst,
        attribution=False,
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
                    node_features, coords, edge_index, node2inst, attribution=True
                )
            finally:
                self.train(was_training)
        return self._forward(
            node_features, coords, edge_index, node2inst, attribution=False
        )

    def attribute(self, node_features, coords, edge_index, node2inst):
        """Predict and return (logits, scores) with completeness attribution."""
        return self.forward(
            node_features, coords, edge_index, node2inst, attribution=True
        )

    def _lift(self, node_features, coords, edge_index, node2inst):
        if self.dim == 3:
            return self.lift(
                node_features,
                coords,
                edge_index,
                node2inst,
                minimum_angle=self.minimum_angle,
            )
        return self.lift(node_features, coords, edge_index, node2inst)

    def _cosmo_features(self, L):
        features = L.features
        for i, layer in enumerate(self.cosmo_layers):
            features = layer(L.source, L.target, features, L.hood_coords)
            if i < len(self.cosmo_layers) - 1:
                features = self.activations[i](features)
        return features

    def _pool_index(self, L, num_nodes, num_inst):
        if self.pool_to == "node":
            return L.lifted2node, num_nodes
        return L.lifted2inst, num_inst

    def _forward(self, node_features, coords, edge_index, node2inst, attribution):
        num_nodes = node_features.shape[0]
        num_inst = int(node2inst.max().item()) + 1 if num_nodes > 0 else 0
        L = self._lift(node_features, coords, edge_index, node2inst)
        lifted = self._cosmo_features(L)
        index, size = self._pool_index(L, num_nodes, num_inst)
        pooled = self.lower(lifted, index, size)
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
            if not isinstance(layer, (NeuralFieldCosmo, KernelPointCosmo)):
                raise TypeError(
                    f"Exact attribution is not implemented for {type(layer).__name__}."
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
        if isinstance(layer, KernelPointCosmo):
            dist = torch.cdist(hood_coords.unsqueeze(0), layer.mu)
            nn_idx = dist.argmin(dim=2).squeeze(0)
            w = layer.w[:, nn_idx].permute(1, 0, 2)
            return w, "sum"
        raise TypeError(f"No filter weights for {type(layer).__name__}.")

    def _initial_fields(self, L, lifted, W, num_nodes, num_inst):
        """
        Fields of shape (s, out_channels, last_hidden) on `s` starting lifted
        entities, such that composing them back to the input reconstructs
        `W @ pooled_hidden` (without bias).
        """
        m, hidden = lifted.shape
        device, dtype = lifted.device, lifted.dtype
        index, size = self._pool_index(L, num_nodes, num_inst)

        if self.pool == "sum":
            edges = torch.arange(m, device=device)
            fields = W.to(dtype).unsqueeze(0).expand(m, -1, -1).clone()
            return edges, fields

        if self.pool == "mean":
            count = scatter_sum(
                torch.ones(m, device=device, dtype=dtype),
                index,
                dim=0,
                dim_size=size,
            ).clamp(min=1.0)
            scale = (1.0 / count[index]).unsqueeze(-1).unsqueeze(-1)
            edges = torch.arange(m, device=device)
            fields = W.to(dtype).unsqueeze(0) * scale
            return edges, fields

        if self.pool == "softmax":
            alpha = scatter_softmax(lifted, index, dim_size=size, dim=0)
            edges = torch.arange(m, device=device)
            fields = W.to(dtype).unsqueeze(0) * alpha.unsqueeze(1)
            return edges, fields

        # max: only the winning lifted entity of each pooled channel contributes
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
            fields[torch.arange(s, device=device), :, channel] = W[:, channel].T.to(dtype)
        return edges, fields

    def _compose_fields(self, L, edges, fields):
        """
        Walk Cosmo layers backward, composing filter matrices in the local
        reference frame of each query edge/triangle (see legacy/plot.py).
        """
        m = L.features.shape[0]

        for layer in reversed(list(self.cosmo_layers)):
            if edges.numel() == 0:
                break
            src, dst = index_select_columns(L.source, L.target, edges, m)
            if src.numel() == 0:
                edges = src
                fields = fields[:0]
                break
            R = L.bases[edges[dst]]
            hood = L.coords[L.lifted2node[src]] - L.coords[L.lifted2node[edges[dst]]]
            hood = torch.bmm(R, hood.unsqueeze(-1)).squeeze(-1)
            w, agg = self._filter_weights(layer, hood)
            if agg == "mean":
                deg = scatter_sum(
                    torch.ones_like(dst, dtype=w.dtype),
                    dst,
                    dim=0,
                    dim_size=edges.shape[0],
                )[dst].unsqueeze(-1).unsqueeze(-1)
                w = w / deg.clamp(min=1)
            fields = torch.einsum("boi,bij->boj", fields[dst], w)
            edges = src
        return edges, fields

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
        if lifted.shape[0] == 0:
            scores = torch.zeros(n, self.out_channels, device=device, dtype=dtype)
        else:
            edges, fields = self._initial_fields(L, lifted, W, num_nodes, num_inst)
            edges, fields = self._compose_fields(L, edges, fields)
            scores = torch.zeros(n, self.out_channels, device=device, dtype=dtype)
            if edges.numel() > 0:
                fields = scatter_sum(
                    fields, L.lifted2node[edges], dim_size=n, dim=0
                )
                scores = torch.einsum("nci,ni->nc", fields, node_features)

        ones = torch.ones(n, device=device, dtype=dtype)
        node_count = scatter_sum(ones, node2inst, dim=0, dim_size=num_inst).clamp(min=1.0)
        if self.pool_to == "graph":
            scores = scores + b.unsqueeze(0) / node_count[node2inst].unsqueeze(1)
        else:
            scores = scores + b.unsqueeze(0)
        return scores
