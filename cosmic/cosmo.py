import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

"""
Cosmo can be implemented with various filter functions. The underlying principle is always to compute the filter under transformation of a local reference frame (hood_coords) which is derived from neighboring input points. The forward signature of the layer is always the same and inputs can be obtained from a Lift2D or Lift3D module.
"""


def resolve_field_activation(spec):
    """Map config strings to the last-layer activation on NF weight heads."""
    if spec is None or spec == "tanh":
        return nn.Tanh
    if spec in ("identity", "none", "linear"):
        return nn.Identity
    if isinstance(spec, type):
        return spec
    raise ValueError(
        f"field_activation {spec!r} unsupported; use tanh, identity, or nn.Module."
    )


def kernel_influence_radius(kernel_points):
    """Influence radius for linear KPConv correlation.

    Matches ``KP_extent`` in Thomas et al. 2019: kernel points live in a ball of
    this radius, and a kernel point's weight falls to zero at this distance.
    Defaults to the farthest kernel point from the origin. If every kernel point
    is at the origin, falls back to the nearest-neighbor spacing, then to 1.
    """
    mu = kernel_points.detach().float().reshape(-1, kernel_points.shape[-1])
    extent = mu.norm(dim=-1).max()
    if torch.isfinite(extent) and float(extent) > 0:
        return extent.reshape(())
    if mu.shape[0] >= 2:
        dist = torch.cdist(mu, mu)
        dist.fill_diagonal_(float("inf"))
        spacing = dist.min(dim=1).values.max()
        if torch.isfinite(spacing) and float(spacing) > 0:
            return spacing.reshape(())
    return mu.new_tensor(1.0)


def init_neighbor_gate(module, in_channels, out_channels, gate):
    """Optional mix of the neighborhood aggregate and the current entity state.

    The flag defaults off, so the aggregate is returned unchanged. When on, a
    linear map of the current features produces a per-channel sigmoid. That
    value weights the aggregate; its complement keeps the current state.
    """
    module.gate = bool(gate)
    if not module.gate:
        module.neighbor_gate = None
        module.state_proj = None
        return
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    module.neighbor_gate = nn.Linear(in_channels, out_channels)
    nn.init.zeros_(module.neighbor_gate.weight)
    nn.init.zeros_(module.neighbor_gate.bias)
    if in_channels == out_channels:
        module.state_proj = None
    else:
        module.state_proj = nn.Linear(in_channels, out_channels, bias=False)


def apply_neighbor_gate(module, aggregated, features):
    if not module.gate:
        return aggregated
    update = torch.sigmoid(module.neighbor_gate(features))
    state = features if module.state_proj is None else module.state_proj(features)
    return update * aggregated + (1.0 - update) * state


def init_message_attention(module, in_channels, out_channels, dim, enabled):
    """Optional per-neighbor softmax and a residual of the current state.

    The flag defaults off, so the caller keeps an unweighted sum. When on, a
    score of the source state, the target state, and the local-frame offset
    weights each message. The current state is added back, through a bias-free
    projection when the width changes.
    """
    module.message_attention = bool(enabled)
    if not module.message_attention:
        module.msg_score = None
        module.msg_residual = None
        return
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    hidden = min(32, max(in_channels, 8))
    module.msg_score = nn.Sequential(
        nn.Linear(2 * in_channels + int(dim), hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    )
    nn.init.zeros_(module.msg_score[-1].weight)
    nn.init.zeros_(module.msg_score[-1].bias)
    if in_channels == out_channels:
        module.msg_residual = None
    else:
        module.msg_residual = nn.Linear(in_channels, out_channels, bias=False)


def add_message_residual(module, aggregated, features):
    if not getattr(module, "message_attention", False):
        return aggregated
    if module.msg_residual is None:
        if features.shape[-1] != aggregated.shape[-1]:
            return aggregated
        return aggregated + features
    return aggregated + module.msg_residual(features)


def _row_step(n_msg, width):
    """Rows per chunk. Keeps a gathered ``(rows, width)`` tensor under 2^30 entries."""
    width = max(int(width), 1)
    if n_msg <= 0:
        return 1
    return max(1, min(int(n_msg), (1 << 30) // width))


def scatter_messages(module, source, target, features, hood_coords, messages_fn, n_out):
    """Scatter per-message values onto targets.

    ``messages_fn(start, end)`` returns ``(end - start, n_out)``. Message
    attention softmax-weights those values and adds the current state. Rows
    are chunked so a wide hidden size never builds one index tensor past the
    32-bit CUDA limit.
    """
    m = features.shape[0]
    n_msg = int(source.shape[0])
    aggregated = features.new_zeros(m, n_out)
    if n_msg == 0:
        return add_message_residual(module, aggregated, features)
    alpha = None
    if getattr(module, "message_attention", False):
        step = _row_step(
            n_msg, 2 * int(features.shape[-1]) + int(hood_coords.shape[-1])
        )
        parts = []
        for start in range(0, n_msg, step):
            end = min(n_msg, start + step)
            parts.append(
                module.msg_score(
                    torch.cat(
                        [
                            features.index_select(0, source[start:end]),
                            features.index_select(0, target[start:end]),
                            hood_coords[start:end].to(dtype=features.dtype),
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)
            )
        alpha = scatter_softmax(torch.cat(parts, dim=0), target, dim_size=m)
    scatter_width = max(int(n_out), int(features.shape[-1]))
    msg_width = getattr(module, "psi_in_width", None)
    if msg_width is not None:
        scatter_width = max(scatter_width, int(msg_width))
    step = _row_step(n_msg, scatter_width)
    for start in range(0, n_msg, step):
        end = min(n_msg, start + step)
        messages = messages_fn(start, end)
        if alpha is not None:
            messages = messages * alpha[start:end].unsqueeze(-1)
        aggregated = aggregated + scatter_sum(
            messages, target[start:end], dim_size=m, dim=0
        )
    return add_message_residual(module, aggregated, features)


def aggregate_messages(module, messages, source, target, features, hood_coords):
    """Sum already-built messages. Attention replaces the sum with a weighted one."""
    def messages_fn(start, end):
        return messages[start:end]

    return scatter_messages(
        module,
        source,
        target,
        features,
        hood_coords,
        messages_fn,
        int(messages.shape[-1]),
    )


class _KernelPoint(nn.Module):
    """Shared kernel-point weights. ``mu`` has shape (k, dim), ``w`` is (out, k, in).

    ``bias`` and ``silu`` are applied after the neighborhood sum. The kernel
    weight application itself stays linear.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_points,
        bias=False,
        silu=False,
        norm=False,
        residual=False,
        gate=False,
        message_attention=False,
        learnable_kernel_points=False,
        soft_influences=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.silu = bool(silu)
        self.residual = bool(residual)
        self.learnable_kernel_points = bool(learnable_kernel_points)
        self.soft_influences = bool(soft_influences)
        init_neighbor_gate(self, in_channels, out_channels, gate)
        mu = kernel_points.detach().float().reshape(-1, kernel_points.shape[-1])
        if self.learnable_kernel_points:
            self.mu = nn.Parameter(mu.clone())
            self.register_buffer("mu_shell_norm", mu.norm(dim=-1).clone())
        else:
            self.register_buffer("mu", mu.contiguous())
        self.w = nn.Parameter(torch.empty(out_channels, self.mu.shape[0], in_channels))
        nn.init.xavier_uniform_(self.w)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        # Scale only. The learned threshold stays in ``bias``, which is applied
        # after this norm so LayerNorm does not remove it.
        self.norm = (
            nn.LayerNorm(out_channels, elementwise_affine=False) if norm else None
        )
        init_message_attention(
            self,
            in_channels,
            out_channels,
            self.mu.shape[-1],
            message_attention,
        )
        self.edge_geom_dim = 0
        self.edge_geom_proj = None

    def _kernel_mu(self):
        """Kernel point locations; learnable points stay on their initial shells."""
        if not self.learnable_kernel_points:
            return self.mu
        norm = self.mu.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.mu / norm * self.mu_shell_norm.unsqueeze(-1)

    def _init_edge_geom(self, edge_geom_dim):
        dim = int(edge_geom_dim or 0)
        self.edge_geom_dim = dim
        if dim > 0:
            in_channels = int(self.w.shape[-1])
            self.edge_geom_proj = nn.Linear(dim, in_channels, bias=False)
        else:
            self.edge_geom_proj = None

    def _gather_source(self, features, source, start, end, tri_edge_geom=None):
        gathered = features.index_select(0, source[start:end])
        if tri_edge_geom is not None and self.edge_geom_proj is not None:
            gathered = gathered + self.edge_geom_proj(
                tri_edge_geom.index_select(0, source[start:end])
            )
        return gathered

    def _geom_bias_features(self, tri_edge_geom):
        if tri_edge_geom is None or self.edge_geom_proj is None:
            return None
        return self.edge_geom_proj(tri_edge_geom)

    def _finish(self, aggregated, features):
        """Gate, bias, norm, SiLU, and a same-width residual after the neighborhood sum."""
        aggregated = apply_neighbor_gate(self, aggregated, features)
        if self.norm is not None:
            aggregated = self.norm(aggregated)
        if self.bias is not None:
            aggregated = aggregated + self.bias
        if self.silu:
            aggregated = F.silu(aggregated)
        if (
            self.residual
            and not self.message_attention
            and features is not None
            and features.shape[-1] == aggregated.shape[-1]
        ):
            aggregated = aggregated + features
        return aggregated


class KernelPointCosmo(_KernelPoint):
    """
    Kernel Point Convolution (Thomas et al. 2019) in the Cosmo framework.

    Each neighbor offset ``y`` is filtered by every kernel point:

        g(y) = sum_k max(0, 1 - ||y - mu_k|| / sigma) W_k

    ``sigma`` is the influence radius (KPConv's ``KP_extent``). Correlation
    weights are constants with respect to feature gradients; kernel points are
    fixed buffers. Offsets farther than ``sigma`` from every kernel point
    contribute nothing.

    ``bias`` adds a learned per-channel vector after the neighborhood sum.
    ``silu`` applies SiLU to that sum. Both default off, so the layer stays linear.
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        kernel_points,  # Kernel points of shape (k, dim)
        sigma=None,  # Influence radius; default is the kernel extent
        bias=False,  # Learned bias after the neighborhood sum
        silu=False,  # SiLU after the neighborhood sum
        norm=False,  # LayerNorm of the neighborhood sum, before the bias
        residual=False,  # Add the incoming features when widths match
        gate=False,  # Mix the neighborhood aggregate with the current state
        message_attention=False,  # Softmax over neighbors, then add the state
        edge_geom_dim=0,
        learnable_kernel_points=False,
        soft_influences=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_points,
            bias=bias,
            silu=silu,
            norm=norm,
            residual=residual,
            gate=gate,
            message_attention=message_attention,
            learnable_kernel_points=learnable_kernel_points,
            soft_influences=soft_influences,
        )
        self._init_edge_geom(edge_geom_dim)
        mu_ref = self._kernel_mu()
        if sigma is None:
            sigma = kernel_influence_radius(mu_ref)
        else:
            sigma = torch.as_tensor(
                float(sigma), dtype=torch.float32, device=mu_ref.device
            )
        self.register_buffer("sigma", sigma.reshape(()).float())

    def influences(self, hood_coords):
        """Linear (or softplus) correlation per kernel point, shape (n_msg, k)."""
        mu = self._kernel_mu()
        dist = torch.cdist(hood_coords, mu)
        raw = 1.0 - dist / self.sigma
        if self.soft_influences:
            return F.softplus(raw)
        return raw.clamp(min=0)

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
        tri_edge_geom=None,  # Per-entity invariant edge features (n_ent, edge_geom_dim)
    ):
        def messages_fn(start, end):
            hood = hood_coords[start:end]
            with torch.no_grad():
                influence = self.influences(hood)
            gathered = self._gather_source(
                features, source, start, end, tri_edge_geom=tri_edge_geom
            )
            return torch.einsum("mi,oki,mk->mo", gathered, self.w, influence)

        aggregated = scatter_messages(
            self,
            source,
            target,
            features,
            hood_coords,
            messages_fn,
            self.out_channels,
        )
        return self._finish(aggregated, features)


class KernelPointXCosmo(_KernelPoint):
    """
    Kernel Point Convolution with the KPConvX closest-point trick (Thomas et al. 2024).

    Each neighbor uses only the weight matrix of its nearest kernel point, instead
    of the linear combination over all kernel points used by :class:`KernelPointCosmo`.

    ``bias`` and ``silu`` match :class:`KernelPointCosmo`: both are optional and
    run after the neighborhood sum. The closest-point weight application stays linear.

    Each kernel slice is Xavier-initialized on its own. A single neighbor uses
    one slice, so initializing the whole ``(out, k, in)`` tensor would shrink
    every message by about ``sqrt(k)``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_points,
        bias=False,
        silu=False,
        norm=False,
        residual=False,
        gate=False,
        message_attention=False,
        edge_geom_dim=0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_points,
            bias=bias,
            silu=silu,
            norm=norm,
            residual=residual,
            gate=gate,
            message_attention=message_attention,
        )
        self._init_edge_geom(edge_geom_dim)
        for index in range(self.mu.shape[0]):
            nn.init.xavier_uniform_(self.w[:, index, :])

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
        tri_edge_geom=None,
    ):
        def messages_fn(start, end):
            hood = hood_coords[start:end]
            with torch.no_grad():
                nn_idx = torch.cdist(hood, self.mu).argmin(dim=1)
            gathered = self._gather_source(
                features, source, start, end, tri_edge_geom=tri_edge_geom
            )
            return torch.einsum("ni,oni->no", gathered, self.w[:, nn_idx])

        aggregated = scatter_messages(
            self,
            source,
            target,
            features,
            hood_coords,
            messages_fn,
            self.out_channels,
        )
        return self._finish(aggregated, features)


class NeuralFieldCosmo(nn.Module):
    """
    Implements Neural Field Convolution (Proposed with Cosmo in Kucera et al. 2026) in the Cosmo framework. Weight matrices are computed from input coordinates in the local reference frame using a neural field (parameterized by a neural network).
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        field_channels=32,  # Number of channels in the neural field
        field_layers=3,  # Number of layers in the neural field
        field_dropout=0.0,  # Dropout rate in the neural field
        field_activation=nn.Tanh,  # Activation function in the neural field
        radius=1.0,  # Scale parameter for input coordinates
        dim=3,  # Dimension of the input data (2 or 3)
        field_input_dim=None,
        gate=False,  # Mix the neighborhood mean with the current state
        message_attention=False,  # Softmax over neighbors, then add the state
    ):
        super().__init__()
        self.register_buffer("radius", torch.tensor(radius))
        self.register_buffer("in_channels", torch.tensor(in_channels))
        self.register_buffer("out_channels", torch.tensor(out_channels))
        input_dim = int(field_input_dim) if field_input_dim is not None else int(dim)
        self.field_input_dim = input_dim
        init_neighbor_gate(self, in_channels, out_channels, gate)
        init_message_attention(
            self, in_channels, out_channels, dim, message_attention
        )
        act = resolve_field_activation(field_activation)
        self.neural_field = nn.Sequential(
            nn.Linear(input_dim, field_channels),
            nn.LayerNorm(field_channels),
            nn.ReLU(),
            nn.Dropout(field_dropout),
            *[
                nn.Linear(field_channels, field_channels),
                nn.LayerNorm(field_channels),
                nn.ReLU(),
                nn.Dropout(field_dropout),
            ]
            * (field_layers - 2),
            nn.Linear(field_channels, in_channels * out_channels),
            act(),
        )

    def _field_coords(self, hood_coords):
        if int(hood_coords.shape[-1]) == int(self.field_input_dim):
            return hood_coords
        return hood_coords / self.radius

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
    ):
        m = features.shape[0]
        w = self.neural_field(self._field_coords(hood_coords)).view(
            -1, self.out_channels, self.in_channels
        )
        f = features[source]
        out_channels = torch.einsum("ni,noi->no", f, w)  # m x out
        if self.message_attention:
            aggregated = aggregate_messages(
                self, out_channels, source, target, features, hood_coords
            )
        else:
            aggregated = scatter_mean(out_channels, target, dim_size=m, dim=0)
        return apply_neighbor_gate(self, aggregated, features)


def scale_hood_coords(hood_coords, radius):
    """Express LRF offsets in units of the model ``radius`` (KP extent scale)."""
    r = float(radius.item()) if torch.is_tensor(radius) else float(radius)
    if r <= 0:
        return hood_coords
    return hood_coords / r


class MessageMLPCosmo(nn.Module):
    """Message passing with an MLP on ``[h_dst, h_src, x_local]`` (not NeuralField).

    Update: ``h'_j = phi(h_j, sum_i psi(h_j, h_i, x_ij))`` with ``x`` in the LRF of ``j``.
    Optional ``tri_edge_geom`` (e.g. PiFold backbone RBF per lifted edge/triangle) is
    concatenated into ``psi`` on the source entity, matching KP ``edge_geom_proj`` indexing.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dim=3,
        radius=1.0,
        mlp_hidden=64,
        gate=False,
        edge_geom_dim=0,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.dim = int(dim)
        self.edge_geom_dim = int(edge_geom_dim or 0)
        self.register_buffer("radius", torch.tensor(float(radius)))
        init_neighbor_gate(self, in_channels, out_channels, gate)
        hidden = max(int(mlp_hidden), int(out_channels), 16)
        psi_in = int(in_channels) * 2 + self.dim + self.edge_geom_dim
        self.psi_in_width = int(psi_in)
        self.psi = nn.Sequential(
            nn.Linear(psi_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_channels),
        )
        phi_in = int(in_channels) + self.out_channels
        self.phi = nn.Sequential(
            nn.Linear(phi_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_channels),
        )

    def forward(self, source, target, features, hood_coords, tri_edge_geom=None):
        geom = tri_edge_geom
        if self.edge_geom_dim > 0:
            if geom is None:
                raise ValueError(
                    "MessageMLPCosmo edge_geom_dim > 0 requires tri_edge_geom."
                )
            if geom.shape[-1] != self.edge_geom_dim:
                raise ValueError(
                    f"tri_edge_geom width {geom.shape[-1]} != edge_geom_dim "
                    f"{self.edge_geom_dim}"
                )

        def messages_fn(start, end):
            s = source[start:end]
            t = target[start:end]
            parts = [
                features.index_select(0, t),
                features.index_select(0, s),
                hood_coords[start:end].to(dtype=features.dtype),
            ]
            if geom is not None:
                parts.append(geom.index_select(0, s).to(dtype=features.dtype))
            msg_in = torch.cat(parts, dim=-1)
            return self.psi(msg_in)

        aggregated = scatter_messages(
            self,
            source,
            target,
            features,
            hood_coords,
            messages_fn,
            self.out_channels,
        )
        out = self.phi(torch.cat([features, aggregated], dim=-1))
        return apply_neighbor_gate(self, out, features)


class PointTransformerCosmo(nn.Module):
    """
    Implements Point Transformer Convolution (Zhao et al. 2020) in the Cosmo framework.
    """

    def __init__(
        self,
        in_channels,  # Number of input channels
        out_channels,  # Number of output channels
        radius=1.0,  # Scale parameter for input coordinates
        dim=3,  # Dimension of the input data (2 or 3)
        gate=False,  # Mix the neighborhood aggregate with the current state
        message_attention=False,  # Add the current state on top of the attention sum
    ):
        super().__init__()
        self.register_buffer("radius", torch.tensor(radius))
        self.message_attention = bool(message_attention)
        init_neighbor_gate(self, in_channels, out_channels, gate)
        if self.message_attention and int(in_channels) != int(out_channels):
            self.msg_residual = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.msg_residual = None
        self.delta = nn.Sequential(
            nn.Linear(dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        self.w1 = nn.Linear(in_channels, out_channels, bias=False)
        self.w2 = nn.Linear(in_channels, out_channels, bias=False)
        self.w3 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self,
        source,  # Source edges ij or triangles ijk (m,)
        target,  # Target edges jk or triangles jkl (m,)
        features,  # Edge or triangle features of shape (m, in_channels)
        hood_coords,  # Locally transformed coordinates of shape (m, dim)
    ):
        m = features.shape[0]
        d = self.delta(hood_coords / self.radius)
        w1 = self.w1(features)
        w2 = self.w2(features)
        w3 = self.w3(features)
        a = scatter_softmax(w1[target] - w2[source] + d, target, dim_size=m, dim=0)
        aggregated = scatter_sum(a * (w3[source] + d), target, dim_size=m, dim=0)
        aggregated = add_message_residual(self, aggregated, features)
        return apply_neighbor_gate(self, aggregated, features)
