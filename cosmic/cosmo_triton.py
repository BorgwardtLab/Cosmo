"""Triton-backed Cosmo layers that never materialize per-message weight matrices."""

import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_sum

from .cosmo import (
    KernelPointCosmo,
    KernelPointXCosmo,
    NeuralFieldCosmo,
    add_message_residual,
    aggregate_messages,
    apply_neighbor_gate,
    scatter_messages,
)
from .triton_ops import (
    kernel_point_apply,
    kernel_point_x_apply,
    neural_field_apply,
    resolve_field_act_id,
    use_triton_kernels,
)


class KernelPointCosmoTriton(KernelPointCosmo):
    """KernelPointCosmo with a fused linear-correlation GEMV Triton kernel.

    Same parameters and attribution semantics as :class:`KernelPointCosmo`.
    The ``(n_msg, C_out, C_in)`` blend of kernel weights is never materialized.
    Falls back to the reference implementation on CPU.
    """

    def forward(self, source, target, features, hood_coords, tri_edge_geom=None):
        m = features.shape[0]
        if m == 0 or source.numel() == 0:
            aggregated = add_message_residual(
                self, features.new_zeros(m, self.out_channels), features
            )
            return self._finish(aggregated, features)
        if not use_triton_kernels(features):
            return super().forward(
                source, target, features, hood_coords, tri_edge_geom=tri_edge_geom
            )

        geom_bias = self._geom_bias_features(tri_edge_geom)

        def messages_fn(start, end):
            hood = hood_coords[start:end]
            with torch.no_grad():
                influence = self.influences(hood)
            out = kernel_point_apply(features, source[start:end], self.w, influence)
            if geom_bias is not None:
                out = out + kernel_point_apply(
                    geom_bias, source[start:end], self.w, influence
                )
            return out

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


class KernelPointXCosmoTriton(KernelPointXCosmo):
    """KernelPointXCosmo with a fused closest-point gather-GEMV Triton kernel.

    Same parameters and attribution semantics as :class:`KernelPointXCosmo`.
    Falls back to the reference implementation on CPU.
    """

    def forward(self, source, target, features, hood_coords, tri_edge_geom=None):
        m = features.shape[0]
        if m == 0 or source.numel() == 0:
            aggregated = add_message_residual(
                self, features.new_zeros(m, self.out_channels), features
            )
            return self._finish(aggregated, features)
        if not use_triton_kernels(features):
            return super().forward(
                source, target, features, hood_coords, tri_edge_geom=tri_edge_geom
            )

        geom_bias = self._geom_bias_features(tri_edge_geom)

        def messages_fn(start, end):
            hood = hood_coords[start:end]
            with torch.no_grad():
                nn_idx = torch.cdist(hood, self.mu).argmin(dim=1)
            out = kernel_point_x_apply(features, source[start:end], self.w, nn_idx)
            if geom_bias is not None:
                out = out + kernel_point_x_apply(
                    geom_bias, source[start:end], self.w, nn_idx
                )
            return out

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


class NeuralFieldCosmoTriton(NeuralFieldCosmo):
    """NeuralFieldCosmo with a fused last-linear / activation / GEMV Triton kernel.

    The field trunk stays in PyTorch. The ``(m, C_out, C_in)`` weight tensor is
    never materialized. Attribution still uses the shared ``neural_field``.
    Falls back to the reference implementation on CPU or unsupported activations.
    """

    def forward(self, source, target, features, hood_coords):
        m = features.shape[0]
        out_channels = int(self.out_channels)
        in_channels = int(self.in_channels)
        if m == 0:
            return apply_neighbor_gate(
                self, features.new_zeros(m, out_channels), features
            )
        act = self.neural_field[-1]
        act_id = resolve_field_act_id(act)
        if not use_triton_kernels(features) or act_id is None:
            return super().forward(source, target, features, hood_coords)
        last = self.neural_field[-2]
        if not isinstance(last, nn.Linear):
            return super().forward(source, target, features, hood_coords)
        trunk = self.neural_field[:-2]
        h = trunk(hood_coords / self.radius)
        messages = neural_field_apply(
            h,
            features,
            source,
            last.weight,
            last.bias,
            out_channels,
            in_channels,
            act_id,
        )
        if self.message_attention:
            aggregated = aggregate_messages(
                self, messages, source, target, features, hood_coords
            )
        else:
            aggregated = scatter_mean(messages, target, dim_size=m, dim=0)
        return apply_neighbor_gate(self, aggregated, features)
