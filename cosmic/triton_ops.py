"""Triton kernels for Cosmo filter application (no per-message weight matrix)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


ACT_IDENTITY = 0
ACT_TANH = 1


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def use_triton_kernels(tensor: torch.Tensor) -> bool:
    return (
        TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


if TRITON_AVAILABLE:

    @triton.jit
    def _kp_x_forward_kernel(
        feat_ptr,
        src_ptr,
        w_ptr,
        nn_ptr,
        out_ptr,
        m,
        c_in,
        c_out,
        stride_feat_m,
        stride_feat_c,
        stride_w_o,
        stride_w_k,
        stride_w_i,
        stride_out_m,
        stride_out_c,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        p = tl.program_id(0).to(tl.int64)
        ot = tl.program_id(1).to(tl.int64)
        c_in = tl.cast(c_in, tl.int64)
        c_out = tl.cast(c_out, tl.int64)
        stride_feat_m = tl.cast(stride_feat_m, tl.int64)
        stride_feat_c = tl.cast(stride_feat_c, tl.int64)
        stride_w_o = tl.cast(stride_w_o, tl.int64)
        stride_w_k = tl.cast(stride_w_k, tl.int64)
        stride_w_i = tl.cast(stride_w_i, tl.int64)
        stride_out_m = tl.cast(stride_out_m, tl.int64)
        stride_out_c = tl.cast(stride_out_c, tl.int64)
        if p >= m:
            return
        src = tl.load(src_ptr + p).to(tl.int64)
        k = tl.load(nn_ptr + p).to(tl.int64)
        offs_o = ot * BLOCK_OUT + tl.arange(0, BLOCK_OUT).to(tl.int64)
        mask_o = offs_o < c_out
        o_safe = tl.where(mask_o, offs_o, 0)
        offs_i = tl.arange(0, BLOCK_IN).to(tl.int64)
        acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
        for i0 in range(0, c_in, BLOCK_IN):
            i = i0 + offs_i
            mask_i = i < c_in
            i_safe = tl.where(mask_i, i, 0)
            f = tl.load(
                feat_ptr + src * stride_feat_m + i_safe * stride_feat_c,
                mask=mask_i,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr
                + o_safe[:, None] * stride_w_o
                + k * stride_w_k
                + i_safe[None, :] * stride_w_i,
                mask=mask_o[:, None] & mask_i[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * f[None, :], axis=1)
        tl.store(
            out_ptr + p * stride_out_m + o_safe * stride_out_c,
            acc,
            mask=mask_o,
        )

    @triton.jit
    def _kp_forward_kernel(
        feat_ptr,
        src_ptr,
        w_ptr,
        h_ptr,
        out_ptr,
        m,
        c_in,
        c_out,
        stride_feat_m,
        stride_feat_c,
        stride_w_o,
        stride_w_k,
        stride_w_i,
        stride_h_m,
        stride_h_k,
        stride_out_m,
        stride_out_c,
        BLOCK_M: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        N_K: tl.constexpr,
    ):
        """Linear KPConv over a tile of messages. Does not build (m, C_out, C_in) weights."""
        pid_m = tl.program_id(0).to(tl.int64)
        pid_o = tl.program_id(1).to(tl.int64)
        c_in = tl.cast(c_in, tl.int64)
        c_out = tl.cast(c_out, tl.int64)
        stride_feat_m = tl.cast(stride_feat_m, tl.int64)
        stride_feat_c = tl.cast(stride_feat_c, tl.int64)
        stride_w_o = tl.cast(stride_w_o, tl.int64)
        stride_w_k = tl.cast(stride_w_k, tl.int64)
        stride_w_i = tl.cast(stride_w_i, tl.int64)
        stride_h_m = tl.cast(stride_h_m, tl.int64)
        stride_h_k = tl.cast(stride_h_k, tl.int64)
        stride_out_m = tl.cast(stride_out_m, tl.int64)
        stride_out_c = tl.cast(stride_out_c, tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT).to(tl.int64)
        mask_m = offs_m < m
        mask_o = offs_o < c_out
        m_safe = tl.where(mask_m, offs_m, 0)
        o_safe = tl.where(mask_o, offs_o, 0)
        src = tl.load(src_ptr + m_safe).to(tl.int64)
        offs_i = tl.arange(0, BLOCK_IN).to(tl.int64)
        acc = tl.zeros([BLOCK_M, BLOCK_OUT], dtype=tl.float32)
        for k in range(N_K):
            hk = tl.load(
                h_ptr + m_safe * stride_h_m + k * stride_h_k,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            acc_k = tl.zeros([BLOCK_M, BLOCK_OUT], dtype=tl.float32)
            for i0 in range(0, c_in, BLOCK_IN):
                i = i0 + offs_i
                mask_i = i < c_in
                i_safe = tl.where(mask_i, i, 0)
                f = tl.load(
                    feat_ptr + src[:, None] * stride_feat_m + i_safe[None, :] * stride_feat_c,
                    mask=mask_m[:, None] & mask_i[None, :],
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    w_ptr
                    + o_safe[:, None] * stride_w_o
                    + k * stride_w_k
                    + i_safe[None, :] * stride_w_i,
                    mask=mask_o[:, None] & mask_i[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc_k += tl.dot(f, tl.trans(w), input_precision="ieee")
            acc += acc_k * hk[:, None]
        tl.store(
            out_ptr + m_safe[:, None] * stride_out_m + o_safe[None, :] * stride_out_c,
            acc,
            mask=mask_m[:, None] & mask_o[None, :],
        )

    @triton.jit
    def _nf_forward_kernel(
        h_ptr,
        feat_ptr,
        src_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        m,
        c_in,
        c_out,
        c_field,
        stride_h_m,
        stride_h_c,
        stride_feat_m,
        stride_feat_c,
        stride_w_oi,
        stride_w_c,
        stride_out_m,
        stride_out_c,
        HAS_BIAS: tl.constexpr,
        ACT_ID: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_FIELD: tl.constexpr,
    ):
        p = tl.program_id(0).to(tl.int64)
        o = tl.program_id(1).to(tl.int64)
        c_in = tl.cast(c_in, tl.int64)
        c_field = tl.cast(c_field, tl.int64)
        stride_h_m = tl.cast(stride_h_m, tl.int64)
        stride_h_c = tl.cast(stride_h_c, tl.int64)
        stride_feat_m = tl.cast(stride_feat_m, tl.int64)
        stride_feat_c = tl.cast(stride_feat_c, tl.int64)
        stride_w_oi = tl.cast(stride_w_oi, tl.int64)
        stride_w_c = tl.cast(stride_w_c, tl.int64)
        stride_out_m = tl.cast(stride_out_m, tl.int64)
        stride_out_c = tl.cast(stride_out_c, tl.int64)
        if p >= m or o >= c_out:
            return
        src = tl.load(src_ptr + p).to(tl.int64)
        offs_i = tl.arange(0, BLOCK_IN).to(tl.int64)
        offs_c = tl.arange(0, BLOCK_FIELD).to(tl.int64)
        acc = 0.0
        for i0 in range(0, c_in, BLOCK_IN):
            i = i0 + offs_i
            mask_i = i < c_in
            i_safe = tl.where(mask_i, i, 0)
            f = tl.load(
                feat_ptr + src * stride_feat_m + i_safe * stride_feat_c,
                mask=mask_i,
                other=0.0,
            ).to(tl.float32)
            raw = tl.zeros([BLOCK_IN], dtype=tl.float32)
            if HAS_BIAS:
                raw += tl.load(
                    bias_ptr + o * c_in + i_safe, mask=mask_i, other=0.0
                ).to(tl.float32)
            for c0 in range(0, c_field, BLOCK_FIELD):
                c = c0 + offs_c
                mask_c = c < c_field
                c_safe = tl.where(mask_c, c, 0)
                hvec = tl.load(
                    h_ptr + p * stride_h_m + c_safe * stride_h_c,
                    mask=mask_c,
                    other=0.0,
                ).to(tl.float32)
                w_tile = tl.load(
                    weight_ptr
                    + (o * c_in + i_safe)[:, None] * stride_w_oi
                    + c_safe[None, :] * stride_w_c,
                    mask=mask_i[:, None] & mask_c[None, :],
                    other=0.0,
                ).to(tl.float32)
                raw += tl.sum(w_tile * hvec[None, :], axis=1)
            if ACT_ID == 1:
                filt = tl.extra.cuda.libdevice.tanh(raw)
            else:
                filt = raw
            acc += tl.sum(tl.where(mask_i, filt * f, 0.0))
        tl.store(out_ptr + p * stride_out_m + o * stride_out_c, acc)


def _kp_x_forward_triton(features, source, w, nn_idx):
    n_msg = source.shape[0]
    c_in = features.shape[1]
    c_out = w.shape[0]
    messages = torch.empty(n_msg, c_out, device=features.device, dtype=features.dtype)
    if n_msg == 0 or c_out == 0:
        return messages
    block_in = min(128, max(16, next_pow2(c_in)))
    block_out = min(16, max(1, next_pow2(c_out)))
    _kp_x_forward_kernel[(n_msg, triton.cdiv(c_out, block_out))](
        features,
        source,
        w,
        nn_idx,
        messages,
        n_msg,
        c_in,
        c_out,
        features.stride(0),
        features.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        messages.stride(0),
        messages.stride(1),
        BLOCK_IN=block_in,
        BLOCK_OUT=block_out,
        num_warps=4,
    )
    return messages


def _kp_forward_triton(features, source, w, h):
    n_msg = source.shape[0]
    c_in = features.shape[1]
    c_out = w.shape[0]
    n_k = w.shape[1]
    messages = torch.empty(n_msg, c_out, device=features.device, dtype=features.dtype)
    if n_msg == 0 or c_out == 0 or n_k == 0:
        return messages
    block_m = 64
    block_in = 32 if c_in >= 32 else max(16, next_pow2(c_in))
    block_out = 64 if c_out >= 64 else max(16, next_pow2(c_out))
    _kp_forward_kernel[(triton.cdiv(n_msg, block_m), triton.cdiv(c_out, block_out))](
        features,
        source,
        w,
        h,
        messages,
        n_msg,
        c_in,
        c_out,
        features.stride(0),
        features.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        h.stride(0),
        h.stride(1),
        messages.stride(0),
        messages.stride(1),
        BLOCK_M=block_m,
        BLOCK_IN=block_in,
        BLOCK_OUT=block_out,
        N_K=n_k,
        num_warps=4,
    )
    return messages


def _nf_forward_triton(h, features, source, weight, bias, c_out, c_in, act_id):
    m = h.shape[0]
    c_field = h.shape[1]
    messages = torch.empty(m, c_out, device=h.device, dtype=h.dtype)
    if m == 0 or c_out == 0:
        return messages
    has_bias = bias is not None
    if bias is None:
        bias = weight.new_empty(0)
    block_in = min(128, max(16, next_pow2(c_in)))
    block_field = min(128, max(16, next_pow2(c_field)))
    _nf_forward_kernel[(m, c_out)](
        h,
        features,
        source,
        weight,
        bias,
        messages,
        m,
        c_in,
        c_out,
        c_field,
        h.stride(0),
        h.stride(1),
        features.stride(0),
        features.stride(1),
        weight.stride(0),
        weight.stride(1),
        messages.stride(0),
        messages.stride(1),
        HAS_BIAS=has_bias,
        ACT_ID=act_id,
        BLOCK_IN=block_in,
        BLOCK_FIELD=block_field,
        num_warps=4,
    )
    return messages


def _message_row_chunk(n_msg, width, row_chunk=None):
    """Rows per gather. A single ``(n_msg, width)`` tensor overflows int32 numel
    once width is the 117-d PiFold feature size and a batch has ~2e7 messages.
    """
    n_msg = int(n_msg)
    if n_msg <= 0:
        return 1
    if row_chunk is not None:
        return max(1, min(n_msg, int(row_chunk)))
    width = max(int(width), 1)
    rows = min((1 << 30) // width, (256 * 1024 * 1024) // (width * 4))
    return max(1, min(n_msg, rows))


def _kp_x_backward_torch(grad_msg, features, source, w, nn_idx, row_chunk=None):
    """Memory-efficient KPConvX backward: one GEMM per kernel point."""
    n_msg = source.shape[0]
    grad_w = w.new_zeros(w.shape)
    grad_feat = features.new_zeros(features.shape)
    if n_msg == 0:
        return grad_feat, grad_w
    step = _message_row_chunk(n_msg, features.shape[1], row_chunk)
    k = w.shape[1]
    for start in range(0, n_msg, step):
        end = min(n_msg, start + step)
        src = source[start:end]
        f = features.index_select(0, src)
        gmsg = grad_msg[start:end]
        nn = nn_idx[start:end]
        for ki in range(k):
            mask = nn == ki
            if not torch.any(mask):
                continue
            gk = gmsg[mask]
            fk = f[mask]
            grad_w[:, ki, :] = grad_w[:, ki, :] + gk.T @ fk
            grad_feat.index_add_(0, src[mask], gk @ w[:, ki, :])
    return grad_feat, grad_w


def _kp_backward_torch(grad_msg, features, source, w, h, row_chunk=None):
    """Linear KPConv backward. Correlation ``h`` is a constant."""
    n_msg = source.shape[0]
    grad_w = w.new_zeros(w.shape)
    grad_feat = features.new_zeros(features.shape)
    if n_msg == 0:
        return grad_feat, grad_w
    step = _message_row_chunk(n_msg, features.shape[1], row_chunk)
    k = w.shape[1]
    for start in range(0, n_msg, step):
        end = min(n_msg, start + step)
        src = source[start:end]
        f = features.index_select(0, src)
        gmsg = grad_msg[start:end]
        hh = h[start:end]
        for ki in range(k):
            gk = gmsg * hh[:, ki].to(dtype=gmsg.dtype).unsqueeze(1)
            grad_w[:, ki, :] = grad_w[:, ki, :] + gk.T @ f
            grad_feat.index_add_(0, src, gk @ w[:, ki, :])
    return grad_feat, grad_w


def _nf_backward_torch(grad_msg, h, features, source, weight, bias, c_out, c_in, act_id):
    """Memory-efficient NF backward: GEMM tiles over a few output channels."""
    f = features.index_select(0, source)
    m = h.shape[0]
    # Keep the (m, tile, C_in) scratch under ~16 MiB so W is never rebuilt.
    max_elems = 4 * 1024 * 1024
    tile = max(1, min(c_out, max_elems // max(m * max(c_in, 1), 1)))
    grad_h = torch.zeros_like(h)
    grad_f = torch.zeros_like(f)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    for o0 in range(0, c_out, tile):
        o1 = min(c_out, o0 + tile)
        sl = slice(o0 * c_in, o1 * c_in)
        raw = F.linear(h, weight[sl], None if bias is None else bias[sl])
        raw = raw.view(m, o1 - o0, c_in)
        filt = torch.tanh(raw) if act_id == ACT_TANH else raw
        g = grad_msg[:, o0:o1].unsqueeze(-1)
        grad_f = grad_f + (g * filt).sum(dim=1)
        if act_id == ACT_TANH:
            d_raw = g * f.unsqueeze(1) * (1.0 - filt * filt)
        else:
            d_raw = g * f.unsqueeze(1)
        d_flat = d_raw.reshape(m, -1)
        grad_h = grad_h + d_flat @ weight[sl]
        grad_weight[sl] = d_flat.T @ h
        if grad_bias is not None:
            grad_bias[sl] = d_flat.sum(dim=0)
    grad_feat = features.new_zeros(features.shape)
    grad_feat.index_add_(0, source, grad_f)
    return grad_h, grad_feat, grad_weight, grad_bias


class _KernelPointXApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, source, w, nn_idx):
        source = source.contiguous().long()
        nn_idx = nn_idx.contiguous().long()
        features = features.contiguous()
        w = w.contiguous()
        messages = _kp_x_forward_triton(features, source, w, nn_idx)
        ctx.save_for_backward(features, source, w, nn_idx)
        return messages

    @staticmethod
    def backward(ctx, grad_msg):
        features, source, w, nn_idx = ctx.saved_tensors
        grad_feat, grad_w = _kp_x_backward_torch(
            grad_msg.contiguous(), features, source, w, nn_idx
        )
        return grad_feat, None, grad_w, None


class _KernelPointApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, source, w, h):
        source = source.contiguous().long()
        h = h.contiguous()
        features = features.contiguous()
        w = w.contiguous()
        messages = _kp_forward_triton(features, source, w, h)
        ctx.save_for_backward(features, source, w, h)
        return messages

    @staticmethod
    def backward(ctx, grad_msg):
        features, source, w, h = ctx.saved_tensors
        grad_feat, grad_w = _kp_backward_torch(
            grad_msg.contiguous(), features, source, w, h
        )
        return grad_feat, None, grad_w, None


class _NeuralFieldApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, features, source, weight, bias, c_out, c_in, act_id):
        source = source.contiguous().long()
        h = h.contiguous()
        features = features.contiguous()
        weight = weight.contiguous()
        has_bias = bias is not None
        if has_bias:
            bias = bias.contiguous()
        else:
            bias = weight.new_empty(0)
        messages = _nf_forward_triton(
            h, features, source, weight, bias if has_bias else None, c_out, c_in, act_id
        )
        ctx.save_for_backward(h, features, source, weight, bias)
        ctx.c_out = int(c_out)
        ctx.c_in = int(c_in)
        ctx.act_id = int(act_id)
        ctx.has_bias = has_bias
        return messages

    @staticmethod
    def backward(ctx, grad_msg):
        h, features, source, weight, bias = ctx.saved_tensors
        if not ctx.has_bias:
            bias = None
        grad_h, grad_feat, grad_weight, grad_bias = _nf_backward_torch(
            grad_msg.contiguous(),
            h,
            features,
            source,
            weight,
            bias,
            ctx.c_out,
            ctx.c_in,
            ctx.act_id,
        )
        return grad_h, grad_feat, None, grad_weight, grad_bias, None, None, None


def kernel_point_x_apply(features, source, w, nn_idx):
    return _KernelPointXApply.apply(features, source, w, nn_idx)


def kernel_point_apply(features, source, w, h):
    return _KernelPointApply.apply(features, source, w, h)


def neural_field_apply(h, features, source, weight, bias, c_out, c_in, act_id):
    return _NeuralFieldApply.apply(
        h, features, source, weight, bias, int(c_out), int(c_in), int(act_id)
    )


def resolve_field_act_id(module: torch.nn.Module) -> int | None:
    if isinstance(module, torch.nn.Tanh):
        return ACT_TANH
    if isinstance(module, torch.nn.Identity):
        return ACT_IDENTITY
    return None
