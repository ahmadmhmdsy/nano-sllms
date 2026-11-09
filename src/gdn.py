"""Self-contained Gated DeltaNet + Gated Attention blocks (Milestone G1).

Deterministic pure-PyTorch port of torch_chunk_gated_delta_rule -- the
algorithm Megatron-LM uses in deterministic_mode
(research/raw/milestone_g1_megatron_gdn.json, megatron/core/ssm/gated_delta_net.py;
identical math to the installed transformers 5.16.1 qwen3_next torch
fallback) -- per research/gdn_sandbox_design.md section 2.4:

* pure PyTorch only: no triton, no fla, no flash-attn, no HF hub-kernel
  decorators (design section 4 R7);
* chunk_size 32, WY representation, the intra-chunk inverse computed in
  fp32 via torch.linalg.solve_triangular on the strictly-lower decayed
  system matrix (design R2; algebraically identical to the FLA /
  Megatron forward-substitution loop, proven in scripts/test_gdn_math.py);
* per-head decay  alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias));
* write gate      beta_t  = sigmoid(b_t);
* short depthwise causal conv (k=4) on q/k/v + SiLU, L2-normed q/k;
* sigmoid output gate + zero-centered RMSNormGated (design R5);
* ALL delta-rule state/math in fp32 with fp16 IO casts (design R1/R2);
  the fp32 section runs under torch.autocast(..., enabled=False) so an
  ambient fp16 autocast can never downcast the state matmuls.

Kernel I/O convention: q/k [B, H, L, Dk], v [B, H, L, Dv], g/beta [B, H, L].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GDN_CHUNK_SIZE = 32                 # design 2.4 / Megatron deterministic path
GDN_A_LOG_INIT = math.log(0.08)     # decay ~= 0.90 at step 0 (design 2.3)
GDN_DT_BIAS_INIT = 1.0
L2NORM_EPS = 1e-6


def _fp32_guard(device_type: str):
    """Disable an ambient autocast inside the fp32 GDN state math.

    torch.autocast would silently downcast every matmul below to fp16 even
    when the inputs are fp32; design R1/R2 requires the delta-rule state
    to stay fp32 on sm_75. On CPU the guard is a no-op.
    """
    if device_type not in ("cuda", "cpu", "xpu", "mps", "npu"):
        device_type = "cpu"
    return torch.autocast(device_type=device_type, enabled=False)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = L2NORM_EPS) -> torch.Tensor:
    """FLA-aligned L2 norm (same math as the qwen3_next l2norm helper)."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ---------------------------------------------------------------------------
# The delta-rule recurrence: naive per-token reference + chunkwise kernel
# ---------------------------------------------------------------------------

def naive_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
    initial_state: torch.Tensor | None = None,
):
    """Per-token fp32 reference recurrence (validation ground truth).

    Mirrors torch_recurrent_gated_delta_rule from the installed
    transformers qwen3_next torch fallback == the Megatron payload's naive
    delta_rule_recurrence extended with the per-head decay:

        S~_t = alpha_t * S_{t-1}
        v~_t = v_t - S~_t^T k_t              (decayed erase)
        S_t  = S~_t + beta_t * k_t v~_t^T    (gated rank-1 write)
        y_t  = S_t^T q_t
    """
    with _fp32_guard(query.device.type):
        orig_dtype = query.dtype
        q, k, v, beta, g = [x.float() for x in (query, key, value, beta, g)]
        if use_qk_l2norm_in_kernel:
            q = l2norm(q)
            k = l2norm(k)
        bsz, heads, seq_len, d_k = q.shape
        if initial_state is None:
            state = q.new_zeros(bsz, heads, d_k, v.shape[-1])
        else:
            state = initial_state.float()
        q = q * (d_k ** -0.5)
        out = torch.zeros_like(v)
        for t in range(seq_len):
            state = state * g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            kv_mem = (state * k[:, :, t].unsqueeze(-1)).sum(dim=-2)
            delta = (v[:, :, t] - kv_mem) * beta[:, :, t].unsqueeze(-1)
            state = state + k[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
            out[:, :, t] = (state * q[:, :, t].unsqueeze(-1)).sum(dim=-2)
        return out.to(orig_dtype), state


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = GDN_CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
):
    """Deterministic chunkwise Gated DeltaNet (WY representation).

    Pure-PyTorch port of Megatron's torch_chunk_gated_delta_rule / the
    transformers qwen3_next torch fallback. The only intentional
    difference from the reference: FLA's iterative forward-substitution
    loop for the intra-chunk inverse is replaced by an fp32
    torch.linalg.solve_triangular on the same strictly-lower system
    (design R2).

    Recurrence per head (alpha_t = exp(g_t)):
        S~_t = alpha_t S_{t-1};  v~_t = v_t - S~_t^T k_t;
        S_t  = S~_t + beta_t k_t v~_t^T;  y_t = S_t^T q_t
    All math in fp32; the output is cast back to the input dtype.
    """
    with _fp32_guard(query.device.type):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query.float())
            key = l2norm(key.float())
        query, key, value, beta, g = [
            x.float() for x in (query, key, value, beta, g)
        ]

        batch, heads, seq_len, d_k = key.shape
        d_v = value.shape[-1]

        pad = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad:
            query = F.pad(query, (0, 0, 0, pad))
            key = F.pad(key, (0, 0, 0, pad))
            value = F.pad(value, (0, 0, 0, pad))
            beta = F.pad(beta, (0, pad))
            g = F.pad(g, (0, pad))
        total_len = seq_len + pad
        n_chunks = total_len // chunk_size

        scale = d_k ** -0.5
        query = query * scale

        # Intra-chunk WY factor. Per chunk, the write vectors u_t = beta_t * v~_t
        # satisfy (I + M) U = B with the strictly-lower decayed kernel matrix
        #     M[t, s] = beta_t * exp(glog_t - glog_s) * <k_t, k_s>   (s < t)
        # where glog is the inclusive in-chunk cumsum of g. Solve T = (I + M)^-1
        # in fp32 (design R2), then U = T B minus the incoming-state correction.
        q_c, k_c, v_c = (x.reshape(batch, heads, n_chunks, chunk_size, -1)
                         for x in (query, key, value))
        k_beta = (key * beta.unsqueeze(-1)).reshape(
            batch, heads, n_chunks, chunk_size, d_k)
        glog = g.reshape(batch, heads, n_chunks, chunk_size).cumsum(dim=-1)
        decay_mask = ((glog.unsqueeze(-1) - glog.unsqueeze(-2)).tril()
                      .exp().tril())                    # [B,H,NC,C,C] incl-diag
        kk = k_beta @ k_c.transpose(-1, -2)             # [B,H,NC,C,C]
        system = (kk * decay_mask).tril(diagonal=-1)    # strictly lower, = M
        eye = torch.eye(chunk_size, dtype=system.dtype, device=system.device)
        # fp32 triangular inverse (design R2); I + M is unit lower triangular
        wy = torch.linalg.solve_triangular(
            eye + system, eye.expand_as(system), upper=False, unitriangular=True)
        u0 = wy @ (value * beta.unsqueeze(-1)).reshape(
            batch, heads, n_chunks, chunk_size, d_v)
        w_dec = wy @ ((k_beta * glog.exp().unsqueeze(-1)).reshape(
            batch, heads, n_chunks, chunk_size, d_k))

        # ---- loop-invariant operands, hoisted out of the chunk scan ----
        # The per-chunk state recurrence is affine in the incoming state, so
        # every operand that does not depend on it is computed once (batched)
        # instead of once per Python iteration. On this GPU the scan is
        # CPU-dispatch-bound (~17 kernel launches per chunk), so cutting the
        # per-chunk launch count is the lever that matters; the math is
        # unchanged (each batched op is the same fp32 computation the
        # per-chunk loop performed, just issued as a single launch).
        attn_intra = (q_c @ k_c.transpose(-1, -2)) * decay_mask   # [B,H,NC,C,C]
        glog_exp = glog.exp()                                     # [B,H,NC,C]
        glog_last_exp = glog[:, :, :, -1].exp()                   # [B,H,NC]
        k_decay = k_c * (glog[:, :, :, -1].unsqueeze(-1)
                         - glog).exp().unsqueeze(-1)              # [B,H,NC,C,Dk]
        q_glog = q_c * glog_exp.unsqueeze(-1)                     # [B,H,NC,C,Dk]
        # Concatenate operands that share a matmul so the scan issues two
        # launches per chunk instead of four (same math, fewer dispatches):
        #   m1 = [w_dec | q_glog] @ state         (erase + inter-chunk read)
        #   m2 = [attn_intra | k_decay^T] @ v_new (intra-chunk + state write)
        wq = torch.cat([w_dec, q_glog], dim=3)                    # [B,H,NC,2C,Dk]
        ak = torch.cat([attn_intra, k_decay.transpose(-1, -2)], dim=3)  # [B,H,NC,C+Dk,C]

        state = (q_c.new_zeros(batch, heads, d_k, d_v)
                 if initial_state is None else initial_state.float())
        out = torch.empty(batch, heads, n_chunks, chunk_size, d_v,
                          dtype=query.dtype, device=query.device)
        for i in range(n_chunks):
            m1 = wq[:, :, i] @ state                              # [B,H,2C,Dv]
            v_new = u0[:, :, i] - m1[:, :, :chunk_size]           # erase incoming state
            y_inter = m1[:, :, chunk_size:]
            m2 = ak[:, :, i] @ v_new                              # [B,H,C+Dk,Dv]
            out[:, :, i] = y_inter + m2[:, :, :chunk_size]
            state = (state * glog_last_exp[:, :, i].unsqueeze(-1).unsqueeze(-1)
                     + m2[:, :, chunk_size:])

        out = out.reshape(batch, heads, total_len, d_v)[:, :, :seq_len]
        final_state = state if output_final_state else None
        return out.to(initial_dtype), final_state


# ---------------------------------------------------------------------------
# Norms (zero-centered; hub-kernel decorators dropped per design R7)
# ---------------------------------------------------------------------------


class RMSNormZeroCentered(nn.Module):
    """Qwen3-Next zero-centered RMSNorm (weight stored as 0, applied as 1+w)."""

    def __init__(self, dim: int, eps: float = 1e-5, zero_centered: bool = True):
        super().__init__()
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _fp32_guard(x.device.type):
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            w = self.weight.float()
            h = h * ((1.0 + w) if self.zero_centered else w)
        return h.type_as(x)


class RMSNormGated(nn.Module):
    """Zero-centered gated RMSNorm (Qwen3NextRMSNormGated math, hub-kernel
    decorator dropped per design R7).

    Activation is SIGMOID, per the design (research section 3.1 / design
    section 1.2: the output uses a sigmoid gate, which beats SiLU); the
    shipped HF class defaults to silu -- the design overrides it.
    """

    def __init__(self, dim: int, eps: float = 1e-5, zero_centered: bool = True):
        super().__init__()
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor,
                gate: torch.Tensor | None = None) -> torch.Tensor:
        with _fp32_guard(hidden_states.device.type):
            input_dtype = hidden_states.dtype
            h = hidden_states.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            w = self.weight.float()
            h = h * ((1.0 + w) if self.zero_centered else w)
            if gate is not None:
                h = h * torch.sigmoid(gate.float())
        return h.to(input_dtype)


# ---------------------------------------------------------------------------
# GDN layer (Megatron GatedDeltaNet, single-GPU port)
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (design 2.2 / 2.3).

    Separate q/k/v/g(b-a) projections + depthwise causal conv k=4 + fp32
    chunkwise delta rule + zero-centered RMSNormGated. beta/alpha are per
    VALUE head (Qwen3-Next/Megatron convention); q/k are repeat_interleave'd
    to the v-head count (expand_v).
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-5,
        use_short_conv: bool = True,
        use_sigmoid_gate: bool = True,
    ):
        super().__init__()
        self.use_short_conv = use_short_conv
        self.use_sigmoid_gate = use_sigmoid_gate
        assert num_v_heads % num_k_heads == 0, "expand_v must be an integer"
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # sigmoid output gate
        self.b_proj = nn.Linear(hidden_size, num_v_heads, bias=False)     # write gate beta
        self.a_proj = nn.Linear(hidden_size, num_v_heads, bias=False)     # decay input
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=conv_kernel_size, groups=self.conv_dim,
            padding=conv_kernel_size - 1,   # causal: crop the tail (Megatron pattern)
        )
        self.dt_bias = nn.Parameter(torch.full((num_v_heads,), GDN_DT_BIAS_INIT))
        self.A_log = nn.Parameter(torch.full((num_v_heads,), GDN_A_LOG_INIT))
        self.o_norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # padding states are zeroed before the projections (Megatron/Qwen3-Next)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)

        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        z = self.g_proj(hidden_states)
        b = self.b_proj(hidden_states)
        a = self.a_proj(hidden_states)

        # ---- fp32 core: conv + decay/gates + chunked delta rule + gated norm
        with _fp32_guard(hidden_states.device.type):
            if self.use_short_conv:
                mixed = torch.cat(
                    (q.reshape(bsz, seq_len, self.key_dim),
                     k.reshape(bsz, seq_len, self.key_dim),
                     v.reshape(bsz, seq_len, self.value_dim)), dim=-1).float()
                conv = self.conv1d(mixed.transpose(1, 2))[..., :seq_len]  # causal crop
                conv = F.silu(conv).transpose(1, 2)
                cq, ck, cv = torch.split(
                    conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            else:
                cq = q.reshape(bsz, seq_len, self.key_dim).float()
                ck = k.reshape(bsz, seq_len, self.key_dim).float()
                cv = v.reshape(bsz, seq_len, self.value_dim).float()
            q = cq.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
            k = ck.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
            v = cv.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
            z = z.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

            beta = torch.sigmoid(b.float())
            # .float() guards A_log.exp() against fp16 -inf (Qwen3-Next comment)
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

            repeat = self.num_v_heads // self.num_k_heads
            if repeat > 1:
                q = q.repeat_interleave(repeat, dim=2)
                k = k.repeat_interleave(repeat, dim=2)

            core_attn_out, _ = chunk_gated_delta_rule(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                g=g.transpose(1, 2), beta=beta.transpose(1, 2),
                chunk_size=GDN_CHUNK_SIZE, use_qk_l2norm_in_kernel=True,
            )                                             # [B, V, L, Dv]
            core_attn_out = core_attn_out.transpose(1, 2)  # [B, L, V, Dv]
            gate = z.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim) \
                if self.use_sigmoid_gate else None      # [B, L, V, Dv]
            out = self.o_norm(core_attn_out, gate)
            out = out.reshape(bsz, seq_len, self.value_dim)

        # ---- fp16 IO: back to the ambient (possibly autocast) dtype for out_proj
        return self.out_proj(out.to(hidden_states.dtype))


# ---------------------------------------------------------------------------
# Gated Attention layer (GQA SDPA + per-head sigmoid output gate)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Default RoPE (theta 10000); fp32 cos/sin, cast to the query dtype."""

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, L] -> cos/sin [B, L, head_dim]
        freqs = position_ids[:, :, None].float() * self.inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        with _fp32_guard(position_ids.device.type):
            cos, sin = emb.cos(), emb.sin()
        return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    return hidden[:, :, None, :, :].expand(
        hidden.shape[0], hidden.shape[1], n_rep, hidden.shape[2], hidden.shape[3]
    ).reshape(hidden.shape[0], hidden.shape[1] * n_rep, hidden.shape[2], hidden.shape[3])


class GatedAttention(nn.Module):
    """GQA SDPA attention + per-head sigmoid output gate before o_proj
    (design 1.2 / 2.3: the gate is what kills the attention sink)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
        use_sigmoid_gate: bool = True,
    ):
        super().__init__()
        self.use_sigmoid_gate = use_sigmoid_gate
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim ** -0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, num_attention_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, scale=self.scaling,
                dropout_p=dropout_p, is_causal=True,
            )
        else:
            # additive causal + padding mask (2D [B, L] mask)
            min_val = torch.finfo(query.dtype).min
            causal = torch.triu(
                torch.full((seq_len, seq_len), min_val,
                           dtype=query.dtype, device=query.device),
                diagonal=1,
            )
            pad = (1.0 - attention_mask[:, None, :].to(query.dtype)) * min_val
            attn_mask = causal[None, None, :, :] + pad[:, :, None, :]
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, scale=self.scaling,
                dropout_p=dropout_p,
            )

        if self.use_sigmoid_gate:
            gate = torch.sigmoid(self.gate_proj(hidden_states))            # [B, L, H]
            attn_output = attn_output * gate.transpose(1, 2).unsqueeze(-1)  # [B, H, L, D]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# SwiGLU FFN (identical structure to the GQA baseline's Llama MLP)
# ---------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
