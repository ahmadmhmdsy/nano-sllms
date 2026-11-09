"""CPU math validation for the GDN chunkwise kernel (TASKS row 16, step 4).

Pure torch, CPU-only, fp32. Proves on random tensors:
  1. chunk_gated_delta_rule == naive_gated_delta_rule (per-token reference)
     -> max abs err and rel err, require rel < 1e-4;
  2. edge case alpha -> 1 (g == 0): both reduce to the Megatron payload's
     ungated naive delta_rule_recurrence (copied verbatim below);
  3. edge case beta -> 1 (write gate fully open);
  4. sequence length NOT a multiple of chunk_size (L=70, chunk 32);
  5. initial_state propagation matches the per-token reference;
  6. the solve_triangular intra-chunk inverse matches the FLA/Megatron
     iterative forward-substitution loop (verbatim copy of the installed
     transformers qwen3_next torch fallback construction);
  7. finite autograd gradients through the chunked kernel.

Run: & .\\.venv\\Scripts\\python.exe scripts\\test_gdn_math.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.gdn import chunk_gated_delta_rule, naive_gated_delta_rule

torch.manual_seed(0)
REL_TOL = 1e-4
FAILURES = []


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().max().clamp_min(1e-12)
    return float((a - b).abs().max() / denom)


def make_inputs(b=2, h=3, l=64, dk=32, dv=48, g_lo=-2.0, g_hi=-0.01, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, l, dk, generator=gen)
    k = torch.randn(b, h, l, dk, generator=gen)
    v = torch.randn(b, h, l, dv, generator=gen)
    g = g_lo + (g_hi - g_lo) * torch.rand(b, h, l, generator=gen)  # log-decay < 0 (decay in (0, 1))
    beta = torch.sigmoid(torch.randn(b, h, l, generator=gen))
    return q, k, v, g, beta


def check(name, got, want, tol=REL_TOL):
    abs_e = float((got - want).abs().max())
    r_e = rel_err(got, want)
    ok = r_e < tol and math.isfinite(abs_e)
    verdict = "PASS" if ok else "FAIL"
    print(f"{verdict} {name:46s} max_abs={abs_e:.3e} rel={r_e:.3e}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Verbatim reference 1: FLA naive per-token delta rule (Megatron payload,
# research/raw/milestone_g1_megatron_gdn.json -> delta_rule_recurrence;
# the einops rearrange was not used there, plain indexing -- same math).
# ---------------------------------------------------------------------------
def delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d_k = q.shape
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    if initial_state is not None:
        S += initial_state
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", _q, S)
    S = None if output_final_state is False else S
    return o.to(orig_dtype), S


# ---------------------------------------------------------------------------
# Verbatim reference 2: the FLA/Megatron iterative intra-chunk inverse
# (installed transformers qwen3_next torch fallback construction, hub-kernel
# decorator dropped; returns the T our solve_triangular must reproduce).
# ---------------------------------------------------------------------------
def iterative_wy_inverse(k_beta, k_c, glog, chunk_size):
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=k_beta.device), diagonal=0)
    attn = -((k_beta @ k_c.transpose(-1, -2)) *
             (glog.unsqueeze(-1) - glog.unsqueeze(-2)).tril().exp().tril())
    attn = attn.masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    return attn


def main() -> None:
    print(f"torch {torch.__version__} | CPU fp32 | chunk_size=32 | rel_tol={REL_TOL}")

    # ---- 1. main equivalence: chunked kernel vs per-token reference
    q, k, v, g, beta = make_inputs()
    o_c, s_c = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=32,
                                      output_final_state=True)
    o_n, s_n = naive_gated_delta_rule(q, k, v, g, beta)
    check("1. chunk == naive (L=64, 2 chunks)", o_c, o_n)
    check("1b. final state == naive", s_c, s_n)

    # aggressive decay + mixed betas (numerically harder)
    q2, k2, v2, g2, beta2 = make_inputs(g_lo=-6.0, g_hi=-0.001, seed=7)
    o_c2, _ = chunk_gated_delta_rule(q2, k2, v2, g2, beta2, chunk_size=32)
    o_n2, _ = naive_gated_delta_rule(q2, k2, v2, g2, beta2)
    check("1c. chunk == naive (decays in [e^-6, e^-0.001])", o_c2, o_n2)

    # single chunk (L == chunk_size) and one-token inputs
    q3, k3, v3, g3, beta3 = make_inputs(l=32, seed=3)
    o_c3, _ = chunk_gated_delta_rule(q3, k3, v3, g3, beta3, chunk_size=32)
    o_n3, _ = naive_gated_delta_rule(q3, k3, v3, g3, beta3)
    check("1d. chunk == naive (L == chunk_size == 32)", o_c3, o_n3)
    q4, k4, v4, g4, beta4 = make_inputs(l=1, seed=4)
    o_c4, _ = chunk_gated_delta_rule(q4, k4, v4, g4, beta4, chunk_size=32)
    o_n4, _ = naive_gated_delta_rule(q4, k4, v4, g4, beta4)
    check("1e. chunk == naive (L == 1)", o_c4, o_n4)

    # ---- 2. edge case: alpha -> 1 (g == 0) == the payload's ungated naive rule
    q5, k5, v5, _, beta5 = make_inputs(seed=11)
    g5 = torch.zeros_like(beta5)
    o_c5, _ = chunk_gated_delta_rule(q5, k5, v5, g5, beta5, chunk_size=32,
                                     use_qk_l2norm_in_kernel=False)
    o_payload, _ = delta_rule_recurrence(q5, k5, v5, beta5)
    check("2. alpha->1, chunk == payload naive (ungated)", o_c5, o_payload)

    # ---- 3. edge case: beta -> 1 (write gate fully open)
    q6, k6, v6, g6, _ = make_inputs(seed=13)
    beta6 = torch.ones_like(g6)
    o_c6, _ = chunk_gated_delta_rule(q6, k6, v6, g6, beta6, chunk_size=32)
    o_n6, _ = naive_gated_delta_rule(q6, k6, v6, g6, beta6)
    check("3. beta->1, chunk == naive", o_c6, o_n6)

    # ---- 4. sequence length NOT a multiple of chunk_size
    q7, k7, v7, g7, beta7 = make_inputs(l=70, seed=17)
    o_c7, s_c7 = chunk_gated_delta_rule(q7, k7, v7, g7, beta7, chunk_size=32,
                                        output_final_state=True)
    o_n7, s_n7 = naive_gated_delta_rule(q7, k7, v7, g7, beta7)
    check("4. chunk == naive (L=70, tail chunk 6)", o_c7, o_n7)
    check("4b. final state (L=70)", s_c7, s_n7)

    # ---- 5. initial_state propagation
    gen = torch.Generator().manual_seed(23)
    s0 = torch.randn(2, 3, 32, 48, generator=gen)
    o_c8, s_c8 = chunk_gated_delta_rule(q7, k7, v7, g7, beta7, chunk_size=32,
                                        initial_state=s0, output_final_state=True)
    o_n8, s_n8 = naive_gated_delta_rule(q7, k7, v7, g7, beta7, initial_state=s0)
    check("5. initial_state, chunk == naive (out)", o_c8, o_n8)
    check("5b. initial_state, chunk == naive (state)", s_c8, s_n8)

    # ---- 6. solve_triangular inverse == FLA/Megatron iterative loop
    q9, k9, v9, g9, beta9 = make_inputs(l=64, seed=29)
    k_c = k9.reshape(2, 3, 2, 32, 32)
    k_beta = (k9 * beta9.unsqueeze(-1)).reshape(2, 3, 2, 32, 32)
    glog = g9.reshape(2, 3, 2, 32).cumsum(-1)
    t_iter = iterative_wy_inverse(k_beta, k_c, glog, 32)
    decay_mask = (glog.unsqueeze(-1) - glog.unsqueeze(-2)).tril().exp().tril()
    system = ((k_beta @ k_c.transpose(-1, -2)) * decay_mask).tril(diagonal=-1)
    eye = torch.eye(32, dtype=system.dtype)
    t_solve = torch.linalg.solve_triangular(eye + system, eye.expand_as(system),
                                            upper=False, unitriangular=True)
    check("6. WY inverse: solve_triangular == iterative loop", t_solve, t_iter)

    # ---- 7. finite autograd grads through the chunked kernel
    q10 = torch.randn(2, 3, 64, 32, requires_grad=True)
    k10 = torch.randn(2, 3, 64, 32, requires_grad=True)
    v10 = torch.randn(2, 3, 64, 48, requires_grad=True)
    raw_g = torch.randn(2, 3, 64, requires_grad=True)
    g10 = -torch.abs(raw_g) - 0.01            # leaf: raw_g
    raw_beta = torch.randn(2, 3, 64, requires_grad=True)
    beta10 = torch.sigmoid(raw_beta)          # leaf: raw_beta
    o10, _ = chunk_gated_delta_rule(q10, k10, v10, g10, beta10, chunk_size=32)
    loss = (o10 * torch.randn_like(o10)).pow(2).mean()
    loss.backward()
    names = ["q", "k", "v", "g", "beta"]
    grads = [q10.grad, k10.grad, v10.grad, raw_g.grad, raw_beta.grad]
    ok = all(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0
             for x in grads)
    norms = [f"{float(x.abs().sum()):.3e}" if x is not None else "None" for x in grads]
    grad_map = dict(zip(names, norms))
    verdict = "PASS" if ok else "FAIL"
    print(f"{verdict} 7. finite nonzero autograd grads          "
          f"grad abs-sum per input = {grad_map}")
    if not ok:
        FAILURES.append("7. grads")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)}: {FAILURES})")
        raise SystemExit(1)
    print("RESULT: PASS (all CPU math checks within rel < 1e-4)")


if __name__ == "__main__":
    main()
