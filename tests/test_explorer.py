"""tests for webui/explorer.py — graph math validated against real models."""
import _common  # noqa: F401

import artifacts as A
import explorer as E
from _common import run

ROOT = A.ROOT
SMOKE = ROOT / "runs" / "smoke" / "final"
TARGET = ROOT / "runs" / "target" / "final"


def _smoke_dims_header():
    return A.hf_config_dims(SMOKE), A.safetensors_header(SMOKE)


def t_projected_formula_matches_all_three():
    gates = [("smoke", 12_323_072), ("pilot", 100_682_496), ("target", 226_526_208)]
    for phase, want in gates:
        dims = A.yaml_config_dims(A.phase_config(phase))
        got = E.projected_params(dims)
        assert got == want, (phase, got, want)


def t_build_graph_smoke_real():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    # tok + embed + 4 layers x 4 blocks + final norm + lm_head + logits
    assert len(blocks) == 2 + 4 * dims["layers"] + 3, len(blocks)
    assert blocks[0]["id"] == "tok" and blocks[-1]["id"] == "logits"
    embed = blocks[1]
    assert embed["params"] == 32768 * 256 and embed["real"] is True
    attn = next(b for b in blocks if b["kind"] == "attn")
    assert attn["params"] > 0 and attn["tensors"], "real attn tensors missing"
    assert all(b["plain"] and b["tech"] for b in blocks), "info text missing"


def t_build_graph_projected_flagged():
    dims = A.yaml_config_dims(A.phase_config("smoke"))
    blocks = E.build_graph(dims, None, label="smoke (config)", lora=None)
    # tok/logits need no header (real=True); tied lm_head + logits are 0-param
    assert [b["id"] for b in blocks if b["real"]] == ["tok", "logits"]
    assert [b["id"] for b in blocks
            if b["params"] == 0] == ["tok", "lm_head", "logits"]
    assert all(b["params"] > 0 for b in blocks
               if b["id"] not in ("tok", "lm_head", "logits"))


def t_kv_cache_token_target_16kib():
    dims = A.hf_config_dims(TARGET)
    kib = E.kv_cache_kib_per_token(dims)
    assert abs(kib - 16.0) < 0.1, kib  # measured on this machine


def t_shape_trace():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    walk = E.shape_walk(blocks)
    assert walk[0][1] == "[text]" and walk[-1][1] == "[1, 64, 32768]"
    mid = next(w for w in walk if w[0].endswith("attn"))
    assert mid[1] == f"[1, 64, {dims['hidden']}]", mid


def t_render_svg():
    dims, header = _smoke_dims_header()
    blocks = E.build_graph(dims, header, label="smoke/final", lora=None)
    svg = E.render_svg(blocks, selected_id="L1.attn")
    assert "L1.attn" in svg and "data-bid" in svg, "no clickable ids"
    assert "http" not in svg, "external asset leaked into SVG"
    assert svg.count("<svg") == 1


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
