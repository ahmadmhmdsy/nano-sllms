"""tests for webui/artifacts.py — real files under runs/, headers only."""
import _common  # noqa: F401  (path setup)

import artifacts as A
from _common import run

ROOT = A.ROOT
SMOKE = ROOT / "runs" / "smoke" / "final"
PILOT = ROOT / "runs" / "pilot" / "final"
TARGET = ROOT / "runs" / "target" / "final"
LORA = ROOT / "runs" / "h2_copy_lora" / "final"


def t_header_smoke():
    h = A.safetensors_header(SMOKE)
    assert h, "no safetensors under runs/smoke/final"
    emb = h.get("model.embed_tokens.weight")
    assert emb and emb["shape"] == [32768, 256], f"embed shape {emb}"
    assert emb["dtype"] in ("F16", "F32", "BF16"), emb["dtype"]  # safetensors dtype strings


def t_totals_match_ladder():
    gates = [("smoke", SMOKE, 12_300_000, 700_000),
             ("pilot", PILOT, 100_700_000, 1_200_000),
             ("target", TARGET, 226_500_000, 1_500_000)]
    for name, path, want, tol in gates:
        h = A.safetensors_header(path)
        assert h, f"{name}: no header"
        total = sum(A.tensor_params(h).values())
        assert abs(total - want) <= tol, f"{name}: {total} vs {want} (+/-{tol})"


def t_module_aggregation():
    h = A.safetensors_header(SMOKE)
    mods = A.module_param_totals(h)
    assert mods.get("embed") == 32768 * 256, mods.get("embed")
    assert mods.get("L0.self_attn"), "attn module missing"
    assert mods.get("L3.mlp"), "mlp module missing"
    assert mods.get("model.norm"), "final norm missing (key must be model.norm)"


def t_hf_dims_and_yaml_dims_agree():
    hf = A.hf_config_dims(SMOKE)
    assert hf and hf["layers"] == 4 and hf["hidden"] == 256 and hf["kv_heads"] == 2, hf
    y = A.yaml_config_dims(A.phase_config("smoke"))
    for k in ("layers", "hidden", "heads", "kv_heads", "ffn", "ctx", "vocab", "tie"):
        assert hf[k] == y[k], (k, hf[k], y[k])


def t_lora_note():
    note = A.lora_note(LORA)
    if note is None:
        assert not (LORA / "adapter_config.json").is_file()
        return
    assert note["r"] and note["targets"], note


def t_missing_dir_is_none():
    assert A.safetensors_header(ROOT / "runs" / "definitely-not-here") is None
    assert A.load_json(ROOT / "runs" / "definitely-not-here" / "x.json") is None


if __name__ == "__main__":
    run({k[2:]: v for k, v in sorted(globals().items()) if k.startswith("t_")})
