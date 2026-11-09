"""Model factory for the S/P/T ladder (PLAN.md §3).

One architecture for all phases: plain GQA decoder (RoPE + RMSNorm + SwiGLU +
tied embeddings) via transformers' LlamaForCausalLM. Chosen over the
resources' Flash-Next/GDN skeletons deliberately: those are pseudo-code, this
is battle-tested, supports SDPA attention (Turing-safe; flash-attn needs
Ampere+), and plugs into HF Trainer's checkpoint/resume machinery unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers.configuration_utils import strict
except ImportError:
    from huggingface_hub.dataclasses import strict

from transformers import AutoModelForCausalLM, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from src.gdn import (
    GDN_A_LOG_INIT,
    GDN_CHUNK_SIZE,
    GDN_DT_BIAS_INIT,
    GatedAttention,
    GatedDeltaNet,
    RMSNormGated,
    RMSNormZeroCentered,
    RotaryEmbedding,
    SwiGLUMLP,
)


def build_config(cfg: dict, vocab_size: int | None = None) -> LlamaConfig:
    m = cfg["model"]
    config = LlamaConfig(
        vocab_size=int(vocab_size or cfg["tokenizer"]["vocab_size"]),
        hidden_size=int(m["hidden"]),
        intermediate_size=int(m["ffn"]),
        num_hidden_layers=int(m["layers"]),
        num_attention_heads=int(m["heads"]),
        num_key_value_heads=int(m["kv_heads"]),
        max_position_embeddings=int(m["ctx"]),
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=bool(m.get("tie_embeddings", True)),
        attention_dropout=float(m.get("dropout", 0.0)),
        use_cache=False,  # disabled while training; eval.py re-enables for generation
    )
    # Track B (TASKS row 30): config-gated YaRN re-param, defaults OFF (no
    # rope.yarn block -> every existing config stays byte-identical).
    # Delegates to HF's native yarn (5.16.1 _compute_yarn_parameters):
    # per-frequency-band interpolation ramp (beta_fast 32 / beta_slow 1) +
    # the YaRN attention temperature (mscale-derived attention_factor).
    # Zero extra VRAM — a re-parametrization of the existing rotary (note 04 §1).
    yarn = cfg.get("rope", {}).get("yarn")
    if yarn:
        config.rope_parameters = {
            "rope_type": "yarn",
            # LlamaConfig (5.16) keeps theta inside rope_parameters — there is
            # no rope_theta attribute; 10000.0 matches the constructor constant.
            "rope_theta": 10000.0,
            "factor": float(yarn["factor"]),
            "original_max_position_embeddings": int(yarn["original_max"]),
        }
    return config


def build_model(cfg: dict, vocab_size: int | None = None):
    """Freshly-initialized causal LM for training from scratch.

    SDPA is the fastest attention backend the Quadro RTX 4000 (sm_75) takes.
    arch 'gqa' (default) keeps the original LlamaForCausalLM path byte-for-byte;
    arch 'gdn_hybrid' (Milestone G1) dispatches to build_gdn_hybrid_model.
    """
    arch = str(cfg.get("model", {}).get("arch", "gqa")).lower()
    if arch == "gdn_hybrid":
        model = build_gdn_hybrid_model(cfg, vocab_size)
        model.config.use_cache = False
        return model
    config = build_config(cfg, vocab_size)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    model.config.use_cache = False
    return model


# --- Track A context-eval knobs (TASKS row 29) ------------------------------
# Eval-path-only helpers behind research/distill_survey/adoption_plan.md A.
# train.py never calls them; with knobs off every existing config is
# byte-identical, so the training contract (PLAN 5.3) is untouched.

def apply_rope_scaling(model, rope_type: str = "dynamic-ntk",
                       factor: float = 1.0) -> dict:
    """A1 Dynamic-NTK: swap the loaded model's RoPE to HF dynamic rope.

    transformers 5.x: config.rope_parameters["rope_type"] = "dynamic" makes
    the rotary module recompute inv_freq with the NTK-scaled base
    (base * (factor*seq/max_pos - (factor-1))**(dim/(dim-2))) whenever a
    forward's max position exceeds max_position_embeddings, and reset to the
    original base below it (modeling_rope_utils.dynamic_frequency_update).
    factor=1 is exactly Qwen's shipped dynamic-NTK (note 04).
    Weights untouched. Returns the saved rope_parameters for reset_rope_scaling.
    """
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = model.config
    saved = dict(getattr(config, "rope_parameters", {}) or {})
    params = dict(saved)
    params.setdefault("rope_theta", 10000.0)
    if rope_type in (None, "", "default"):
        params["rope_type"] = "default"
        params.pop("factor", None)
    else:
        params["rope_type"] = ("dynamic" if rope_type in
                               ("dynamic", "dynamic-ntk", "ntk") else rope_type)
        params["factor"] = float(factor)
    config.rope_parameters = params
    # Layers share this one model-level instance; a fresh module re-reads the
    # new rope_type at init and rebuilds inv_freq from the scaled config.
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config)
    return saved


def reset_rope_scaling(model, saved: dict | None = None) -> None:
    """Restore default RoPE after apply_rope_scaling (probe matrices alternate
    knobs within one process; the fresh module rebuilds inv_freq)."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = model.config
    config.rope_parameters = dict(saved or {"rope_theta": 10000.0,
                                            "rope_type": "default"})
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config)


def build_streaming_sink_mask(seq_len: int, window: int, sink_tokens: int = 4,
                              device=None) -> torch.Tensor:
    """A2 StreamingLLM: bool [1,1,L,L] SDPA mask, True = attend (eval-only).

    Query i attends key j iff j <= i (causal) AND (j >= i-window+1 OR j <
    sink_tokens): the last `window` tokens plus `sink` first tokens
    (attention sinks, arXiv 2309.17453). transformers passes prepared 4D
    masks through create_causal_mask untouched and SDPA consumes bool masks
    directly (verified in the installed 5.16.1 source).
    """
    q = torch.arange(seq_len, device=device)
    row, col = q[:, None], q[None, :]
    keep = (col <= row) & ((col >= row - (window - 1)) | (col < sink_tokens))
    return keep.view(1, 1, seq_len, seq_len)


def streaming_position_ids(seq_len: int, window: int, sink_tokens: int = 4,
                           batch_size: int = 1, device=None) -> torch.Tensor:
    """StreamingLLM cache-relative (pos_shift) positions, [batch, L].

    Token i holds position i until the sink+window cache is full; every later
    token enters at the fixed cache length sink+window (exactly each token's
    encode-time RoPE position in streaming decode). Cap = sink+window, not
    -1: the first overflowing token enters a full cache of sink+window entries.
    """
    cap = sink_tokens + window
    pos = torch.arange(seq_len, device=device).clamp(max=cap)
    return pos.unsqueeze(0).expand(batch_size, -1)


def maybe_wrap_peft(model, cfg: dict):
    """Row 10 LoRA hook: wrap with a PeftModel iff the config has a peft block.

    Only adapter params stay trainable; enable_input_require_grads() is
    required so gradients flow into the frozen base when gradient
    checkpointing is on. No-op without the block (full fine-tune path).
    """
    peft = cfg.get("peft")
    if not peft:
        return model
    from peft import LoraConfig, get_peft_model

    lcfg = LoraConfig(
        r=int(peft["r"]),
        lora_alpha=int(peft["lora_alpha"]),
        lora_dropout=float(peft.get("lora_dropout", 0.0)),
        target_modules=list(peft["target_modules"]),
        bias=peft.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def load_finetune_init(model, init_dir) -> dict:
    """Track B (row 30): seed a freshly built model with a saved apex's weights.

    Used to fine-tune an existing checkpoint (e.g. runs/target/final) under a
    NEW config (YaRN rope @ longer ctx) without touching the base in place
    (adoption_plan B base discipline). The architecture must match; only
    config-level fields may differ. Tied-head note: safetensors saved from a
    tied model omit lm_head.weight (same storage as embed_tokens.weight), so
    loading embed_tokens updates the head through the existing tie.
    """
    from safetensors.torch import load_file

    init_dir = Path(init_dir)
    shards = sorted(init_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {init_dir}")
    sd = {}
    for f in shards:
        sd.update(load_file(str(f)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = {"lm_head.weight"}  # tied to embed_tokens (same storage)
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"finetune init mismatch from {init_dir}: "
            f"missing={bad_missing} unexpected={sorted(unexpected)}")
    return {"tensors": len(sd), "missing": sorted(missing),
            "shards": [f.name for f in shards]}


def save_final(trainer, final_dir) -> None:
    """keep-final-forever (§5.3.5): in-memory model is the BEST checkpoint.

    PeftModel -> merge_and_unload() first so final_dir always holds a FULL
    safetensors that eval.py/infer.py load unchanged. Saved outside the
    save_total_limit rotation.
    """
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(str(final_dir))


# ---------------------------------------------------------------------------
# Milestone G1: S-scale GDN / gated-attention hybrid (research/
# gdn_sandbox_design.md sections 2.1-2.5). Config-driven via model.arch;
# 'gqa' (default) leaves the Llama path above untouched.
# ---------------------------------------------------------------------------


@strict
class GDNHybridConfig(PretrainedConfig):
    """Config for the 3:1 Gated DeltaNet / Gated Attention hybrid.

    transformers 5.16 configs are strict dataclasses: class-level defaults +
    __post_init__ (a handwritten 4.x-style __init__ collides with the
    from_dict(**config_dict) reload path).
    """

    model_type = "gdn_hybrid"

    vocab_size: int = 32768
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    # GDN-specific (design section 2.1)
    num_k_heads: int = 4                    # GDN key/QK heads
    num_v_heads: int = 8                    # value heads = k_heads * expand_v
    head_k_dim: int = 64                    # d_k = d_v at S (design 2.1)
    head_v_dim: int = 64
    conv_kernel_size: int = 4
    hybrid_step: int = 1                    # design knob (Qwen 3:1 = 1)
    hybrid_period: int = 4                  # layers per 3:1 block
    use_short_conv: bool = True
    use_sigmoid_gate: bool = True
    use_zero_centered_rmsnorm: bool = True
    gdn_state_dtype: str = "fp32"
    use_cache: bool = False

    def __post_init__(self, **kwargs):
        self.arch = "gdn_hybrid"
        super().__post_init__(**kwargs)


def _init_gdn_module(module) -> None:
    """Design 2.3 inits on top of the default PreTrainedModel ones."""
    if isinstance(module, GatedDeltaNet):
        # b_proj=0 -> beta = sigmoid(0) = 0.5; a_proj=0 + A_log/dt_bias ->
        # alpha = exp(-0.08 * softplus(1)) = 0.90 at step 0 (design 2.3)
        nn.init.zeros_(module.b_proj.weight)
        nn.init.zeros_(module.a_proj.weight)
        with torch.no_grad():
            module.dt_bias.fill_(GDN_DT_BIAS_INIT)
            module.A_log.fill_(GDN_A_LOG_INIT)
    elif isinstance(module, (RMSNormZeroCentered, RMSNormGated)):
        nn.init.zeros_(module.weight)   # zero-centered: applied as 1 + weight
    elif isinstance(module, GatedAttention):
        nn.init.zeros_(module.gate_proj.weight)  # neutral per-head gate 0.5


class GDNHybridDecoderLayer(GradientCheckpointingLayer):
    """Pre-norm block; token mixer is GDN or GA per hybrid_step (design 2.2)."""

    def __init__(self, config: GDNHybridConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # 3:1 block: GA on the last layer of each period-4 group, i.e.
        # [GDN, GDN, GDN, GA] at S (design 2.2 binding statement; the literal
        # (i+1) % hybrid_step formula with hybrid_step=1 would make EVERY
        # layer GA, contradicting it).
        self.layer_type = ("gated_attention"
                           if (layer_idx + 1) % config.hybrid_period == 0
                           else "gated_deltanet")
        if self.layer_type == "gated_attention":
            self.token_mixer = GatedAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                attention_dropout=config.attention_dropout,
                use_sigmoid_gate=config.use_sigmoid_gate,
            )
        else:
            self.token_mixer = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=config.num_k_heads,
                num_v_heads=config.num_v_heads,
                head_k_dim=config.head_k_dim,
                head_v_dim=config.head_v_dim,
                conv_kernel_size=config.conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
                use_short_conv=config.use_short_conv,
                use_sigmoid_gate=config.use_sigmoid_gate,
            )
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormZeroCentered(
            config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=config.use_zero_centered_rmsnorm)
        self.post_attention_layernorm = RMSNormZeroCentered(
            config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=config.use_zero_centered_rmsnorm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "gated_attention":
            hidden_states = self.token_mixer(
                hidden_states, position_embeddings, attention_mask=attention_mask)
        else:
            hidden_states = self.token_mixer(
                hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GDNHybridModel(PreTrainedModel):
    config_class = GDNHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: GDNHybridConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GDNHybridDecoderLayer(config, i)
             for i in range(config.num_hidden_layers)])
        self.norm = RMSNormZeroCentered(
            config.hidden_size, eps=config.rms_norm_eps,
            zero_centered=config.use_zero_centered_rmsnorm)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings, config.rope_theta)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device).unsqueeze(0).expand(bsz, -1)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(position_ids, inputs_embeds.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class GDNHybridForCausalLM(PreTrainedModel, GenerationMixin):
    """Causal LM wrapper: tied head, shifted CE loss, Trainer/generate-ready.

    past_key_values/use_cache are accepted and IGNORED (design section 4 R8):
    the GDN state is not carried across calls; generation recomputes the full
    prefix each step (fine at ctx 256 / 64 new tokens).
    """

    config_class = GDNHybridConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # transformers 5.16: dict of {tied_key: source_key} (not the old list)
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: GDNHybridConfig):
        super().__init__(config)
        self.model = GDNHybridModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        _init_gdn_module(module)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        use_cache=None,
        **kwargs,
    ) -> dict:
        # no-cache generation: feed the FULL sequence every step
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids}

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None,
            hidden_states=None, attentions=None,
        )


def build_gdn_hybrid_config(cfg: dict, vocab_size: int | None = None) -> GDNHybridConfig:
    m = cfg["model"]
    linear_heads = int(m["linear_heads"])
    expand_v = int(m.get("expand_v", 2))
    head_dim = int(m["head_dim_gdn"])
    return GDNHybridConfig(
        vocab_size=int(vocab_size or cfg["tokenizer"]["vocab_size"]),
        hidden_size=int(m["hidden"]),
        intermediate_size=int(m["ffn"]),
        num_hidden_layers=int(m["layers"]),
        num_attention_heads=int(m["heads"]),
        num_key_value_heads=int(m["kv_heads"]),
        max_position_embeddings=int(m["ctx"]),
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=bool(m.get("tie_embeddings", True)),
        attention_dropout=float(m.get("dropout", 0.0)),
        num_k_heads=linear_heads,
        num_v_heads=linear_heads * expand_v,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        conv_kernel_size=4,
        hybrid_step=int(m.get("hybrid_step", 1)),
        hybrid_period=int(m.get("hybrid_period", 3 * int(m.get("hybrid_step", 1)) + 1)),
        use_short_conv=bool(m.get("use_short_conv", True)),
        use_sigmoid_gate=bool(m.get("use_sigmoid_gate", True)),
        use_zero_centered_rmsnorm=bool(m.get("use_zero_centered_rmsnorm", True)),
        gdn_state_dtype=str(m.get("gdn_state_dtype", "fp32")),
    )


def _register_gdn_auto_classes() -> None:
    """Register the hybrid arch in the transformers auto maps (idempotent).

    auto_map alone is NOT enough for eval.py: transformers 5.16.1 hard-gates
    every auto_map/dynamic-code load behind trust_remote_code=True (verified
    2026-09-09, dynamic_module_utils.resolve_trust_remote_code -- no env
    bypass). In-process registration makes
    AutoModelForCausalLM.from_pretrained work with the default
    trust_remote_code=None (has_local_code -> False -> native mapping), for
    every process that imports src.model (train, sanity, any eval-side
    loader that does so).
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        # 5.16 signatures: AutoConfig.register(model_type, config, exist_ok),
        # AutoModel*.register(config_class, model_class, exist_ok)
        AutoConfig.register(GDNHybridConfig.model_type, GDNHybridConfig,
                            exist_ok=True)
        AutoModelForCausalLM.register(GDNHybridConfig, GDNHybridForCausalLM,
                                      exist_ok=True)
    except Exception:  # noqa: BLE001 - registration is best-effort; never break import
        pass


_register_gdn_auto_classes()


def build_gdn_hybrid_model(cfg: dict, vocab_size: int | None = None):
    """Fresh GDN/gated-attention hybrid CausalLM (design 2.2).

    Stamps config.auto_map so eval.py's plain
    AutoModelForCausalLM.from_pretrained(final_dir) resolves the class from
    this repo-local module (ROOT is on eval.py's sys.path).
    """
    config = build_gdn_hybrid_config(cfg, vocab_size)
    if config.gdn_state_dtype != "fp32":
        raise ValueError(
            "gdn_state_dtype must be 'fp32' on sm_75 (design section 4 R1); "
            f"got {config.gdn_state_dtype!r}")
    model = GDNHybridForCausalLM(config)
    config.auto_map = {
        "AutoConfig": "src.model.GDNHybridConfig",
        "AutoModelForCausalLM": "src.model.GDNHybridForCausalLM",
    }
    return model
