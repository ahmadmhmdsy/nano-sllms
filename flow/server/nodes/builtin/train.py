"""Train node definition: shard-dir in, checkpoint run out.

Owns the webui-parity PRESETS / LR_PRESETS tables moved here verbatim
from the pre-registry config_gen, plus the cross-node ctx-choices check
(the preset ctx_choices must admit the tokenize node seq_len, reached
through the validation context).

Config-order contract (pinned by the goldens + config_gen tests): this
node validates in TWO passes - the walker early pass checks the steps
knob first, before any other participant semantic check (today the
steps check was the first raised message); the full pass runs preset /
lr_preset / ctx checks last, in the historic order.
"""

from flow.server.nodes.base import NodeDefinition, PortSpec, PropSpec
from flow.server.nodes.ports import (B_CHECKPOINT, B_FILE_LIST, CKPT_DIR,
                                     SHARD_DIR, T_CKPT_DIR, T_SHARD_DIR)

# Real key-name source of truth: webui/app.py PRESETS / LR_PRESETS
# (dims copied verbatim from the shipped configs).
PRESETS = {
    "Small (~12M, smoke)": dict(layers=4, hidden=256, heads=4, kv_heads=2,
                                ffn=1024, accum=32, ctx_choices=[256, 512]),
    "Medium (~101M, pilot)": dict(layers=12, hidden=768, heads=12, kv_heads=4,
                                  ffn=2048, accum=32, ctx_choices=[512]),
    "Large (~226M, target)": dict(layers=16, hidden=1024, heads=16, kv_heads=4,
                                  ffn=4096, accum=16, ctx_choices=[512, 1024]),
}
LR_PRESETS = {"Pretrain 4e-4": 4.0e-4, "Conservative 2e-4": 2.0e-4,
              "Fine-tune 3e-5": 3.0e-5}


class TrainNode(NodeDefinition):
    """train: packed shards -> finetuned checkpoint run."""

    kind = "train"
    title = "Train"
    category = "train"
    config_participant = True
    gate = "gpu"
    label_semantic = None
    features = ()
    ports = [
        PortSpec(SHARD_DIR, "in", T_SHARD_DIR, B_FILE_LIST),
        PortSpec(CKPT_DIR, "out", T_CKPT_DIR, B_CHECKPOINT),
    ]
    props = [
        PropSpec("preset", "string", required=True),
        PropSpec("steps", "number", required=True),
        PropSpec("lr_preset", "string", required=True),
    ]
    required_upstream = {SHARD_DIR: ("tokenize", SHARD_DIR)}

    def validate_semantic(self, node, ctx):
        """Two passes per the config-order contract (module docstring).

        early: the steps-knob presence check that historically fired
        before every other reason (pinned by the untuned-flow golden).
        full assumes steps was checked: preset / lr_preset string+known
        checks, then the cross-node ctx-choices check against the
        tokenize node seq_len via ctx.chain.
        """
        props = node.get("props", {})
        if ctx.phase == "early":
            if not isinstance(props, dict) or 'steps' not in props:
                return [
                    "steps: missing - %s node '%s' requires the steps knob"
                    % (self.kind, node.get('id'))]
            return []
        errors = []
        preset_name = props.get("preset")
        if not isinstance(preset_name, str):
            errors.append("preset: expected a string preset name, got %r"
                          % (preset_name,))
        elif preset_name not in PRESETS:
            errors.append("preset: unknown preset %r" % (preset_name,))
        lr_name = props.get("lr_preset")
        if not isinstance(lr_name, str):
            errors.append("lr_preset: expected a string LR preset name, "
                          "got %r" % (lr_name,))
        elif lr_name not in LR_PRESETS:
            errors.append("lr_preset: unknown lr_preset %r" % (lr_name,))
        if errors:
            # Historic order: the ctx-choices check ran only after the
            # preset / lr_preset checks had passed.
            return errors
        p = PRESETS[preset_name]
        tprops = ctx.chain["tokenize"]["props"]
        tctx = int(tprops["seq_len"])
        if tctx not in p["ctx_choices"]:
            errors.append("ctx %d not valid for preset %r (choices %s)"
                          % (tctx, preset_name, p["ctx_choices"]))
        return errors

    def build_section(self, ctx):
        """The model / train / eval blocks (webui build_config parity)."""
        props = ctx.chain[self.kind]["props"]
        p = PRESETS[props["preset"]]
        steps = int(props["steps"])
        tctx = int(ctx.chain["tokenize"]["props"]["seq_len"])
        return {
            "model": {"layers": p["layers"], "hidden": p["hidden"],
                      "heads": p["heads"], "kv_heads": p["kv_heads"],
                      "ffn": p["ffn"], "ctx": tctx, "dropout": 0.0,
                      "tie_embeddings": True},
            "train": {"output_dir": "runs/%s" % ctx.slug,
                      "final_dir": "runs/%s/final" % ctx.slug,
                      "max_steps": steps, "batch": 1, "eval_batch": 4,
                      "accum": p["accum"],
                      "lr": LR_PRESETS[props["lr_preset"]],
                      "scheduler": "cosine",
                      "warmup_steps": max(10, steps // 20),
                      "weight_decay": 0.1, "max_grad_norm": 1.0,
                      "logging_steps": 50, "eval_steps": 500,
                      "save_steps": 500, "save_total_limit": 3,
                      "fp16": True, "grad_ckpt": True,
                      "optim": "adamw_torch", "dataloader_num_workers": 0,
                      "seed": 42},
            "eval": {"max_new_tokens": 64,
                     "prompts": ["def fibonacci(n):", "class Stack:"]},
        }


DEFINITION = TrainNode()
