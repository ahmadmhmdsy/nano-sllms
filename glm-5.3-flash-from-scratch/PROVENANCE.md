# Provenance - vendored GLM snapshot

This directory is a vendor snapshot of the public MIT project
https://github.com/vukrosic/glm-5.3-flash-from-scratch
at commit 96d9c7c3bf850ca2500a41b8184b4efb8e07890f (upstream main at the time of import).

Why vendored: the source repository tracked this line as a bare gitlink
(mode 160000, commit a7e7bd45..., no .gitmodules); that pinned tree is not
published upstream and was absent from the machine, so the pointer is unclonable.
The upstream snapshot restores the readable GLM-5.3-Flash-from-scratch model,
pretraining and RLOO RL code; the later local-only extensions (playground
ladders, sandbox runner, aspect LoRA and byte-tokenizer arms) were never
published and are NOT part of this snapshot.

The upstream project is independent of Z.ai and is not affiliated with it.
Its LICENSE (MIT) is included in this directory. Never push to the upstream remote.

