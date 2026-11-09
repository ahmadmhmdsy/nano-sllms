# E-093: three arms (single / vote / centered) from ONE loaded model.
# *** FEW-MINUTE SCREEN, NOT A FULL GATE RUN. ***
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\python.exe diacritizer\scripts\e93_centered.py `
    --ckpt runs/diac/e73_soup/c4b_uniform `
    --config runs/diac/e67_gen/c4b_ctx256.yaml `
    --src abdou_test --limit 400 `
    --window 256 --stride 128 `
    --out-dir runs/diac/e93_centered
if ($LASTEXITCODE -ne 0) { Write-Host "E93 FAILED"; exit 1 }
Write-Host "E93 DONE"
