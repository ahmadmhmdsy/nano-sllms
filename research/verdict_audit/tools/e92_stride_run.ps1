# E-092 vote-density screen: fixed window 256, stride {128,64,32}, abdou_test first 400 lines.
# *** FEW-MINUTE SCREEN, NOT A FULL GATE RUN. ***
$ErrorActionPreference = "Stop"
$base = "runs\diac\e92_stride"
foreach ($s in 128, 64, 32) {
  $out = Join-Path $base "$s"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Write-Host "=== stride $s ==="
  & .\.venv\Scripts\python.exe diacritizer\scripts\e67_p6_vote.py `
      --ckpt runs/diac/e73_soup/c4b_uniform `
      --config runs/diac/e67_gen/c4b_ctx256.yaml `
      --src abdou_test --limit 400 `
      --window 256 --stride $s `
      --out-single "$out/abdou_test.single.txt" `
      --out-vote "$out/abdou_test.vote.txt"
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at stride $s"; exit 1 }
}
Write-Host "E92 SWEEP DONE"
