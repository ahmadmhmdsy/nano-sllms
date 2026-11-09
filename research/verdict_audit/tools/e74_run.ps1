$ErrorActionPreference = "Stop"
$steps = @(5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000)
New-Item -ItemType Directory -Force -Path "runs/diac/e74_devcheck/pred" | Out-Null
foreach ($s in $steps) {
  Write-Host ("=== abdou step " + $s + " ===")
  & .\.venv\Scripts\python.exe diacritizer\scripts\bench.py --ckpt ("runs/diac/e74_devcheck/step" + $s) --config runs/diac/e67_gen/c4b_ctx256.yaml --src abdou_test --limit 4000 --out ("runs/diac/e74_devcheck/pred/step" + $s + ".txt")
}
Write-Host "E74 RUN DONE"
