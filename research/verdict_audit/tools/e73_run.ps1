$ErrorActionPreference = "Stop"
$arms = @("c4b_uniform", "c4_uniform", "all24_uniform")
$corpora = @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")
foreach ($a in $arms) {
  New-Item -ItemType Directory -Force -Path ("runs/diac/e73_soup/" + $a + "/pred") | Out-Null
  foreach ($c in $corpora) {
    Write-Host ("=== " + $a + " " + $c + " ===")
    & .\.venv\Scripts\python.exe diacritizer\scripts\bench.py --ckpt ("runs/diac/e73_soup/" + $a) --config runs/diac/e67_gen/c4b_ctx256.yaml --src $c --out ("runs/diac/e73_soup/" + $a + "/pred/" + $c + ".txt")
  }
}
Write-Host "E73 RUN DONE"
