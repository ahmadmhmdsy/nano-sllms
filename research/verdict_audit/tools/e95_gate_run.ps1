$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$ckpt = "runs/diac/e73_soup/c4b_uniform"
$config = "runs/diac/e67_gen/c4b_ctx256.yaml"
$corpora = @("fadel_test", "sadeed25", "wikinews2014", "wikinews2024", "abdou_test")
foreach ($c in $corpora) {
  $out = "runs/diac/e95_gate/$c"
  Write-Output "=== $c : decode $(Get-Date -Format o) ==="
  & $py diacritizer/scripts/e94_nospace.py --ckpt $ckpt --config $config --src $c --out-dir $out --window 256 --stride 128 --arms single,nospace
  if ($LASTEXITCODE -ne 0) { Write-Output "DECODE FAILED: $c"; exit 1 }
  Write-Output "=== $c : score (registered splitlines) ==="
  & $py research/verdict_audit/tools/e95_gate_score.py --dir $out --corpus $c --arms single,nospace
  if ($LASTEXITCODE -ne 0) { Write-Output "SCORE FAILED: $c"; exit 1 }
}
Write-Output "=== E-095 ALL DONE $(Get-Date -Format o) ==="
