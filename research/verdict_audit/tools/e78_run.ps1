$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$tool = "research\verdict_audit\tools\e78_logit_ens.py"
$cfg = "runs/diac/e67_gen/c4b_ctx256.yaml"
$out = "runs/diac/e78_logitens/L1"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($s in @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")) {
  Write-Host "=== L1 $s"
  & $py -u $tool --ckpt runs\diac\e73_soup\c4b_uniform --ckpt runs\diac\e73_soup\c4_uniform `
      --config $cfg --src $s --out "$out/$s.single.txt" --tag "L1_c4b_c4"
  if ($LASTEXITCODE -ne 0) { throw "L1 failed $s (exit $LASTEXITCODE)" }
}
Write-Host "E78 RUN DONE"