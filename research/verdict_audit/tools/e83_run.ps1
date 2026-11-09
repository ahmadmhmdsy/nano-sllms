# E-083 - weighted cross-basin LOGIT ensemble: one pass over 5 corpora, then score all schemes.
$ErrorActionPreference = "Stop"
$py  = ".\.venv\Scripts\python.exe"
$cfg = "runs\diac\e67_gen\c4b_ctx256.yaml"
$out = "runs\diac\e83_weighted"
$log = "$out\e83_run.log"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$schemes = "base24,c4b12,c412,b2,b4,b8,ramp_b1"

# preflight: every declared member must exist, else the arm set is not the one registered
$steps = 5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000
$missing = 0
foreach ($s in $steps) {
  foreach ($p in @("runs\diac\e70_c4b_ctx256\gate_probe\weights_step$s.pt", "runs\diac\e69_c4_ctx256\gate_probe\weights_step$s.pt")) {
    if (-not (Test-Path $p)) { Write-Host "PREFLIGHT MISSING $p"; $missing++ }
  }
}
if ($missing -gt 0) { throw "E-083 aborted: $missing member files missing" }
Write-Host "PREFLIGHT OK: 24 members present; schemes=$schemes"

$t_all = Get-Date
foreach ($src in @("fadel_test","sadeed25","wikinews2024","wikinews2014","abdou_test")) {
  $t0 = Get-Date
  $cmd = "research\verdict_audit\tools\e83_logit_weighted.py"
  & $py -u $cmd --config $cfg --src $src --out-root $out --batch 64 --device cuda --schemes $schemes 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { throw "E-083 driver failed on $src (exit $LASTEXITCODE)" }
  $dt = [int]((Get-Date) - $t0).TotalSeconds
  Write-Host ("E083 {0} elapsed {1}s at {2}" -f $src, $dt, (Get-Date -Format "HH:mm:ss"))
}
Write-Host ("E083 INFERENCE DONE total {0}s" -f [int]((Get-Date) - $t_all).TotalSeconds)

& $py -u research\verdict_audit\tools\e83_score_all.py 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { throw "E-083 scoring failed (exit $LASTEXITCODE)" }
Write-Host "E83 DONE"
