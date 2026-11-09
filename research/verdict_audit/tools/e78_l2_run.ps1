$ErrorActionPreference = "Stop"
$py   = ".\.venv\Scripts\python.exe"
$tool = "research\verdict_audit\tools\e78_logit_ens.py"
$cfg  = "runs/diac/e67_gen/c4b_ctx256.yaml"
$out  = "runs/diac/e78_logitens/L2"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# L2 = logit-mean of the 24 INDIVIDUAL probes (12 c4b + 12 c4), i.e. the exact
# member set that E-073 averaged at the WEIGHT level into all24_uniform. L1
# averaged the two same-basin soups (2 members); L2 is 24 members = 12x L1.
$steps = 5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000
$members = @()
foreach ($s in $steps) { $members += "runs\diac\e70_c4b_ctx256\gate_probe\weights_step$s.pt" }
foreach ($s in $steps) { $members += "runs\diac\e69_c4_ctx256\gate_probe\weights_step$s.pt" }
if ($members.Count -ne 24) { throw "expected 24 members, got $($members.Count)" }
foreach ($m in $members) { if (-not (Test-Path $m)) { throw "missing member $m" } }
$ck = @()
foreach ($m in $members) { $ck += @("--ckpt", $m) }

foreach ($s in @("fadel_test","sadeed25","wikinews2024","wikinews2014","abdou_test")) {
  $t0 = Get-Date
  Write-Host "=== L2 $s  start $($t0.ToString('HH:mm:ss'))"
  & $py -u $tool @ck --config $cfg --src $s --out "$out/$s.single.txt" --tag "L2_24probes"
  if ($LASTEXITCODE -ne 0) { throw "L2 failed $s (exit $LASTEXITCODE)" }
  Write-Host "=== L2 $s  done $((Get-Date).ToString('HH:mm:ss'))  elapsed $([int]((Get-Date) - $t0).TotalSeconds)s"
}
Write-Host "E78 L2 RUN DONE"

& $py -u research\verdict_audit\tools\score_vote_dir.py --base $out --kinds single `
   --out "runs/diac/e78_logitens/L2/verdict.json" `
   --title "E-078 L2 cross-basin LOGIT ensemble of 24 individual probes (single pass)" `
   --ref L1_logit_single_gate=0.351324 --ref L1_logit_single_abdou=0.350703 `
   --ref c4b_soup_single_gate=0.352209 --ref c4b_soup_single_abdou=0.349044 `
   --ref all24_weight_soup_gate=0.449353 --ref all24_weight_soup_abdou=0.439397 `
   2>&1 | Tee-Object -FilePath runs\diac\e78_logitens\L2\e78_l2_score.log
Write-Host "E78 L2 SCORE DONE"