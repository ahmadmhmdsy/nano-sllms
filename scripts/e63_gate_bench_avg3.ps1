# E-63 avg-3 checkpoint model: 4+1 gate bench (full multi-metric)
$ErrorActionPreference = "Continue"
cd E:\python_projects\nano_SLMs
$py = ".venv\Scripts\python.exe"
$phase = "e63_avg3_tr"
$ckpt = "runs\diac\e63_avg3_tr\final"
$cfg = "runs\diac\e63_avg3_tr\final\config.yaml"
New-Item -ItemType Directory -Force -Path runs\diac\e63_avg3_tr\gate_bench | Out-Null
foreach ($g in "fadel_test","sadeed25","wikinews2024","wikinews2014","abdou_test") {
  Write-Host "=== GATE $g ==="
  & $py -u diacritizer\scripts\bench.py --ckpt $ckpt --src $g --config $cfg --out runs\diac\e63_avg3_tr\gate_bench\$g.prd.txt 2>&1 | Tee-Object -Append -FilePath runs\e63_avg3_gates.log
  & $py -u scratch\e63_gate_score.py runs\diac\e63_avg3_tr\gate_bench\$g 2>&1 | Tee-Object -Append -FilePath runs\e63_avg3_gates.log
}
Write-Host "=== AVG3 GATE BENCH COMPLETE ==="