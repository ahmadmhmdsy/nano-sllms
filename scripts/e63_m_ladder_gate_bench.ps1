# Optimized 2026-09-23: --batch 512 (length-equal piece batching in bench.py;
$ErrorActionPreference = "Continue"
cd E:\python_projects\nano_SLMs
$py = ".venv\Scripts\python.exe"
foreach ($ph in "e63_m1_tr","e63_m2_tr","e63_m3_tr") {
  $ckpt = "runs\diac\$ph\final"
  $cfg = "runs\diac\$ph\final\config.yaml"
  New-Item -ItemType Directory -Force -Path runs\diac\$ph\gate_bench | Out-Null
  foreach ($g in "fadel_test","sadeed25","wikinews2024","wikinews2014","abdou_test") {
    Write-Host "=== $ph GATE $g ==="
    & $py -u diacritizer\scripts\bench.py --ckpt $ckpt --src $g --config $cfg --batch 512 --out runs\diac\$ph\gate_bench\$g.prd.txt 2>&1 | Tee-Object -Append -FilePath runs\e63_m_ladder_gates.log
    & $py -u scratch\e63_gate_score.py runs\diac\$ph\gate_bench\$g 2>&1 | Tee-Object -Append -FilePath runs\e63_m_ladder_gates.log
  }
}
Write-Host "=== M-LADDER GATE BENCH COMPLETE ==="