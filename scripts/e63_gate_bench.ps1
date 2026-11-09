# E-63 final 4+1 gate bench: strict bench + full multi-metric (DER/DER_nocase/WER/text_preservation)
$ErrorActionPreference = "Continue"
cd E:\python_projects\nano_SLMs
$py = ".venv\Scripts\python.exe"
$ckpt = "runs\diac\stage3best_ls\final"
$cfg = "runs\diac\stage3best_ls\final\config.yaml"
if (-not (Test-Path $cfg)) { $cfg = "configs\diac_stage3best_ls.yaml" }
New-Item -ItemType Directory -Force -Path runs\diac\stage3best_ls\gate_bench | Out-Null
foreach ($g in "fadel_test","sadeed25","wikinews2024","wikinews2014","abdou_test") {
  Write-Host "=== GATE $g ==="
  & $py -u diacritizer\scripts\bench.py --ckpt $ckpt --src $g --config $cfg --out runs\diac\stage3best_ls\gate_bench\$g.prd.txt 2>&1 | Tee-Object -Append -FilePath runs\e63_gates.log
  & $py -u scratch\e63_gate_score.py runs\diac\stage3best_ls\gate_bench\$g 2>&1 | Tee-Object -Append -FilePath runs\e63_gates.log
}
Write-Host "=== GATE BENCH COMPLETE ==="