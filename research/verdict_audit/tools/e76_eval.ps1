$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$ck = "runs/diac/e76_greedy/greedy"
$cfg = "runs/diac/e67_gen/c4b_ctx256.yaml"
$out = "runs/diac/e76_greedy/pred"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($s in @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")) {
  Write-Host "=== voting $s"
  & $py diacritizer\scripts\e67_p6_vote.py --ckpt $ck --config $cfg --src $s `
      --out-single "$out/$s.single.txt" --out-vote "$out/$s.vote.txt" `
      --window 256 --stride 128
  if ($LASTEXITCODE -ne 0) { throw "vote failed for $s (exit $LASTEXITCODE)" }
}
Write-Host "E76 EVAL RUN DONE"