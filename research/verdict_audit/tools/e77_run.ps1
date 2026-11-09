$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$ck = "runs/diac/e73_soup/c4b_uniform"
$cfg = "runs/diac/e67_gen/c4b_ctx256.yaml"
$corpora = @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")
foreach ($arm in @(@(192, 96), @(128, 64))) {
  $win = $arm[0]; $st = $arm[1]
  $out = "runs/diac/e77_window/w$win"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  foreach ($s in $corpora) {
    Write-Host "=== w$win $s"
    & $py diacritizer\scripts\e67_p6_vote.py --ckpt $ck --config $cfg --src $s `
        --out-single "$out/$s.single.txt" --out-vote "$out/$s.vote.txt" `
        --window $win --stride $st
    if ($LASTEXITCODE -ne 0) { throw "vote failed $s w$win (exit $LASTEXITCODE)" }
  }
}
Write-Host "E77 RUN DONE"