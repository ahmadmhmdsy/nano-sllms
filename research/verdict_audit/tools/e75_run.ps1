$ErrorActionPreference = "Stop"
$srcs = @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")
New-Item -ItemType Directory -Force -Path "runs/diac/e75_vote" | Out-Null
foreach ($s in $srcs) {
  Write-Host ("=== soup+vote " + $s + " ===")
  & .\.venv\Scripts\python.exe diacritizer\scripts\e67_p6_vote.py --ckpt runs/diac/e73_soup/c4b_uniform --config runs/diac/e67_gen/c4b_ctx256.yaml --src $s --out-single ("runs/diac/e75_vote/" + $s + ".single.txt") --out-vote ("runs/diac/e75_vote/" + $s + ".vote.txt") --window 256 --stride 128
}
Write-Host "E75 RUN DONE"
