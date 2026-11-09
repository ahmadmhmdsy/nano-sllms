$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$tools = "research\verdict_audit\tools"
$corpora = @("fadel_test", "sadeed25", "wikinews2024", "wikinews2014", "abdou_test")

function Step($n) { Write-Host ""; Write-Host ("##### " + $n) }

Step "STEP 1/6 - E-076 eval: vote the greedy soup on all five corpora (window 256 / stride 128)"
& "$tools\e76_eval.ps1"
if ($LASTEXITCODE -ne 0) { throw "e76_eval failed (exit $LASTEXITCODE)" }

Step "STEP 2/6 - score E-076 greedy vs uniform"
& $py -u "$tools\score_vote_dir.py" --base runs\diac\e76_greedy\pred `
    --out runs\diac\e76_greedy\verdict.json `
    --title "E-076 GREEDY SOUP voted 256/128 vs E-075 UNIFORM SOUP voted" `
    --ref uniform_voted_gate=0.346827 --ref uniform_voted_abdou=0.341475 `
    2>&1 | Tee-Object -FilePath runs\diac\e76_greedy\e76_score.log
if ($LASTEXITCODE -ne 0) { throw "e76 score failed" }

Step "STEP 3/6 - E-077 voting-window sweep (192/96, 128/64)"
& "$tools\e77_run.ps1"
if ($LASTEXITCODE -ne 0) { throw "e77_run failed (exit $LASTEXITCODE)" }

Step "STEP 4/6 - score E-077 windows"
foreach ($w in @(192, 128)) {
  & $py -u "$tools\score_vote_dir.py" --base "runs\diac\e77_window\w$w" `
      --out "runs\diac\e77_window\w$w\verdict.json" `
      --title "E-077 WINDOW w$w" `
      --ref deployed_w256_vote_gate=0.346827 --ref deployed_w256_vote_abdou=0.341475 `
      2>&1 | Tee-Object -FilePath "runs\diac\e77_window\w$w\score.log"
  if ($LASTEXITCODE -ne 0) { throw "e77 score w$w failed" }
}

Step "STEP 5/6 - E-078 cross-basin LOGIT ensemble L1 (c4b + c4 soups, single pass)"
& "$tools\e78_run.ps1"
if ($LASTEXITCODE -ne 0) { throw "e78_run failed (exit $LASTEXITCODE)" }

Step "STEP 6/6 - score E-078 L1"
& $py -u "$tools\score_vote_dir.py" --base runs\diac\e78_logitens\L1 --kinds single `
    --out runs\diac\e78_logitens\verdict.json `
    --title "E-078 L1 cross-basin logit ensemble (single pass)" `
    --ref c4b_soup_single_gate=0.352209 --ref c4b_soup_single_abdou=0.349044 `
    2>&1 | Tee-Object -FilePath runs\diac\e78_logitens\e78_score.log
if ($LASTEXITCODE -ne 0) { throw "e78 score failed" }

Write-Host ""
Write-Host "##### CHAIN DONE (E-076 eval, E-077, E-078). E-079 training NOT started - it is a separate protected window."