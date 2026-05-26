# mock 检测/追踪 + 全链路真实 VL（关键帧筛选 / 轨迹质检 / 全局关系 / 片段关系 LLM）
# $env:DASHSCOPE_API_KEY = "sk-你的key"
# .\scripts\run_with_vl.ps1

param(
    [string]$ApiKey = "",
    [string]$Video = "data/validation_dummy.mp4",
    [string]$RunDir = "runs/live_vl",
    [string]$VideosList = "",
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $ApiKey) { $ApiKey = $env:DASHSCOPE_API_KEY }
if (-not $ApiKey) {
    Write-Host "ERROR: 请设置 DASHSCOPE_API_KEY 或 -ApiKey" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Video) -and -not $VideosList) {
    python scripts/make_validation_dummy.py
}

$argsList = @(
    "scripts/run_vidvrd_auto.py",
    "--run_dir", $RunDir,
    "--config", "configs/run_with_vl.json",
    "--resume",
    "--api_key", $ApiKey
)
if ($SkipEval) { $argsList += "--skip_eval" }
if ($VideosList) {
    $argsList += @("--videos", $VideosList)
} else {
    $argsList += @("--video", $Video)
}

Write-Host "RUN: python $($argsList -join ' ')" -ForegroundColor Cyan
python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/generate_run_report.py --run_dir $RunDir
Write-Host ""
Write-Host "完成。检查 VL 是否传图:" -ForegroundColor Green
Write-Host "  $RunDir/videos/*/keyframe_screen/screen_result.json  -> vl_screen.used_images"
Write-Host "  $RunDir/videos/*/track_qc/track_qc.json              -> vl_review.used_images"
Write-Host "  $RunDir/videos/*/global_relation/relations_global.json -> _global_review.used_images"
Write-Host "  $RunDir/pred/relations_pred.json"
Write-Host "  $RunDir/reports/run_report.md"
