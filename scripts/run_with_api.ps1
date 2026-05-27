# 全自动跑通（mock 检测/追踪 + 真实 DashScope 关系 LLM）
# 用法（不要把 key 写进本文件）：
#   $env:DASHSCOPE_API_KEY = "sk-你的key"
#   .\scripts\run_with_api.ps1
# 或：
#   .\scripts\run_with_api.ps1 -ApiKey "sk-你的key"

param(
    [string]$ApiKey = "",
    [string]$Video = "data/validation_dummy.mp4",
    [string]$RunDir = "runs/live_api",
    [string]$VideosList = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $ApiKey) { $ApiKey = $env:DASHSCOPE_API_KEY }
if (-not $ApiKey) {
    Write-Host "ERROR: 请设置环境变量 DASHSCOPE_API_KEY 或传入 -ApiKey" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $Video)) {
    Write-Host "生成测试视频..."
    python scripts/make_validation_dummy.py
}

$argsList = @(
    "scripts/run_vidvrd_auto.py",
    "--run_dir", $RunDir,
    "--config", "configs/run_with_api.json",
    "--resume",
    "--skip_eval",
    "--api_key", $ApiKey
)
if ($VideosList) {
    $argsList += @("--videos", $VideosList)
} else {
    $argsList += @("--video", $Video)
}

Write-Host "RUN: python $($argsList -join ' ')" -ForegroundColor Cyan
python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "完成。查看:" -ForegroundColor Green
Write-Host "  $RunDir/run_manifest.json"
Write-Host "  $RunDir/pred/relations_pred.json"
