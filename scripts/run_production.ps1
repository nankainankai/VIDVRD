# 正式批量标注（真实检测 + 关系 LLM + Gold 评测）
# $env:DASHSCOPE_API_KEY = "sk-xxx"
# $env:DINOX_API_TOKEN = "xxx"   # production_full 使用 dinox 时
# .\scripts\run_production.ps1 -Video "D:\v.mp4"

param(
    [string]$ApiKey = "",
    [string]$Video = "",
    [string]$VideosList = "data/videos.txt",
    [string]$RunDir = "runs/production",
    [string]$Config = "configs/production_full.json"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not $ApiKey) { $ApiKey = $env:DASHSCOPE_API_KEY }
if (-not $ApiKey) {
    Write-Host "ERROR: 设置 DASHSCOPE_API_KEY 或 -ApiKey" -ForegroundColor Red
    exit 1
}

$argsList = @(
    "scripts/run_vidvrd_auto.py",
    "--run_dir", $RunDir,
    "--config", $Config,
    "--resume",
    "--api_key", $ApiKey
)
if ($Video) {
    $argsList += @("--video", $Video)
} else {
    if (-not (Test-Path $VideosList)) {
        Write-Host "ERROR: 提供 -Video 或准备 $VideosList" -ForegroundColor Red
        exit 1
    }
    $argsList += @("--videos", $VideosList)
}

python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts/generate_run_report.py --run_dir $RunDir
Write-Host "报告: $RunDir/reports/run_report.md" -ForegroundColor Green
if (Test-Path "$RunDir/reports/presence_report.md") {
    Write-Host "评测: $RunDir/reports/presence_report.md" -ForegroundColor Green
}
