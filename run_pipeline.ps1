<#
.SYNOPSIS
    Runs the AutoResearch Pipeline (CPU Lightweight Edition).

.DESCRIPTION
    Downloads data if needed, then kicks off the autonomous experiment loop.
    Usage: .\run_pipeline.ps1 [-Loop 5]  # run 5 iterations (default: 3)
#>

param(
    [int]$Loop = 3
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   AutoResearch Pipeline (CPU Edition)   " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Prepare data
Write-Host "`n[1/2] Preparing data (TinyStories)..." -ForegroundColor Yellow
uv run prepare.py --num-shards 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed during prepare.py execution." -ForegroundColor Red
    exit 1
}

# 2. Start Agent Loop
Write-Host "`n[2/2] Starting AutoResearch Agent Loop ($Loop iterations)..." -ForegroundColor Yellow
uv run agent_loop.py --loop $Loop
