<#
.SYNOPSIS
    Runs the AutoResearch Pipeline automatically utilizing the local Claude CLI login.

.DESCRIPTION
    This script runs agent_loop.py to generate a training experiment prompt (claude_task.md),
    runs the local 'claude -p' command using the logged-in Max account,
    and then repeats the process for autonomous self-improvement.
#>

param(
    [int]$Loop = 3
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   AutoResearch Pipeline (Claude CLI)    " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Prepare data
Write-Host "`n[1/2] Preparing data (TinyStories)..." -ForegroundColor Yellow
uv run prepare.py --num-shards 1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed during prepare.py execution." -ForegroundColor Red
    exit 1
}

# 2. Start Agent Loop
Write-Host "`n[2/2] Starting autonomous loop ($Loop iterations)..." -ForegroundColor Yellow

for ($i = 1; $i -le $Loop; $i++) {
    Write-Host "`n>>> ITERATION $i / $Loop <<<" -ForegroundColor Green
    
    # Run Orchestrator to evaluate previous and generate prompt for next
    Write-Host "--> Running Antigravity Orchestrator..." -ForegroundColor Gray
    uv run agent_loop.py --loop 1
    
    if (Test-Path "claude_task.md") {
        Write-Host "--> Sending task to Claude Code CLI... (This may take 1-2 minutes)" -ForegroundColor Magenta
        
        # Use cmd to safely redirect file to claude
        cmd.exe /c "claude.cmd -p < claude_task.md > claude_response.md"
        
        Write-Host "--> Claude finished editing train.py" -ForegroundColor Magenta
        
        # Clean up the task file so it's fresh for next loop
        Remove-Item "claude_task.md" -Force
    } else {
        Write-Host "No claude_task.md found. Loop might have finished or errored." -ForegroundColor Red
        break
    }
}

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "   Pipeline Finished!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
