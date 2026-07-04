param(
    [ValidateRange(1, 25)]
    [int]$Runs = 10,
    [string]$BranchName = "codex/courtvision-agent-loop"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path -LiteralPath "prompt.txt")) {
    throw "prompt.txt was not found in $repoRoot"
}

$codexCommand = Get-Command codex.cmd -ErrorAction SilentlyContinue
if (-not $codexCommand) {
    $npmCodex = Join-Path $env:APPDATA "npm\codex.cmd"
    if (Test-Path -LiteralPath $npmCodex) {
        $codexPath = $npmCodex
    }
    else {
        throw "Codex CLI was not found. Add $env:APPDATA\npm to PATH."
    }
}
else {
    $codexPath = $codexCommand.Source
}

$currentBranch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine the current git branch."
}

if ($currentBranch -eq "main" -or $currentBranch -eq "master") {
    & git show-ref --verify --quiet "refs/heads/$BranchName"
    if ($LASTEXITCODE -eq 0) {
        & git switch $BranchName
    }
    else {
        & git switch -c $BranchName
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Could not switch from $currentBranch to $BranchName."
    }
}

New-Item -ItemType Directory -Force -Path "runs" | Out-Null
$progressPath = Join-Path $repoRoot "runs\agent-progress.md"
if (-not (Test-Path -LiteralPath $progressPath)) {
    @(
        "# CourtVision agent progress"
        ""
        "Each run should inspect the repository and append its completed work here."
        ""
    ) | Set-Content -LiteralPath $progressPath
}

for ($i = 1; $i -le $Runs; $i++) {
    Write-Host ""
    Write-Host "================================="
    Write-Host "Starting Codex Run $i of $Runs"
    Write-Host "Branch: $(& git branch --show-current)"
    Write-Host "================================="
    Write-Host ""

    Get-Content -Raw -LiteralPath "prompt.txt" | & $codexPath exec -
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "Codex Run $i failed with exit code $exitCode. The loop was stopped."
    }

    Write-Host ""
    Write-Host "Completed Run $i of $Runs"
    Write-Host ""
}

Write-Host "All $Runs runs completed on branch $(& git branch --show-current)."
