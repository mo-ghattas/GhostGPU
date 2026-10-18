param(
    [ValidateSet('cpu','cuda','both')][string]$Backend = 'cpu',
    [ValidateSet('Debug','Release')][string]$Configuration = 'Release',
    [switch]$SkipKernel
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportRoot = Join-Path $projectRoot 'artifacts/validation'
$previousSelector = [Environment]::GetEnvironmentVariable('GG_BACKEND', 'Process')
Push-Location $projectRoot
try {
    New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
    $backends = if ($Backend -eq 'both') { @('cpu','cuda') } else { @($Backend) }
    foreach ($selected in $backends) {
        & dotnet run --project GhostGPU.Validation -c $Configuration -- --backend $selected --no-benchmarks --output "artifacts/validation/runtime-$selected.json"
        if ($LASTEXITCODE -ne 0) { throw "Runtime validation failed ($selected)." }
        $env:GG_BACKEND = if ($selected -eq 'cpu') { 'CpuDebug' } else { 'CudaOnly' }
        & dotnet run --project GhostGPU.Booleans.Validation -c $Configuration -- --output "artifacts/validation/mesh-$selected.json"
        if ($LASTEXITCODE -ne 0) { throw "Mesh Boolean validation failed ($selected)." }
    }
    if (-not $SkipKernel) {
        & dotnet run --project GhostGPU.Kernel.Validation -c $Configuration -- --golden-dir tests/fixtures/golden --out artifacts/validation/kernel.json
        if ($LASTEXITCODE -ne 0) { throw 'Kernel harness failed.' }
        Write-Warning 'Kernel harness coverage is incomplete: M3 contains placeholder cases; CUDA can also be unavailable. Inspect the report and docs/KNOWN-ISSUES.md.'
    }
    Write-Output "Validation reports: $reportRoot"
}
finally {
    [Environment]::SetEnvironmentVariable('GG_BACKEND', $previousSelector, 'Process')
    Pop-Location
}
