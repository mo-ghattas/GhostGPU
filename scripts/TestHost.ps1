param(
    [ValidateSet('cpu','both')][string]$Backend = 'cpu',
    [string]$RhinoInstallDir = 'C:\Program Files\Rhino 8',
    [switch]$WriteExample
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$oldInstall = [Environment]::GetEnvironmentVariable('RHINO_INSTALL_DIR', 'Process')
$oldIconInstall = [Environment]::GetEnvironmentVariable('RhinoInstallDir', 'Process')
Push-Location $projectRoot
try {
    $env:RHINO_INSTALL_DIR = $RhinoInstallDir
    $env:RhinoInstallDir = $RhinoInstallDir
    & dotnet build tests/GhostGPU.HostValidation -c Release --nologo "-p:RhinoInstallDir=$RhinoInstallDir"
    if ($LASTEXITCODE -ne 0) { throw 'Host validation build failed.' }
    # RhinoCore must run under dotnet, not a generated application host executable.
    $runnerArgs = @('tests/GhostGPU.HostValidation/bin/Release/net8.0-windows/GhostGPU.HostValidation.dll')
    if ($Backend -eq 'both') { $runnerArgs += '--cuda' }
    if ($WriteExample) { $runnerArgs += '--write-example' }
    & dotnet @runnerArgs
    if ($LASTEXITCODE -ne 0) { throw 'Rhino host validation failed.' }
}
finally {
    [Environment]::SetEnvironmentVariable('RHINO_INSTALL_DIR', $oldInstall, 'Process')
    [Environment]::SetEnvironmentVariable('RhinoInstallDir', $oldIconInstall, 'Process')
    Pop-Location
}
