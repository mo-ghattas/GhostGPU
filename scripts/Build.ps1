param(
    [ValidateSet('Debug','Release')][string]$Configuration = 'Release',
    [ValidateSet('cpu','cuda','both')][string]$ValidationBackend = 'both',
    [string]$RhinoInstallDir,
    [switch]$SkipValidation,
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$version = ([xml](Get-Content -LiteralPath (Join-Path $projectRoot 'Directory.Build.props'))).Project.PropertyGroup.Version
$stageRoot = Join-Path $projectRoot "artifacts/GhostGPU-$version"
Push-Location $projectRoot
try {
    $buildArgs = @('build','GhostGPU.sln','-c',$Configuration,'--nologo')
    if ($RhinoInstallDir) { $buildArgs += "-p:RhinoInstallDir=$RhinoInstallDir" }
    & dotnet @buildArgs
    if ($LASTEXITCODE -ne 0) { throw 'Build failed. Full plug-in builds require installed Rhino 8 SDK assemblies.' }
    if (-not $SkipValidation) {
        & (Join-Path $PSScriptRoot 'Test.ps1') -Backend $ValidationBackend -Configuration $Configuration
    }
    # Guard the resolved target before clearing a previous staging directory.
    if (Test-Path -LiteralPath $stageRoot) {
        $resolvedStage = [IO.Path]::GetFullPath($stageRoot)
        $artifactRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'artifacts')) + [IO.Path]::DirectorySeparatorChar
        if (-not $resolvedStage.StartsWith($artifactRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Stage path is outside artifacts.' }
        Remove-Item -LiteralPath $resolvedStage -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null
    $outputRoot = Join-Path $projectRoot "GhostGPU.Grasshopper/bin/$Configuration/net48"
    $hostAssemblies = @('RhinoCommon.dll','Grasshopper.dll','GH_IO.dll','Rhino.UI.dll','System.Drawing.dll')
    foreach ($file in Get-ChildItem -LiteralPath $outputRoot -File) {
        if ($file.Name -in $hostAssemblies) { continue }
        if ($file.Extension -in @('.dll','.gha','.pdb','.config')) {
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $stageRoot $file.Name)
        }
    }
    foreach ($name in @('README.md','LICENSE','THIRD_PARTY_NOTICES.md','CONTRIBUTING.md')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $stageRoot
    }
    foreach ($name in @('docs','examples')) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $stageRoot -Recurse
    }
    $assets = Get-Content -LiteralPath (Join-Path $projectRoot 'GhostGPU.Grasshopper/obj/project.assets.json') -Raw | ConvertFrom-Json
    $licenseRoot = Join-Path $stageRoot 'licenses'
    New-Item -ItemType Directory -Path $licenseRoot -Force | Out-Null
    foreach ($library in $assets.libraries.PSObject.Properties) {
        if ($library.Value.type -ne 'package') { continue }
        foreach ($folder in $assets.packageFolders.PSObject.Properties.Name) {
            $packageRoot = Join-Path $folder $library.Value.path
            if (-not (Test-Path -LiteralPath $packageRoot)) { continue }
            $packageLicenseRoot = Join-Path $licenseRoot ($library.Name.Replace('/','-'))
            New-Item -ItemType Directory -Path $packageLicenseRoot -Force | Out-Null
            Get-ChildItem -LiteralPath $packageRoot -File | Where-Object {
                $_.Name -match '^(LICENSE|NOTICE|COPYING)' -or $_.Extension -eq '.nuspec'
            } | Copy-Item -Destination $packageLicenseRoot
            break
        }
    }
    $manifest = Get-ChildItem -LiteralPath $stageRoot -File -Recurse | ForEach-Object {
        (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.FullName.Substring($stageRoot.Length + 1).Replace('\','/')
    }
    $manifest | Set-Content -LiteralPath (Join-Path $stageRoot 'SHA256SUMS.txt') -Encoding UTF8
    Write-Output "Staged: $stageRoot"
    if ($Install) {
        if (Get-Process Rhino -ErrorAction SilentlyContinue) { throw 'Close Rhino before installing the plug-in.' }
        $libraryRoot = Join-Path $env:APPDATA 'Grasshopper/Libraries/GhostGPU'
        New-Item -ItemType Directory -Path $libraryRoot -Force | Out-Null
        Get-ChildItem -LiteralPath $stageRoot -File | Where-Object { $_.Extension -in @('.dll','.gha','.pdb') } | Copy-Item -Destination $libraryRoot -Force
        Get-ChildItem -LiteralPath $libraryRoot -File | Unblock-File
        Write-Output "Installed: $libraryRoot"
    }
}
finally { Pop-Location }
