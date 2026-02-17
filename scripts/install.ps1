param(
  [string]$Version = "latest",
  [string]$Repo = "aloutndoye/deepseek-cli",
  [string]$InstallDir = "$env:LOCALAPPDATA\\Programs\\deepseek\\bin",
  [string]$Target = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Target)) {
  if ($env:PROCESSOR_ARCHITECTURE -match "ARM64") {
    Write-Host "warning: using x86_64 Windows binary on ARM; emulation may be required."
  }
  $Target = "x86_64-pc-windows-msvc"
}
$asset = "deepseek-$Target.zip"

if ($Version -eq "latest") {
  $baseUrl = "https://github.com/$Repo/releases/latest/download"
} else {
  $baseUrl = "https://github.com/$Repo/releases/download/$Version"
}

$assetUrl = "$baseUrl/$asset"
$checksumUrl = "$baseUrl/checksums.txt"

if ($DryRun) {
  Write-Host "dry-run"
  Write-Host "repo: $Repo"
  Write-Host "version: $Version"
  Write-Host "target: $Target"
  Write-Host "asset: $assetUrl"
  Write-Host "install_dir: $InstallDir"
  exit 0
}

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("deepseek-install-" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmpRoot | Out-Null

$assetPath = Join-Path $tmpRoot $asset
$checksumsPath = Join-Path $tmpRoot "checksums.txt"

Invoke-WebRequest -Uri $assetUrl -OutFile $assetPath
try {
  Invoke-WebRequest -Uri $checksumUrl -OutFile $checksumsPath
} catch {
  Write-Host "warning: could not download checksums.txt; skipping checksum verification"
}

if (Test-Path $checksumsPath) {
  $line = Get-Content $checksumsPath | Where-Object { $_ -match "\s$([regex]::Escape($asset))$" } | Select-Object -First 1
  if ($line) {
    $expected = ($line -split "\s+")[0].ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 -Path $assetPath).Hash.ToLower()
    if ($expected -ne $actual) {
      throw "checksum verification failed (expected=$expected actual=$actual)"
    }
  }
}

Expand-Archive -Path $assetPath -DestinationPath $tmpRoot -Force
$binary = Get-ChildItem -Path $tmpRoot -Recurse -Filter "deepseek.exe" | Select-Object -First 1
if (-not $binary) {
  throw "unable to locate extracted deepseek.exe"
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$destination = Join-Path $InstallDir "deepseek.exe"
Copy-Item -Path $binary.FullName -Destination $destination -Force

Write-Host "Installed deepseek to $destination"
Write-Host "If needed, add to PATH: $InstallDir"
Write-Host "Uninstall: Remove-Item $destination"
