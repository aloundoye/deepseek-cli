param(
  [string]$Version = "latest",
  [string]$Repo = "aloundoye/codingbuddy",
  [string]$InstallDir = "$env:LOCALAPPDATA\Programs\codingbuddy\bin",
  [string]$Target = "",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Target)) {
  $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
  if ($arch -eq [System.Runtime.InteropServices.Architecture]::Arm64) {
    $Target = "aarch64-pc-windows-msvc"
  } elseif ($arch -eq [System.Runtime.InteropServices.Architecture]::X64) {
    $Target = "x86_64-pc-windows-msvc"
  } else {
    throw "unsupported architecture for prebuilt binaries: $arch"
  }
}
$asset = "codingbuddy-$Target.zip"

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

$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("codingbuddy-install-" + [guid]::NewGuid().ToString())
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
$binary = Get-ChildItem -Path $tmpRoot -Recurse -Filter "codingbuddy.exe" | Select-Object -First 1
if (-not $binary) {
  throw "unable to locate extracted codingbuddy.exe"
}

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$destination = Join-Path $InstallDir "codingbuddy.exe"
Copy-Item -Path $binary.FullName -Destination $destination -Force

# Auto-add to user PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$InstallDir*") {
  [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$currentPath", "User")
  Write-Host "Added $InstallDir to user PATH."
}

# Cleanup
Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "CodingBuddy installed! Run: codingbuddy"
Write-Host "Location: $destination"
Write-Host "Uninstall: Remove-Item `"$destination`""
