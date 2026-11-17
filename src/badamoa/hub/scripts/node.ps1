# BADAMOA node installer (Windows). Served by the Hub with HUB_URL filled in.
# Run in an *Administrator* PowerShell:
#   & ([scriptblock]::Create((irm <hub>/install/node.ps1))) -Token <join-token>
param(
    [Parameter(Mandatory = $true)][string]$Token
)
$ErrorActionPreference = "Stop"

$HubUrl = "{{HUB_URL}}"
$Version = "{{VERSION}}"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "관리자 권한 PowerShell에서 실행하세요."
}

$Root = Join-Path $env:ProgramFiles "BADAMOA"
$env:UV_INSTALL_DIR = Join-Path $Root "bin"
$env:UV_TOOL_DIR = Join-Path $Root "tools"
$env:UV_TOOL_BIN_DIR = Join-Path $Root "bin"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "python"
$env:UV_NO_MODIFY_PATH = "1"
New-Item -ItemType Directory -Force -Path $env:UV_INSTALL_DIR | Out-Null

$uv = Join-Path $env:UV_INSTALL_DIR "uv.exe"
if (-not (Test-Path $uv)) {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if ($existing) { $uv = $existing.Source }
    else {
        Write-Host "==> installing uv"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
}

# Installed already (the installer run again): stop it first, Windows cannot replace a running program.
schtasks /End /TN "BADAMOA Node" 2>$null | Out-Null
Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($Root) -and $_.Name -ne "uv" } | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "==> installing badamoa node $Version from $HubUrl"
& $uv tool install --force --python 3.12 --index "$HubUrl/install/simple/" "badamoa==$Version"
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }

$badamoa = Join-Path $env:UV_TOOL_BIN_DIR "badamoa.exe"

Write-Host "==> joining hub"
& $badamoa node join --hub $HubUrl --token $Token
if ($LASTEXITCODE -ne 0) { throw "join failed" }

Write-Host "==> opening TCP 8475 (other nodes take this PC's registered images from here, not through the Hub)"
Get-NetFirewallRule -DisplayName "BADAMOA node data" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName "BADAMOA node data" -Direction Inbound -Protocol TCP -LocalPort 8475 -Action Allow | Out-Null

Write-Host "==> registering startup task"
& $badamoa node install-service
if ($LASTEXITCODE -ne 0) { throw "service install failed" }

Write-Host "BADAMOA node installed. Check the Hub UI -> 노드."
