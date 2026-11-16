# BADAMOA node update (Windows): the program only; this machine stays registered as the same node.
# For nodes installed before they could update themselves (0.30.0 and earlier). In an *Administrator* PowerShell:
#   irm <hub>/install/node-update.ps1 | iex
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

$uv = Join-Path $env:UV_INSTALL_DIR "uv.exe"
if (-not (Test-Path $uv)) {
    $existing = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $existing) { throw "uv가 없습니다. 이 PC는 노드 설치 스크립트(node.ps1)로 설치되지 않았습니다." }
    $uv = $existing.Source
}

Write-Host "==> stopping the node (Windows cannot replace a running program)"
schtasks /End /TN "BADAMOA Node" 2>$null | Out-Null
Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($Root) -and $_.Name -ne "uv" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "==> updating badamoa node to $Version from $HubUrl"
& $uv tool install --force --python 3.12 --index "$HubUrl/install/simple/" "badamoa==$Version"
if ($LASTEXITCODE -ne 0) { throw "uv tool install failed" }

Write-Host "==> starting the node"
schtasks /Run /TN "BADAMOA Node" | Out-Null

Write-Host "BADAMOA node updated to $Version (same registration). From now on it follows the Hub by itself."
