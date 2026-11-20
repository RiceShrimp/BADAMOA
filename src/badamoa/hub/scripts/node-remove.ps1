# Take the BADAMOA node off this PC: the startup task, the program, its settings.
# Data folders (images and labels registered from here) are not touched. In an *Administrator* PowerShell:
#   irm <hub>/install/node-remove.ps1 | iex
$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "관리자 권한 PowerShell에서 실행하세요."
}

$Root = Join-Path $env:ProgramFiles "BADAMOA"
$Home_ = Join-Path $env:ProgramData "BADAMOA\node"

Write-Host "==> stopping and removing the startup task"
schtasks /End /TN "BADAMOA Node" 2>$null | Out-Null
schtasks /Delete /F /TN "BADAMOA Node" 2>$null | Out-Null
Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($Root) } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Get-NetFirewallRule -DisplayName "BADAMOA node data" -ErrorAction SilentlyContinue | Remove-NetFirewallRule

Write-Host "==> removing the program and its settings"
# the node's work folders BADAMOA made on other disks (named badamoa-work; an admin's other folder stays)
$Used = Join-Path $Home_ "work-dirs.json"
if (Test-Path $Used) {
    foreach ($dir in (Get-Content $Used -Raw | ConvertFrom-Json)) {
        if ((Split-Path $dir -Leaf) -eq "badamoa-work" -and (Test-Path $dir)) { Remove-Item $dir -Recurse -Force }
    }
}
foreach ($dir in @($Root, $Home_)) {
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
}

Write-Host "BADAMOA node removed from this PC. Data folders were not touched."
