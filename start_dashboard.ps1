param(
    [string]$Date = "",
    [int]$Port = 5050,
    [switch]$DebugMode
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$oldProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -like "python*" -and
        $_.CommandLine -like "*web_worldcup_dashboard.py*"
    }

foreach ($process in $oldProcesses) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

$argsList = @("src\web_worldcup_dashboard.py", "--port", "$Port")
if (-not [string]::IsNullOrWhiteSpace($Date)) {
    $argsList += @("--date", $Date)
}
if ($DebugMode) {
    $argsList += "--debug"
}

Write-Host "Starting dashboard..."
if ([string]::IsNullOrWhiteSpace($Date)) {
    Write-Host "URL: http://127.0.0.1:$Port/ (auto date)"
} else {
    Write-Host "URL: http://127.0.0.1:$Port/?date=$Date"
}
Write-Host "Old dashboard processes stopped: $($oldProcesses.Count)"

python @argsList
