# Registers a daily task in Windows Task Scheduler (for the current user).
# Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\register_schedule.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\register_schedule.ps1 -Time 07:30
param(
    [string]$Time = "08:00",
    [string]$TaskName = "Hotel overbooking - daily report"
)
$root = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $root "scripts\daily_run.bat"
if (-not (Test-Path (Join-Path $root ".venv\Scripts\python.exe"))) {
    Write-Warning "No .venv found in $root - create the environment: python -m venv .venv; pip install -r requirements.txt"
}
$action   = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$bat`"" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Daily -At $Time
# StartWhenAvailable: if the computer was off at 08:00, the task runs right after it is switched on
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Overbooking plan and list of bookings to contact (outputs\site\index.html)" -Force | Out-Null

Write-Host "Registered task '$TaskName' - daily at $Time."
Write-Host "Test now:     Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Log:          $root\logs\daily_run.log"
Write-Host "Report:       $root\outputs\site\index.html"
