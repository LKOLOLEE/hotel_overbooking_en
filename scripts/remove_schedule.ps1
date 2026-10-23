# Removes the task from Windows Task Scheduler.
#   powershell -ExecutionPolicy Bypass -File scripts\remove_schedule.ps1
param([string]$TaskName = "Hotel overbooking - daily report")
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Removed task '$TaskName'."
