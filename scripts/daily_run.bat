@echo off
rem Daily report run - called by Windows Task Scheduler.
rem Extra arguments are passed on to daily_run.py, e.g.:
rem   daily_run.bat --publish-dir "\\server\www\report"
setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
set PYTHONUTF8=1
call ".venv\Scripts\activate.bat"
echo ===== %date% %time% ===== >> "logs\daily_run.log"
python daily_run.py --replay %* >> "logs\daily_run.log" 2>&1
endlocal
