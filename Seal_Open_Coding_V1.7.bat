@echo off
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\pythonw.exe" (
  start "Seal Open Coding V1.7" "..\.venv\Scripts\pythonw.exe" "%~dp0open_coding_gui.py"
) else (
  start "Seal Open Coding V1.7" pythonw "%~dp0open_coding_gui.py"
)
endlocal
