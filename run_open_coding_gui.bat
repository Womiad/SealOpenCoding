@echo off
setlocal
cd /d "%~dp0"
if exist "..\.venv\Scripts\pythonw.exe" (
  start "" "..\.venv\Scripts\pythonw.exe" "%~dp0open_coding_gui.py"
) else (
  start "" pythonw "%~dp0open_coding_gui.py"
)
endlocal
