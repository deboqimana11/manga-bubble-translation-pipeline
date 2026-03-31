@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& '%~dp0\.venv\Scripts\python.exe' '%~dp0\ui_translate_manga.py'"
