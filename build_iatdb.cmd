@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "PLUGIN_DIR=%~dp0"
for %%I in ("%PLUGIN_DIR%..\..\..\python_embeded\python.exe") do set "PYTHON=%%~fI"
set "LOG=%TEMP%\ComfyUI-IAT-build.log"

if "%~1"=="" (
  set "RESULT=2"
  echo Drag one dataset folder or the IAT-datasets root onto this file.
  goto :finish
)

if not exist "%PYTHON%" (
  echo [IAT] Embedded Python was not found: %PYTHON%
  set "RESULT=2"
  goto :finish
)

echo [IAT] Building dataset. Full output will be saved to:
echo %LOG%
>"%LOG%" echo [IAT] ComfyUI-IAT dataset build log
>>"%LOG%" echo [IAT] Command: "%PYTHON%" "%PLUGIN_DIR%scripts\build_iatdb.py" %*
"%PYTHON%" "%PLUGIN_DIR%scripts\build_iatdb.py" %* >"%LOG%" 2>&1
set "RESULT=%ERRORLEVEL%"
type "%LOG%"

if not "%RESULT%"=="0" echo [IAT] Build failed with exit code %RESULT%. The window will stay open.
if "%RESULT%"=="0" echo [IAT] Build finished successfully. The window will stay open.

:finish
echo.
pause
exit /b %RESULT%
