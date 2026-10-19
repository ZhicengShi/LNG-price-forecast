@echo off
setlocal
if not defined LNG_PYTHON set "LNG_PYTHON=python"
pushd "%~dp0.."
if errorlevel 1 exit /b 1
"%LNG_PYTHON%" -B -m service.app --config "%~dp0..\service\config.yaml" %*
set "LNG_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %LNG_EXIT_CODE%
