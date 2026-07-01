@echo off
REM ===========================================================================
REM  LaunchPoint - one-click dependency installer (Windows)
REM
REM  Creates a local virtual environment (.venv) and installs everything the
REM  Controller-Origin Finder needs. The geospatial stack (rasterio/GDAL,
REM  pyproj, shapely, geopandas) ships binary wheels for CPython 3.10-3.12, so
REM  this script picks a compatible interpreter automatically.
REM
REM  Usage:  double-click this file, or run  install_dependencies.bat
REM ===========================================================================
setlocal EnableExtensions

cd /d "%~dp0"
echo === LaunchPoint dependency installer ===
echo Project: %cd%
echo.

REM --- 1. Find a compatible Python interpreter (3.10 - 3.12) -----------------
set "PYEXE="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
    )
)

if not defined PYEXE (
    REM Fall back to whatever 'python' is on PATH, if it is in range.
    python -c "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<(3,13) else 1)" >nul 2>&1 && set "PYEXE=python"
)

if not defined PYEXE (
    echo.
    echo ERROR: Could not find a compatible Python.
    echo        LaunchPoint needs CPython 3.10, 3.11, or 3.12.
    echo        Install one from https://www.python.org/downloads/ and re-run.
    echo        ^(Python 3.13+ has no rasterio/GDAL wheels yet.^)
    exit /b 1
)

echo Using interpreter: %PYEXE%
%PYEXE% --version
echo.

REM --- 2. Create the virtual environment ------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo Virtual environment .venv already exists - reusing it.
) else (
    echo Creating virtual environment in .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo ERROR: failed to create the virtual environment.
        exit /b 1
    )
)

set "VENV_PY=.venv\Scripts\python.exe"

REM --- 3. Install dependencies ----------------------------------------------
echo.
echo Ensuring pip is present ...
"%VENV_PY%" -m ensurepip --default-pip >nul 2>&1

echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 ( echo ERROR: pip upgrade failed. & exit /b 1 )

echo.
echo Installing runtime + test dependencies ^(this can take a few minutes^) ...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: dependency install failed. & exit /b 1 )

echo.
echo Installing LaunchPoint ^(editable^) ...
"%VENV_PY%" -m pip install -e .
if errorlevel 1 ( echo ERROR: package install failed. & exit /b 1 )

REM --- 3b. Optional GPU acceleration (NVIDIA GPUs only) ----------------------
REM  The CUDA viewshed needs numba-cuda + the CUDA 12 toolkit wheels (the [gpu]
REM  extra). We only install it when an NVIDIA GPU is actually present, and a
REM  failure here is non-fatal: LaunchPoint runs identically (just slower) on the
REM  CPU kernel, so a machine without an NVIDIA card installs cleanly.
echo.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo No NVIDIA GPU detected ^(nvidia-smi not found^) - skipping GPU acceleration.
    echo LaunchPoint will use the CPU viewshed, which is fully supported.
) else (
    echo NVIDIA GPU detected - installing CUDA viewshed acceleration ^(~150 MB^) ...
    "%VENV_PY%" -m pip install -e ".[gpu]"
    if errorlevel 1 (
        echo WARNING: GPU acceleration install failed - continuing on CPU only.
        echo          You can retry later with:  .venv\Scripts\pip install -e ".[gpu]"
    )
)

REM --- 4. Smoke test ---------------------------------------------------------
echo.
echo Verifying the install ...
"%VENV_PY%" -c "import numpy, rasterio, pyproj, shapely, geopandas, numba, launchpoint; print('LaunchPoint', launchpoint.__version__, 'ready; GDAL', rasterio.__gdal_version__)"
if errorlevel 1 ( echo ERROR: import check failed. & exit /b 1 )

REM  Report the viewshed backend (non-fatal - CPU is a valid outcome).
"%VENV_PY%" -c "from launchpoint.viewshed.gpu import gpu_status; ok, msg = gpu_status(); print('GPU acceleration: ' + msg if ok else 'GPU acceleration: off (CPU mode) - ' + msg)"

echo.
echo ===========================================================================
echo  Done. To use LaunchPoint:
echo.
echo    .venv\Scripts\activate
echo    launchpoint demo --out demo_origin.tif
echo.
echo  Or run the test suite:
echo.
echo    .venv\Scripts\activate
echo    pytest
echo ===========================================================================
endlocal
