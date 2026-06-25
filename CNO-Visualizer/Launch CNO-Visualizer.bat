@echo off
REM ============================================================================
REM  Double-click to launch CNO-Visualizer with a file picker (no terminal).
REM  Right-click -> "Send to > Desktop (create shortcut)" to pin it; you can set
REM  a custom icon on the shortcut via Properties > Change Icon.
REM ============================================================================

REM Prefer the project's known conda env; fall back to whatever is on PATH.
set "PYW=%USERPROFILE%\miniconda3\envs\physics\pythonw.exe"

if exist "%PYW%" (
    start "" "%PYW%" -m cno_visualizer.launcher %*
) else (
    REM Fall back: rely on conda being initialised in the shell.
    start "" cmd /c "conda activate physics && cno-visualizer-gui %*"
)
