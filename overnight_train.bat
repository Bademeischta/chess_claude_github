@echo off
REM ===============================================================
REM  Chess AI — overnight training pipeline
REM
REM  Double-click this file (or run it from cmd) and go to sleep.
REM  Runs in order:
REM    1) Stockfish-vs-Stockfish game generation
REM    2) Train on those games (supervised, 3 epochs)
REM    3) Lichess puzzle conversion (auto-downloads DB)
REM    4) Train on puzzles (supervised, 3 epochs)
REM    5) Self-play with crash-resilient supervisor (open-ended)
REM
REM  Log: runs\overnight_YYYYMMDD_HHMMSS.log
REM  Stop in the morning: Ctrl+C in this window (saves checkpoint).
REM ===============================================================

REM Make sure we run from the project root no matter where this is
REM double-clicked from.
cd /d "%~dp0"

REM Show a banner so the user knows it's the right script.
echo.
echo ===============================================================
echo   Chess AI Overnight Training
echo ===============================================================
echo.
echo   Steps 1-4 take ~5-6 hours. Step 5 (self-play) runs until you
echo   press Ctrl+C in this window.
echo.
echo   You can close this window AT ANY TIME with Ctrl+C —
echo   completed steps are saved, and re-running this .bat picks
echo   up where it left off.
echo.
echo   Logs go to: runs\overnight_*.log
echo ===============================================================
echo.

python -u tools\overnight_train.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if %EXITCODE% NEQ 0 (
    echo ===============================================================
    echo   Pipeline ended with exit code %EXITCODE%.
    echo   Check the latest log under runs\ for details.
    echo ===============================================================
) else (
    echo ===============================================================
    echo   Pipeline completed cleanly. Good morning!
    echo ===============================================================
)

REM Keep the window open if the user double-clicked us, so they
REM can read the final message instead of the window vanishing.
pause
exit /b %EXITCODE%
