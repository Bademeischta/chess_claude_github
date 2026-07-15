@echo off
REM ============================================================
REM  Chess AI — UCI engine launcher for chess GUIs
REM  (Arena, Cute Chess, BanksiaGUI, Nibbler, En Croissant, ...)
REM
REM  In your GUI: add a NEW UCI ENGINE and point it at THIS file.
REM  The engine auto-loads the newest checkpoint in checkpoints\.
REM
REM  cd /d "%~dp0" makes the working dir the project root so the
REM  relative checkpoints\ path resolves no matter where the GUI
REM  launches this from. -u keeps stdin/stdout unbuffered (UCI
REM  pipes must not block-buffer).
REM ============================================================
cd /d "%~dp0"
"C:\Users\silas\AppData\Local\Programs\Python\Python311\python.exe" -u tools\uci_engine.py %*
