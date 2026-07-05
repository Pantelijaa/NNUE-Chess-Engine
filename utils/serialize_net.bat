@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM Konvertuje Lightning checkpoint (.ckpt) u Stockfish .nnue mrezu.
REM
REM Upotreba:
REM   serialize_net.bat                          -> last.ckpt  -> training_out\model.nnue
REM   serialize_net.bat <ckpt>                   -> <ckpt>     -> training_out\model.nnue
REM   serialize_net.bat <ckpt> <izlaz.nnue>      -> <ckpt>     -> <izlaz.nnue>
REM
REM VAZNO: --features mora da odgovara feature setu na kome je mreza trenirana.
REM ─────────────────────────────────────────────────────────────────────────────

set "PY=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\.venv\Scripts\python.exe"
set "NP=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\utils\easy_train_data\experiments\experiment_test\nnue-pytorch"

set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"
set "TORCHDYNAMO_DISABLE=1"

cd /d "%NP%"

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=training_out\lightning_logs\version_4\checkpoints\last.ckpt"

set "OUT=%~2"
if "%OUT%"=="" set "OUT=training_out\model.nnue"

"%PY%" serialize.py "%CKPT%" "%OUT%" --features "Full_Threats+HalfKAv2_hm^"

echo.
echo Gotovo: %OUT%
pause
