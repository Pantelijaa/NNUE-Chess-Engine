@echo off
cd /d "%~dp0"

set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"

set "CC=C:/msys64/ucrt64/bin/gcc.exe"
set "CXX=C:/msys64/ucrt64/bin/g++.exe"
set "MAKE=C:/msys64/usr/bin/make.exe"

..\.venv\Scripts\python.exe easy_train.py ^
    --training-dataset=./../nnue/binpack/dfrc99-16tb7p.v2.min.dd.high-simple-eval-v4.min-v2.binpack ^
    --validation-dataset=./../nnue/binpack/test80-oct2023-2tb7p.high-simple-eval-v4.min-v2.binpack ^
    --num-workers=4 ^
    --threads=2 ^
    --gpus="0," ^
    --runs-per-gpu=2 ^
    --batch-size=16384 ^
    --max_epoch=10 ^
    --do-network-training=True ^
    --do-network-testing=True ^
    --tui=True ^
    --network-save-period=1 ^
    --random-fen-skipping=3 ^
    --start-lambda=1.0 ^
    --end-lambda=0.75 ^
    --experiment-name=test ^
    --fail-on-experiment-exists=False ^
    --build-engine-arch=x86-64-avx2 ^
    --build-threads=2 ^
    --epoch-size=1638400 ^
    --validation-size=16384 ^
    --network-testing-threads=24 ^
    --network-testing-explore-factor=1.5 ^
    --network-testing-book="https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip" ^
    --network-testing-nodes-per-move=20000 ^
    --network-testing-hash-mb=8 ^
    --network-testing-games-per-round=200 ^
    --engine-base-branch=official-stockfish/Stockfish/master ^
    --engine-test-branch=official-stockfish/Stockfish/master

pause
