@echo off

set "PY=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\.venv\Scripts\python.exe"
set "NP=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\utils\easy_train_data\experiments\experiment_test\nnue-pytorch"
set "TRAIN=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\nnue\binpack\dfrc99-16tb7p.v2.min.dd.high-simple-eval-v4.min-v2.binpack"
set "VAL=C:\Users\nikol\PycharmProjects\NNUE-Chess-Engine\nnue\binpack\test80-oct2023-2tb7p.high-simple-eval-v4.min-v2.binpack"

set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"

REM torch.compile/inductor je nestabilan na Windows-u (trazi C++/Triton) -> eager rezim
set "TORCHDYNAMO_DISABLE=1"

cd /d "%NP%"

"%PY%" train.py "%TRAIN%" ^
    --validation-datasets "%VAL%" ^
    --gpus 0, ^
    --num-workers 8 ^
    --threads 4 ^
    --batch-size 16384 ^
    --max-epochs 60 ^
    --random-fen-skipping 2 ^
    --features "Full_Threats+HalfKAv2_hm^" ^
    --lr 0.000875 ^
    --gamma 0.992 ^
    --start-lambda 1.0 ^
    --end-lambda 0.75 ^
    --seed 42 ^
    --epoch-size 100000000 ^
    --validation-size 1000000 ^
    --network-save-period 10 ^
    --save-last-network True ^
    --default-root-dir "%NP%\training_out"

pause
