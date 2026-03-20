@echo off
REM train_model.bat - Convenient training script for BlueprintGPT
REM
REM Usage:
REM   train_model.bat              - Train with default settings
REM   train_model.bat --gpu        - Train with GPU (requires CUDA PyTorch)
REM   train_model.bat --quick      - Quick test training (3 epochs, small model)

setlocal enabledelayedexpansion

cd /d %~dp0

set PYTHON=.venv\Scripts\python.exe
set TRAIN_DATA=learned/data/kaggle_train_expanded.jsonl
set VAL_DATA=learned/data/kaggle_val_expanded.jsonl
set SAVE_PATH=learned/model/checkpoints/improved_v1.pt

REM Check if data exists
if not exist "%TRAIN_DATA%" (
    echo Expanded training data not found. Generating...
    %PYTHON% -m learned.data.expand_dataset --input learned/data/kaggle_train.jsonl --output %TRAIN_DATA%
    %PYTHON% -m learned.data.expand_dataset --input learned/data/kaggle_train_val.jsonl --output %VAL_DATA% --no-jitter
)

REM Parse arguments
set DEVICE=cpu
set EPOCHS=30
set BATCH=16
set LAYERS=6
set QUICK=0

:parse_args
if "%~1"=="" goto :run
if /i "%~1"=="--gpu" (
    set DEVICE=cuda
    shift
    goto :parse_args
)
if /i "%~1"=="--quick" (
    set QUICK=1
    set EPOCHS=5
    set LAYERS=4
    shift
    goto :parse_args
)
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:run
echo ========================================
echo BlueprintGPT Model Training
echo ========================================
echo Device: %DEVICE%
echo Epochs: %EPOCHS%
echo Training data: %TRAIN_DATA%
echo Output: %SAVE_PATH%
echo ========================================

if %QUICK%==1 (
    echo Running quick test mode...
    %PYTHON% -u -m learned.model.train_improved ^
        --train learned/data/kaggle_train.jsonl ^
        --val learned/data/kaggle_train_val.jsonl ^
        --epochs %EPOCHS% --batch 16 --layers 4 ^
        --save learned/model/checkpoints/test_quick.pt ^
        --device %DEVICE% --augment --expansion 4
) else (
    echo Running full training (this may take several hours on CPU)...
    %PYTHON% -u -m learned.model.train_improved ^
        --train %TRAIN_DATA% ^
        --val %VAL_DATA% ^
        --epochs %EPOCHS% --batch %BATCH% --layers %LAYERS% ^
        --save %SAVE_PATH% ^
        --device %DEVICE% --no-augment
)

echo ========================================
echo Training complete!
echo Checkpoint saved to: %SAVE_PATH%
echo ========================================

endlocal
