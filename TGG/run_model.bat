@echo off
echo Setting environment variables...
set TRAIN_DATA_PATH=data
set TRAIN_LOG_PATH=logs
set TRAIN_TF_EVENTS_PATH=tensorboard
set TRAIN_CKPT_PATH=checkpoints

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Running model with InfoNCE loss...
python main08281.py --batch_size 32 --lr 0.001 --maxlen 50 --hidden_units 64 --num_epochs 1 --temperature 0.1 --device cpu

echo Done!
pause
