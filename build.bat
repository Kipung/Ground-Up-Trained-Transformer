@echo off

:: Display the project file location
echo Project file location: %~dp0

:: Check if the virtual environment already exists
if not exist ".\cuda" (
    echo Generating Virtual Environment for Python 
    color 07
    color 0C
    echo [DO NOT CLOSE]

    :: Create virtual environment with Python (cuda for GPU Acceleration)
    py -m venv %~dp0cuda
) else (
    color 0E
    echo Environment already exists
)
color 07

:: Activate the virtual environment
color 0E
echo ACTIVATING ENVIRONMENT 
color 0C
echo [DO NOT CLOSE]
color 07

call cuda\Scripts\activate.bat

:: Upgrade pip to the latest version
python -m pip install --upgrade pip

python -m pip install numpy
python -m pip install matplotlib

:: We want to limit our reliance on torch and eventually get rid of it, but for now its helpful
:: py -m pip install torch 

:: Torch 2.4.0 with CUDA 12.1
python -m pip install torch==2.0.0+cu129 -f https://download.pytorch.org/whl/torch_stable.html


:: py -m pip install pylzma <- this is a C++ compression algorithm that i dont wanna use


@echo off
echo Should be done now...
pause

