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
py -m pip install --upgrade pip

py -m pip install numpy
py -m pip install matplotlib

:: We want to limit our reliance on torch and eventually get rid of it, but for now its helpful
py -m pip install torch 

:: py -m pip install pylzma <- this is a C++ compression algorithm that i dont wanna use

