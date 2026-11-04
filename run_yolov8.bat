@echo off
setlocal enabledelayedexpansion
title Fall Detection - YOLOv8 Pipeline
echo ======================================================================
echo Launching YOLOv8 Pose Fall Detection (Interactive Runner)
echo Virtual Environment: .venv
echo ======================================================================
cd /d "%~dp0yolov8-pose-fall-detection"
"..\.venv\Scripts\python.exe" run.py %*
if errorlevel 1 pause

