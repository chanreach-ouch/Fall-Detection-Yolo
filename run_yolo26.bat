@echo off
setlocal enabledelayedexpansion
title Fall Detection - YOLO26 Pipeline
echo ======================================================================
echo Launching YOLO26 Pose Fall Detection & Activity Recognition
echo Virtual Environment: .venv
echo ======================================================================
cd /d "%~dp0yolo26-pose-fall-detection"
"..\.venv\Scripts\python.exe" fall_detection.py --model models\yolo26s-pose.pt --input input\video_2.avi --output output\result_video_2.mp4 %*
if errorlevel 1 pause

