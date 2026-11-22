@echo off
chcp 65001 >nul
title Twitch Auto-Clipper CLI
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py %*
pause
