@echo off
rem 照 models.txt 安裝本專案需要的模型（雙擊執行）。
rem 只檢查不安裝：install_models.bat --check
rem 已安裝的也更新：install_models.bat --update
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
python install_models.py %*
echo.
pause
