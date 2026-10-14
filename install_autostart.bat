@echo off
rem 設定成登入 Windows 時自動在背景啟動 LINE bot（伺服器 + cloudflared，沒有視窗），並立刻啟動一次。
rem 取消自動啟動：install_autostart.bat --remove
chcp 65001 >nul
cd /d "%~dp0"
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CalorieCalculation LINE bots.lnk"

if /i "%~1"=="--remove" (
  if exist "%SHORTCUT%" del "%SHORTCUT%"
  echo 已取消開機自動啟動（目前在跑的 bot 不受影響，要停止請執行 stop_bots.bat）。
  pause
  exit /b 0
)

for /f "delims=" %%P in ('python -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),'pythonw.exe'))"') do set "PYTHONW=%%P"
if not exist "%PYTHONW%" (
  echo [錯誤] 找不到 pythonw.exe，請確認 Python 已安裝並在 PATH 裡。
  pause
  exit /b 1
)

powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:SHORTCUT); $s.TargetPath=$env:PYTHONW; $s.Arguments='\"%~dp0run_bots.py\"'; $s.WorkingDirectory='%~dp0'; $s.Save()"
echo 已設定：登入 Windows 時自動啟動。
echo   捷徑位置：%SHORTCUT%
echo.
echo 現在先啟動一次（背景執行，沒有視窗）…
start "" "%PYTHONW%" "%~dp0run_bots.py"
echo 啟動中，約 1～2 分鐘後就緒。狀態請執行 status_bots.bat，或看 logs\run_bots.log。
pause
