@echo off
rem 停止背景執行的 LINE bot（run_bots、bot 伺服器、cloudflared 一起停）。
rem 不會取消開機自動啟動；下次登入 Windows 還是會自動跑起來。
chcp 65001 >nul
powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*.exe' -and ($_.CommandLine -like '*run_bots.py*' -or $_.CommandLine -like '*webhook_app.py*')) -or ($_.Name -eq 'cloudflared.exe' -and $_.CommandLine -like '*127.0.0.1:5000*') };" ^
  "if (-not $p) { Write-Output '目前沒有在執行。' } else { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('已停止 ' + $_.Name + ' (' + $_.ProcessId + ')') } }"
pause
