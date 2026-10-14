@echo off
rem 查看 LINE bot 是否在執行，以及最近的啟動／Webhook 設定紀錄。
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -Command ^
  "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://127.0.0.1:5000/health; Write-Output 'bot 伺服器：執行中' } catch { Write-Output 'bot 伺服器：沒有回應' };" ^
  "if (Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\") { Write-Output 'cloudflared：執行中' } else { Write-Output 'cloudflared：沒有執行' }"
echo.
echo ---- logs\run_bots.log 最後 15 行 ----
powershell -NoProfile -Command "if (Test-Path logs\run_bots.log) { Get-Content logs\run_bots.log -Encoding utf8 -Tail 15 } else { '（還沒有紀錄）' }"
pause
