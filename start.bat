@echo off
cd /d D:\ScholarPulse
echo ============================================
echo   ScholarPulse 学脉 正在启动...
echo   学生端: http://127.0.0.1:8000
echo   教师端: http://127.0.0.1:8000/teacher
echo ============================================
python webapp\app.py
pause
