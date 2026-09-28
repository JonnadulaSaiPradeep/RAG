@echo off
echo =========================================================
echo   KnowFlow AI - Enterprise Knowledge Intelligence Platform
echo   Author: JonnadulaSaiPradeep (231fa18264@gmail.com)
echo =========================================================
echo Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat
echo Starting Streamlit Web Application...
streamlit run app.py
pause
