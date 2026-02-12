# Script to restart Streamlit with fresh backend connection
Write-Host "Restarting Streamlit app..." -ForegroundColor Green
Set-Location "C:\Users\Akram Alimaad\Desktop\Phishing Detector"

# Activate virtual environment and run Streamlit
& ".\venv\Scripts\streamlit.exe" run frontend\streamlit_app.py
