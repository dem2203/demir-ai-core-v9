# -*- coding: utf-8 -*-
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Demir AI v10 - Dashboard")

# Dizin ayarları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "dashboard_data.json")

# Klasörleri oluştur (Yoksa hata vermesin)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Anlık verileri JSON dosyasından oku"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JSONResponse(content=data)
        else:
            return JSONResponse(content={"status": "waiting_data"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Standalone çalıştırılacaksa
    uvicorn.run(app, host="0.0.0.0", port=8000)
