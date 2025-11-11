# src/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import os
from src.inference import AFRREnergyPricePredictor

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
app = FastAPI(title="Finland aFRR Up-Price Predictor")

# Base directory setup (go up one level from /src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Static and template folders
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ------------------------------------------------------------
# Load model (ONNX)
# ------------------------------------------------------------
predictor = AFRREnergyPricePredictor(model_type="onnx")

# ------------------------------------------------------------
# Pydantic model
# ------------------------------------------------------------
class InputPayload(BaseModel):
    features: list[float]

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: InputPayload):
    features = np.array(data.features, dtype=np.float32)
    result = predictor.predict(features)
    return {"predicted_up_price": result}
