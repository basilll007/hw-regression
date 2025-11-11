# src/main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from inference import AFRREnergyPricePredictor

app = FastAPI(title="Finland aFRR Up-Price Predictor")

# Static + template setup
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

# Load model once
predictor = AFRREnergyPricePredictor(model_type="onnx")

class InputPayload(BaseModel):
    features: list[float]

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: InputPayload):
    features = np.array(data.features, dtype=np.float32)
    result = predictor.predict(features)
    return {"predicted_up_price": result}
