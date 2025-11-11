# src/inference.py
import torch
import onnxruntime as ort
import numpy as np
import joblib
import os

class AFRREnergyPricePredictor:
    """Handles model loading and inference for Finland aFRR Up-price prediction."""

    def __init__(self, model_type="onnx"):
        self.model_type = model_type.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve base directory dynamically
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models")
        scaler_dir = os.path.join(base_dir, "scalers")

        # Load scalers
        self.scaler_X = joblib.load(os.path.join(scaler_dir, "scaler_X.pkl"))
        self.scaler_y = joblib.load(os.path.join(scaler_dir, "scaler_y.pkl"))

        # Load model
        if self.model_type == "torch":
            model_path = os.path.join(model_dir, "afrr_up_model.pt")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        elif self.model_type == "onnx":
            model_path = os.path.join(model_dir, "afrr_up_model.onnx")
            self.session = ort.InferenceSession(model_path)
        else:
            raise ValueError("model_type must be 'torch' or 'onnx'")

    def predict(self, features: np.ndarray):
        """features: array-like shape (n_features,) or (1,n_features)"""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        x_scaled = self.scaler_X.transform(features).astype(np.float32)

        if self.model_type == "torch":
            with torch.no_grad():
                inp = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
                output = self.model(inp).cpu().numpy()
        else:
            output = self.session.run(None, {"input": x_scaled})[0]

        y_pred = self.scaler_y.inverse_transform(output)[0, 0]
        return float(y_pred)
