# ⚡ Finland aFRR Up-Price Predictor

A machine learning application that predicts **Automatic Frequency Restoration Reserve (aFRR) Up-regulation prices** for the Finnish energy market.
The model uses **PyTorch** for regression, exports to **ONNX** for production inference, and is served via **FastAPI** with a simple **HTML/JS frontend**.

---

## 🧠 Overview

The aFRR market helps maintain grid stability by adjusting power frequency when renewable generation fluctuates.
This project predicts hourly **Up-regulation prices** using a combination of:

* Market variables (spot price, regulation prices)
* Meteorological indicators (wind, temperature, pressure)
* Demand metrics (consumption & forecasts)
* Temporal features (hour, weekday, month, weekend flags)

---

## 🏗️ What We Built

1. **Data Processing (Colab):**

   * Loaded & explored the dataset (`Finland aFRR energy market and weather data`)
   * Created time-based, lagged, and rolling features (total 54 inputs)
   * Normalized inputs using `StandardScaler`
   * Split into train/val/test sets

2. **Model Development:**

   * Built a deep regression model in **PyTorch**
   * Trained on Google Colab (GPU-enabled)
   * Evaluated using MAE, RMSE, R² metrics
   * Exported the model in both `.pt` and `.onnx` formats

3. **Inference & API:**

   * Built a **FastAPI** backend (`src/main.py`)
   * Added an inference engine (`src/inference.py`) for both Torch & ONNX
   * Integrated a **frontend** with:

     * Manual input (comma-separated features)
     * JSON upload prediction
     * “Load Sample” & “Download Sample JSON” buttons

4. **Deployment Setup:**

   * Organized folder structure for production:

     ```
     finland-afrr-predictor/
     ├── models/
     ├── scalers/
     ├── src/
     ├── static/
     ├── templates/
     ├── requirements.txt
     └── Procfile
     ```
   * Ready for cloud deployment via **Render.com**

---

## 🚀 How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/finland-afrr-predictor.git
   cd finland-afrr-predictor/src
   ```

2. **Install dependencies**

   ```bash
   pip install -r ../requirements.txt
   ```

3. **Start FastAPI**

   ```bash
   uvicorn main:app --reload
   ```

4. **Open your browser**

   ```
   http://127.0.0.1:8000
   ```

5. **Use the app**

   * Paste 54 comma-separated features
   * Or upload a JSON file
   * Or click “Load Sample” for demo data


