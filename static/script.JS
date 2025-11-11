// static/script.js

// ------------- SAMPLE VECTOR -------------
const sampleFeatures = [
  -5.23, 1.82, 3.4, 0.0, 1008.12, 7.45, 84.7, 243.1,
  5.2, 10.0, 0, 24850.3, 3248613.6, 25324.4, 23, 2,
  11, 0, 28.1, -21.4, 3.6, 0.0, 1007.9, 7.8, 84.9,
  243.4, 0.0, 0.0, 24.6, -18.7, 3.2, 0.0, 1008.1,
  8.1, 85.2, 244.2, 0.0, 0.0, 25.1, -17.3, 3.0,
  0.0, 1008.4, 8.3, 85.0, 244.8, 0.0, 0.0, 24.8,
  -18.5, 3.1, 0.0, 1008.0, 8.0
];

// ------------- PREDICT (manual text) -------------
async function predict() {
  const featureText = document.getElementById("features").value.trim();
  const features = featureText.split(",").map(Number).filter(v => !isNaN(v));

  if (features.length === 0) {
    alert("Please enter valid numeric values separated by commas.");
    return;
  }

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });

  const data = await response.json();
  showResult(data);
}

// ------------- LOAD SAMPLE BUTTON -------------
function loadSample() {
  document.getElementById("features").value = sampleFeatures.join(", ");
  document.getElementById("result").innerText = "✅ Sample data loaded into input field.";
}

// ------------- DOWNLOAD SAMPLE JSON -------------
function downloadSample() {
  const jsonContent = JSON.stringify({ features: sampleFeatures }, null, 2);
  const blob = new Blob([jsonContent], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sample_features.json";
  a.click();
  URL.revokeObjectURL(url);
}

// ------------- UPLOAD JSON -------------
async function uploadFile() {
  const fileInput = document.getElementById("jsonFile");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please choose a JSON file first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload_json", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  showResult(data);
}

// ------------- DISPLAY RESULT -------------
function showResult(data) {
  const resultDiv = document.getElementById("result");
  if (data.error) {
    resultDiv.innerText = "⚠️ " + data.error;
    resultDiv.style.color = "#d93025";
  } else {
    resultDiv.innerText = `🔹 Predicted aFRR Up Price: ${data.predicted_up_price.toFixed(2)} EUR/MWh`;
    resultDiv.style.color = "#2e7d32";
  }
}
