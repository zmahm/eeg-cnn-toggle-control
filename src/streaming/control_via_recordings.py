import os
import webbrowser
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import Flask, render_template, jsonify, request
from pathlib import Path
from model_gen.torch_baseline_model_gen import EEGClassifier1DCNN  # Use the proper shared model definition

app = Flask(__name__, template_folder="control_pages", static_folder="static")

# Globals
MODEL = None
CLASS_NAMES = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_model_name = None

# Paths
MODEL_BASE_DIR = Path("./data/model").resolve()
PROCESSED_BASE_DIR = Path("./data/processed/training").resolve()


@app.route("/")
def index():
    """Render the control interface and list available trained models."""
    available_models = [
        folder.name for folder in MODEL_BASE_DIR.iterdir()
        if folder.is_dir() and (folder / "eeg_1dcnn_model.pth").exists()
    ]
    return render_template("control_via_recordings.html", model_folders=available_models)


@app.route("/select_model", methods=["POST"])
def select_model():
    """Load the selected model and prepare class names for classification."""
    global selected_model_name, MODEL, CLASS_NAMES

    model_name = request.json.get("model_name")
    selected_model_name = model_name

    # Resolve paths
    model_path = MODEL_BASE_DIR / model_name / "eeg_1dcnn_model.pth"
    data_path = PROCESSED_BASE_DIR / model_name / "preprocessed_data.npy"
    labels_path = PROCESSED_BASE_DIR / model_name / "labels.npy"

    # Sanity check: ensure files exist
    if not model_path.exists() or not data_path.exists() or not labels_path.exists():
        return jsonify({"error": "Required model or data files not found."}), 400

    # Load class names from labels file
    label_strings = np.load(labels_path)
    CLASS_NAMES = sorted(set(label_strings))

    # Infer number of input channels from saved data shape
    sample_data = np.load(data_path)
    if sample_data.ndim != 3:
        return jsonify({"error": "Invalid data shape."}), 500

    input_channels = sample_data.shape[2]  # (num_windows, time, channels)

    # Initialise model and load weights
    MODEL = EEGClassifier1DCNN(input_channels=input_channels, num_classes=len(CLASS_NAMES)).to(DEVICE)
    MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    MODEL.eval()

    return jsonify({"message": f"Model '{model_name}' loaded successfully."})


@app.route("/get_current_model")
def get_current_model():
    """Return the currently selected model name."""
    return jsonify({"selected_model": selected_model_name})


@app.route("/classify_sample", methods=["POST"])
def classify_sample():
    """Select a random sample of a requested class, classify it, and return the prediction."""
    if selected_model_name is None or MODEL is None:
        return jsonify({"error": "No model is currently loaded."}), 400

    request_data = request.get_json()
    target_class = request_data.get("class_name")

    if not target_class:
        return jsonify({"error": "Missing 'class_name' in request body."}), 400

    # Load sample data and labels
    processed_dir = PROCESSED_BASE_DIR / selected_model_name
    try:
        data = np.load(processed_dir / "preprocessed_data.npy")
        labels = np.load(processed_dir / "labels.npy")
    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

    # Find windows for the requested class
    matching_indices = np.where(labels == target_class)[0]
    if len(matching_indices) == 0:
        return jsonify({"error": f"No samples found for class '{target_class}'."}), 404

    # Choose one at random
    sample_index = np.random.choice(matching_indices)
    sample_window = data[sample_index]  # (time, channels)

    # Transpose to (channels, time) and add batch dimension
    input_tensor = torch.tensor(sample_window.T).unsqueeze(0).float().to(DEVICE)

    # Run through model
    with torch.no_grad():
        logits = MODEL(input_tensor)
        predicted_idx = torch.argmax(logits, dim=1).item()
        predicted_class = CLASS_NAMES[predicted_idx]

    return jsonify({
        "true_class": target_class,
        "predicted_class": predicted_class
    })


if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
