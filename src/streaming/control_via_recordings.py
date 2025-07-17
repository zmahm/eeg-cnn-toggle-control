import os
import webbrowser
from flask import Flask, render_template, jsonify, request
from pathlib import Path

app = Flask(__name__, template_folder="control_pages", static_folder="static")

# Base directories
MODEL_BASE_DIR = Path("./data/model").resolve()
PROCESSED_BASE_DIR = Path("./data/processed/training").resolve()

# State variables to be used client-side via template rendering
selected_model_name = None

@app.route("/")
def index():
    # List all subdirectories that contain the model file
    model_folders = []
    for folder in MODEL_BASE_DIR.iterdir():
        if folder.is_dir() and (folder / "eeg_1dcnn_model.pth").exists():
            model_folders.append(folder.name)
    return render_template("control_via_recordings.html", model_folders=model_folders)

@app.route("/select_model", methods=["POST"])
def select_model():
    global selected_model_name
    model_name = request.json.get("model_name")

    model_path = MODEL_BASE_DIR / model_name / "eeg_1dcnn_model.pth"
    data_path = PROCESSED_BASE_DIR / model_name / "preprocessed_data.npy"

    if not model_path.exists() or not data_path.exists():
        return jsonify({"error": "Model or data not found."}), 400

    selected_model_name = model_name
    return jsonify({"message": f"Model '{model_name}' loaded successfully."})

@app.route("/get_current_model")
def get_current_model():
    return jsonify({"selected_model": selected_model_name})

if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
