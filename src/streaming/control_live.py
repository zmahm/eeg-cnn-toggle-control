import os
import time
import numpy as np
import torch
import sys
from threading import Thread, Lock
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.nn.functional import softmax

# LSL import for EEG stream
from pylsl import StreamInlet, resolve_byprop

# Import EEG model definition
from model_gen.torch_baseline_model_gen import EEGClassifier1DCNN

from preprocessing.preprocessing import bandpass_filter, normalise

# ----- Configuration -----

MODEL_DIR = Path("./data/model").resolve()
PROCESSED_DIR = Path("./data/processed/training").resolve()
SAMPLE_WINDOW_SIZE = 1000  # Matches training window size (e.g., 2 seconds at 500Hz)
UPDATE_INTERVAL = 0.25  # seconds between predictions
# Match the indices used during training
TARGET_CHANNEL_INDICES = [14, 15, 16, 40, 41, 42, 47, 48]
CHANNEL_COUNT = len(TARGET_CHANNEL_INDICES)


# ----- Global state -----

app = Flask(__name__, template_folder="control_pages", static_folder="static")

selected_model_path = None
selected_class_names = []
latest_prediction = "none"
prediction_lock = Lock()

eeg_stream_active = False
stream_thread = None

# ----- Model and streaming setup -----

def load_model(model_folder):
    """
    Load the trained EEGClassifier1DCNN model and associated class labels.
    """
    model_path = MODEL_DIR / model_folder / "eeg_1dcnn_model.pth"
    label_path = PROCESSED_DIR / model_folder / "labels.npy"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Labels not found at: {label_path}")

    # Load and sort class names to match training
    label_strings = np.load(label_path)
    class_names = sorted(set(label_strings))

    # Define the model using known input shape and number of classes
    model = EEGClassifier1DCNN(input_channels=CHANNEL_COUNT, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    print(f"Model loaded: {model_path}")
    print(f"Class names: {class_names}")

    return model, class_names


def stream_and_classify(model):
    global latest_prediction, eeg_stream_active

    from preprocessing import bandpass_filter, normalise  # Import preprocessing methods

    try:
        print("Looking for EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=10)
        if not streams:
            print("[ERROR] No EEG stream found.")
            eeg_stream_active = False
            return

        inlet = StreamInlet(streams[0])
        buffer = []

        last_prediction = None
        prediction_hold_counter = 0
        prediction_hold_limit = 5  # Debounce duration

        while eeg_stream_active:
            sample, _ = inlet.pull_sample(timeout=1.0)
            if sample is None:
                print("[WARNING] No sample received.")
                continue

            # Extract only the 8 target EEG channels
            try:
                selected_sample = [sample[i] for i in TARGET_CHANNEL_INDICES]
            except IndexError:
                print(f"[ERROR] Sample has {len(sample)} channels, expected indices {TARGET_CHANNEL_INDICES}")
                continue

            print("[Stream] Got sample")
            buffer.append(selected_sample)

            if len(buffer) >= SAMPLE_WINDOW_SIZE:
                # Prepare recent window
                raw_window = np.array(buffer[-SAMPLE_WINDOW_SIZE:], dtype=np.float32)  # (1000, 8)

                # Apply preprocessing
                filtered = bandpass_filter(raw_window)
                normalised = normalise(filtered)

                # Reshape for model: (1, channels, samples)
                input_tensor = torch.tensor(normalised).transpose(0, 1).unsqueeze(0).float()

                # Classify
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = softmax(logits, dim=1)
                    predicted_idx = torch.argmax(probs, dim=1).item()
                    predicted_class = selected_class_names[predicted_idx]

                print("[DEBUG] Softmax probabilities:", probs.cpu().numpy())
                print("[DEBUG] Predicted class:", predicted_class)

                # Debounce logic
                if predicted_class != last_prediction or prediction_hold_counter >= prediction_hold_limit:
                    with prediction_lock:
                        latest_prediction = predicted_class
                    print(f"[Prediction] Updated: {predicted_class}")
                    last_prediction = predicted_class
                    prediction_hold_counter = 0
                else:
                    prediction_hold_counter += 1
                    print(f"[Prediction] Held: {last_prediction} ({prediction_hold_counter})")

                time.sleep(UPDATE_INTERVAL)
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"[FATAL] Stream thread crashed: {e}")
        eeg_stream_active = False





# ----- Flask routes -----

@app.route('/')
def index():
    available_models = [f.name for f in MODEL_DIR.iterdir() if (f / "eeg_1dcnn_model.pth").exists()]
    return render_template("control_live.html", model_folders=available_models)

@app.route('/select_model', methods=['POST'])
def select_model():
    global selected_model_path, selected_class_names, stream_thread, eeg_stream_active

    model_name = request.json.get("model_name")

    try:
        model, selected_class_names = load_model(model_name)
        selected_model_path = model_name
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Stop any existing EEG stream thread if already running
    if stream_thread and stream_thread.is_alive():
        print("Stopping previous stream...")
        eeg_stream_active = False
        stream_thread.join()  # Wait for it to terminate cleanly
        print("Previous stream stopped.")

    # Start new stream
    eeg_stream_active = True
    stream_thread = Thread(target=stream_and_classify, args=(model,))
    stream_thread.start()

    return jsonify({"message": f"Model '{model_name}' loaded and streaming started."})


@app.route('/status')
def status():
    with prediction_lock:
        return jsonify({"current_prediction": latest_prediction, "model": selected_model_path})

@app.route('/stop', methods=['POST'])
def stop():
    global eeg_stream_active
    eeg_stream_active = False
    return '', 204

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(debug=True)
