import os
import time
import numpy as np
import sys
from threading import Thread
from flask import Flask, render_template, jsonify, request
from datetime import datetime

# Optional LSL import (only used if not in test mode)
try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    StreamInlet = None

# Configuration settings
CLASSES = ["rest", "left_hand", "right_hand", "leg_movement", "both_hands"]
SAMPLES_PER_TRIAL = 60000  # 2 minutes at 500 Hz (estimated)
NUM_TRIALS_PER_CLASS = 1  # One trial per class (allows for more windowing)

# Array-based channel indices that match the target EEG channels:
# Index : Channel Name
# 14    : C3
# 15    : Cz
# 16    : C4
# 40    : FC3
# 41    : FCz
# 42    : FC4
# 47    : CP3
# 48    : CP4
# grnd and cpz are needed for impedence testing
TARGET_CHANNEL_INDICES = [14, 15, 16, 40, 41, 42, 47, 48]

TEST_MODE = False  # Set to False when using a real EEG device

# Function to generate the proper save directory based on timestamp

def generate_save_path(base_dir):
    """
    Create a time-stamped subdirectory under 'training' for this session.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, "training", timestamp)
    os.makedirs(path, exist_ok=True)
    return path

# Resolve and prepare the appropriate save directory for this session
BASE_DIR = "data/raw"

# Flask app for local web interface
app = Flask(__name__, template_folder='training_pages')

# Status dictionary used for UI feedback
recording_data = {
    'status': 'idle',
    'current_class': '',
    'trial': 0,
    'countdown': 0.0
}

def record_trials():
    SAVE_DIR = generate_save_path(BASE_DIR)
    inlet = None
    channel_indices = list(range(len(TARGET_CHANNEL_INDICES)))  # Dummy fallback if in test mode

    if not TEST_MODE:
        print("Looking for EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=5)
        if not streams:
            raise RuntimeError("No EEG stream found. Ensure EEG device is streaming over LSL.")
        inlet = StreamInlet(streams[0])
        channel_indices = TARGET_CHANNEL_INDICES

    for class_label in CLASSES:
        print(f"\nTask: {class_label}")
        print("Get ready to focus on the mental task when prompted.")
        time.sleep(3)

        recording_data['current_class'] = class_label

        for trial in range(NUM_TRIALS_PER_CLASS):
            recording_data['trial'] = trial + 1
            samples = []
            print(f"Recording trial {trial+1}/{NUM_TRIALS_PER_CLASS} for class '{class_label}'")

            while len(samples) < SAMPLES_PER_TRIAL:
                if TEST_MODE:
                    sample = np.random.randn(len(channel_indices))
                else:
                    sample, _ = inlet.pull_sample()
                    sample = [sample[i] for i in channel_indices]
                samples.append(sample)

            samples = np.array(samples, dtype=np.float32)
            print("Saved sample shape:", samples.shape)

            # Save the recorded trial to file
            filename = f"{class_label}_trial{trial+1}.npy"
            filepath = os.path.join(SAVE_DIR, filename)
            np.save(filepath, samples)
            print(f"Trial {trial+1} saved as: {filename}")
            time.sleep(1)

    recording_data['status'] = 'completed'
    print("\nAll recordings completed.")
    print(f"Saved EEG data is located in: {SAVE_DIR}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(recording_data)

@app.route('/start', methods=['POST'])
def start():
    if recording_data['status'] == 'idle':
        recording_data['status'] = 'recording'
        thread = Thread(target=record_trials)
        thread.start()
    return '', 204

@app.route('/reset', methods=['POST'])
def reset():
    # Reset status so a new session can be started
    recording_data['status'] = 'idle'
    recording_data['current_class'] = ''
    recording_data['trial'] = 0
    recording_data['countdown'] = 0.0
    return '', 204

if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(debug=True)