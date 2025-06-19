import os
import time
import numpy as np
import sys
from threading import Thread
from flask import Flask, render_template, jsonify, request
from datetime import datetime

# Optional LSL import (handled later)
try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    StreamInlet = None
    

# Configuration settings
CLASSES = ["rest", "left_hand", "right_hand", "leg_movement", "both_hands"]
RECORD_SECONDS = 5  # Duration of each trial in seconds
NUM_TRIALS_PER_CLASS = 10  # Number of recordings per class
SAMPLE_RATE = 250  # Expected sample rate from the EEG device
TARGET_CHANNELS = ['C3', 'C4', 'Cz', 'FCz', 'CPz', 'Fz', 'Pz', 'Oz']
# C3/C4: hand motor areas, Cz: foot/leg control, FCz: motor planning/intention,
# CPz: sensorimotor integration, Fz: attention/intention, Pz: cognitive control, Oz: resting/focus state

TEST_MODE = False  # Set to False when using a real EEG device
SINGLE_SAMPLE_PER_CLASS = False  # If True, only one sample will be recorded per class (for emulation/testing)

# Function to generate the proper save directory based on mode and timestamp
def generate_save_path(base_dir, is_emulation=False):
    """
    Create a time-stamped subdirectory under 'training' or 'emulate',
    depending on whether full training or a single-sample emulation run is being performed.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subfolder = "emulate" if is_emulation else "training"
    path = os.path.join(base_dir, subfolder, timestamp)
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

def find_channel_indices(inlet):
    # Map target channel labels to their indices in the stream
    info = inlet.info()
    indices = []
    channel = info.desc().child("channels").child("channel")
    for i in range(info.channel_count()):
        label = channel.child_value("label")
        if label in TARGET_CHANNELS:
            indices.append(i)
        channel = channel.next_sibling()
    return indices

def record_trials():
    SAVE_DIR = generate_save_path(BASE_DIR, is_emulation=SINGLE_SAMPLE_PER_CLASS)
    # Connect to EEG stream via LSL (if not in test mode)
    inlet = None
    channel_indices = list(range(len(TARGET_CHANNELS)))  # Dummy if in test mode

    if not TEST_MODE:
        print("Looking for EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=5)
        if not streams:
            raise RuntimeError("No EEG stream found. Ensure your EEG device is streaming over LSL.")
        inlet = StreamInlet(streams[0])
        channel_indices = find_channel_indices(inlet)

    # Loop through each mental task
    for class_label in CLASSES:
        print(f"\nTask: {class_label}")
        print("Get ready to focus on the mental task when prompted.")
        time.sleep(3)

        # Determine number of trials based on SINGLE_SAMPLE_PER_CLASS
        num_trials = 1 if SINGLE_SAMPLE_PER_CLASS else NUM_TRIALS_PER_CLASS
        recording_data['current_class'] = class_label

        for trial in range(num_trials):
            recording_data['trial'] = trial + 1
            samples = []
            print(f"Recording trial {trial+1}/{num_trials} for class '{class_label}'")

            # Record EEG data for fixed duration with live countdown
            start_time = time.time()
            while time.time() - start_time < RECORD_SECONDS:
                elapsed = time.time() - start_time
                recording_data['countdown'] = max(0, RECORD_SECONDS - elapsed)

                if TEST_MODE:
                    sample = np.random.randn(len(channel_indices))
                else:
                    sample, _ = inlet.pull_sample()
                    sample = [sample[i] for i in channel_indices]
                samples.append(sample)
                time.sleep(1.0 / SAMPLE_RATE)

            samples = np.array(samples)

            # Save trial data to file
            filename = f"{class_label}_trial{trial+1}.npy"
            filepath = os.path.join(SAVE_DIR, filename)
            np.save(filepath, samples)

            print(f"Trial {trial+1} saved as: {filename}")
            time.sleep(1)  # Small pause before next trial

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
    # Reset all status values so a new session can be started
    recording_data['status'] = 'idle'
    recording_data['current_class'] = ''
    recording_data['trial'] = 0
    recording_data['countdown'] = 0.0
    return '', 204


if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://localhost:5000")
    app.run(debug=True)
