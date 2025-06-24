import os
import numpy as np
import scipy.signal as signal
from pathlib import Path
from datetime import datetime

# Sample rate of the EEG data (in Hz)
SAMPLE_RATE = 500

# Duration and step of each processing window in seconds
WINDOW_SIZE_SECONDS = 2
STEP_SIZE_SECONDS = 1

# Compute the number of samples per window and step
WINDOW_SIZE = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)
STEP_SIZE = int(STEP_SIZE_SECONDS * SAMPLE_RATE)

# Minimum acceptable length for any trial (e.g. 1 full window)
MIN_REQUIRED_SAMPLES = WINDOW_SIZE


def bandpass_filter(data, lowcut=0.5, highcut=40, fs=SAMPLE_RATE, order=5):
    # Apply a Butterworth bandpass filter to remove noise outside EEG range
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def normalise(data):
    # Standardise the data to have zero mean and unit variance
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def window_data(data, window_size, step_size):
    # Slice the data into overlapping windows
    segments = []
    for start in range(0, data.shape[0] - window_size + 1, step_size):
        segment = data[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

def preprocess_recordings(folder_name):
    # Define input and output paths relative to the script location
    script_dir = Path(__file__).resolve().parent
    raw_dir = (script_dir / '../../data/raw/training' / folder_name).resolve()
    print(raw_dir)
    save_dir = (script_dir / '../../data/processed/training' / folder_name).resolve()
    for file in raw_dir.glob("*.npy"):
        data = np.load(file)
        print(f"{file.name}: shape {data.shape}")

    # Create output directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []
    labels = []

    # Loop over all .npy files in the folder
    for file in sorted(raw_dir.glob("*.npy")):
        print(f"Found file: {file.name}")
        data = np.load(file)

        if data.shape[0] < MIN_REQUIRED_SAMPLES:
            print(f"Skipping {file.name}: only {data.shape[0]} samples (need at least {MIN_REQUIRED_SAMPLES})")
            continue

        # Apply preprocessing steps
        filtered = bandpass_filter(data)
        normed = normalise(filtered)
        segments = window_data(normed, WINDOW_SIZE, STEP_SIZE)

        if len(segments) == 0:
            print(f"Skipping {file.name}: no valid segments produced after windowing")
            continue

        # Extract label from filename (e.g. 'left_hand_trial1.npy')
        label = file.stem.split('_trial')[0]
        label_array = [label] * len(segments)

        all_segments.extend(segments)
        labels.extend(label_array)

        print(f"Processed {file.name} → {len(segments)} segments")

    if not all_segments:
        print("No valid data found. Exiting.")
        return

    all_trials_array = np.array(all_segments, dtype=np.float32)
    labels_array = np.array(labels)

    print("Final shape before saving:", all_trials_array.shape)
    np.save(save_dir / 'preprocessed_data.npy', all_trials_array)
    np.save(save_dir / 'labels.npy', labels_array)
    print(f"Saved preprocessed dataset to: {save_dir}")

if __name__ == "__main__":
    import argparse

    # Allows to input the folder name
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Name of the recording folder inside data/raw")
    args = parser.parse_args()

    preprocess_recordings(args.folder)
