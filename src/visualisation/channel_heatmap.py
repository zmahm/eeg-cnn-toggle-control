import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Channel names in order
CHANNEL_NAMES = ["C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CP4"]

def load_data(processed_dir):
    data = np.load(processed_dir / "preprocessed_data.npy")  # (N, 1000, 8)
    labels = np.load(processed_dir / "labels.npy")  # (N,)
    return data, labels

def compute_class_channel_activity(data, labels):
    class_channel_activity = defaultdict(lambda: [])

    for segment, label in zip(data, labels):
        abs_amplitude = np.mean(np.abs(segment), axis=0)  # (8,)
        class_channel_activity[label].append(abs_amplitude)

    heatmap_matrix = []
    class_names = sorted(class_channel_activity.keys())

    for cls in class_names:
        avg_per_channel = np.mean(class_channel_activity[cls], axis=0)
        heatmap_matrix.append(avg_per_channel)

    heatmap_matrix = np.array(heatmap_matrix).T  # shape: (channels, classes)
    return heatmap_matrix, class_names

def plot_class_channel_heatmap(matrix, class_names):
    plt.figure(figsize=(10, 6))
    im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(CHANNEL_NAMES)), CHANNEL_NAMES)
    plt.xlabel("Class")
    plt.ylabel("EEG Channel")
    plt.colorbar(im, label="Mean Absolute Amplitude")
    plt.title("EEG Activity Heatmap (Channel × Class)")
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python eeg_class_channel_heatmap.py <recording_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    base = Path(__file__).resolve().parent
    processed = base / '../../data/processed/training' / folder

    if not processed.exists():
        print(f"Processed folder not found: {processed}")
        sys.exit(1)

    data, labels = load_data(processed)
    matrix, class_names = compute_class_channel_activity(data, labels)
    plot_class_channel_heatmap(matrix, class_names)

if __name__ == "__main__":
    main()
