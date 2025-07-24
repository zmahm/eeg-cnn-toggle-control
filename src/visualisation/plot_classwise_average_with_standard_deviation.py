import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# EEG config
SAMPLE_RATE = 500
WINDOW_SIZE = 1000  # 2 seconds
DURATION_SECONDS = WINDOW_SIZE / SAMPLE_RATE
TIME_AXIS = np.linspace(0, DURATION_SECONDS, WINDOW_SIZE)
CHANNEL_NAMES = ["C3", "Cz", "C4", "FC3", "FCz", "FC4", "CP3", "CP4"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_classwise_average.py <recording_folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    script_dir = Path(__file__).resolve().parent
    processed_dir = (script_dir / '../../data/processed/training' / folder_name).resolve()

    data_path = processed_dir / 'preprocessed_data.npy'
    labels_path = processed_dir / 'labels.npy'

    if not data_path.exists() or not labels_path.exists():
        print(f"Missing files in {processed_dir}")
        sys.exit(1)

    data = np.load(data_path)      # shape: (num_segments, 1000, 8)
    labels = np.load(labels_path)  # shape: (num_segments,)

    print(f"Loaded {len(data)} segments from {folder_name}")
    print("Unique labels:", set(labels))

    # Group by label
    class_data = defaultdict(list)
    for segment, label in zip(data, labels):
        class_data[label].append(segment)

    # Plot one figure per label
    for label, segments in class_data.items():
        segments = np.stack(segments)  # shape: (num_segments, 1000, 8)
        mean_wave = np.mean(segments, axis=0)
        std_wave = np.std(segments, axis=0)

        plt.figure(figsize=(12, 5))
        for ch in range(8):
            plt.plot(TIME_AXIS, mean_wave[:, ch], label=CHANNEL_NAMES[ch])
            plt.fill_between(
                TIME_AXIS,
                mean_wave[:, ch] - std_wave[:, ch],
                mean_wave[:, ch] + std_wave[:, ch],
                alpha=0.2
            )

        plt.title(f"Class: {label} ({len(segments)} segments)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (normalized)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
