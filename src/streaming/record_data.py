import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import os
import sys

# Configuration settings
SAVE_DIR = "data/raw"  # Directory where recordings will be saved
CLASSES = ["rest", "left_hand", "right_hand", "leg_movement", "both_hands"]
RECORD_SECONDS = 5  # Duration of each trial in seconds
NUM_TRIALS_PER_CLASS = 10  # Number of recordings per class
SAMPLE_RATE = 250  # Expected sample rate from the EEG device
TARGET_CHANNELS = ['C3', 'C4', 'Cz', 'FCz', 'CPz', 'Fz', 'Pz', 'Oz']  
# C3/C4: hand motor areas, Cz: foot/leg control, FCz: motor planning/intention, CPz: sensorimotor integration, 
# Fz: attention/intention, Pz: cognitive control, Oz: resting/focus state


# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Connect to EEG stream via LSL
print("Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("No EEG stream found. Ensure your EEG device is streaming over LSL.")
inlet = StreamInlet(streams[0])
info = inlet.info()

# Map target channel labels to their indices in the stream
channel_indices = []
channel = info.desc().child("channels").child("channel")
for i in range(info.channel_count()):
    label = channel.child_value("label")
    if label in TARGET_CHANNELS:
        channel_indices.append(i)
    channel = channel.next_sibling()

if not channel_indices:
    raise ValueError("None of the specified target channels were found in the EEG stream.")

print(f"EEG stream connected. Using channels: {channel_indices} corresponding to {TARGET_CHANNELS}")

# Loop through each mental task
for class_label in CLASSES:
    print(f"\nTask: {class_label}")
    print("Get ready to focus on the mental task when prompted.")
    time.sleep(3)

    # Record multiple trials per class
    for trial in range(NUM_TRIALS_PER_CLASS):
        input(f"\nPress Enter to start trial {trial+1}/{NUM_TRIALS_PER_CLASS} for '{class_label}'...")

        print(f"Recording for {RECORD_SECONDS} seconds. Focus on: {class_label}")
        samples = []

        # Record EEG data for fixed duration with live countdown
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # Update countdown timer
            remaining = max(0, RECORD_SECONDS - elapsed)
            sys.stdout.write(f"\rTime remaining: {remaining:.1f} seconds")
            sys.stdout.flush()

            # Stop after duration
            if elapsed >= RECORD_SECONDS:
                break

            # Read full sample and filter to target channels only
            sample, _ = inlet.pull_sample()
            filtered_sample = [sample[i] for i in channel_indices]
            samples.append(filtered_sample)

        print()  # Print newline after timer
        samples = np.array(samples)

        # Save trial data to file
        filename = f"{class_label}_trial{trial+1}.npy"
        filepath = os.path.join(SAVE_DIR, filename)
        np.save(filepath, samples)

        print(f"Trial {trial+1} saved as: {filename}")
        time.sleep(1)  # Small pause before next trial

print("\nAll recordings completed.")
print(f"Saved EEG data is located in: {SAVE_DIR}")
