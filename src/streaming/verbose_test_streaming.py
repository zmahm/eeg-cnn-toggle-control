import os
import time
import numpy as np
from pylsl import StreamInlet, resolve_byprop

# Set up output directory
SAVE_DIR = "test_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# Connect to EEG stream
print("Looking for EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=5)
if not streams:
    raise RuntimeError("No EEG stream found. Make sure your EEG device is streaming over LSL.")

print("Stream found.")
inlet = StreamInlet(streams[0])
info = inlet.info()
channel_count = info.channel_count()
print(f"\nNumber of channels: {channel_count}\n")

# Verbose channel info printout
print("Channel details:")
channel_indices = []
ch = info.desc().child("channels").child("channel")
for i in range(channel_count):
    label = ch.child_value("label")
    ch_type = ch.child_value("type")
    unit = ch.child_value("unit")

    # Print full channel info
    print(f"  Channel {i:02}: type='{ch_type}', unit='{unit}', label='{label}'")

    # Collect EEG channel indices only (type == 'ref')
    if ch_type == 'ref':
        channel_indices.append(i)

    ch = ch.next_sibling()

print(f"\nEEG channel indices (type='ref'): {channel_indices}")

# Save stream metadata as XML
xml_path = os.path.join(SAVE_DIR, "stream_info.xml")
with open(xml_path, "w", encoding="utf-8") as f:
    f.write(info.as_xml())
print(f"\nSaved stream metadata to: {xml_path}")

# Record EEG samples from 'ref' channels only
sample_rate = 250
record_seconds = 3
num_samples = sample_rate * record_seconds
samples = []

print(f"\nRecording {record_seconds} seconds of EEG data from {len(channel_indices)} channels...")
while len(samples) < num_samples:
    sample, _ = inlet.pull_sample(timeout=1.0)
    if sample:
        selected = [sample[i] for i in channel_indices]
        samples.append(selected)
    else:
        print("No sample received.")
    time.sleep(0.001)

samples = np.array(samples)
print(f"Recorded data shape: {samples.shape}")

# Save .npy file
npy_path = os.path.join(SAVE_DIR, "test_data.npy")
np.save(npy_path, samples)
print(f"Saved raw data to: {npy_path}")

# Save .csv file
csv_path = os.path.join(SAVE_DIR, "test_data.csv")
np.savetxt(csv_path, samples, delimiter=",")
print(f"Saved CSV data to: {csv_path}")
