import numpy as np
from pathlib import Path
data = np.load("data/raw/training/zpot2/right_hand_trial1.npy", allow_pickle=True)
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("First element type:", type(data[0]) if len(data) > 0 else "empty")

for file in sorted(Path("data/raw/training/zpot2").glob("*.npy")):
    data = np.load(file)
    print(file.name, data.shape)
