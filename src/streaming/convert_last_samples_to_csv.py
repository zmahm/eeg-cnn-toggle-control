import os
import numpy as np
from datetime import datetime

def get_latest_subfolder(parent_folder):
    """
    Returns the full path to the most recently modified subfolder inside the given parent folder.
    """
    subfolders = [
        os.path.join(parent_folder, name)
        for name in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, name))
    ]
    if not subfolders:
        return None
    latest = max(subfolders, key=os.path.getmtime)
    return latest

def convert_npy_to_csv_in_folder(folder_path):
    """
    Converts all .npy files in the given folder to .csv files in the same location.
    """
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            npy_path = os.path.join(folder_path, file)
            csv_path = os.path.join(folder_path, file.replace(".npy", ".csv"))
            data = np.load(npy_path)
            np.savetxt(csv_path, data, delimiter=",")
            print(f"Converted: {file} → {os.path.basename(csv_path)}")

def main():
    base_path = os.path.join("data", "raw")
    for subfolder in ["training", "emulate"]:
        target_base = os.path.join(base_path, subfolder)
        if os.path.exists(target_base):
            latest_session = get_latest_subfolder(target_base)
            if latest_session:
                print(f"Processing latest folder in {subfolder}: {latest_session}")
                convert_npy_to_csv_in_folder(latest_session)
            else:
                print(f"No session folders found in {subfolder}")
        else:
            print(f"Directory does not exist: {target_base}")

if __name__ == "__main__":
    main()
