# EEG 1D-CNN Pipeline - Recording → Preprocessing → Training → Live Control

A full, end-to-end pipeline to collect EEG data over LSL, preprocess it into overlapping windows, train a lightweight 1D-CNN classifier, and run real-time (or emulated) control via a Flask UI. The UI can optionally send classification-driven commands to an ESP8266 over WebSocket for robotic control.

**Status:** working pipeline; `control_live.py` is not fully tested. Use `control_via_recordings.py` to emulate the live loop from saved data.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [EEG & Windowing Assumptions](#eeg--windowing-assumptions)
- [End-to-End Workflow](#end-to-end-workflow)
  - [1) Record training data](#1-record-training-data)
  - [2) Preprocess to windows](#2-preprocess-to-windows)
  - [3) Train the 1D-CNN](#3-train-the-1d-cnn)
  - [4a) Emulated “live” control (recommended)](#4a-emulated-live-control-recommended)
  - [4b) Live classification from LSL (experimental)](#4b-live-classification-from-lsl-experimental)
- [Frontend & ESP8266 Control](#frontend--esp8266-control)
- [Visualizations](#visualizations)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

---

## Features

- Data collection over LSL with a simple web UI (`record_training_data.py`).
- Preprocessing: band-pass filter, normalization, and sliding windows (`preprocessing/preprocessing.py`).
- Training: 1D-CNN with early stopping, metrics plots, and saved weights (`model_gen/cnn_model_gen.py`).
- Live classification from LSL stream (`control_live.py`) or emulated from disk (`control_via_recordings.py`).
- Browser UI for model selection and monitoring (`control_pages/*.html`, `static/control_live.js`).
- ESP8266 integration: predictions mapped to JSON commands sent over WebSocket.

---

## Requirements

- Python 3.11
- Packages:
  - numpy, scipy, matplotlib, flask, scikit-learn
  - torch (CPU or CUDA)
  - pylsl (for LSL)

Install (CPU-only example):

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    source .venv/bin/activate
    pip install numpy scipy matplotlib flask scikit-learn pylsl
    pip install torch --index-url https://download.pytorch.org/whl/cpu

If you have CUDA, install the matching Torch build for your CUDA version. 

---

## EEG & Windowing Assumptions

- Sample rate: 500 Hz
- Target channels (8): indices `[14, 15, 16, 40, 41, 42, 47, 48]` mapping to `["C3","Cz","C4","FC3","FCz","FC4","CP3","CP4"]`
- Live & training window length: **1000 samples** (≈ 2.0 s at 500 Hz)

In `preprocessing/preprocessing.py`, set to match:

    SAMPLE_RATE = 500
    WINDOW_SIZE_SECONDS = 2.0   # 2 seconds → 1000 samples
    STEP_SIZE_SECONDS   = 0.25  # 75% overlap (adjust as needed)

This ensures the saved arrays are `(N, 1000, 8)`, consistent with the model and `control_live.py` (`SAMPLE_WINDOW_SIZE = 1000`).

---

## End-to-End Workflow

### 1) Record training data

    python record_training_data.py
    # browser opens at http://localhost:5000

- Classes: `["rest","left_hand","right_hand","leg_movement","both_hands"]`
- Defaults: `SAMPLES_PER_TRIAL = 60000` (~2 min @ 500 Hz), `NUM_TRIALS_PER_CLASS = 1`
- Output: `data/raw/training/<TIMESTAMP>/*.npy`

Sanity-check your stream first if needed:

    python test_streaming.py
    python verbose_test_streaming.py

### 2) Preprocess to windows

    python -m preprocessing.preprocessing <TIMESTAMP>

- Filter → normalize → window
- Output: `data/processed/training/<TIMESTAMP>/preprocessed_data.npy`, `labels.npy`

### 3) Train the 1D-CNN

    python -m model_gen.cnn_model_gen <TIMESTAMP>

- Architecture: two Conv1d blocks → global average pool → dropout(0.2) → dense
- Early stopping: `patience=30`, `min_delta=0.001`
- Defaults: `epochs=30`, `batch_size=32`, `lr=5e-4`, `weight_decay=1e-4`
- Saves the best model state by validation loss, then writes weights to:
  - `data/model/<TIMESTAMP>/eeg_1dcnn_model.pth`

### 4a) Emulated “live” control (recommended)

    python control_via_recordings.py
    # http://localhost:5000

- Select `<TIMESTAMP>` model.
- Choose a class; server picks a random window from disk and classifies it.
- JSON response: `{"true_class": "...", "predicted_class": "..."}`.

### 4b) Live classification from LSL (experimental)

    python control_live.py
    # http://localhost:5000

- `/select_model`: loads weights and label set.
- Background thread:
  - Pulls LSL samples
  - Selects target channels
  - `bandpass_filter` → `normalise`
  - Classifies most recent 1000 samples
  - Debounce/hold logic stabilizes UI updates
- `/status`: returns the latest prediction.

---

## Frontend & ESP8266 Control

UI (served by Flask):

- `control_live.html`: select model, display current prediction, WebSocket status, and console.
- `static/control_live.js`:
  - Polls `/status` every 500 ms
  - When prediction changes, maps it to a servo action and sends JSON over WebSocket.

Default WebSocket endpoint (edit as needed in `static/control_live.js`):

    const WEBSOCKET_URL = "ws://192.168.4.1:81";

Prediction → Action mapping:

    const actionMap = {
      left_hand:   { servoId: 1, direction: "towards_min" },
      right_hand:  { servoId: 2, direction: "towards_max" },
      both_hands:  { servoId: 3, direction: "towards_max" },
      leg_movement:{ servoId: 0, direction: "towards_max" },
      rest: null
    };

Payload sent to ESP8266:

    {
      "action": "toggle_servo",
      "speed": "medium",
      "servoId": 1,
      "direction": "towards_min"
    }

---

## Visualizations

Class × Channel heatmap:

    python channel_heatmap.py <TIMESTAMP>

Per-class mean ± std over time:

    python plot_classwise_average_with_standard_deviation.py <TIMESTAMP>

---

## Troubleshooting

- “No EEG stream found.”
  - Ensure the EEG device is streaming via LSL.
  - Try `test_streaming.py` or `verbose_test_streaming.py`.
  - Increase resolve timeouts if necessary.

- IndexError: channel indices out of range
  - The incoming stream may have fewer channels or a different ordering.
  - Update `TARGET_CHANNEL_INDICES` to match your device’s LSL channel order.

- Shape mismatch (expected `(N, 1000, 8)` but got `(N, 500, 8)`)
  - Align window sizes across preprocessing, training, and live scripts.

- Plots not visible (headless environment)
  - Save figures with `plt.savefig(...)` instead of `plt.show()`.

- ESP8266 not receiving commands
  - Check `WEBSOCKET_URL`, network connectivity, and JSON schema.

---

## Notes

- `control_live.py` is not fully tested; start with `control_via_recordings.py` to validate model and labels.
- Labels are extracted from filenames during preprocessing (`<label>_trial*.npy`), so maintain consistent naming.
- The training script prints the loaded labels and restores the best model state before saving.


