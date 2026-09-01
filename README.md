# AI Driver Monitoring System (AI-DMS)

A real-time, modular driver drowsiness and distraction detection system built with **OpenCV**, **MediaPipe Face Mesh**, and **Python**, with an optional **machine learning** fatigue classifier.

The system watches the driver via webcam, computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose, fuses them into a 0–100 fatigue score, sounds an alarm when fatigue becomes critical, and logs every event to SQLite with automatic screenshot capture.

## Features

- Real-time webcam monitoring with FPS overlay
- 468-point face mesh tracking (MediaPipe, no Haar Cascade / dlib)
- Eye Aspect Ratio (EAR) based blink detection and eye-closure duration tracking
- Mouth Aspect Ratio (MAR) based yawn detection and counting
- 3D head pose estimation (yaw/pitch/roll) with Forward/Left/Right/Up/Down classification
- Rule-based fatigue scoring engine (SAFE / WARNING / CRITICAL), swappable for a trained ML model with zero changes to calling code
- Looping audible alarm with cooldown (no overlapping alarms)
- SQLite event logging (timestamp, EAR, MAR, blink count, fatigue score, alert type, screenshot path)
- Automatic incident screenshot capture
- Optional Random Forest / XGBoost ML classifier (Phase 9) trained on your own labeled session data

## Project Structure

```
Driver_Drowsiness_System/
├── app.py                  # Main application entry point
├── train_model.py          # Phase 9: ML model training script
├── config.py                # Central configuration (all thresholds/paths)
├── requirements.txt
├── detector/
│   ├── face_mesh.py         # MediaPipe Face Mesh wrapper (468 landmarks)
│   ├── eye_detector.py      # EAR + blink + closure duration
│   ├── yawn_detector.py     # MAR + yawn detection
│   ├── head_pose.py         # solvePnP-based head pose estimation
│   └── fatigue_engine.py    # Rule-based / ML fatigue scoring
├── utils/
│   ├── camera.py            # Webcam capture wrapper
│   ├── drawing.py           # All on-screen overlay/UI drawing
│   ├── geometry.py          # Distance/midpoint/clamp helpers
│   ├── alarm.py              # Pygame-based alarm system
│   └── logger.py            # Rotating file + console logger
├── database/
│   └── database.py          # SQLite persistence layer
├── assets/
│   ├── alert.wav             # Alarm sound (placeholder beep included)
│   └── screenshots/          # Auto-saved incident screenshots
├── logs/                     # Rotating application log files
├── reports/                  # Generated reports (future use)
└── models/                   # Trained ML model artifacts (.joblib)
```

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** MediaPipe requires Python 3.10 or 3.11. If installation fails on a newer Python version, install Python 3.11 specifically and recreate the virtual environment with it.

## Running the Application

```bash
python app.py
```

- Press **`q`** to quit.
- A live window shows the camera feed with face landmarks, EAR/MAR values, blink/yawn counts, head direction, and the fatigue score/status.
- When fatigue status becomes **CRITICAL**, an alarm sounds and a screenshot + database event are saved automatically.

## Configuration

All thresholds (EAR, MAR, head pose angles, fatigue weights, alarm cooldown, etc.) live in **`config.py`**. Nothing is hardcoded inside detector modules — tune the entire system from this one file.

## Training the ML Fatigue Classifier (Phase 9, optional)

1. Collect labeled session data into a CSV with columns:
   `ear, mar, blink_rate, eye_closure_duration, head_yaw, head_pitch, label`
   where `label` is one of `NORMAL`, `WARNING`, `CRITICAL`.

2. Train the model:

```bash
python train_model.py --data path/to/your_dataset.csv
```

3. The trained model is saved to `models/fatigue_classifier.joblib`.

4. Activate it by setting in `config.py`:

```python
USE_ML_MODEL = True
```

No other code changes are required — `FatigueEngine` automatically loads and uses the trained model instead of the rule-based formula.

## Database

Events are logged to `database/dms_events.db` (SQLite). Each row contains:

| Column | Description |
|---|---|
| timestamp | Unix timestamp of the event |
| ear | Eye Aspect Ratio at time of event |
| mar | Mouth Aspect Ratio at time of event |
| blink_count | Cumulative blink count |
| fatigue_score | Computed fatigue score (0–100) |
| alert_type | SAFE / WARNING / CRITICAL |
| screenshot_path | Path to saved screenshot (CRITICAL events only) |

## Architecture Notes

- **Separation of concerns:** detection logic (`detector/`), I/O/UI (`utils/`), and persistence (`database/`) are fully decoupled. `app.py` only orchestrates calls between them.
- **No hardcoded magic numbers:** every threshold lives in `config.py`.
- **ML-ready by design:** `FatigueEngine.evaluate()` has a stable public interface regardless of whether scoring is rule-based or ML-based, so the application layer never needs to change.

## License

This project is provided as a portfolio/educational reference implementation.
