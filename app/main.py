from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.fft import fft, fftfreq

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

FINGER_MODEL_PATH = MODEL_DIR / "tremor_random_forest_93.pkl"
FINGER_SCALER_PATH = MODEL_DIR / "robust_scaler.pkl"
ROMBERG_MODEL_PATH = MODEL_DIR / "romberg_modelnn.pkl"
ROMBERG_SCALER_PATH = MODEL_DIR / "romberg_scalernn.pkl"
TANDEM_MODEL_PATH = MODEL_DIR / "tandem_back_ensemble.joblib"

FINGER_FEATURES = [
    "Main_Freq_Hz", "Jitter", "Max_Dist_Error", "Smoothness_Acc",
    "Path_Spread_X", "Path_Spread_Y", "Velocity_Std", "Signal_Entropy",
]

# The uploaded Romberg model/scaler expect 20 features. These are the first 20
# features from the supplied notebook's pose-sway extractor.
ROMBERG_FEATURES = [
    "sh_tilt_mean", "sh_tilt_std", "sh_tilt_max", "sh_tilt_range",
    "sh_tilt_p75", "sh_tilt_p90", "hip_tilt_mean", "hip_tilt_std",
    "hip_tilt_max", "hip_tilt_range", "hip_tilt_p75", "hip_tilt_p90",
    "hip_osc_std", "hip_osc_range", "wrist_mean", "wrist_std",
    "elbow_std", "tilt_vel_mean", "tilt_vel_std", "lean_gt_15",
]

TANDEM_FEATURES = [
    "sh_tilt_mean", "sh_tilt_std", "sh_tilt_max", "sh_tilt_range", "sh_tilt_p75", "sh_tilt_p90",
    "hip_tilt_mean", "hip_tilt_std", "hip_tilt_max", "hip_tilt_range", "hip_tilt_p75", "hip_tilt_p90",
    "hip_osc_std", "hip_osc_range",
    "trunk_ap_lean_mean", "trunk_ap_lean_std", "trunk_ap_lean_max", "trunk_ap_lean_range",
    "ankle_rhythm_std", "ankle_rhythm_freq", "ankle_rhythm_entropy",
    "step_asymmetry",
    "wrist_spread_mean", "wrist_spread_std", "elbow_spread_std",
    "sh_tilt_vel_mean", "sh_tilt_vel_std",
    "lean_gt15", "lean_gt25", "lean_gt40",
    "hip_gt15", "hip_gt25",
    "knee_spread_mean", "knee_spread_std",
    "spine_lat_mean", "spine_lat_std",
    "hip_vel_mean", "hip_vel_std", "hip_acc_mean",
]

RESOLUTIONS = [(640, 480), (960, 540), (1280, 720), (480, 270)]

app = FastAPI(title="Parkinson Video AI Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

finger_model = joblib.load(FINGER_MODEL_PATH)
finger_scaler = joblib.load(FINGER_SCALER_PATH)
romberg_model = joblib.load(ROMBERG_MODEL_PATH)
romberg_scaler = joblib.load(ROMBERG_SCALER_PATH)
try:
    _tandem_obj = joblib.load(TANDEM_MODEL_PATH)
    TANDEM_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Tandem model failed to load: {e}")
    _tandem_obj = None
    TANDEM_AVAILABLE = False
tandem_model = _tandem_obj.get("model", _tandem_obj) if isinstance(_tandem_obj, dict) else _tandem_obj
tandem_model_name = _tandem_obj.get("name", "Tandem ML Model") if isinstance(_tandem_obj, dict) else "Tandem ML Model"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def _round_dict(d: Dict[str, Any], ndigits: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), ndigits)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _label_payload(test_name: str, prob: np.ndarray, features: Dict[str, float], frames_used: int, chart_data: Dict[str, List[float]] | None = None) -> Dict[str, Any]:
    p_healthy = float(prob[0])
    p_patient = float(prob[1])
    label = "PATIENT" if p_patient >= 0.5 else "HEALTHY"
    confidence = max(p_healthy, p_patient) * 100.0
    # Healthy score for UI circle: high = better/normal.
    score = p_healthy * 100.0
    return {
        "test": test_name,
        "label": label,
        "prediction": label,
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "p_healthy": round(p_healthy, 6),
        "p_patient": round(p_patient, 6),
        "frames_used": int(frames_used),
        "features": _round_dict(features),
        "chart_data": chart_data or {},
    }


def ultra_recovery_processing(frame: np.ndarray) -> np.ndarray:
    gamma = 1.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def extract_finger_features(video_path: str) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    distances: List[float] = []
    coords: List[List[float]] = []

    with mp_pose.Pose(min_detection_confidence=0.5, model_complexity=2) as pose, \
         mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=1) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.3) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            enhanced = ultra_recovery_processing(frame)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            pose_res = pose.process(rgb)
            offset_x, offset_y = 0, 0
            roi_rgb = rgb

            if pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                xs = [int(l.x * w) for l in [lm[11], lm[12], lm[0]]]
                ys = [int(l.y * h) for l in [lm[11], lm[12], lm[0]]]
                x1, y1 = max(0, min(xs) - 150), max(0, min(ys) - 150)
                x2, y2 = min(w, max(xs) + 150), min(h, max(ys) + 200)
                roi_rgb = rgb[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1

            if roi_rgb.size == 0:
                continue

            face_res = face_mesh.process(roi_rgb)
            hand_res = hands.process(roi_rgb)

            if face_res.multi_face_landmarks and hand_res.multi_hand_landmarks:
                nose = face_res.multi_face_landmarks[0].landmark[1]
                finger = hand_res.multi_hand_landmarks[0].landmark[8]

                nx = nose.x * roi_rgb.shape[1] + offset_x
                ny = nose.y * roi_rgb.shape[0] + offset_y
                fx = finger.x * roi_rgb.shape[1] + offset_x
                fy = finger.y * roi_rgb.shape[0] + offset_y

                dist = float(np.sqrt((fx - nx) ** 2 + (fy - ny) ** 2))
                distances.append(dist)
                coords.append([float(fx - nx), float(fy - ny)])

    cap.release()

    if len(distances) < 15:
        raise ValueError("Not enough hand/face landmarks detected. Make sure face and index finger are visible.")

    dists = np.array(distances, dtype=float)
    coords_arr = np.array(coords, dtype=float)

    yf = fft(dists - np.mean(dists))
    xf = fftfreq(len(dists), 1 / fps)
    idx = np.argmax(np.abs(yf)[1:]) + 1 if len(dists) > 10 else 0
    main_freq = abs(xf[idx]) if idx > 0 else 0.0

    velocity = np.diff(dists)
    acceleration = np.diff(velocity)
    path_spread = np.std(coords_arr, axis=0)

    features = {
        "Main_Freq_Hz": round(float(main_freq), 2),
        "Jitter": round(float(np.mean(np.abs(np.diff(dists)))), 6),
        "Max_Dist_Error": round(float(np.max(dists)), 4),
        "Smoothness_Acc": round(float(np.mean(acceleration)) if len(acceleration) else 0.0, 6),
        "Path_Spread_X": round(float(path_spread[0]), 4),
        "Path_Spread_Y": round(float(path_spread[1]), 4),
        "Velocity_Std": round(float(np.std(velocity)) if len(velocity) else 0.0, 4),
        "Signal_Entropy": round(float(np.std(dists)), 4),
    }

    chart = {
        "distance_signal": [round(float(x), 3) for x in dists[:120]],
        "velocity_signal": [round(float(x), 3) for x in velocity[:120]],
    }
    return features, len(distances), chart


def extract_romberg_features(video_path: str, n_frames: int = 30) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        raise ValueError("Video has too few frames.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    lm_seq: List[np.ndarray] = []

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.4,
                      min_tracking_confidence=0.4) as pose:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            for res in RESOLUTIONS:
                rgb = cv2.cvtColor(cv2.resize(frame, res), cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)
                if r.pose_landmarks:
                    arr = np.array([[l.x, l.y, l.z, l.visibility] for l in r.pose_landmarks.landmark], dtype=float)
                    lm_seq.append(arr)
                    break
    cap.release()

    if len(lm_seq) < 5:
        raise ValueError("Not enough body landmarks detected. Make sure the full body is visible.")

    lm3 = [s[:, :3] for s in lm_seq]
    torso_widths = np.array([abs(lm[11, 0] - lm[12, 0]) for lm in lm3])
    torso_w = float(np.mean(torso_widths) + 1e-6)

    sh_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[11, 1] - lm[12, 1]), abs(lm[11, 0] - lm[12, 0]) + 1e-6))
        for lm in lm3
    ])
    hip_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[23, 1] - lm[24, 1]), abs(lm[23, 0] - lm[24, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_x = np.array([(lm[23, 0] + lm[24, 0]) / 2 for lm in lm3])
    hip_x_c = hip_x - np.mean(hip_x)
    hip_osc_std = np.std(hip_x_c) / torso_w
    hip_osc_range = (np.max(hip_x_c) - np.min(hip_x_c)) / torso_w

    wrist_spread = np.array([abs(lm[15, 0] - lm[16, 0]) for lm in lm3])
    wrist_spread_std = np.std(wrist_spread) / torso_w
    wrist_spread_mean = np.mean(wrist_spread) / torso_w

    elbow_spread = np.array([abs(lm[13, 0] - lm[14, 0]) for lm in lm3])
    elbow_spread_std = np.std(elbow_spread) / torso_w

    sh_tilt_diff = np.abs(np.diff(sh_tilts))
    sh_tilt_velocity_mean = sh_tilt_diff.mean() if len(sh_tilt_diff) else 0.0
    sh_tilt_velocity_std = sh_tilt_diff.std() if len(sh_tilt_diff) else 0.0

    # Full notebook had 26. Model expects first 20.
    values = np.array([
        sh_tilts.mean(), sh_tilts.std(), sh_tilts.max(), sh_tilts.max() - sh_tilts.min(),
        np.percentile(sh_tilts, 75), np.percentile(sh_tilts, 90),
        hip_tilts.mean(), hip_tilts.std(), hip_tilts.max(), hip_tilts.max() - hip_tilts.min(),
        np.percentile(hip_tilts, 75), np.percentile(hip_tilts, 90),
        hip_osc_std, hip_osc_range,
        wrist_spread_mean, wrist_spread_std, elbow_spread_std,
        sh_tilt_velocity_mean, sh_tilt_velocity_std,
        np.mean(sh_tilts > 15),
    ], dtype=float)

    features = {name: round(float(val), 6) for name, val in zip(ROMBERG_FEATURES, values)}
    chart = {
        "shoulder_tilt_signal": [round(float(x), 3) for x in sh_tilts[:120]],
        "hip_tilt_signal": [round(float(x), 3) for x in hip_tilts[:120]],
        "hip_sway_signal": [round(float(x), 6) for x in hip_x_c[:120]],
    }
    return features, len(lm_seq), chart


def extract_tandem_features(video_path: str, n_frames: int = 30) -> Tuple[Dict[str, float], int, Dict[str, List[float]]]:
    """Extract the 39 Tandem features required by the latest tandem_back_ensemble.joblib model."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        raise ValueError("Video has too few frames.")

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    lm_seq: List[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as pose:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            for res in RESOLUTIONS:
                rgb = cv2.cvtColor(cv2.resize(frame, res), cv2.COLOR_BGR2RGB)
                r = pose.process(rgb)
                if r.pose_landmarks:
                    arr = np.array(
                        [[l.x, l.y, l.z, l.visibility] for l in r.pose_landmarks.landmark],
                        dtype=float,
                    )
                    lm_seq.append(arr)
                    break

    cap.release()

    if len(lm_seq) < 5:
        raise ValueError("Not enough body landmarks detected. Make sure the full body is visible.")

    lm3 = [s[:, :3] for s in lm_seq]

    torso_widths = np.array([abs(lm[11, 0] - lm[12, 0]) for lm in lm3])
    torso_w = float(np.mean(torso_widths) + 1e-6)

    sh_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[11, 1] - lm[12, 1]), abs(lm[11, 0] - lm[12, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_tilts = np.array([
        np.degrees(np.arctan2(abs(lm[23, 1] - lm[24, 1]), abs(lm[23, 0] - lm[24, 0]) + 1e-6))
        for lm in lm3
    ])

    hip_x = np.array([(lm[23, 0] + lm[24, 0]) / 2 for lm in lm3])
    hip_x_c = hip_x - np.mean(hip_x)
    hip_osc_std = np.std(hip_x_c) / torso_w
    hip_osc_range = (np.max(hip_x_c) - np.min(hip_x_c)) / torso_w

    wrist_spread = np.array([abs(lm[15, 0] - lm[16, 0]) for lm in lm3])
    wrist_spread_mean = np.mean(wrist_spread) / torso_w
    wrist_spread_std = np.std(wrist_spread) / torso_w

    elbow_spread = np.array([abs(lm[13, 0] - lm[14, 0]) for lm in lm3])
    elbow_spread_std = np.std(elbow_spread) / torso_w

    sh_tilt_diff = np.abs(np.diff(sh_tilts))
    sh_tilt_velocity_mean = sh_tilt_diff.mean() if len(sh_tilt_diff) else 0.0
    sh_tilt_velocity_std = sh_tilt_diff.std() if len(sh_tilt_diff) else 0.0

    large_lean_15 = np.mean(sh_tilts > 15)
    large_lean_25 = np.mean(sh_tilts > 25)
    large_lean_40 = np.mean(sh_tilts > 40)
    large_hip_15 = np.mean(hip_tilts > 15)
    large_hip_25 = np.mean(hip_tilts > 25)

    sh_mids = np.array([(lm[11] + lm[12]) / 2 for lm in lm3])
    hip_mids = np.array([(lm[23] + lm[24]) / 2 for lm in lm3])
    spine = sh_mids - hip_mids
    spine_lateral = np.abs(spine[:, 0]) / (np.abs(spine[:, 1]) + 1e-6)

    # Trunk AP lean features.
    trunk_ap = np.abs(spine[:, 1])

    # Ankle rhythm features.
    ankle_dist = np.array([abs(lm[27, 0] - lm[28, 0]) for lm in lm3])
    ankle_rhythm_std = np.std(ankle_dist)

    fft_vals = np.abs(np.fft.fft(ankle_dist - np.mean(ankle_dist)))
    ankle_rhythm_freq = float(np.argmax(fft_vals[1:]) + 1) if len(fft_vals) > 1 else 0.0

    prob = ankle_dist / (np.sum(ankle_dist) + 1e-6)
    ankle_rhythm_entropy = float(-np.sum(prob * np.log(prob + 1e-6)))

    # Step asymmetry.
    left_steps = np.array([lm[27, 1] for lm in lm3])
    right_steps = np.array([lm[28, 1] for lm in lm3])
    step_asymmetry = np.mean(np.abs(left_steps - right_steps))

    # Knee spread.
    knee_spread = np.array([abs(lm[25, 0] - lm[26, 0]) for lm in lm3])
    knee_spread_mean = np.mean(knee_spread) / torso_w
    knee_spread_std = np.std(knee_spread) / torso_w

    # Hip velocity and acceleration.
    hip_vel = np.diff(hip_x_c)
    hip_acc = np.diff(hip_vel)
    hip_vel_mean = np.mean(np.abs(hip_vel)) if len(hip_vel) else 0.0
    hip_vel_std = np.std(hip_vel) if len(hip_vel) else 0.0
    hip_acc_mean = np.mean(np.abs(hip_acc)) if len(hip_acc) else 0.0

    values = np.array([
        sh_tilts.mean(), sh_tilts.std(), sh_tilts.max(), sh_tilts.max() - sh_tilts.min(),
        np.percentile(sh_tilts, 75), np.percentile(sh_tilts, 90),
        hip_tilts.mean(), hip_tilts.std(), hip_tilts.max(), hip_tilts.max() - hip_tilts.min(),
        np.percentile(hip_tilts, 75), np.percentile(hip_tilts, 90),
        hip_osc_std, hip_osc_range,
        trunk_ap.mean(), trunk_ap.std(), trunk_ap.max(), trunk_ap.max() - trunk_ap.min(),
        ankle_rhythm_std, ankle_rhythm_freq, ankle_rhythm_entropy,
        step_asymmetry,
        wrist_spread_mean, wrist_spread_std, elbow_spread_std,
        sh_tilt_velocity_mean, sh_tilt_velocity_std,
        large_lean_15, large_lean_25, large_lean_40,
        large_hip_15, large_hip_25,
        knee_spread_mean, knee_spread_std,
        spine_lateral.mean(), spine_lateral.std(),
        hip_vel_mean, hip_vel_std, hip_acc_mean,
    ], dtype=float)

    features = {name: round(float(val), 6) for name, val in zip(TANDEM_FEATURES, values)}

    chart = {
        "shoulder_tilt_signal": [round(float(x), 3) for x in sh_tilts[:120]],
        "hip_tilt_signal": [round(float(x), 3) for x in hip_tilts[:120]],
        "hip_sway_signal": [round(float(x), 6) for x in hip_x_c[:120]],
        "spine_lateral_signal": [round(float(x), 6) for x in spine_lateral[:120]],
        "ankle_distance_signal": [round(float(x), 6) for x in ankle_dist[:120]],
    }

    return features, len(lm_seq), chart


async def _save_upload(video: UploadFile) -> str:
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        return tmp.name


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "Parkinson AI backend is running"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "finger_model": FINGER_MODEL_PATH.exists(),
        "finger_scaler": FINGER_SCALER_PATH.exists(),
        "romberg_model": ROMBERG_MODEL_PATH.exists(),
        "romberg_scaler": ROMBERG_SCALER_PATH.exists(),
        "tandem_model": TANDEM_MODEL_PATH.exists(),
        "tandem_model_name": tandem_model_name,
    }


@app.post("/analyze/finger")
async def analyze_finger(video: UploadFile = File(...)) -> Dict[str, Any]:
    path = await _save_upload(video)
    try:
        features, frames_used, chart = extract_finger_features(path)
        X = pd.DataFrame([[features[k] for k in FINGER_FEATURES]], columns=FINGER_FEATURES)
        X_scaled = finger_scaler.transform(X)
        prob = finger_model.predict_proba(X_scaled)[0]
        return _label_payload("finger_to_nose", prob, features, frames_used, chart)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/analyze/romberg")
async def analyze_romberg(video: UploadFile = File(...)) -> Dict[str, Any]:
    path = await _save_upload(video)
    try:
        features, frames_used, chart = extract_romberg_features(path)
        X = np.array([[features[k] for k in ROMBERG_FEATURES]], dtype=float)
        X_scaled = romberg_scaler.transform(X)
        prob = romberg_model.predict_proba(X_scaled)[0]
        payload = _label_payload("romberg", prob, features, frames_used, chart)
        payload["warning"] = "Romberg model expects 20 features; backend uses first 20 features from supplied notebook order. Confirm with original training CSV for maximum accuracy."
        return payload
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/analyze/tandem")
async def analyze_tandem(video: UploadFile = File(...)) -> Dict[str, Any]:
    if not TANDEM_AVAILABLE or tandem_model is None:
        raise HTTPException(
            status_code=503,
            detail="Tandem model is not loaded. Check tandem_back_ensemble.joblib and requirements versions.",
        )

    path = await _save_upload(video)
    try:
        features, frames_used, chart = extract_tandem_features(path)

        # Use DataFrame to preserve feature names/order for sklearn pipelines.
        X = pd.DataFrame([[features[k] for k in TANDEM_FEATURES]], columns=TANDEM_FEATURES)

        if hasattr(tandem_model, "predict_proba"):
            prob_raw = tandem_model.predict_proba(X)[0]
            classes = list(getattr(tandem_model, "classes_", ["HEALTHY", "PATIENT"]))

            if "HEALTHY" in classes:
                p_healthy = float(prob_raw[classes.index("HEALTHY")])
            else:
                p_healthy = float(prob_raw[0])

            if "PATIENT" in classes:
                p_patient = float(prob_raw[classes.index("PATIENT")])
            else:
                p_patient = float(prob_raw[1]) if len(prob_raw) > 1 else 1.0 - p_healthy

            prob = np.array([p_healthy, p_patient], dtype=float)
        else:
            pred = tandem_model.predict(X)[0]
            p_patient = 1.0 if str(pred).upper() == "PATIENT" else 0.0
            p_healthy = 1.0 - p_patient
            prob = np.array([p_healthy, p_patient], dtype=float)

        payload = _label_payload("tandem", prob, features, frames_used, chart)
        payload["model_name"] = tandem_model_name
        payload["features_count"] = len(TANDEM_FEATURES)
        return payload

    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
