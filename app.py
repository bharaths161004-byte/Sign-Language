"""
Flask backend for ISL to Malayalam translator.
Uses the sklearn model from SIGN LANGUAGE/sign_language_model.pkl
with MediaPipe Hands for feature extraction (matching predict.py logic).
"""

import os
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import pickle
import json
import asyncio
import io
import mediapipe as mp
from collections import Counter
import edge_tts
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ── Gemini setup ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA_dBBlPuwYWxjLlXj2FWjfxKU5ldYJPjM")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ── Load sklearn model & labels ───────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "SIGN LANGUAGE")

with open(os.path.join(MODEL_DIR, "sign_language_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "class_labels.json"), "r") as f:
    class_labels = json.load(f)

# ── MediaPipe Hands ───────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

# ── Feature extraction (mirrors predict.py / train.py) ────────
FINGER_TRIPLETS = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4],
    [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12],
    [0, 13, 14], [13, 14, 15], [14, 15, 16],
    [0, 17, 18], [17, 18, 19], [18, 19, 20],
]

KEY_PAIRS = [
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (8, 16), (8, 20),
    (12, 16), (12, 20), (16, 20),
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (5, 9), (9, 13), (13, 17),
]

FEAT_SIZE = 96  # 63 coords + 15 angles + 18 distances


def _compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.arccos(np.clip(cos_a, -1, 1))


def _hand_feat(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        return None
    pts /= scale
    raw = pts.flatten()
    angles = np.array(
        [_compute_angle(pts[t[0]], pts[t[1]], pts[t[2]]) for t in FINGER_TRIPLETS],
        dtype=np.float32,
    )
    dists = np.array(
        [np.linalg.norm(pts[a] - pts[b]) for a, b in KEY_PAIRS],
        dtype=np.float32,
    )
    return np.concatenate([raw, angles, dists])


def extract_features(hand_landmarks):
    hand_feats = []
    for landmarks in hand_landmarks:
        wrist_x = landmarks[0].x
        feat = _hand_feat(landmarks)
        if feat is not None:
            hand_feats.append((wrist_x, feat))
    if not hand_feats:
        return None
    hand_feats.sort(key=lambda x: x[0])
    combined = np.zeros(FEAT_SIZE * 2, dtype=np.float32)
    for i, (_, feat) in enumerate(hand_feats[:2]):
        combined[i * FEAT_SIZE : (i + 1) * FEAT_SIZE] = feat
    return combined


# ── Smoothing buffer (server-side, per-session simple approach) ─
BUFFER_SIZE = 5
pred_buffer: list[str] = []


# ── Routes ────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    global pred_buffer
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Resize to 640x480 to match predict.py camera resolution
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            pred_buffer.clear()
            return jsonify({"detected": False, "label": None, "confidence": 0})

        feat = extract_features(result.hand_landmarks)
        if feat is None:
            pred_buffer.clear()
            return jsonify({"detected": False, "label": None, "confidence": 0})

        feat_2d = feat.reshape(1, -1)
        proba = model.predict_proba(feat_2d)[0]
        raw_pred = class_labels[np.argmax(proba)]
        confidence = float(np.max(proba))

        # Smoothing
        pred_buffer.append(raw_pred)
        if len(pred_buffer) > BUFFER_SIZE:
            pred_buffer.pop(0)
        smoothed = Counter(pred_buffer).most_common(1)[0][0]

        return jsonify({
            "detected": True,
            "label": smoothed,
            "confidence": confidence,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/refine", methods=["POST"])
def refine_sentence():
    try:
        data = request.get_json()
        words = data.get("words", [])

        if not words:
            return jsonify({"refined": "", "malayalam": ""})

        word_string = ", ".join(words)

        prompt = (
            f"Representing Indian Sign Language gestures, here is a sequence of words/phrases:\n"
            f"{word_string}\n\n"
            f"Task: Convert these words into a single, grammatically correct, and natural-sounding English sentence first, "
            f"then translate it into Malayalam.\n"
            f"Return ONLY a JSON object with two keys: \"english\" and \"malayalam\". No markdown, no explanation."
        )

        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Try to parse as JSON
        import re
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            english = parsed.get("english", word_string)
            malayalam = parsed.get("malayalam", "")
        else:
            english = raw
            malayalam = raw

        return jsonify({
            "refined": english,
            "malayalam": malayalam,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/clear_buffer", methods=["POST"])
def clear_buffer():
    global pred_buffer
    pred_buffer.clear()
    return jsonify({"status": "ok"})


@app.route("/tts", methods=["POST"])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice="ml-IN-SobhanaNeural")
            audio_bytes = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes.write(chunk["data"])
            return audio_bytes.getvalue()

        audio_data = asyncio.run(_synthesize())
        return Response(audio_data, mimetype="audio/mpeg")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"ISL Backend ready — {len(class_labels)} gesture classes loaded.")
    app.run(debug=True, host="0.0.0.0", port=5000)
