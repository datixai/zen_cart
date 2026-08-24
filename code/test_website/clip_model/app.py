"""
CLIP-Only Visual Search Test Site — Datix AI
A separate, standalone test page that uses ONLY the CLIP (ViT-B/32) model —
frozen image embeddings + a Logistic Regression head — kept apart from the
main two-model (EfficientNet-B0 / MobileNetV2) test site so CLIP can be
tried on its own.

FOLDER LAYOUT THIS EXPECTS:

    zen_cart/
      code/
        visual_search_models/
          models/
            clip_vit_b32/                          <- CLIP base model + processor
            clip_logistic_regression_head.joblib    <- the trained linear-probe head
            class_names.json
            model_info.json
        test_website/
          app.py                    (the 2-model site)
          templates/index.html
          clip_model/                <- this app lives here
            app.py                   (this file)
            templates/index.html
      Data_scrap/
        conductor_dataset/

RUN:
    pip install flask torch transformers scikit-learn joblib pillow --break-system-packages
    python app.py
Then open http://127.0.0.1:5001 in your browser.

NOTE: This runs on port 5001, not 5000 — so it can run at the same time as
the main test_website\app.py (port 5000) without a port conflict, if you
want both open side by side.
"""

import os
import json
import base64

from flask import Flask, request, jsonify, render_template
from PIL import Image

import torch

# ---------------------------------------------------------------------------
# CONFIG — this file lives one level deeper than the main test_website\app.py,
# so the relative paths climb one extra level to reach the shared models/
# and dataset folders.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.py lives in: D:\zen_cart\code\test_website\clip_model\
# models live in:  D:\zen_cart\code\visual_search_models\models\
# dataset lives in: D:\zen_cart\Data_scrap\conductor_dataset\
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "visual_search_models", "models"))
CONDUCTOR_DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "..", "Data_scrap", "conductor_dataset"))

CATEGORIES = ["Locomotives", "Passenger Train Cars", "Freight Cars", "Automobiles"]

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load shared metadata
# ---------------------------------------------------------------------------
with open(os.path.join(MODELS_DIR, "class_names.json")) as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)

CLIP_DIR = os.path.join(MODELS_DIR, "clip_vit_b32")
CLIP_HEAD_PATH = os.path.join(MODELS_DIR, "clip_logistic_regression_head.joblib")

# ---------------------------------------------------------------------------
# Lazy-loaded CLIP model — only loaded into memory on the first prediction
# request, so the site starts up instantly even though CLIP itself is large.
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None
_clip_head = None


def clip_is_available():
    return os.path.isdir(CLIP_DIR) and os.path.isfile(CLIP_HEAD_PATH)


def load_clip():
    global _clip_model, _clip_processor, _clip_head

    if _clip_model is not None:
        return _clip_model, _clip_processor, _clip_head

    if not clip_is_available():
        raise ValueError(f"CLIP model files not found in {MODELS_DIR}")

    from transformers import CLIPModel, CLIPProcessor
    import joblib

    _clip_model = CLIPModel.from_pretrained(CLIP_DIR).to(DEVICE).eval()
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_DIR)
    _clip_head = joblib.load(CLIP_HEAD_PATH)

    return _clip_model, _clip_processor, _clip_head


def get_pooled_features(output):
    """Handles whatever shape get_image_features() returns across different
    transformers versions — a plain tensor, an object with .image_embeds,
    .pooler_output, or falls back to mean-pooling .last_hidden_state."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unrecognized CLIP output type: {type(output)}")


def predict_with_clip(pil_image, top_k=3):
    clip_model, clip_processor, clip_head = load_clip()

    inputs = clip_processor(images=[pil_image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw_output = clip_model.get_image_features(**inputs)
        feats = get_pooled_features(raw_output).cpu().numpy()

    probs = clip_head.predict_proba(feats)[0]
    top_idxs = probs.argsort()[::-1][:top_k]

    predictions = []
    for idx in top_idxs:
        class_label = clip_head.classes_[idx]
        # clip_head was trained on integer class indices (0..N-1), matching
        # CLASS_NAMES' order — NOT on the real products_id strings directly.
        # class_label is that integer index, so it must be translated back
        # into the real products_id via CLASS_NAMES. Using it raw would both
        # return the wrong ID and crash JSON serialization (numpy int64).
        products_id = CLASS_NAMES[int(class_label)]
        predictions.append({"products_id": products_id, "probability": float(probs[idx])})

    return predictions


def find_reference_image(products_id):
    """Looks across all 4 category folders for this item's real reference
    photo (the one WITHOUT '_aug_' in its filename)."""
    for category in CATEGORIES:
        folder = os.path.join(CONDUCTOR_DATASET_DIR, category, str(products_id))
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if "_aug_" not in fname and fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                with open(os.path.join(folder, fname), "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(fname)[1].lstrip(".")
                return {
                    "category": category,
                    "image_data_url": f"data:image/{ext};base64,{encoded}",
                }
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", clip_available=clip_is_available())


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_files = request.files.getlist("images")

    if not uploaded_files:
        return jsonify({"error": "No image(s) uploaded."}), 400
    if not clip_is_available():
        return jsonify({"error": f"CLIP model files not found in {MODELS_DIR}"}), 400

    results = []

    for file in uploaded_files:
        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            results.append({"filename": file.filename, "error": f"Could not read image: {e}"})
            continue

        try:
            predictions = predict_with_clip(pil_image)
            for pred in predictions:
                pred["reference"] = find_reference_image(pred["products_id"])
            results.append({"filename": file.filename, "predictions": predictions})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"results": results})


if __name__ == "__main__":
    print(f"Models directory: {MODELS_DIR}")
    print(f"Conductor dataset directory: {CONDUCTOR_DATASET_DIR}")
    print(f"CLIP available: {clip_is_available()}")
    print(f"Total classes: {NUM_CLASSES}")
    app.run(debug=True, port=5001)