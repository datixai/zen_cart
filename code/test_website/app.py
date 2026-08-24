"""
Visual Search Test Site — Datix AI
Local test harness for trying the trained models (EfficientNet-B0, MobileNetV2)
before integrating into the real Zen Cart site. Lets you pick one or more
photos, one or more models, run predictions, and see the matched item's real
reference photo pulled from conductor_dataset.

FOLDER LAYOUT THIS EXPECTS:

    zen_cart/
      code/
        visual_search_models/
          models/                <- unzipped trained_models_export.zip goes here
            EfficientNet-B0_best.pt
            MobileNetV2_best.pt
            class_names.json
            model_info.json
        test_website/             <- this app lives here
          app.py                  (this file)
          templates/index.html
      Data_scrap/
        conductor_dataset/       <- your scraped dataset

RUN:
    pip install flask torch torchvision transformers scikit-learn joblib pillow --break-system-packages
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import io
import json
import base64

from flask import Flask, request, jsonify, render_template
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models as tvmodels

# ---------------------------------------------------------------------------
# CONFIG — adjust these two paths if your local folder layout differs
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.py lives in: D:\zen_cart\code\test_website\
# models live in:  D:\zen_cart\code\visual_search_models\models\   (sibling folder under code\)
# dataset lives in: D:\zen_cart\Data_scrap\conductor_dataset\      (under Data_scrap\, off the project root)
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "visual_search_models", "models"))
CONDUCTOR_DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "Data_scrap", "conductor_dataset"))

CATEGORIES = ["Locomotives", "Passenger Train Cars", "Freight Cars", "Automobiles"]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load shared metadata (class names + model info) once at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODELS_DIR, "class_names.json")) as f:
    CLASS_NAMES = json.load(f)  # index -> products_id, in the exact order models were trained with

with open(os.path.join(MODELS_DIR, "model_info.json")) as f:
    MODEL_INFO = json.load(f)

IMG_SIZE = MODEL_INFO.get("input_size", 224)
NORM_MEAN = MODEL_INFO.get("normalize_mean", [0.485, 0.456, 0.406])
NORM_STD = MODEL_INFO.get("normalize_std", [0.229, 0.224, 0.225])
NUM_CLASSES = len(CLASS_NAMES)

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model builders — must match the exact architecture used during training,
# since we're loading a state_dict into it.
# ---------------------------------------------------------------------------
def build_efficientnet_b0(num_classes):
    model = tvmodels.efficientnet_b0(weights=None)  # weights=None: we load our own trained weights next
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes):
    model = tvmodels.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Lazy-loaded model registry — models only get loaded into memory the first
# time they're actually requested, not all at once at startup.
# ---------------------------------------------------------------------------
_loaded_models = {}


def get_available_models():
    """Returns which model files were actually found on disk, so the UI only
    offers models that are really available."""
    available = {}
    eff_path = os.path.join(MODELS_DIR, "EfficientNet-B0_best.pt")
    mob_path = os.path.join(MODELS_DIR, "MobileNetV2_best.pt")

    if os.path.isfile(eff_path):
        available["EfficientNet-B0"] = eff_path
    if os.path.isfile(mob_path):
        available["MobileNetV2"] = mob_path

    return available


def load_model(model_name):
    """Loads (and caches) the requested model. model_name is one of
    'EfficientNet-B0', 'MobileNetV2'."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    available = get_available_models()
    if model_name not in available:
        raise ValueError(f"Model '{model_name}' not found on disk in {MODELS_DIR}")

    if model_name == "EfficientNet-B0":
        model = build_efficientnet_b0(NUM_CLASSES)
        model.load_state_dict(torch.load(available[model_name], map_location=DEVICE))
        model.to(DEVICE).eval()
        _loaded_models[model_name] = ("cnn", model)

    elif model_name == "MobileNetV2":
        model = build_mobilenet_v2(NUM_CLASSES)
        model.load_state_dict(torch.load(available[model_name], map_location=DEVICE))
        model.to(DEVICE).eval()
        _loaded_models[model_name] = ("cnn", model)

    return _loaded_models[model_name]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_with_cnn(model, pil_image, top_k=3):
    tensor = eval_transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k=min(top_k, NUM_CLASSES))
    return [
        {"products_id": CLASS_NAMES[idx.item()], "probability": prob.item()}
        for prob, idx in zip(top_probs, top_idxs)
    ]


def find_reference_image(products_id):
    """Looks across all 4 category folders for this item's real reference
    photo (the one WITHOUT '_aug_' in its filename), returns a base64 data URL
    for direct display in the browser, plus which category it was found in."""
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
    available_models = list(get_available_models().keys())
    return render_template("index.html", available_models=available_models)


@app.route("/predict", methods=["POST"])
def predict():
    selected_models = request.form.getlist("models")
    uploaded_files = request.files.getlist("images")

    if not selected_models:
        return jsonify({"error": "No model selected."}), 400
    if not uploaded_files:
        return jsonify({"error": "No image(s) uploaded."}), 400

    results = []

    for file in uploaded_files:
        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            results.append({"filename": file.filename, "error": f"Could not read image: {e}"})
            continue

        image_result = {"filename": file.filename, "models": {}}

        for model_name in selected_models:
            try:
                kind, model = load_model(model_name)
                predictions = predict_with_cnn(model, pil_image)

                # Attach the real reference photo for each predicted item, so
                # you can visually confirm whether the prediction looks right.
                for pred in predictions:
                    ref = find_reference_image(pred["products_id"])
                    pred["reference"] = ref

                image_result["models"][model_name] = predictions

            except Exception as e:
                image_result["models"][model_name] = {"error": str(e)}

        results.append(image_result)

    return jsonify({"results": results})


if __name__ == "__main__":
    print(f"Models directory: {MODELS_DIR}")
    print(f"Conductor dataset directory: {CONDUCTOR_DATASET_DIR}")
    print(f"Available models found: {list(get_available_models().keys())}")
    print(f"Total classes: {NUM_CLASSES}")
    app.run(debug=True, port=5000)