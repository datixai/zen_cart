"""
OpenCLIP Retrieval Test Site — Datix AI
Single-model test page for OpenCLIP ViT-B/32 (LAION-2B) — the model
recommended in the benchmark comparison. Uses the retrieval approach:
photos are matched by finding the closest stored embedding in the gallery
index exported from the Kaggle benchmark notebook.

FOLDER LAYOUT THIS EXPECTS:

    zen_cart/
      code/
        retrival_models/
          openclip_vit_b32_laion/
            gallery_prototypes.npy
            gallery_products_ids.json
            model_info.json
        openclip_code/        <- this app lives here
          app.py               (this file)
          templates/index.html
      Data_scrap/
        conductor_dataset/

RUN:
    pip install flask torch transformers pillow numpy --break-system-packages
    python app.py
Then open http://127.0.0.1:5012

NOTE: OpenCLIP's own weights are pulled fresh from Hugging Face on first
run via from_pretrained() — not bundled in the export zip. Cached after.
"""

import os
import json
import base64

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_EXPORT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "retrival_models", "openclip_vit_b32_laion"))
CONDUCTOR_DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "Data_scrap", "conductor_dataset"))

HF_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
MODEL_DISPLAY_NAME = "OpenCLIP ViT-B/32 (LAION-2B)"

CATEGORIES = ["Locomotives", "Passenger Train Cars", "Freight Cars", "Automobiles"]
TOP_K = 3
PORT = 5012

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load the precomputed gallery index
# ---------------------------------------------------------------------------
def gallery_is_available():
    return (os.path.isfile(os.path.join(MODEL_EXPORT_DIR, "gallery_prototypes.npy")) and
            os.path.isfile(os.path.join(MODEL_EXPORT_DIR, "gallery_products_ids.json")))


_gallery_prototypes = None
_gallery_pids = None
if gallery_is_available():
    _gallery_prototypes = np.load(os.path.join(MODEL_EXPORT_DIR, "gallery_prototypes.npy"))
    with open(os.path.join(MODEL_EXPORT_DIR, "gallery_products_ids.json")) as f:
        _gallery_pids = json.load(f)


# ---------------------------------------------------------------------------
# Lazy-loaded live model
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None


def get_pooled_features(output):
    """Handles whatever shape get_image_features() returns across different
    transformers versions — a plain tensor, or a wrapped output object."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unrecognized model output type: {type(output)}")


def load_model():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    from transformers import CLIPModel, CLIPProcessor
    _clip_model = CLIPModel.from_pretrained(HF_REPO).to(DEVICE).eval()
    _clip_processor = CLIPProcessor.from_pretrained(HF_REPO)
    return _clip_model, _clip_processor


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def embed_image(pil_image):
    model, processor = load_model()
    inputs = processor(images=[pil_image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw_output = model.get_image_features(**inputs)
    feats = get_pooled_features(raw_output).cpu().numpy()
    return l2_normalize(feats)[0]


def find_reference_image(products_id):
    for category in CATEGORIES:
        folder = os.path.join(CONDUCTOR_DATASET_DIR, category, str(products_id))
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if "_aug_" not in fname and fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                with open(os.path.join(folder, fname), "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(fname)[1].lstrip(".")
                return {"category": category, "image_data_url": f"data:image/{ext};base64,{encoded}"}
    return None


def predict(pil_image, top_k=TOP_K):
    query_embed = embed_image(pil_image)
    sims = _gallery_prototypes @ query_embed
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [
        {"products_id": _gallery_pids[i], "similarity": float(sims[i])}
        for i in top_idxs
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_display_name=MODEL_DISPLAY_NAME,
        gallery_available=gallery_is_available(),
        gallery_size=len(_gallery_pids) if _gallery_pids else 0,
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        return jsonify({"error": "No image(s) uploaded."}), 400
    if not gallery_is_available():
        return jsonify({"error": f"Gallery index not found in {MODEL_EXPORT_DIR}"}), 400

    results = []
    for file in uploaded_files:
        try:
            pil_image = Image.open(file.stream).convert("RGB")
        except Exception as e:
            results.append({"filename": file.filename, "error": f"Could not read image: {e}"})
            continue

        try:
            predictions = predict(pil_image)
            for pred in predictions:
                pred["reference"] = find_reference_image(pred["products_id"])
            results.append({"filename": file.filename, "predictions": predictions})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"results": results})


if __name__ == "__main__":
    print(f"Model: {MODEL_DISPLAY_NAME} ({HF_REPO})")
    print(f"Gallery index directory: {MODEL_EXPORT_DIR}")
    print(f"Gallery available: {gallery_is_available()} ({len(_gallery_pids) if _gallery_pids else 0} items)")
    print(f"Conductor dataset directory: {CONDUCTOR_DATASET_DIR}")
    app.run(debug=True, port=PORT)
