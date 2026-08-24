r"""
Add a new item to the CLIP gallery index — Datix AI
Run this whenever Harry adds a new locomotive/car/etc. to the inventory.
Computes that item's embedding using the exact same CLIP model the search
app uses, then appends it to the existing gallery files. No retraining.

USAGE:
    1. Take 1+ photos of the new item, put them in a folder, e.g.:
       D:\zen_cart\code\clip_code\new_item_photos\5001\
           photo1.jpg
           photo2.jpg   (optional — more angles = a slightly more robust
                          reference embedding, but even 1 photo works)

    2. Edit NEW_PRODUCTS_ID and PHOTOS_DIR below.

    3. Run:
       python add_new_item.py

    4. Restart app.py (Ctrl+C, then `python app.py` again) — the gallery is
       loaded into memory once at startup, so a running server won't see
       the new item until it's restarted.

    5. Test: upload a photo of item 5001 in the browser and confirm it
       comes back as a strong match.
"""

import os
import json
import shutil

import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# EDIT THESE TWO VALUES
# ---------------------------------------------------------------------------
NEW_PRODUCTS_ID = "5001"
PHOTOS_DIR = "new_item_photos/5001"  # folder containing 1+ photos of the new item

# ---------------------------------------------------------------------------
# Same paths/config as app.py
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_EXPORT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "retrival_models", "clip_vit_b32"))
HF_REPO = "openai/clip-vit-base-patch32"

PROTOTYPES_PATH = os.path.join(MODEL_EXPORT_DIR, "gallery_prototypes.npy")
PIDS_PATH = os.path.join(MODEL_EXPORT_DIR, "gallery_products_ids.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pooled_features(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unrecognized model output type: {type(output)}")


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    # ---- Load existing gallery ----
    if not os.path.isfile(PROTOTYPES_PATH) or not os.path.isfile(PIDS_PATH):
        raise FileNotFoundError(f"Existing gallery files not found in {MODEL_EXPORT_DIR}")

    prototypes = np.load(PROTOTYPES_PATH)
    with open(PIDS_PATH) as f:
        pids = json.load(f)

    print(f"Loaded existing gallery: {len(pids)} items, embedding dim {prototypes.shape[1]}")

    if NEW_PRODUCTS_ID in pids:
        raise ValueError(
            f"'{NEW_PRODUCTS_ID}' is already in the gallery (index {pids.index(NEW_PRODUCTS_ID)}). "
            f"If you're re-adding it with new photos, remove it first or use a different script path."
        )

    # ---- Collect the new item's photos ----
    if not os.path.isdir(PHOTOS_DIR):
        raise FileNotFoundError(f"Photos folder not found: {PHOTOS_DIR}")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    photo_paths = [
        os.path.join(PHOTOS_DIR, f) for f in sorted(os.listdir(PHOTOS_DIR))
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    if not photo_paths:
        raise ValueError(f"No image files found in {PHOTOS_DIR}")

    print(f"Found {len(photo_paths)} photo(s) for item '{NEW_PRODUCTS_ID}': "
          f"{[os.path.basename(p) for p in photo_paths]}")

    # ---- Load CLIP (same model the app uses) ----
    print(f"Loading CLIP ({HF_REPO})...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(HF_REPO).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(HF_REPO)

    # ---- Compute an embedding for each photo, average into one prototype ----
    # (matches how the original 827-item gallery was built — one averaged
    # vector per item, from however many photos it has)
    images = [Image.open(p).convert("RGB") for p in photo_paths]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw_output = model.get_image_features(**inputs)
    embeds = get_pooled_features(raw_output).cpu().numpy()  # (num_photos, dim)

    new_prototype = embeds.mean(axis=0, keepdims=True)  # average across photos -> (1, dim)
    new_prototype = l2_normalize(new_prototype)

    # ---- Back up the existing files before modifying anything ----
    backup_dir = os.path.join(MODEL_EXPORT_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy(PROTOTYPES_PATH, os.path.join(backup_dir, "gallery_prototypes.npy.bak"))
    shutil.copy(PIDS_PATH, os.path.join(backup_dir, "gallery_products_ids.json.bak"))
    print(f"Backed up existing gallery files to {backup_dir}")

    # ---- Append and save ----
    updated_prototypes = np.concatenate([prototypes, new_prototype], axis=0)
    updated_pids = pids + [NEW_PRODUCTS_ID]

    np.save(PROTOTYPES_PATH, updated_prototypes)
    with open(PIDS_PATH, "w") as f:
        json.dump(updated_pids, f)

    print(f"\n✅ Added '{NEW_PRODUCTS_ID}' to the gallery.")
    print(f"   Gallery size: {len(pids)} -> {len(updated_pids)} items")
    print(f"   Saved to: {PROTOTYPES_PATH}")
    print(f"\n👉 Restart app.py for the running server to pick up this change,")
    print(f"   then test by uploading a photo of item {NEW_PRODUCTS_ID} in the browser.")


if __name__ == "__main__":
    main()
