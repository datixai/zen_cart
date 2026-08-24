# Zen Cart — AI Visual Search (conductor.fun)

Camera-based locomotive/rolling-stock identification for Piotr/Harry's
private Zen Cart inventory site — point a phone camera at a physical item,
get redirected to its product page. An addition to the existing QR-code
lookup system, not a replacement for it.

**Client:** Piotr / Harry (conductor.fun), USA
**Built by:** Datix AI

---

## Goal

The live site (`inventory.lackawannarailroad.com`) is a private Zen Cart
(PHP/MySQL) catalog for a model train collection — browsing only, checkout
disabled. Visitors currently look up items by scanning a QR code sticker.
This project adds a second lookup path: photograph the item, and the system
finds the matching product automatically.

**Core requirement that shaped the whole approach:** the model needs to
recognize new inventory Harry adds *after* launch, without retraining.

---

## Where We Are — Summary

| Phase | Status |
|---|---|
| Data collection (scraping conductor.fun) | ✅ Done — 838 items, 4 categories |
| Data augmentation (1 → 5 images/item) | ✅ Done |
| Classifier-based approach (EfficientNet-B0, MobileNetV2, CLIP, ConductorNet) | ✅ Benchmarked — **abandoned in favor of retrieval** |
| Retrieval-based approach (CLIP, OpenCLIP, SigLIP, DINOv2, YOLO-World) | ✅ Benchmarked on Kaggle |
| Local test sites (CLIP / DINOv2 / OpenCLIP, retrieval-based) | ✅ Built, running locally |
| "Add new item" mechanism | ✅ Built and tested (CLIP; DINOv2 + OpenCLIP versions pending) |
| Client-facing comparison deck | ✅ Delivered (`docs/visual_search_model_comparison.pptx`) |
| Zen Cart PHP integration (`visual_search.php`) | ⏳ Not started |
| Production model hosting (Replicate or similar) | ⏳ Not started |
| Sync mechanism: Zen Cart DB → search index | ⏳ Not started (currently manual only) |

---

## Why Retrieval, Not Classification

The first approach trained classifiers (EfficientNet-B0, MobileNetV2, a
custom-built "ConductorNet") to pick among a **fixed** list of known items.
This worked well (up to 97.5% test accuracy on the full catalog), but has a
hard architectural limit: a classifier's output layer has one slot per
trained class. It is **structurally incapable** of recognizing an item added
after training, no matter how good the model is.

Since Harry will keep adding inventory, this was a dead end for production.

**The switch:** instead of training a classifier, each model is used as a
**frozen embedding extractor** — it converts any photo into a numeric
"fingerprint" vector. A new photo is identified by finding the closest
stored fingerprint (cosine similarity), not by picking from a fixed list.

**What this buys us:** adding a new item becomes *one photo → one embedding
→ append to the index*. No retraining, no redeployment.

---

## Results

### Classifier bake-off (abandoned approach, kept for reference)

Full 4-category catalog (827 items), leakage-safe split (1 real photo/item
held out as test):

| Model | Test Acc. | Notes |
|---|---|---|
| EfficientNet-B0 | 97.5% | Best classifier — but still can't handle new items |
| MobileNetV2 | 96.9% | |
| CLIP (linear probe) | 95.3% | |
| ConductorNet (custom, from scratch) | — | Built for learning purposes; underperformed transfer learning as expected |

### Retrieval bake-off (current approach)

5 models, evaluated as frozen embedding extractors + nearest-neighbor
retrieval, same leakage-safe split (827 items, 3,308 gallery images, 827
held-out query images):

| Model | Top-1 Acc. | Top-3 Acc. | Size | Speed | Cost / 1,000 searches (T4) |
|---|---|---|---|---|---|
| **OpenCLIP ViT-B/32 (LAION-2B)** ★ | **93.5%** | **99.4%** | 577 MB | 5.3 ms | **$0.0012** |
| CLIP ViT-B/32 | 92.5% | 99.0% | 577 MB | 5.3 ms | $0.0012 |
| DINOv2 Base | 94.0% | 99.4% | 330 MB | 18.1 ms | $0.0041 |
| SigLIP Base | 94.0% | 99.5% | 775 MB | 14.2 ms | $0.0032 |
| YOLO-World (improvised) | 52.0% | 62.9% | 49 MB | 14.5 ms | $0.0033 |

★ **Recommended model** — near-top accuracy, fastest, cheapest. Full
comparison charts and reasoning: `docs/visual_search_model_comparison.pptx`.

**Note on YOLO-World:** it's an open-vocabulary object *detector*, not an
embedding model — included for a fair, tested comparison rather than
assumed to underperform. It did, as expected.

---

## Folder Structure

```
zen_cart/
├── README.md
├── .gitignore
├── Data_scrap/
│   ├── conductor_dataset/         # scraped site data, augmented (838 items)
│   │   ├── Locomotives/
│   │   ├── Passenger Train Cars/
│   │   ├── Freight Cars/
│   │   ├── Automobiles/
│   │   ├── products_index.csv     # products_id → name → category
│   │   ├── augmentation_log.csv
│   │   └── failed_downloads.csv
│   └── lionel_dataset/            # supplementary scraped Lionel SKU data (unused so far)
├── code/
│   ├── clip_code/                 # CLIP retrieval test site (port 5010)
│   │   ├── app.py
│   │   ├── add_new_item.py
│   │   └── templates/index.html
│   ├── dinov2_code/                # DINOv2 retrieval test site (port 5011)
│   ├── openclip_code/              # OpenCLIP retrieval test site (port 5012) — recommended model
│   ├── test_website/                # earlier classifier-based test site (2-model + CLIP-only variant)
│   ├── retrival_models/             # exported gallery indexes (.npy embeddings + product IDs)
│   │   ├── clip_vit_b32/
│   │   ├── dinov2_base/
│   │   └── openclip_vit_b32_laion/
│   ├── visual_search_models/        # trained classifier checkpoints (EfficientNet-B0, MobileNetV2 — legacy)
│   └── notebooks_codes/             # Kaggle notebooks (EDA, augmentation, training, comparisons)
├── docs/                            # proposal, handover doc, architecture diagrams, client deck
│   └── visual_search_model_comparison.pptx
├── site-backup/                     # pulled copy of live Zen Cart files (reference/rollback only)
├── training-photos/                 # early manual reshoot photos from Harry
└── notes/                           # working notes, accuracy logs, scratch files
```

---

## How the Current System Works

```
Phone photo
    │
    ▼
Embedding model (OpenCLIP, frozen — no fine-tuning)
    │
    ▼
512-dim vector
    │
    ▼
Compare against gallery_prototypes.npy (827 known items)
    │
    ▼
Ranked by cosine similarity → top-3 shown, or auto-redirect if confidence is high
```

The gallery index (`retrival_models/<model>/gallery_prototypes.npy` +
`gallery_products_ids.json`) is the actual "known items" database for
search — it is **separate from** `Data_scrap/conductor_dataset`, which is
now only used to fetch a reference thumbnail to display alongside a match.
The dataset folder is not searched directly.

---

## Running the Test Sites

Each model has its own standalone Flask app, single-purpose, for comparing
real-world results before committing to one for production.

| Model | Folder | Port |
|---|---|---|
| CLIP | `code/clip_code/` | 5010 |
| DINOv2 | `code/dinov2_code/` | 5011 |
| OpenCLIP (recommended) | `code/openclip_code/` | 5012 |

```powershell
cd code/openclip_code
pip install flask torch transformers pillow numpy
python app.py
```

Then open `http://127.0.0.1:5012`. First run downloads that model's weights
from Hugging Face (one-time, cached afterward at
`~/.cache/huggingface/hub/`) — the gallery index itself was already
computed on Kaggle and does not require retraining.

---

## Adding New Inventory Items

Since this is a retrieval system, new items are added without retraining:

1. Photograph the new item, save 1–3 photos into
   `code/<model>_code/new_item_photos/<products_id>/`
2. Edit `NEW_PRODUCTS_ID` and `PHOTOS_DIR` at the top of `add_new_item.py`
3. Run `python add_new_item.py` — computes the embedding, backs up the old
   gallery files automatically, appends the new item
4. Restart `app.py` (gallery loads into memory once at startup)

**Currently only built for CLIP.** DINOv2 and OpenCLIP versions are
straightforward ports of the same script (different HF repo, different
embedding dimension) — not yet built.

**Important limitation to solve before production:** this process is
entirely manual right now. When Harry adds a new item through Zen Cart's
real admin panel, nothing currently updates the search index automatically
— the two systems are not connected yet. See Roadmap below.

---

## Environment / Secrets

This repo does not contain any credentials, API keys, or client data.

Site/cPanel/Zen Cart admin credentials are shared privately (not in this
repo, not in chat) — ask Ahmed if you don't have them yet.

---

## Roadmap — What's Left

1. **Build the Zen Cart → search index sync.** Currently the only way to
   add an item is manual (`add_new_item.py` + a hand-placed photo folder).
   Production needs this driven by Zen Cart's actual `products` table —
   either a scheduled sync script or a hook in the admin "save product"
   action.
2. **Port `add_new_item.py` to DINOv2 and OpenCLIP.**
3. **Host the recommended model (OpenCLIP) in production** — Replicate or a
   small VPS were the two options discussed; Replicate avoids managing
   server infrastructure but has per-second GPU billing (cost estimates in
   the client deck are based on this).
4. **Build `visual_search.php`** on the live Zen Cart site — photo upload →
   API call to the hosted model → confidence-threshold logic (≥85% direct
   redirect, 50–85% "top 3 matches" picker, <50% "no confident match").
5. **Real-world phone photo testing** before launch — all benchmark numbers
   so far are against one held-out *catalog* photo per item, not a casual
   phone photo, so expect some accuracy drop in real use.
6. **Data quality follow-ups** flagged by the original EDA, not yet
   resolved: 11 items with no photo at all, ~24 items sharing identical
   photos across different products (Passenger Train Cars especially),
   several near-identical sequential model variants likely to be confused.

---

## Notes / Learnings

- Training photos and scraped datasets are real client inventory images —
  do not share outside this project.
- `site-backup/`, `training-photos/`, `Data_scrap/`, `lionel_dataset/`, and
  any `.zip` exports are gitignored on purpose (large, not code).
- Augmentation (`augment_dataset.py`) generates synthetic variants
  (flip/rotate/brightness/zoom) from each item's single real photo to meet
  training minimums — this does not add genuinely new visual information;
  real multi-angle reshoots remain the actual fix for weak items.
- Retrieval evaluation was leakage-safe throughout: each item's one real
  catalog photo was held out as the test query; only synthetic augmented
  variants were used to build the searchable gallery index.