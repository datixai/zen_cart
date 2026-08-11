# Zen Cart — AI Visual Search (conductor.fun)

Adding camera-based locomotive identification to Piotr/Harry's private Zen Cart
inventory site, as an addition to the existing QR-code lookup system (not a
replacement).

Client: Piotr / Harry (conductor.fun), USA
Built by: Datix AI

---

## Project Summary

The site is a private Zen Cart (PHP/MySQL) inventory catalog for a train
collection — browsing only, no checkout. This project adds a visual search
feature: a phone photo of a locomotive gets matched against a trained image
classifier, and the visitor is redirected to (or shown candidates for) the
correct product page.

Full project background, architecture diagrams, technical implementation
details, and the step-by-step build plan are in `/docs`.

## Folder Structure

```
zen_cart/
├── README.md
├── .gitignore
├── site-backup/        # pulled copy of live Zen Cart files (reference/rollback only)
├── training-photos/    # locomotive photos organized by SKU, for model training
├── code/                # actual working code — new/modified PHP files live here
├── docs/                # project docs (proposal, handover doc, diagrams)
└── notes/               # working notes, accuracy logs, misc scratch files
```

## Environment / Secrets

This repo does not contain any credentials, API keys, or client data.

Create a local `.env` file (already gitignored) in `/code` with:

```
AZURE_CV_PREDICTION_KEY=
AZURE_CV_ENDPOINT=
AZURE_CV_PROJECT_ID=
AZURE_CV_ITERATION_NAME=
```

Site/cPanel/Zen Cart admin credentials are shared privately (not in this
repo, not in chat) — ask Ahmed if you don't have them yet.

## Getting Started

1. Clone the repo.
2. Copy `.env.example` (if present) to `.env` in `/code` and fill in the
   Azure Custom Vision values above.
3. Pull the current live site into `site-backup/` via FTP for reference
   (see `/docs` for cPanel/FTP setup steps).
4. Work on new/changed files inside `code/`.
5. See `/docs` for the full step-by-step build plan, architecture diagrams,
   model comparison, and PHP integration reference.

## Notes

- Training photos are real client inventory images — do not share outside
  this project.
- Do not commit anything from `site-backup/` or `training-photos/` — both
  are gitignored on purpose (large, client-owned files, not code).
