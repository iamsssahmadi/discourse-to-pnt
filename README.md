# PNT Score Predictor

**Live demo:** https://iamsssahmadi.github.io/discourse-to-pnt/ (runs entirely in your browser)

Paste a discourse transcript (Cinderella story retelling, or PBJ procedure
description) and get predicted Philadelphia Naming Test (PNT) scores —
correct responses plus 7 error subtypes.

**Everything runs in your browser.** The transcript you paste is never
uploaded anywhere — feature extraction, the sentence embedding, and the
trained regression models all run client-side (via `transformers.js` and
`onnxruntime-web`). Only static model files are fetched (from this site,
plus the public embedding-model weights on first load).

Research use only. Not clinically validated. Does not replace formal PNT
administration.

## How it was built

The prediction models (Ridge / RandomForest / XGBoost, one per PNT score
type) were trained in Python on paired discourse-transcript + PNT-score
data, then converted to ONNX and plain JSON so they run natively in the
browser with no server. See the parent project's `train_final_models.py`
and `web/dev/export_web_models.py` for the training/export pipeline.
