import os
import uuid
import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, redirect, url_for, jsonify

from src.models.unet import UNet
from src.models.efficientnet_ckd import EfficientNetCKD


# -----------------------------------------------------------
# Paths & config
# -----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

SEG_MODEL_PATH = os.path.join(ROOT, "models_ckd", "unet_kidney_10k_30ep.pth")
# Model priority: focal (96.07% acc) > enhanced > v2
FOCAL_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_focal.pth")
ENHANCED_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_enhanced.pth")
V2_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_v2.pth")


def _resolve_cls_model_path() -> str:
    """Choose the best available classifier weights, error clearly if none exist."""
    for candidate in (FOCAL_MODEL_PATH, ENHANCED_MODEL_PATH, V2_MODEL_PATH):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No EfficientNet CKD weights found. Expected one of: "
        f"{FOCAL_MODEL_PATH}, {ENHANCED_MODEL_PATH}, {V2_MODEL_PATH}"
    )


CLS_MODEL_PATH = _resolve_cls_model_path()

CT_SLICES_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")

STATIC_RESULTS_DIR = os.path.join(ROOT, "static", "results")
os.makedirs(STATIC_RESULTS_DIR, exist_ok=True)

# Directory for built‑in demo CT slices used by the "Sample test" button.
DEMO_SAMPLES_DIR = os.path.join(ROOT, "static", "demo_samples")
os.makedirs(DEMO_SAMPLES_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# same classes as training
NUM_CLASSES = 4
STAGE_INFO = {
    1: {
        "title": "Stage 1 • Well Preserved",
        "summary": "Kidney structure appears intact with minimal detectable damage.",
        "care": "Continue hydration, routine labs, and annual screening."
    },
    2: {
        "title": "Stage 2 • Mild Decline",
        "summary": "Early structural changes with small areas of scarring or swelling.",
        "care": "Lifestyle tuning, blood pressure control, monitor kidney labs every 6 months."
    },
    3: {
        "title": "Stage 3 • Moderate Damage",
        "summary": "Noticeable loss of filtering surface; scarring is broader but kidneys still compensating.",
        "care": "Nephrology follow-up, medication review, tighter control of blood pressure and diabetes."
    },
    4: {
        "title": "Stage 4 • Advanced Damage",
        "summary": "Extensive scarring and reduced functional reserve; risk for progression is high.",
        "care": "Specialist co-management, renal-protective therapy, and preparation for renal replacement options."
    },
}

# -----------------------------------------------------------
# Load models once
# -----------------------------------------------------------
print(f"[INFO] Using device: {DEVICE}")

_unet = UNet(in_channels=1, out_channels=1, base_ch=64).to(DEVICE)
_unet.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
_unet.eval()
print(f"[INFO] Loaded UNet from: {SEG_MODEL_PATH}")

_effnet = EfficientNetCKD(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
_effnet.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
_effnet.eval()
print(f"[INFO] Loaded EfficientNet CKD from: {CLS_MODEL_PATH}")

app = Flask(__name__)


# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def preprocess_ct_slice(img):
    """
    Input: image from cv2 (BGR or grayscale).
    Output: tensor (1, 1, 512, 512) on DEVICE, plus resized grayscale image.
    """
    if img is None:
        raise ValueError("preprocess_ct_slice received None image")

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512))
    img_norm = gray.astype(np.float32) / 255.0
    img_norm = np.expand_dims(np.expand_dims(img_norm, axis=0), axis=0)  # (1,1,H,W)
    tensor = torch.from_numpy(img_norm).to(DEVICE)
    return tensor, gray


def run_unet_and_get_mask(img_tensor):
    with torch.no_grad():
        logits = _unet(img_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)  # (H,W)
    return mask_np


def create_overlay(ct_gray, mask):
    """
    ct_gray: (H,W) uint8
    mask:    (H,W) {0,1}
    returns: overlay BGR image
    """
    h, w = ct_gray.shape
    ct_color = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2BGR)

    red = np.zeros_like(ct_color)
    red[:, :, 2] = 255  # pure red

    alpha = 0.5
    mask_3 = np.stack([mask] * 3, axis=-1)

    overlay = np.where(mask_3 == 1, (1 - alpha) * ct_color + alpha * red, ct_color)
    overlay = overlay.astype(np.uint8)
    return overlay


def extract_kidney_roi(ct_gray, mask, out_size=224):
    """
    Extract kidney ROI using bounding box of mask.
    If mask empty -> center crop.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # fallback: center crop square
        h, w = ct_gray.shape
        side = min(h, w) // 2
        x1 = w // 2 - side // 2
        y1 = h // 2 - side // 2
        x2 = x1 + side
        y2 = y1 + side
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # small padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(ct_gray.shape[1] - 1, x2 + pad)
        y2 = min(ct_gray.shape[0] - 1, y2 + pad)

    roi = ct_gray[y1:y2, x1:x2]
    roi = cv2.resize(roi, (out_size, out_size))
    return roi


def classify_roi(roi_gray):
    """
    roi_gray: (H,W) uint8  224x224
    returns: probs (4,), pred_stage_index (1..4)
    """
    img = roi_gray.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)   # (1,H,W)
    img = np.expand_dims(img, axis=0)   # (1,1,H,W)
    tensor = torch.from_numpy(img).to(DEVICE)

    with torch.no_grad():
        logits = _effnet(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    # our stages are {1,2,3,4}, but model indices 0..3
    pred_idx_0 = int(np.argmax(probs))
    pred_stage = pred_idx_0 + 1
    return probs, pred_stage


def stage_summary_text(stage: int) -> str:
    """Return a concise, human-readable summary for the predicted stage."""
    info = STAGE_INFO.get(stage)
    if info is None:
        return "Kidney stage could not be summarized."
    return f"{info['summary']} Recommended next steps: {info['care']}"


def base_context() -> dict:
    """Base template context used for the single-slice page."""
    return {
        "result_ready": False,
        "stage_info": STAGE_INFO,
    }


def analyze_ct_image(img_gray: np.ndarray) -> dict:
    """
    Run the full segmentation → ROI → classification pipeline for a single CT slice.

    Returns a context dict that can be merged into the template context.
    """
    img_tensor, ct_gray = preprocess_ct_slice(img_gray)
    mask = run_unet_and_get_mask(img_tensor)
    overlay = create_overlay(ct_gray, mask)
    roi = extract_kidney_roi(ct_gray, mask)

    probs, pred_stage = classify_roi(roi)
    summary = stage_summary_text(pred_stage)

    uid = uuid.uuid4().hex[:8]
    ct_path = os.path.join(STATIC_RESULTS_DIR, f"{uid}_ct.png")
    mask_path = os.path.join(STATIC_RESULTS_DIR, f"{uid}_mask.png")
    overlay_path = os.path.join(STATIC_RESULTS_DIR, f"{uid}_overlay.png")
    roi_path = os.path.join(STATIC_RESULTS_DIR, f"{uid}_roi.png")

    cv2.imwrite(ct_path, ct_gray)
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(roi_path, roi)

    return {
        "result_ready": True,
        "pred_stage": pred_stage,
        "pred_stage_label": STAGE_INFO[pred_stage]["title"],
        "stage_details": STAGE_INFO[pred_stage],
        "summary": summary,
        "img_ct": url_for("static", filename=f"results/{os.path.basename(ct_path)}"),
        "img_mask": url_for("static", filename=f"results/{os.path.basename(mask_path)}"),
        "img_overlay": url_for("static", filename=f"results/{os.path.basename(overlay_path)}"),
        "img_roi": url_for("static", filename=f"results/{os.path.basename(roi_path)}"),
    }


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    # landing page -> single slice page
    return redirect(url_for("single_slice"))


@app.route("/slice", methods=["GET", "POST"])
def single_slice():
    """
    Page 1: upload a single CT slice or trigger a built-in demo sample.
    Shows: stage, stage meaning, probs, images, summary.
    """
    context = base_context()

    if request.method == "POST":
        start_t = time.perf_counter()
        use_demo = request.form.get("use_demo") == "1"

        if use_demo:
            # If the client already selected a specific demo image, use that;
            # otherwise fall back to picking one on the server.
            demo_rel = request.form.get("demo_rel", "").strip()
            sample_path = None

            if demo_rel:
                # demo_rel is like "demo_samples/xxx.png" or "results/yyy_ct.png"
                if demo_rel.startswith("demo_samples/"):
                    sample_path = os.path.join(DEMO_SAMPLES_DIR, demo_rel.split("/", 1)[1])
                elif demo_rel.startswith("results/"):
                    sample_path = os.path.join(STATIC_RESULTS_DIR, demo_rel.split("/", 1)[1])

            if not sample_path:
                # Prefer explicit demo_samples/, but fall back to any *_ct.png
                # images already present in static/results/ (from previous runs).
                samples: list[str] = []
                if os.path.isdir(DEMO_SAMPLES_DIR):
                    samples.extend(
                        os.path.join(DEMO_SAMPLES_DIR, f)
                        for f in os.listdir(DEMO_SAMPLES_DIR)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    )

                if not samples and os.path.isdir(STATIC_RESULTS_DIR):
                    samples.extend(
                        os.path.join(STATIC_RESULTS_DIR, f)
                        for f in os.listdir(STATIC_RESULTS_DIR)
                        if f.lower().endswith("_ct.png")
                    )

                if not samples:
                    context["demo_error"] = (
                        "Demo sample not configured yet. "
                        "Please add at least one CT slice PNG/JPG to static/demo_samples/ "
                        "or generate a result once so we can reuse it as a demo."
                    )
                    return render_template("index.html", **context)

                sample_path = random.choice(samples)
            img_gray = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                context["demo_error"] = "Could not read the demo CT slice."
                return render_template("index.html", **context)

            analysis_ctx = analyze_ct_image(img_gray)
            analysis_ctx["demo_source"] = os.path.basename(sample_path)
            context.update(analysis_ctx)
        else:
            file = request.files.get("ct_image")
            if not file or file.filename == "":
                context["error"] = "Please choose a CT slice image first."
                return render_template("index.html", **context)

            raw_bytes = np.frombuffer(file.read(), np.uint8)
            img_gray = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)

            if img_gray is None:
                context["error"] = "Could not read the image. Please upload .png or .jpg."
                return render_template("index.html", **context)

            analysis_ctx = analyze_ct_image(img_gray)
            context.update(analysis_ctx)

        # Enforce a minimum response time so the UI spinner is visible.
        elapsed = time.perf_counter() - start_t
        min_seconds = 2.0
        if elapsed < min_seconds:
            time.sleep(min_seconds - elapsed)

    return render_template("index.html", **context)


@app.route("/api/select-demo", methods=["GET"])
def select_demo():
    """
    API helper: pick a demo CT slice (without running the model) so the client
    can preview it before running segmentation.
    """
    candidates: list[tuple[str, str]] = []  # (source, filename)

    if os.path.isdir(DEMO_SAMPLES_DIR):
        for f in os.listdir(DEMO_SAMPLES_DIR):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(("demo_samples", f))

    if not candidates and os.path.isdir(STATIC_RESULTS_DIR):
        for f in os.listdir(STATIC_RESULTS_DIR):
            if f.lower().endswith("_ct.png"):
                candidates.append(("results", f))

    if not candidates:
        return jsonify({"ok": False, "message": "No demo CT slices found."}), 404

    source, fname = random.choice(candidates)
    rel = f"{source}/{fname}"
    img_url = url_for("static", filename=rel)
    return jsonify({"ok": True, "demo_rel": rel, "image_url": img_url})


@app.route("/about", methods=["GET"])
def about():
    """Static about page describing the application."""
    return render_template("about.html", stage_info=STAGE_INFO)


if __name__ == "__main__":
    app.run(debug=True)
