import io
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import json

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224

MODEL_PATH = "model/plantdoc_model_tf2_m2.h5"
LABELS_PATH = "model/labels.json"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
    labels = {int(k): v for k, v in labels.items()}


# -----------------------------
# DAMAGE MASK GENERATION (simple edge-based damage detection)
# -----------------------------
def generate_damage_mask(img_pil):
    """
    Converts leaf to grayscale → applies edge detection → returns mask.
    This is NOT a real segmentation model but works for demo.
    """

    gray = ImageOps.grayscale(img_pil)
    gray_np = np.array(gray)

    # Simple edge mask to simulate "damage"
    edges = np.abs(np.diff(gray_np.astype("int32"), axis=0))
    edges = np.pad(edges, ((0, 1), (0, 0)))

    # threshold for pseudo damage
    mask = (edges > 25).astype(np.uint8)

    damage_percent = float(np.sum(mask) / mask.size * 100)

    # Convert mask to color overlay image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    overlay = Image.new("RGBA", img_pil.size)
    overlay_np = np.array(overlay)

    # Create transparent red overlay on damaged areas
    overlay_np[mask == 1] = [255, 0, 0, 120]

    overlay_img = Image.fromarray(overlay_np, "RGBA")

    # Merge overlay with original image
    combined = Image.alpha_composite(img_pil.convert("RGBA"), overlay_img)

    # Encode overlay image to base64
    buffered = io.BytesIO()
    combined.save(buffered, format="PNG")
    overlay_b64 = base64.b64encode(buffered.getvalue()).decode()

    # Severity classification
    if damage_percent < 20:
        severity = "Low"
    elif damage_percent < 50:
        severity = "Medium"
    else:
        severity = "High"

    return damage_percent, severity, overlay_b64


# -----------------------------
# MAIN /analyze ENDPOINT
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))

    # Prediction pipeline
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))
    pred_label = labels[pred_idx]
    confidence = float(preds[0][pred_idx])

    # Damage severity + mask overlay
    damage_percent, severity, overlay_b64 = generate_damage_mask(img_pil)

    return {
        "class": pred_label,
        "confidence": confidence,
        "damage_percent": round(damage_percent, 2),
        "severity": severity,
        "overlay_image": f"data:image/png;base64,{overlay_b64}"
    }
