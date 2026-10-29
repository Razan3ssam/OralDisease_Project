import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

model = tf.keras.models.load_model("models/tuned_EfficientNetB0.keras")
H = model.input_shape[1] or 224
W = model.input_shape[2] or 224

with open("class_names.json", encoding="utf-8") as f:
    class_names = json.load(f)

app = FastAPI(title="Oral Disease Classifier")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Please upload a JPG or PNG image only.")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image is too large (max 8MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB").resize((W, H))
    except Exception:
        raise HTTPException(400, "Could not read the image.")

    # EfficientNet in Keras rescales internally, so we pass raw 0-255 values
    x = np.expand_dims(np.asarray(img, dtype=np.float32), 0)
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))

    return {
        "label": class_names[top],
        "confidence": float(probs[top]),
        "all": [
            {"label": c, "prob": float(p)}
            for c, p in sorted(zip(class_names, probs), key=lambda t: -t[1])
        ],
    }