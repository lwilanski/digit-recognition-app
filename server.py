import numpy as np
from src.MLP import MLP
from src.utils import load_weights_biases
from src.utils import resize_center_image

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64

w, b = load_weights_biases()

model = MLP(w, b)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_b64: str

def preprocess_from_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw))

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(black, img)

    img = img.convert("RGB")
    arr_rgb = np.array(img, dtype=np.uint8)
    img = Image.fromarray(arr_rgb, mode="RGB")

    img = img.convert("L", dither=Image.NONE)

    arr = np.array(img, dtype=np.uint8)

    arr[arr < 2] = 0
    arr_resized = resize_center_image(arr, 20)
    arr_normalized = arr_resized / 255.0
    arr_flattened = arr_normalized.reshape(-1)
    return arr_flattened

@app.post("/predict")
def predict(req: PredictRequest):
    if req.image_b64:
        x = preprocess_from_b64(req.image_b64)
    else:
        return {"error": "Provide image_b64 or pixels_28x28"}

    proba = model.predict(x)
    pred = int(np.argmax(proba))
    return {"prediction": pred, "proba": proba.tolist()}