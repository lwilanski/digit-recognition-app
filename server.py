import numpy as np
from src.Network import NeuralNetwork
from src.utils.evaluate import load_weights_biases
from src.utils.images import show_image
from src.utils.preprocess import resize_center_image

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64

layers = [400, 512, 10]
rng = np.random.default_rng(seed=15)

net = NeuralNetwork(layers, rng)
w, b = load_weights_biases("input400_hl512_ep8_lr0.001_b10.90_b20.990_alpha0.10_bs128_train99.53_test98.07")

net.weights = w
net.biases = b

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
    arr = np.array(img, dtype=np.float32)
    np.save("test-array.npy", arr)
    arr_resized = resize_center_image(arr, 20)
    show_image(arr_resized)
    arr_normalized = arr_resized / 255.0
    arr_flattened = arr_normalized.reshape(-1)
    return arr_flattened

@app.post("/predict")
def predict(req: PredictRequest):
    if req.image_b64:
        x = preprocess_from_b64(req.image_b64)
    else:
        return {"error": "Provide image_b64 or pixels_28x28"}

    proba, _, _ = net.calculate_one_sample(x)
    pred = int(np.argmax(proba))
    return {"prediction": pred, "proba": proba.tolist()}