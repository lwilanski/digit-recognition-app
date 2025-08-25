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

from pathlib import Path

layers = [400, 512, 256, 128, 10]
rng = np.random.default_rng(seed=15)

net = NeuralNetwork(layers, rng)
w, b = load_weights_biases("newer_data_input400_hl512_256_128_ep4_lr0.001_b10.90_b20.999_alpha0.10_bs128_train99.84_test98.45")

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

save_dir = Path("C:/Users/Lukasz/Documents/GitHub/digit-recognition-app/additional-examples")

if (save_dir / "additional_examples.npz").exists():
    data = np.load(save_dir / "additional_examples.npz")
    X = data["X"].tolist()
    y = data["y"].tolist()
else:
    X, y = [], []

class PredictRequest(BaseModel):
    image_b64: str
    # label: int

def preprocess_from_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(raw))

    # 1) Wyrzuć alfę i profil kolorów, skomponuj na czarnym tle (pewność)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    black = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(black, img)

    # 2) Usuń potencjalne profile/ICC/gamma przez "re-encode":
    #    (zamiana na RGB bez ICC, potem z powrotem do PIL)
    img = img.convert("RGB")  # bez alfa
    arr_rgb = np.array(img, dtype=np.uint8)  # surowe bajty, bez ICC
    img = Image.fromarray(arr_rgb, mode="RGB")

    # 3) Szarość bez ditheringu
    img = img.convert("L", dither=Image.NONE)

    # 4) (opcjonalnie) zmiana rozmiaru do 28x28 bez antyaliasu
    # img = img.resize((28, 28), resample=Image.NEAREST)

    arr = np.array(img, dtype=np.uint8)

    # 5) Twardy próg — wyczyść „1-ki” i mikroszum (0..1 -> 0)
    #    Jeśli chcesz obraz binarny: próg np. 5 z marginesem.
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

    # X.append(x)
    # y.append(req.label)
    # np.savez(save_dir / "additional_examples.npz", 
    #      X=np.array(X, dtype=np.float32), 
    #      y=np.array(y, dtype=np.uint8))
    proba, _, _ = net.calculate_one_sample(x)
    pred = int(np.argmax(proba))
    return {"prediction": pred, "proba": proba.tolist()}