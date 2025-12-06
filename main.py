import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model import Sign2Gloss

# -----------------------------
# SAFE PREPROCESS FUNCTION
# -----------------------------
def preprocess_keypoints(kps, max_frames=128):
    import numpy as np
    import torch

    arr = np.array(kps, dtype=np.float32)

    # TRAINING PIPELINE: reshape from (T,33,3) → (T,99)
    if arr.ndim == 3:
        T, K, C = arr.shape
        arr = arr.reshape(T, K * C)   # (T, 99)
    else:
        T = arr.shape[0]              # already (T,99)

    # TRAINING PIPELINE: pad or truncate to 128 frames
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        arr = arr[idx]
    elif T < max_frames:
        pad = np.zeros((max_frames - T, 99), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,128,99)


# -----------------------------
# LOAD VOCAB
# -----------------------------
with open("vocabulary.json", "r") as f:
    itos = json.load(f)

stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)

# -----------------------------
# LOAD MODEL
# -----------------------------
DEVICE = "cpu"
model = Sign2Gloss(99, 256, vocab_size).to(DEVICE)
model.load_state_dict(torch.load("best_sign2gloss.pt", map_location=DEVICE))
model.eval()

app = FastAPI()

class KeypointsInput(BaseModel):
    keypoints: list  # list of frames

def ids_to_gloss(ids):
    tokens = []
    for i in ids:
        if i in (stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]):
            continue
        tokens.append(itos[i])
    return " ".join(tokens)

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: KeypointsInput):

    # PREPROCESS SAFELY
    kp = preprocess_keypoints(data.keypoints).to(DEVICE)

    # RUN MODEL
    ids = model.generate(kp, stoi)  # or model.generate(kp) depending on your model
    gloss = ids_to_gloss(ids)

    return {"gloss": gloss}


