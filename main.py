import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model import Sign2Gloss

# -----------------------------
# SAFE PREPROCESS FUNCTION
# -----------------------------
def preprocess_keypoints(kps, target_dim=99, max_frames=128):
    import numpy as np
    import torch

    arr = np.array(kps, dtype=np.float32)
    T, D = arr.shape

    # Fix feature dimension
    if D < target_dim:
        arr = np.pad(arr, ((0,0),(0,target_dim-D)), mode='constant')
    elif D > target_dim:
        arr = arr[:, :target_dim]

    # Fix frame count
    if T < max_frames:
        pad = np.zeros((max_frames - T, target_dim), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        arr = arr[idx]

    return torch.tensor(arr).unsqueeze(0)  # (1,128,99)

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
