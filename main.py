import json
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model import Sign2Gloss

# Load vocabulary
with open("vocabulary.json", "r") as f:
    itos = json.load(f)

stoi = {t: i for i, t in enumerate(itos)}
vocab_size = len(itos)

# Model config (same as training)
INPUT_DIM = 99
HIDDEN_DIM = 256
DEVICE = "cpu"

# Load model
model = Sign2Gloss(INPUT_DIM, HIDDEN_DIM, vocab_size).to(DEVICE)
model.load_state_dict(torch.load("best_sign2gloss.pt", map_location=DEVICE))
model.eval()

app = FastAPI()

class KeypointsInput(BaseModel):
    keypoints: list  # shape [T, 99]

def ids_to_gloss(ids):
    tokens = []
    for i in ids:
        if i in (stoi["<pad>"], stoi["<bos>"], stoi["<eos>"]):
            continue
        tokens.append(itos[i])
    return " ".join(tokens)

@app.post("/predict")
def predict(data: KeypointsInput):
    arr = np.array(data.keypoints, dtype=np.float32)  # [T, 99]
    kp = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, 99]

    ids = model.generate(kp, stoi)
    gloss = ids_to_gloss(ids)
    return {"gloss": gloss}
