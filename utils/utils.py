from pathlib import Path
import pickle
import torch

def load_pickle(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open('rb') as f:
        result = pickle.load(f)
    return result

def padding(x, max_len):
    return torch.tensor([s + [0]*(max_len-len(s)) for s in x])