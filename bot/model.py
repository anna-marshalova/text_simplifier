import os
import torch
from transformers import AutoTokenizer

from seq2seq.utils import MODEL_CONFIG

def get_model(model_id, from_checkpoints = True, checkpoints_path = None, device = None):
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(checkpoints_path) and from_checkpoints:
        print(f'Loading model from checkpoints {checkpoints_path}')
        return torch.load(checkpoints_path, map_location = device)
    print(f'Loading pretrained model {MODEL_CONFIG[model_id]["pretrained_model_name"]}')
    checkpoints_path = MODEL_CONFIG[model_id]['pretrained_model_name']
    return MODEL_CONFIG[model_id]['model_class'].from_pretrained(checkpoints_path).to(device)

def get_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(MODEL_CONFIG[model_id]['pretrained_model_name'])