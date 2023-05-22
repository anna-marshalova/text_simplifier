import gc
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, BartForConditionalGeneration

BATCH_SIZE = 32
RANDOM_STATE = 42

MODEL_CONFIG = {
    'rut5-base':{'pretrained_model_name':'sberbank-ai/ruT5-base', 'model_class':T5ForConditionalGeneration},
    'mt5-small':{'pretrained_model_name':'google/mt5-small', 'model_class':MT5ForConditionalGeneration},
    'paraphraser': {'pretrained_model_name':'cointegrated/rut5-base-paraphraser', 'model_class':T5ForConditionalGeneration},
    'bart': {'pretrained_model_name':'sn4kebyt3/ru-bart-large', 'model_class':BartForConditionalGeneration}
}

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def sent_length_info(sents, tokenizer):
    sent_lengths = [len(tokenizer(sent)[0]) for sent in sents]
    print(f'Average text length: {np.mean(sent_lengths)}')
    print(f'Median text length: {np.median(sent_lengths)}')
    print(f'Max text length: {max(sent_lengths)}')