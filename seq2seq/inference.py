from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader

from seq2seq.utils import BATCH_SIZE

def batch_inference(texts, model, tokenizer, max_new_tokens = 50, batch_size = BATCH_SIZE):
    generated_texts = []
    loader = DataLoader(texts, batch_size = batch_size)
    for batch in tqdm(loader):
        input_ids, attention_mask = tokenizer(batch, return_tensors = 'pt', padding=True).values()
        with torch.no_grad():
            outputs = model.generate(input_ids = input_ids.to(model.device), attention_mask = attention_mask.to(model.device), max_new_tokens = max_new_tokens)
        generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        generated_texts.extend(generated_batch)
    return generated_texts

def example(source, taget, model, tokenizer):
    print(f'SOURCE: {source}\nTARGET: {taget}')
    input_ids, attention_mask = tokenizer(source, return_tensors = 'pt').values()
    with torch.no_grad():
        output = model.generate(input_ids = input_ids.to(model.device), attention_mask = attention_mask.to(model.device), max_new_tokens = len(input_ids)*2,min_length=0)
    return tokenizer.decode(output.squeeze(0), skip_special_tokens = True)