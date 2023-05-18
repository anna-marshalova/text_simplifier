import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import ROOT, paths


class Simplifier:
    def __init__(self, model_name, model_id):
        self.DEVICE = 'cpu'
        self.ROOT = ROOT
        self.model_name = model_name
        self.model_id = model_id
        self.checkpoints_path = os.path.join(paths['checkpoints'], f'{self.model_id}.pt')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint_path).to(self.DEVICE)
        self.model = torch.load(self.checkpoints_path, map_location = self.DEVICE)

    def tokenize(self, text):
        inputs = self.tokenizer(
            text,
            return_attention_mask=True,
            return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def simplify(self, text, max_new_tokens=10):
        input_ids, attention_mask = self.tokenize(text)
        with torch.no_grad():
            output = self.model.generate(input_ids=input_ids.unsqueeze(0).to(self.DEVICE),
                                     attention_mask=attention_mask.unsqueeze(0).to(self.DEVICE),
                                     do_sample = False,
                                     max_new_tokens=len(input_ids)*2)
        return self.tokenizer.decode(output.squeeze(0), skip_special_tokens=True)