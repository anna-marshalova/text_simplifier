import os
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils import ROOT


class Simplifier:
    def __init__(self, model_name):
        self.DEVICE = 'cpu'
        self.ROOT = ROOT
        self.model_name = model_name
        self.checkpoint_path = os.path.join(self.ROOT, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.checkpoint_path).to(self.DEVICE)

    def tokenize(self, text):
        inputs = self.tokenizer(
            text,
            return_attention_mask=True,
            return_tensors='pt')
        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

    def simplify(self, text, max_new_tokens=10):
        input_ids, attention_mask = self.tokenize(text)
        output = self.model.generate(input_ids=input_ids.unsqueeze(0).to(self.DEVICE),
                                     attention_mask=attention_mask.unsqueeze(0).to(self.DEVICE),
                                     max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output.squeeze(0), skip_special_tokens=True)