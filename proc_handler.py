import transformer_tokenizer
import transformer_vocab
from fastai.text import Tokenizer, NumericalizeProcessor, TokenizeProcessor
from transformers import AutoConfig,AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from functools import partial
import torch.nn as nn

class ProcHandler():
    def __init__(self):
        self.tokenizer_class = AutoTokenizer
        self.config_class = AutoConfig
        self.model_class = AutoModel

    def fetch_pretrained(self, class_obj) -> object:
        pretrained = class_obj.from_pretrained(self.model_name)
        return pretrained

class SeqClassHandler(ProcHandler):
    def __init__(self, model_name):
        super().__init__()
        setattr(self, 'model_class', AutoModelForSequenceClassification)
        self.model_name = model_name
        self.pt_tokenizer, self.pt_config, self.pt_model = [self.fetch_pretrained(i) for i in [self.tokenizer_class,self.config_class,self.model_class]]
        self.base_tokenizer = transformer_tokenizer.TransformersBaseTokenizer(self.model_name, self.pt_tokenizer)
        self.tokenizer = Tokenizer(tok_func=self.base_tokenizer, pre_rules= [], post_rules= [])
        self.vocab = transformer_vocab.TransformersVocab(self.pt_tokenizer)
        self.model = CustTransformerModel(self.pt_model)
    
    def __repr__(self):
        return f'Model: {self.model_name}\nOriginal Tokenizer: {self.pt_tokenizer}'

    def __call__(self):
        # Processors for Tokenising & Processing
        tokenizer_proc = TokenizeProcessor(tokenizer = self.tokenizer, include_bos=False,include_eos=False)
        numerilize_proc = NumericalizeProcessor(vocab = self.vocab)
        return [tokenizer_proc, numerilize_proc]

# Overwrite forward method & return scores from tuple
class CustTransformerModel(nn.Module):
    def __init__(self, pt_model):
        super().__init__()
        self.pt_model = pt_model

    def forward(self, input_ids):
        # Generally returns loss, logits, attentions
        # When not passing labels in forward - returns logits as first element
        x = self.pt_model(input_ids)[0]
        return x