import transformer_tokenizer
from fastai.text import Tokenizer
from transformers import AutoConfig,AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from functools import partial

class Handler():
    def __init__(self):
        self.tokenizer_class = AutoTokenizer
        self.config_class = AutoConfig
        self.model_class = AutoModel

    def fetch_pretrained(self, class_obj) -> object:
        pretrained = class_obj.from_pretrained(self.model_name)
        return pretrained

class SeqClass(Handler):
    def __init__(self, model_name):
        super().__init__()
        setattr(self, 'model_class', AutoModelForSequenceClassification)
        self.model_name = model_name
        self.pt_tokenizer, self.pt_config, self.pt_model = [self.fetch_pretrained(i) for i in [self.tokenizer_class,self.config_class,self.model_class]]
        self.tokenizer = Tokenizer(transformer_tokenizer.TransformersBaseTokenizer(self.model_name, self.pt_tokenizer))