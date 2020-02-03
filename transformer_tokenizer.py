from fastai import *
from fastai.text import BaseTokenizer
from typing import List
from transformers import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

class TransformersBaseTokenizer(BaseTokenizer):
    """Wraps transformer pretrained tokenizer
        Inherits from BaseTokenizer for fast.ai compatibility"""
    
    def __init__(self, model_name, pretrained_tokenizer):
        self.model_name = model_name

    def __name__(self):
        return f'{self.model_name}_BaseTokenizer'

    def __call__(self, *args, **kwargs): 
        return self

    # tokenizer instance passed through to Tokenizer tok_func
    def tokenizer(self, t:str) -> List[str]:
        
        cls_token = self.pretrained_tokenizer.cls_token # classification token (bos)

        sep_token = self.pretrained_tokenizer.sep_token # seperation token
        

        # Roberta requires a space to start the input string (add prefix_space) (Inherits from GPT2Tokenizer)
        if self.model_name in list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP):

            #Take 2 from max_seq_len to account for the cls & sep tokens we add to the sequence
            tokens = self.pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            
        else:
            tokens = self.pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
    
        return [cls_token] + tokens + [sep_token]