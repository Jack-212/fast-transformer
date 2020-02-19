from fastai.text import Vocab
import numpy as np
from typing import List, Collection
import collections

class TransformersVocab(Vocab):
    # itos is a list of token vocab
    # stoi is an id:tok dictionary, built below
    # Overwrite base functions to use pt_tokenizer behaviour
    def __init__(self, tokenizer):
        super().__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None\
        else self.tokenizer.convert_ids_to_tokens(nums)
    
    # Following two are direct from Vocab Object.
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})