import numpy as np
import pandas as pd
import math
import logging

from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict
import os

"""
codes are modified from 
UKPlab/sentence-transformers Code(https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py) 
"""



# pretrained_transformer

class PtTransformer(nn.Module):
    def __init__(self, 
                 model_name_or_path: str, 
                 max_seq_length = None,
                 model_args: Dict = {}, 
                 tokenizer_args: Dict = {}):

        super(PtTransformer, self).__init__()
        
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args)
        self._load_model(model_name_or_path, config, **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)

        #infer max_seq_length from model
        max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)
        self.max_seq_length = max_seq_length
        self.config_keys = ['max_seq_length']



    def _load_model(self, model_name_or_path, config, **model_args):
        #Load pre_trained transformer model

        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, **model_args)



    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self):
        return self.auto_model.config.hidden_size


    def tokenize(self, texts):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]


        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
