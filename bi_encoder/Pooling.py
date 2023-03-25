import torch
from torch import Tensor
from torch import nn
from typing import List, Dict
import os
import json

# mean pooling 
class Pooling(nn.Module):
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_mean_tokens: bool = True,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = 'pooling_mode_mean_tokens'
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens


    def forward(self, features):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []

        if self.pooling_mode_mean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)


        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features


    def get_config_dict(self):
        return {self.config_keys: self.__dict__[self.config_keys]}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)
