import math
import logging
import os
import stat
from collections import OrderedDict
from typing import List, Dict
import requests
import numpy as np
from numpy import ndarray

import transformers
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from tqdm.autonotebook import trange

from sentence_transformers.util import batch_to_device


"""
codes are modified from 
UKPlab/sentence-transformers Code(https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py) 
"""



class BiEncoder(nn.Sequential):

    # without forward method, using Transformer pretrained model 

    def __init__(self, modules):

        # modules = [embedding_model, pooler] / embd model : from pretrained
        modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        """
        OrderedDict: 
        for each module, allocate idx 
        {idx: modules[i]}
        입력순서를 유지하면서 
        """
        
        super().__init__(modules) #modules 에 전달된 모듈들 nn.Sequential에 전달 - modules를 하나로 묶어서 BiEncoder에서 실행 


        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    
    def _first_module(self):
        return self._modules[next(iter(self._modules))] # torch ._modules -> _ : return into Ordered dict / sequential에 전달된 modules 
    """
    torch ._modules -> _ : return into Ordered dict / sequential에 전달된 modules
    next(iter()) : 첫번째 key값에 해당하는 module을 반환 
    """
    
    def tokenize(self, texts):
        """
        Tokenizes the texts (from PtTransformer)
        """
        return self._first_module().tokenize(texts) # 'input_ids', 'token_type_ids', 'attention_mask'


    def batch_collate(self, batch): #batch = list of tuples: [(tokens, label), ...]
        
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)] # text length만큼 list 
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts): 
                texts[idx].append(text) 

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels

      
    def _eval_during_training(self, evaluator, output_path, save_best_model):
        eval_path = output_path
        score, _ = evaluator(self, output_path=eval_path)
        if score > self.best_score:
            self.best_score = score
            if save_best_model:
                self.save(output_path)
        
    def evaluate(self, evaluator, output_path): 
        """
        outpath에 directory만들고 evaluator 반환 
        """
        os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

      
    def fit(self,
            train_objectives,
            evaluator = None,
            epochs = 1,
            warmup_steps: int = 10000,
            optimizer_class= torch.optim.AdamW,
            optimizer_params = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            ):


            self.to(self.device)
            

            # trian_dataloader
            dataloader = train_objectives[0][0]
            dataloader.collate_fn = self.batch_collate # dataloader batch clltfn 할당 

            
            # CosineSimilarityLoss(model=bi_encoder)
            loss_model = train_objectives[0][1]
            loss_model.to(self.device)

            self.best_score = -9999999 # for eval_during_training (score > best_score)

            steps_per_epoch = len(dataloader)
            num_train_steps = int(steps_per_epoch * epochs)

            
            # optimizer 


            param_optimizer = list(loss_model.named_parameters()) # get params from pretrained model 

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},# if True (not in no_decay list)
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # if False (in no_decay)
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

            # scheduler
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
                



            global_step = 0
            data_iterators = iter(dataloader)

            
            #train loop
            for epoch in tqdm(range(epochs), desc="Epoch"):
                training_steps = 0


                loss_model.zero_grad()
                loss_model.train()

                for _ in tqdm(range(steps_per_epoch), desc="Iteration", smoothing=0.05):
                    data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device)
                    features = list(map(lambda batch: batch_to_device(batch, self.device), features)) 

                    loss = loss_model(features, labels) #loss_model = CossineSimilarityLoss(nn.Module)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()


                    scheduler.step()

                    training_steps += 1
                    global_step += 1

                    if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                        self._eval_during_training(evaluator, output_path, save_best_model)


                        loss_model.zero_grad()
                        loss_model.train()


                self._eval_during_training(evaluator, output_path, save_best_model -1)


    def encode(self, 
               sentences,
               batch_size: int = 32,
               convert_to_numpy: bool = True,
               ):
     
        self.eval()
    
        device = self.device
        self.to(device)

        
        length_sorted_idx = np.argsort([-sum([len(s) for s in sen]) for sen in sentences]) 
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx] # sort by length
        """
        np.argsort([[len_of_sent1], [len_of_sent2],... ]) , sort by length of sent , return idx
        """


        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size] # return sentences as size of batch/ len(sentences_batch) = batch_size
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device) # to device each batch tkns

            with torch.no_grad():
                out_features = self.forward(features)

                embeddings = out_features['sentence_embedding'] #sentence embedding from forward to model 
                embeddings = embeddings.detach()

                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings) # add elements to list allembeddings 

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)] # go before length_sorted of sentences
        
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        return all_embeddings
    


    def save(self, path):

        if path is None:
            return

        #Save modules

        for name in self._modules:
            module = self._modules[name]

            module.save(path)
