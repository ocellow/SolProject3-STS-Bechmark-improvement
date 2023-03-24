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


logger = logging.getLogger(__name__)

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
        
        super().__init__(modules) #modules 에 전달된 모듈들 nn.Sequential에 전달 - modules를 하나로 묶어서 SentenceTransformer에서 실행 


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


    def batch_collate(self, batch):
 
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            
            """
            self.tokenize(texts[idx]) : each sequence of a sentence's 'input_ids', 'token_type_ids','attention_mask'
            
            for example: 
            idx:1 , tokenized = {'input_ids': tensor([[   0, 1826,    2]]),
                                 'token_type_ids': tensor([[0, 0, 0]]),
                                 'attention_mask': tensor([[1, 1, 1]])}
            """


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
            dataloaders = [dataloader for dataloader, _ in train_objectives]

    
            for dataloader in dataloaders:
                dataloader.collate_fn = self.batch_collate

            # CosineSimilarityLoss(model=bi_encoder)
            loss_models = [loss for _, loss in train_objectives]
            for loss_model in loss_models:
                loss_model.to(self.device)

            self.best_score = -9999999 # for eval_during_training (score > best_score)

            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
            num_train_steps = int(steps_per_epoch * epochs)

            
            # optimizer for each model 
            optimizers = []
            schedulers = []
            for loss_model in loss_models: # two model for bi-encoding 
                param_optimizer = list(loss_model.named_parameters()) # get params from pretrained model 

                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},# if True (not in no_decay list)
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # if False (in no_decay)
                ]

                optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
                
                scheduler_obj = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
                
                optimizers.append(optimizer)
                schedulers.append(scheduler_obj)


            global_step = 0
            data_iterators = [iter(dataloader) for dataloader in dataloaders]

            num_train_objectives = len(train_objectives) # 2 model

            
            for epoch in tqdm(range(epochs), desc="Epoch"):
                training_steps = 0

                for loss_model in loss_models: # for each model zeor_grad / train()
                    loss_model.zero_grad()
                    loss_model.train()

                for _ in tqdm(range(steps_per_epoch), desc="Iteration", smoothing=0.05):
                    for train_idx in range(num_train_objectives):
                        loss_model = loss_models[train_idx]
                        optimizer = optimizers[train_idx]
                        scheduler = schedulers[train_idx]
                        data_iterator = data_iterators[train_idx]

                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features, labels = data
                        labels = labels.to(self.device)
                        features = list(map(lambda batch: batch_to_device(batch, self.device), features)) ##

                        loss = loss_model(features, labels)
                        loss.backward()
                        
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()

                        
                        scheduler.step()

                    training_steps += 1
                    global_step += 1

                    if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                        self._eval_during_training(evaluator, output_path, save_best_model)

                        for loss_model in loss_models:
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
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]


        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                embeddings = out_features['sentence_embedding']
                embeddings = embeddings.detach()

                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        
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
