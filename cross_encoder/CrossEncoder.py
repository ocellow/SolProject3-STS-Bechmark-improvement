# cross_encoder

import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from .model_evaluation.model_evaluator import ModelEvaluator



class CrossEncoder():
    def __init__(self, model_name:str, 
                num_labels:int = 1, 
                max_length:int = None,
                tokenizer_args:Dict={},
                automodel_args:Dict={}):
        self.config = AutoConfig.from_pretrained(model_name) # load config from pretrained model 


        self.config.num_labels = num_labels # set num_label as 1 for STS regression task 

        # load huggingface model, tokenizer 
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, **automodel_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length # if None, max length of self.model 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def batch_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip()) # 시작과 끝 공백 있을경우 제거

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float).to(self.device) # label to device
        
        # tokenized = {'input_ids':[[]] , 'token_type_ids':[[]], 'attention_mask':[[]]}
        for name in tokenized: 
            tokenized[name] = tokenized[name].to(self.device)  # token to device

        return tokenized, labels

    def batch_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized


    def _eval_during_training(self, evaluator, output_path, save_best_model):
        #if evaluator is not None:
        score, _ = evaluator(self, output_path=output_path)
        if score > self.best_score:
            self.best_score = score
            if save_best_model:
                self.save(output_path)

 
    def fit(self,
            train_dataloader,
            evaluator,
            epochs:int = 1,
            warmup_steps:int=10000,
            optimizer_class = torch.optim.AdamW,
            optimizer_params:Dict[str,object] = {'lr':2e-5},
            weight_decay:float = 0.01,
            evaluation_steps=0,
            output_path:str=None,
            save_best_model:bool=True,
            max_grad_norm:float=1,
            verbose = True):
            

        train_dataloader.collate_fn = self.batch_collate

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True) # make directory 
        
        self.model.to(self.device)

        self.best_score = -9999999 # for eval_during_training (score > best_score)
        num_train_steps = int(len(train_dataloader)*epochs)

        # get params from pretrained model 
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], # if True (not in no_decay list)
                'weight_decay':weight_decay},
            {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], # if False (in no_decay)
                'weight_decay':0.0} ]
        # n : name of bias, weights from all layers in ptmodel
        # p : w&b of each name 
        


        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        
        
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
        
        loss_fct = torch.nn.BCEWithLogitsLoss()  
        #loss_fct = torch.nn.MSELoss() 

        

        #train loop
        
        for epoch in tqdm(range(epochs)):
            training_steps = 0
            step_loss = 0
            self.model.zero_grad()
            self.model.train() 

            # features = input_ids, tkn_ids, attn_mask 
            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
                model_predictions = self.model(**features, return_dict=True)
                logits = model_predictions.logits # get logits from SequenceClassifier output (logits = Tensor([[]]))
                logits=logits.view(-1)  
            
                loss = loss_fct(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                scheduler.step()
                
                training_steps +=1

                step_loss += loss.item()

                if training_steps % 50 == 0 and training_steps !=0:
                    if verbose == True:
                        print(f"Step : {training_steps},  Avg Loss : { step_loss / training_steps:.4f}")

                if evaluation_steps >0 and training_steps % evaluation_steps==0:
                    self._eval_during_training(evaluator, output_path, save_best_model)

                    self.model.zero_grad()
                    self.model.train()
            
            self._eval_during_training(evaluator, output_path, save_best_model) # last val 

    def predict(self, sentences:List[List[str]], 
                batch_size:int = 32,
                num_workers:int=0,
                convert_to_numpy: bool = True):
        
        input_was_string=False
        if isinstance(sentences[0] , str):
            sentences = [sentences]
            input_was_string = True

        input_dataloader = DataLoader(sentences,
                                        batch_size=batch_size,
                                        collate_fn=self.batch_collate_text_only,
                                        num_workers=num_workers,
                                        shuffle=False)
        iterator = tqdm(input_dataloader, desc="Batches")
        
        activation_fct = nn.Sigmoid() 
        """
        transform the model's output logits into probabilty values (0 to 1) by nn.Sigmoid()
        cause sts score range = [0-1] continous values 
        """

        pred_scores = []
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)
                pred_scores.extend(logits)

        
        pred_scores = [score[0] for score in pred_scores]

        if convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores



    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

   
