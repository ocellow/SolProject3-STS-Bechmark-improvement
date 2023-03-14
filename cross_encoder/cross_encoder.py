from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from sentence_transformers import SentenceTransformer


class CrossEncoder():
  def __init__(self, model_name:str, 
               num_labels:int = 1, 
               max_length:int = None,
               tokenizer_args:Dict={},
               automodel_args:Dict={}):
    self.config = AutoConfig.from_pretrained(model_name) # load huggingface model

    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, **automodel_args)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
    self.max_length = max_length

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.default_activiation_function = nn.Sigmoid() if self.config.num_labels ==1 else nn.Identity()
  
  def batch_collate(self, batch):
    texts = [[] for _ in range(len(batch[0].texts))]
    labels = []

    for example in batch:
      for idx, text in enumerate(example.texts):
        texts[idx].append(text.strip())

      labels.append(example.label)

    tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first',
                               return_tensors='pt', max_length=self.max_length)
    for name in tokenized:
      tokenized[name] = tokenized[name].to(self.device)

    return tokenized, labels
  
  def batch_collate_text_only(self, batch):
    texts = [[] for _ in range(len(batch[0]))]

    for example in batch:
      for idx, text in enumerate(example):
        texts[idx].append(text.strip())
      
      tokenized = self.tokenizer(*text, padding=True, truncation='longest_first',
                                 return_tensors='pt', max_length=self.max_length)

      for name in tokenized:
        tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

  def _eval_during_training(self, evaluator, output_path, save_best_model,epoch,steps):
    if evaluator is not None:
      score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
      if score > self.best_score:
        self.best_score = score
        if save_best_model:
          self.save(output_path)


  def fit(self,
          train_dataloader: DataLoader,
          evaluator,
          epochs:int = 1,
          loss_fct=None,
          activation_fct = nn.Identity(),
          scheduler:str = 'WarmupLinear',
          warmup_steps:int=10000,
          optimizer_class:Type[Optimizer] = torch.optim.AdamW,
          optimizer_params:Dict[str,object] = {'lr':2e-5},
          weight_decay:float = 0.01,
          evaluation_steps=0,
          output_path:str=None,
          save_best_model:bool=True,
          max_grad_norm:float=1,
          show_progress_bar:bool=True):
    
    train_data_loader.collate_fn = self.batch_collate
    
    self.model.to(self.device)
    if output_path is not None:
      os.makedirs(output_path, exist_ok=True)
    
    self.best_score = -9999999
    num_train_steps = int(len(train_dataloader)*epochs)

    param_optimizer = list(self.model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay':weight_decay},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay':0.0}
    ]

    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

    if isinstance(scheduler, str):
      scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler,
                                                     warmup_steps = warmup_steps, t_total=num_train_steps)
    if loss_fct is None:
      loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels==1 else nn.CrossEntropyLoss()

    skip_scheduler=False

    #train loop
    for epoch in tqdm(range(epochs)):
      training_steps = 0
      self.model.zero_grad()
      self.model.train() 

      for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
        model_predictions = self.model(**features, return_dict=True)
        logits = activation_fct(model_predictions.logits)
        if self.config.num_labels==1:
          logits=logits.view(-1)
        
        loss = loss_fct(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        optimizer.step()

      optimizer.zero_grad()

      if not skip_scheduler:
        scheduler.step()
      
      training_steps +=1

      if evaluator is not None and evaluation_steps >0 and training_steps % evaluation_steps==0:
        self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)

        self.model.zero_grad()
        self.model.train()
    if evaluator is not None:
      self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)
  
  def predict(self, sentences:List[List[str]],
              batch_size:int = 32,
              show_progress_bar:bool=True,
              num_workers:int=0):
    if isinstance(sentences[0] , str):
      sentences = [sentences]
      input_was_string = True

    input_dataloader = DataLoader(sentences,
                                  batch_size=batch_size,
                                  collate_fn=self.batch_collate_text_only,
                                  num_workers=num_workers,
                                  shuffle=False)
    iterator = tqdm(input_dataloader, desc="Batches")
    activation_fct = self.default_activation_function

    pred_scores = []
    self.model.eval()
    self.model.to(self.device)
    with torch.no_grad():
      for features in iterator:
        model_predictions = self.model(**features, return_dict=True)
        logits = activation_fct(model_predictions.logits)

    if self.config.num_lables==1:
      pred_scores = [score[0] for score in pred_scores]

    if input_was_string:
      pred_scores = pred_scores[0]
    
    return pred_scores

  def save(self, path):

    if path is None:
      return
    
    self.model.save_pretrained(path)
    self.tokenizer.save_pretrained(path)

  def save_pretrained(self, path):
    return self.save(path)
