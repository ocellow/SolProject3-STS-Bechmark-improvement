# model evaluator 
import logging
from scipy.stats import pearsonr
from typing import List
import os
import csv
from sentence_transformers.readers import InputExample

from importlib import reload
reload(logging)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[LoggingHandler()],
)


# class input : sentence_pairs[List]
class ModelEvaluator:
    def __init__(self, sentence_pairs: List[List[str]], scores: List[float],verbose=True):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.verbose = verbose
        
    
    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        
        return cls(sentence_pairs, scores, **kwargs)

    def __call__(self, model, output_path: str = None):
        if self.verbose == True:
            logging.info("ModelEvaluator: Evaluating the model on STS dataset..")
        pred_scores = model.predict(self.sentence_pairs)
        
        eval_pearson, _ = pearsonr(self.scores, pred_scores)
        if self.verbose == True:
            logging.info("Correlation:\tPearson: {:.4f}".format(eval_pearson))
        
        conf_mat = []
        for score, pred in zip(self.scores, pred_scores):
            score = score * 5.0
            pred = pred * 5.0

            if (pred >= 3.0) & (score >= 3.0):
                conf_mat.append('True_P')
            elif (pred < 3.0) & (score < 3.0):
                conf_mat.append('True_N')
            elif (pred >= 3.0) & (score < 3.0):
                conf_mat.append('False_P')
            else:
                conf_mat.append('False_N')
        
        TP = conf_mat.count('True_P')
        TN = conf_mat.count('True_N')
        FP = conf_mat.count('False_P')
        FN = conf_mat.count('False_N')
        if self.verbose == True:
            logging.info("True_P: {}\tTrue_N: {}\tFalse_P: {}\tFalse_N: {}".format(TP,TN,FP,FN))
        pc = TP / (TP + FP)
        rc = TP / (TP + FN)

        f1 = 2 * pc * rc / (pc + rc)
        if self.verbose == True:
            logging.info("F1-score:\t {:.4f}".format(f1))

        return eval_pearson, f1
