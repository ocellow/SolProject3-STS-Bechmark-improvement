import logging
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import paired_cosine_distances
from typing import List
import os
import csv
from sentence_transformers.readers import InputExample,LoggingHandler


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
    def __init__(self, sentence_pairs: List[List[str]], scores: List[float], encoding = 'cross_encoding', verbose=True):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.verbose = verbose
        self.encoding = encoding

        self.sentences1 = [pair[0] for pair in sentence_pairs]
        self.sentences2 = [pair[1] for pair in sentence_pairs]
        
    
    @classmethod
    def from_input_examples(cls, examples: List[InputExample], encoding = 'cross_encoding' , **kwargs):
        
            
        if encoding == 'cross_encoding':
 
            sentence_pairs = []
            scores = []

            for example in examples:
                sentence_pairs.append(example.texts)
                scores.append(example.label)

            return cls(sentence_pairs, scores, encoding=encoding, **kwargs)
        
        
        elif encoding == 'bi_encoding':
            
            sentences1 = []
            sentences2 = []
            scores = []

            for example in examples:
                sentences1.append(example.texts[0])
                sentences2.append(example.texts[1])
                scores.append(example.label)
            
            return cls(list(zip(sentences1, sentences2)), scores, encoding = encoding,**kwargs)
    
    
    def conf_matrix(self,pred_scores):
        
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
        
        return f1 
        

    def __call__(self, model, output_path: str = None):
        if self.verbose == True:
            logging.info("ModelEvaluator: Evaluating the model on STS dataset..")
        
        if self.encoding == 'cross_encoding':
            pred_scores = model.predict(self.sentence_pairs)

            eval_pearson, _ = pearsonr(self.scores, pred_scores)

            if self.verbose == True:
                logging.info("Correlation:\tPearson: {:.4f}".format(eval_pearson))
            
            f1 = self.conf_matrix(pred_scores)

            return eval_pearson, f1
        
        elif self.encoding == 'bi_encoding':
            
            embeddings1 = model.encode(self.sentences1, convert_to_numpy=True)
            embeddings2 = model.encode(self.sentences2, convert_to_numpy=True)
            

            pred_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            eval_pearson_cosine, _ = pearsonr(self.scores, pred_scores)
            
            if self.verbose == True:
                logging.info("Cosine-Similarity:\tPearson: {:.4f}".format(eval_pearson_cosine))
                
            f1 = self.conf_matrix(pred_scores)
            
            return eval_pearson_cosine, f1
