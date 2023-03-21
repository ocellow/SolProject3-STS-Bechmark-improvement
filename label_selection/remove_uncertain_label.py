import numpy as np
import pandas as pd
import math
import logging
from datetime import datetime
from tqdm.auto import tqdm
from importlib import reload
import torch

from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from cross_encoder.CrossEncoder import CrossEncoder


reload(logging)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[LoggingHandler()],
)

logging.info("Run remove_uncertain_label.py")



def remove_uncertain_label (model_list:List[str], sentence_pairs: List[str]):
    
    """
    Remove uncertain label from sentence pairs using a list of cross-encoder models.

    model_list: A list of paths to saved cross-encoder models.
    sentence_pairs: A list of sentence pairs to predict labels for.

    """
    
    
    labels = []
    
    for i,model_save_path in enumerate(model_list):
        
        logging.info("Load model_{}".format(i))
        model = CrossEncoder(model_save_path)
        
        logging.info("Predict_with_model_{}".format(i))
        scores = model.predict(sentence_pairs)
        
        globals()['label_{}'.format(i)] = np.round((scores*5).tolist(),1)
        labels.append(globals()['label_{}'.format(i)])
        
    
    
    label_mean = []
    label_std = []
    removed_mean = []

    for i in range(len(sentence_pairs)):
        label_each_pair = [labels[j][i] for j in range(len(labels))]
        label_std.append(np.round( (np.std(label_each_pair)),2 ) )
        label_mean.append(np.round(sum(label_each_pair)/len(label_each_pair),2))


        min = label_mean[i] - label_std[i]
        max = label_mean[i] + label_std[i]
        
        
        filtered_labels = [k for k in label_each_pair if min <= k <= max]

        removed_mean.append(np.round( sum(filtered_labels) / len(filtered_labels) ,1))
        
    return removed_mean




