"""
use jaccard distance to make new devset 
for below 3.0 score, high diistance
above 3.0, low distance pairs are prioritized 
prevent from model to predict based on wordoverlap 
"""

import pandas as pd
from konlpy import *


# tokenize by morpheme-level 
def jaccard_distance(sentence_pair):
    mecab = Mecab()
    tokens1 = set(mecab.morphs(sentence_pair[0]))
    tokens2 = set(mecab.morphs(sentence_pair[1]))
    
    # Calculate the Jaccard distance between the two sets of morphemes
    jaccard_distance = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    return jaccard_distance
  
#리스트를 받으면 크기순(오름,내림) index 반환 
def top_n_idx(lst, n, reverse:bool=True):

    indexed_lst = list(enumerate(lst))

    def get_key(pair):
        return pair[1] # 원소기준으로 정렬하기 위한 함수 
    
    if reverse is True:
        sorted_idx = sorted(indexed_lst, key=get_key, reverse=True)  # 원소 높은순 .
    else:
        sorted_idx = sorted(indexed_lst, key=get_key, reverse=False)  # 낮은순  
    
    top_n_idx = [idx for idx, val in sorted_idx[:n]] # top n index 추출
    return top_n_idx

    
# dataframe과 label을 입력하면 top_n을 출력  

# reverse = 'True' : top_n jacc_dist 
# reverse = 'False' : below_n jacc_dist

def rank_jacc(data_path, label:float, top_n:int, reverse:str ='True'):
    df = pd.read_csv(data_path)
    filtered_df = df[df['labels.label'] == label]
    pairs = list(zip(filtered_df['sentence1'],filtered_df['sentence2']))

    # jaccard distance 모든 각 문장 페어에 대해서 구하기
    overlap_score = []
    for sent_pair in pairs:
        overlap_score.append(jaccard_distance(sent_pair))
        

    # distance 높은 순으로 반환 
    if reverse == 'True':
        idx = top_n_idx(overlap_score, top_n, reverse=True)
    
    # 낮은 순 반환 
    elif reverse == 'False':
        idx = top_n_idx(overlap_score, top_n, reverse=False)
    
    top_n_idx_of_df = filtered_df.iloc[idx].index.tolist() # 원본데이터의 idx를 반환 
    

    return df.iloc[top_n_idx_of_df]

