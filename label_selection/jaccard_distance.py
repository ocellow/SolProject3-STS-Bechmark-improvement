from konlpy import *


# tokenize by morpheme-level 
def jaccard_distance(sentence_pair):
    mecab = Mecab()
    tokens1 = set(mecab.morphs(sentence_pair[0]))
    tokens2 = set(mecab.morphs(sentence_pair[1]))
    
    # Calculate the Jaccard distance between the two sets of morphemes
    jaccard_distance = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    return jaccard_distance
  
  
