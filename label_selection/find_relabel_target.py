import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sentence_transformers.readers import InputExample
import torch
import torch.backends.cudnn as cudnn 
import random 
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
import gc
import math
import logging
import tqdm 
from tqdm.auto import tqdm
from importlib import reload

from cross_encoder.train_utils import *
from cross_encoder.CrossEncoder import CrossEncoder
from model_evaluation.model_evaluator import ModelEvaluator
 


# 학습경과 모니터링하는 logger 초기화
reload(logging)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[LoggingHandler()],
)

class FindRelabelTarget():  
    def __init__(self,
                model_save_path, 
                train_path,
                dev_path,
                num_epochs,
                train_batch_size,
                num_folds=10):

        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.model_save_path = model_save_path
        
        self.og_train = pd.read_csv(train_path)
        self.og_train = self.og_train.reset_index()
        self.og_train = self.og_train[['sentence1','sentence2','labels.label']]
        
        self.dev = pd.read_csv(dev_path)
        self.dev = self.dev[['sentence1','sentence2','labels.label']]
        
        self.num_folds = num_folds
        self.filtered_df = self.filter_df_by_label_range()
        
    
    # train 에서 2.0-3.5값만 추출한 데이터프레임 만들기 
    def filter_df_by_label_range(self):
        
        label_col = 'labels.label'
        filtered_df = self.og_train[(self.og_train[label_col] >= 2.0) & (self.og_train[label_col] <= 3.5)]
        return filtered_df
 
    def get_data_by_fold(self, fold_num): ## fold로 나눠서 test 추출  
        # 데이터프레임의 인덱스를 리스트로 변환
        idx = self.filtered_df.index.tolist()
        
        # KFold 객체 생성
        kf = KFold(n_splits=self.num_folds, shuffle=True,random_state=0)
        
        # 각 fold의 인덱스를 저장할 리스트
        folds = []

        # sklearn KFold 객체를 사용하여 데이터를 분할
        for _, fold_idx in kf.split(idx):
            folds.append(fold_idx)

        # 몇번째 fold인지 선택 
        test_idx = folds[fold_num]
        test_data = self.filtered_df.iloc[test_idx]

        # 선택된 test_idx에 해당하는 부분을 test, 나머지를 train + val 
        train_data = self.og_train.drop(index=test_idx)


        return train_data, test_data

    def extract_false(self, pretrained_model_name=None, seed=0):
    # fold num에 따라 데이터 선택 
        
        if pretrained_model_name is None:
            pretrained_model_name = "klue/roberta-base"

        # torch seed fix 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(seed)

        false_idx = [] # false n&p index in og_train
        
        for i in range(self.num_folds):

            # prevent from cuda memory error  
            gc.collect()
            torch.cuda.empty_cache()
            
            # 학습 루프마다 다른 이름으로 저장 
            model_save_path = self.model_save_path + '-' + f'{i+1}th fold'
            
            # [i]th train,val,test 
            logging.info('get_data_by_fold / extract_test')
            train, test = self.get_data_by_fold(i) # extract test from train(2.0-3.5)
            
            # load & initialize pretrained model 
            if i ==0:
                logging.info(f'load pretrained_model:{pretrained_model_name}..')
            else: 
                logging.info(f'model initialization:{i+1}th..')
            
            # load pretrained model / train model by each fold 
            cross_encoder = train_ce(model_save_path,
                                     train,
                                     self.dev,
                                     self.num_epochs,
                                     self.train_batch_size,
                                     pretrained_model_name,
                                     verbose=False)

            # predict test data / return label 
            logging.info(f'extract False Label..')
            test_pairs = list(zip(test['sentence1'], test['sentence2']))
            pred = cross_encoder.predict(test_pairs)
            pred = np.round([num * 5.0 for num in pred],1) # 0-5단위 점수로 복구

            # Flase Negative and False Positive label Extraction 
            answer = test['labels.label'].tolist() # 정답 라벨의 list 

            # list of false from test 
            for j in range(len(test)):
                if (pred[j] >= 3.0) & (answer[j] < 3.0):
                    false_idx.append(test.index[j]) # add index of dataframe 
                elif (pred[j] <= 3.0) & (answer[j] > 3.0):
                    false_idx.append(test.index[j])
            
            


            logging.info(f'finish extracting false label')

        return false_idx

    # use standard deviation of labels from different model / seed to extract uncertain datapoint 
    def extract_uncertain(self, false_idx, num_seeds=None, model_list=None):

        test = self.og_train.iloc[false_idx]
        train = self.og_train.drop(false_idx)

        if num_seeds is None:
            num_seeds = 3

        if model_list is None:
            model_list = ['klue/roberta-base',
                        'klue/bert-base',
                        'monologg/koelectra-base-v3-discriminator']

        preds_all_model = [] # 
        for pretrained_model_name in model_list: # 모델 별 루프
            logging.info(f'Load {pretrained_model_name}.. to extract_uncertain')
            
            seeds = random.sample(range(100),num_seeds) #num_seed 인자로 받기 
            for i, seed in enumerate(seeds): # 시드별 루프 
                
                logging.info(f'set_{i}th seed')
                
                # cuda memory error 방지  
                gc.collect()
                torch.cuda.empty_cache()

                # seed 
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                cudnn.benchmark = False
                cudnn.deterministic = True
                random.seed(seed)
                
                # 학습 루프마다 다른 이름으로 저장 
                model_save_path = self.model_save_path + '-' + f'{pretrained_model_name}' + '-' + f'seed{seed}'
                
                # load pretrained model / train with each random seed  
                cross_encoder = train_ce(model_save_path,
                                         train,
                                         self.dev,
                                         self.num_epochs,
                                         self.train_batch_size,
                                         pretrained_model_name,
                                         verbose=False)
            
                # predict test data / return label 
                logging.info(f'Do test..')
                test_pairs = list(zip(test['sentence1'], test['sentence2']))
                pred = cross_encoder.predict(test_pairs)
                pred = np.round([num * 5.0 for num in pred],1) # 0-5단위 점수로 복구
                
                preds_all_model.append(pred)

        std_per_label = [] # calculate standard deviation of each datapoint
        for i in range(len(preds_all_model[0])):
            pred_per_label = [preds_each_model[i] for preds_each_model in preds_all_model]
            std_per_label.append(np.std(pred_per_label)) # standard deviation of each data

        threshold = np.median(std_per_label) # use meidan as a threshold 
        logging.info(f'threshold:{threshold}')

        uncertain_dp_idx = [test.index[i] for i, std in enumerate(std_per_label) if std > threshold] 

        return uncertain_dp_idx

if __name__ == '__main__':
    num_epochs = 3
    train_batch_size = 64
    model_save_path = '/content/sample_data/cross_encoder'
    train_path = 'https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_train%20(2).csv'
    dev_path = 'https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_val%20(2).csv'
    
    find_re_target = FindRelabelTarget(model_save_path = model_save_path,
                                    train_path = train_path,
                                    dev_path = dev_path,
                                    train_batch_size = train_batch_size,
                                    num_folds = 10)

    false_idx = find_re_target.extract_false()
    relabel_target = find_re_target.extract_uncertain(false_idx)
    pd.read_csv(train_path).loc[relabel_target]
