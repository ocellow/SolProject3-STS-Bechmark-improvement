import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers import InputExample
import torch
import torch.backends.cudnn as cudnn 
import random 
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import gc
import math
import logging 

# 학습경과 모니터링하는 logger 초기화
from importlib import reload
reload(logging)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    handlers=[LoggingHandler()],
)

class FindRelabelTarget():
    def __init__(self, num_epochs, train_batch_size, model_save_path, train_path, dev_path, num_folds=10):
        
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.model_save_path = model_save_path

        self.og_train = pd.read_csv(train_path)
        self.og_train = self.og_train.reset_index() ### 
        self.og_train = self.og_train[['sentence1','sentence2','labels.label']] # train_path = og_train_path ###
        self.dev = pd.read_csv(dev_path) # dev_path = KLUE devset_path
        self.num_folds = num_folds
        
        self.filtered_df = self.filter_df_by_label_range()
        

    
    # train 에서 2.0-3.5값만 추출한 데이터프레임 만들기 
    def filter_df_by_label_range(self):
        
        # label_col: 라벨값이 저장된 열 이름 (문자열)
        # label_range: 추출할 라벨값 범위 (튜플 형태로 최소값과 최대값을 지정)
        label_col = 'labels.label'
        label_range = (2.0,3.5)

        filtered_df = self.og_train[(self.og_train[label_col] >= label_range[0]) & (self.og_train[label_col] <= label_range[1])]
        return filtered_df
 
    def get_data_by_fold(self, fold_num): ##데이터 겹쳐서 분할되는 문제 해결 
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

    # trainset에서 devset분포에 맞게 valset 추출 
    def extract_val_from_train(self, train_data):
        dev = self.dev[['sentence1','sentence2','labels.label']]
        
        # select val by label dist 
        val_ratio = (dev.groupby('labels.label').size() / dev.groupby('labels.label').size().sum()).to_frame('ratio')
        val_ratio['number'] = round(val_ratio['ratio'] * (len(train_data) *0.05)).astype(int) # train데이터의 5%를 val로 사용 
        
        val_data = pd.DataFrame()
        label = list(set(train_data['labels.label'].values))
        
        
        for i in label:
            sample = train_data.loc[train_data['labels.label']==i].sample(n=val_ratio['number'][i], random_state = 0)
            val_data = pd.concat([val_data,sample])
        
        
        train_data = train_data.drop(val_data.index)
        return train_data, val_data
    
    # train, val 한번에 input_examples로 변환 
    def create_input_examples(self,train,val):

        dfs = [train, val]
        input_examples = [[] for _ in range(2)]

        for i, df in enumerate(dfs):
            for _, row in tqdm(df.iterrows()):
                sentence1, sentence2, score = row[0], row[1], row[2]/5.0
                input_examples[i].append(InputExample(texts=[sentence1, sentence2], label=score))

        return input_examples[0], input_examples[1] # train, val


    def train(self, pretrained_model_name, model_save_path, train):
        # load pretrained model 
        cross_encoder = CrossEncoder(pretrained_model_name, num_labels=1)

        logging.info('extract_val_from_train_data')
        train, val = self.extract_val_from_train(train) # extract val from train 

        logging.info('create_input_examples')
        sts_train_examples, sts_val_examples = self.create_input_examples(train,val)

        logging.info('assign_DataLoader_and_val_evaluator')
        train_dataloader = DataLoader(sts_train_examples,
                                      shuffle=True,
                                      batch_size = self.train_batch_size)
        val_evaluator = CECorrelationEvaluator.from_input_examples(sts_val_examples)
        
        warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs / self.train_batch_size*0.1) # 10%of train

        logging.info(f'start train model..')
        cross_encoder.fit(
            train_dataloader = train_dataloader,
            evaluator=val_evaluator,
            epochs=self.num_epochs,
            evaluation_steps=int(len(train_dataloader)*0.1),
            optimizer_params = {'lr':5e-5},
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            show_progress_bar = True
        )

        return cross_encoder



    def extract_false(self, pretrained_model_name=None):
    # fold num에 따라 데이터 선택 
        
        if pretrained_model_name is None:
            pretrained_model_name = "klue/roberta-base"

        seed = 0
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

            # load & initialize pretrained model 
            if i ==0:
                logging.info(f'load pretrained_model:{pretrained_model_name}..')
            else: 
                logging.info(f'model initialization:{i+1}th..')
            
            # [i]th train,val,test 
            logging.info('get_data_by_fold / extract_test')
            train, test = self.get_data_by_fold(i) # extract test from train(2.0-3.5)

            # load pretrained model / train model by each fold 
            cross_encoder = self.train(pretrained_model_name, model_save_path, train)
            
            # predict test data / return label 
            logging.info(f'extract False Label..')
            test_pairs = list(zip(test['sentence1'], test['sentence2']))
            pred = cross_encoder.predict(test_pairs, show_progress_bar=True)
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
            seeds = random.sample(range(100),num_seeds) #num_seed 인자로 받기 

            for seed in seeds: # 시드별 루프 
                
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
                
                            # load pretrained model / train model by each fold 
                cross_encoder = self.train(pretrained_model_name, model_save_path, train)
            
                # predict test data / return label 
                logging.info(f'Do test..')
                test_pairs = list(zip(test['sentence1'], test['sentence2']))
                pred = cross_encoder.predict(test_pairs, show_progress_bar=True)
                pred = np.round([num * 5.0 for num in pred],1) # 0-5단위 점수로 복구
                
                preds_all_model.append(pred)

        std_per_label = [] # calculate standard deviation of each datapoint
        for i in range(len(preds_all_model[0])):
            pred_per_label = [preds_each_model[i] for preds_each_model in preds_all_model]
            std_per_label.append(np.std(pred_per_label)) # standard deviation of each data

        threshold = np.median(std_per_label)*2 # use meidan*2 as a threshold 
        logging.info(f'threshold:{threshold}')

        uncertain_dp_idx = [test.index[i] for i, std in enumerate(std_per_label) if std > threshold] 

        return uncertain_dp_idx


if __name__ == '__main__':
    num_epochs = 1
    train_batch_size = 64
    model_save_path = '/content/sample_data/cross_encoder'
    train_path = 'https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_train%20(2).csv'
    dev_path = 'https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_val%20(2).csv'


    find_re_target = FindRelabelTarget(num_epochs,
                                       train_batch_size,
                                       model_save_path,
                                       train_path,
                                       dev_path,
                                       num_folds=2)
    
    false_idx = find_re_target.extract_false()
    relabel_target = find_re_target.extract_uncertain(false_idx)
