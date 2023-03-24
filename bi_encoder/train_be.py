import random
import math
import logging
from importlib import reload
from tqdm.auto import tqdm 

from cross_encoder.train_utils import extract_val_from_train, create_input_examples
from bi_encoder import PtTransformer, BiEncoder, CosineSimilarityLoss, Pooling
from sentence_transformers import InputExample,LoggingHandler
from model_evaluation.model_evaluator import ModelEvaluator


def train_be(model_save_path,
             train, # DataFrame
             dev, # DataFrame
             num_epochs,
             train_batch_size,
             pretrained_model_name='klue/roberta-base',
             verbose = True):
    # load pretrained model 
    embedding_model = PtTransformer(pretrained_model_name)
    
    pooler = Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens = True)

    bi_encoder = BiEncoder(modules = [embedding_model, pooler])
    # bi_encoder = SentenceTransformer(modules = [embedding_model, pooler])

    logging.info('extract_val_from_train')
    _train, val = extract_val_from_train(train, dev) # extract val from train 

    logging.info('create_input_examples')
    sts_train_examples, sts_val_examples = create_input_examples(_train,val)

    logging.info('assign_DataLoader_and_val_evaluator')
    train_dataloader = DataLoader(sts_train_examples,
                                    shuffle=True,
                                    batch_size = train_batch_size)
    # Evaluator by sts-validation
    logging.info("Creating bi-encoding evaluators")
    val_evaluator = ModelEvaluator.from_input_examples(sts_val_examples,encoding='bi_encoding')
    
    train_loss = CosineSimilarityLoss(model=bi_encoder)


    warmup_steps = math.ceil(len(train_dataloader) * num_epochs / train_batch_size*0.1) # 10%of train

    logging.info(f'start train model..')
    bi_encoder.fit(
        train_objectives = [(train_dataloader,train_loss)],
        evaluator=val_evaluator,
        epochs=num_epochs,
        evaluation_steps=int(len(train_dataloader)*0.1),
        optimizer_params = {'lr':5e-5},
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        show_progress_bar = True
    )

    return bi_encoder
