# bi-encoder
![image](https://user-images.githubusercontent.com/100064247/226699977-63687249-0125-4116-8996-1866615c11b5.png)

```python
%run PtTransformer.py
%run Pooling.py
%run BiEncoder.py
%run CosineSimilarityLoss.py
%run model_evaluator.py
%run train_be.py
```
## fast train
```python
model_save_path = '' #path
train = pd.read_csv('https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_train%20(2).csv')
train = train[['sentence1','sentence2','labels.label']]
dev = pd.read_csv('https://raw.githubusercontent.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/main/data/KLUE_STS_val%20(2).csv')
dev = dev[['sentence1','sentence2','labels.label']]
num_epochs = 1
train_batch_size = 32

# train BiEncoder
train_be(model_save_path,
         train, # DataFrame
         dev, # DataFrame
         num_epochs, 
         train_batch_size)
```
## do manually
```python
#get model
embedding_model = PtTransformer(pretrained_model_name) #from huggingface
pooler = Pooling(embedding_model._get_word_embedding_dimenstion())
bi_encoder = BiEncoder(modules = [embedding_model, pooler])

...

bi_encoder.fit(train_objectives = [(train_dataloader,train_loss)],
               evaluator=val_evaluator,
               epochs=num_epochs,
               warmup_steps = warmup_steps,
               output_path = model_save_path) # return bi_encoder

```

## evaluation / testing 
```python
test_evaluator = ModelEvaluator.from_input_examples(test_examples, encoding='bi_encoding',verbose=True)
test_evaluator(trained_bi_encoder)
```
