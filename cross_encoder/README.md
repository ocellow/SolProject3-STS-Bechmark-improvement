## cross_encoder

![image](https://user-images.githubusercontent.com/100064247/226699977-63687249-0125-4116-8996-1866615c11b5.png)


- run training directly 
```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/
%run ../SolProject3-STS-Bechmark-improvement/cross_encoder/train_utils.py
```

```python
extract_val_from_train(train,dev) # train에서 devset label분포에 맞게 train,val return 



create_input_examples(train,val) # train, val 한번에 input_examples로 변환 


train_ce(model_save_path,       # train cross_encoder
         train, # DataFrame
         dev, # DataFrame
         num_epochs,
         train_batch_size,
         pretrained_model_name
         verbose = True)
```


- or  

```python
%run ../SolProject3-STS-Bechmark-improvement/cross_encoder/CrossEncoder.py
```

```python
model = CrossEncoder(pretrained_model_name) # str
model.fit(train_dataloader, evaluator)

#predict
model.predict(sentences) # List[List[str] 
```

