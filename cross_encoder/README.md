## cross_encoder

- run training directly 
```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/
%run ../SolProject3-STS-Bechmark-improvement/cross_encoder/train_utils.py
```


# or  

```python
%run ../SolProject3-STS-Bechmark-improvement/cross_encoder/CrossEncoder.py
```

```python
model = CrossEncoder(pretrained_model_name) # str
model.fit(train_dataloader, evaluator)

#predict
model.predict(sentences) # List[List[str] 
```

