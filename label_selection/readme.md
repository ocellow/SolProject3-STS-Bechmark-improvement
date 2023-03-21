# Label Selection 

## find_relabel_target
```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement/
%run SolProject3-STS-Bechmark-improvement/label_selection/find_relabel_target.py
```
<img width="349" alt="image" src="https://user-images.githubusercontent.com/100064247/226592917-e73d5b79-0e28-4fd2-9a11-959b6052ac91.png">







## remove uncertain pseudo label
- 모델 성능 향상을 위한 self-training방식 적용 시 불확실성이 높은 pseudo-label을 선별 
- 사용방법

```python
%run remove_uncertain_label.py
```

```python
# unlabeled sentence pairs
sentence_pairs = [ *unlabeled sentence pairs* ]

# make list of model save path
model_list = [ *model_save_path_1*, *model_save_path_2* , *model_save_path_3* , ... ] # trained model

remove_uncertain_label(model_list,sentence_pairs)

```
