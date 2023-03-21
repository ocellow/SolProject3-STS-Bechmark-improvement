### ModelEvaluator
- eval during training / loss, pearson_r, f1_score 

```python
! git clone https://github.com/tommyEzreal/SolProject3-STS-Bechmark-improvement
%cd /SolProject3-STS-Bechmark-improvement/model_evaluation/
%run model_evaluator.py
```
```python
#Evaluate by sts-test
test_evaluator = ModelEvaluator.from_input_examples(sts_test_examples)
pearson_r , f1_score = test_evaluator(trained_model) 
```
### output example
<img width="803" alt="image" src="https://user-images.githubusercontent.com/100064247/226134635-549be079-9dba-4d59-ba4c-3639b7aec78d.png">







### For Evaluation Experiments
 
- 새로 제작한 devset을 testset으로 사용하여 STS 모델의 성능을 평가할 수 있습니다. 
- 기존 devset에서 잘못된 라벨들을 수정하고 분포에 맞춰 데이터를 변경하였습니다.
- 피어슨 상관계수를 비롯해 confusion_matrix, f1score, mse, 구간별 correlation, acc를 한번에 확인할 수 있습니다.
```python
%run evaluate_exp.py
```

```python
# 학습완료된 model 불러오기 (cross-encoder)
model_save_path = ' '
model_evaluation(model_save_path ,
                 encoding ='cross-encoding')

# bi-encoder
model_save_path = ' '
model_evaluation(model_save_path ,
                 encoding ='bi-encoding')
```
<br/>

## output example
<img width="674" alt="image" src="https://user-images.githubusercontent.com/100064247/209533964-e17a150f-556a-461c-ab3d-e6191b66240c.png">
