# VERY_NALEE_s / Bigleader9
## Kaggle - Fashion Recommendation
- H&M Personalized Fashion Recommendation



## 1. 접근법

1. 비슷한 이미지의 상품을 추천(CNN) or 아이템끼리 clustering이 가능할까...?
2. 유저 클러스터링을 하면 가능할지도...?
   1. ~~구매 내역 데이터는 결국 한 줄 한줄이 구매에 대한 시계열 예측(LSTM?)~~
3. 빈도수 및 유사도 분석(most often purchased and similarity)
4. 추천시스템 ~ 추천 알고리즘(Recommendation algorithm)
   1. Collaborative Filtering : 협업 필터링
   2. Content-based Recommender Systems: 컨텐츠 기반 추천 시스템
      1. 과거에 경험했던 아이템 중 비슷한 아이템을 현재 시점에서 추천
      2. 유저의 성향을 배우는 문제
   3. Knowledge-based systems : 지식 기반 추천 시스템



### 1.1) Find Each Customer's Last Week of Purchased

(1) Recommend Most often previously purchased Items

(2) Recommend Items Purchased Together

> cosine similarity

(3) Recommend Last Week's Most Popular Items





# **<a id='content' style="color:#023e8a;"> Table of Content </a>**

### [**<span style="color:#023e8a"> 1. EDA </span>**](#EDA)
* [**<span style="color:#023e8a;">1. First steps</span>**](#First)  
* [**<span style="color:#023e8a;">2. Articles</span>**](#Articles)  
* [**<span style="color:#023e8a;">3. Customers</span>**](#Customers)  
* [**<span style="color:#023e8a;">4. Transactions</span>**](#Transactions)  
* [**<span style="color:#023e8a;">5. Images with description and price</span>**](#Images)  

### [**<span style="color:#023e8a"> 2. Preprocessing </span>**](#Preprocessing)
* [**<span style="color:#023e8a;">1. First steps</span>**](#First)  
* [**<span style="color:#023e8a;">2. Articles</span>**](#Articles)  

### [**<span style="color:#023e8a"> 3. Modeling  </span>**](#EDA)
* [**<span style="color:#023e8a;">1. Ensemble </span>**](#First)  
* [**<span style="color:#023e8a;">2. RandomForest </span>**](#Articles)  
* [**<span style="color:#023e8a;">3. XGBoost </span>**](#Articles)  
* [**<span style="color:#023e8a;">4. lightGBM </span>**](#Articles)  
* [**<span style="color:#023e8a;">5. CatBoost </span>**](#Articles)  


### [**<span style="color:#023e8a"> 4. Evaluation  </span>**](#Evaluation)
* [**<span style="color:#023e8a;">1. Confusion matrix </span>**](#ConfusionMatrix)  





cosine similarity

```python
import numpy as np

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```



