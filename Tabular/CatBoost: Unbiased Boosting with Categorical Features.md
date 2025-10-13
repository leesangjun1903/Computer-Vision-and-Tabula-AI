# CatBoost: Unbiased Boosting with Categorical Features

**핵심 주장 및 주요 기여**  
CatBoost는 기존 그래디언트 부스팅 방식이 가진 **타깃 누수(target leakage)** 문제를 해결하기 위해 두 가지 핵심 기법을 제안한다. 첫째, **Ordered Boosting**을 도입하여 부스팅 단계에서 발생하는 예측 분포 편향(prediction shift)을 제거하고, 둘째, **Ordered Target Statistics**를 이용해 범주형 변수 처리 시 생기는 타깃 누수를 방지한다. 이 두 기법의 결합으로 CatBoost는 XGBoost, LightGBM 대비 다양한 데이터셋에서 일관된 성능 향상을 보인다.[1]

## 1. 해결하고자 하는 문제  
기존 그래디언트 부스팅은 학습 과정에서  
- 모델 업데이트 과정에서 현재 모델이 사용한 훈련 타깃을 다시 참조함으로써 발생하는 **예측 분포 편향(prediction shift)**  
- 범주별 통계(target statistics)를 산출할 때 동일 레코드의 타깃을 포함하여 계산하는 **타깃 누수**  
를 겪는다. 이로 인해 훈련 시와 테스트 시 모델이 보는 분포가 달라져 일반화 성능이 저하된다.[1]

## 2. 제안 방법

### 2.1 Ordered Target Statistics  
각 훈련 샘플 $$(x_k, y_k) $$에 대해 무작위 순열 $$\pi$$를 정의하고, 해당 샘플의 범주형 값에 대한 타깃 통계를  

$$
\mathrm{TS}(x_k)
= \frac{\sum_{j: \pi(j) < \pi(k)} \mathbb{I}[x_j = x_k] \, y_j + a \, p}{\sum_{j: \pi(j) < \pi(k)} \mathbb{I}[x_j = x_k] + a}
$$

로 계산한다. 여기서 $$p$$는 전체 데이터의 평균 타깃, $$a$$는 스무딩 파라미터이다. 이 방식은 훈련 샘플의 자기 자신을 제외하고 이전 순서의 샘플만 사용하므로 타깃 누수를 제거한다.[1]

### 2.2 Ordered Boosting  
부스팅 매 스텝에서 잔차(residual)를 계산할 때, 각 샘플을 제외하고 학습된 **supporting model**을 이용한다. 무작위 순열 $$\pi$$에 따라  
- 샘플 $$i$$의 잔차 계산에는 순서상 이전 $$i-1$$개의 샘플로 학습된 모델 $$M_{i-1}$$ 사용  
- 전체 훈련 샘플에 대해 $$n$$개의 supporting model을 유지  
를 기본 개념으로 한다. 이론적으로는 $$n$$개의 모델을 학습해야 하지만, 실용화를 위해 로그 단위로 축소하여 유지하며 계산 복잡도를 표준 GBDT와 동일한 $$O(n)$$ 수준으로 유지한다.[1]

## 3. 모델 구조  
- **Base Learner**: 균형 잡힌 *Oblivious Decision Tree*(모든 레벨에서 동일 분할 기준)  
- **Boosting Modes**:  
  - *Plain*: 기존 GBDT에 Ordered TS만 도입  
  - *Ordered*: Ordered Boosting + Ordered TS  
- **범주형 조합(feature combinations)**: Greedy 방식으로 두 개 이상의 범주형 피처 조합하여 높은 차수 의존성 포착.

## 4. 성능 향상 및 한계  

### 4.1 성능 향상  
CatBoost는 XGBoost, LightGBM 대비  
- 성능 지표(logloss, zero-one loss)에서 모든 벤치마크 데이터셋에서 우수한 성능 달성  
- 특히 소규모 데이터셋에서 Ordered 모드가 Plain 모드 대비 더 큰 이득을 보이며 일반화 성능이 개선됨  
- 범주형 처리를 위한 Ordered TS가 Holdout, Leave-one-out 방식 대비 가장 높은 성능 제공

### 4.2 한계 및 고려 사항  
- **계산 복잡도**: Ordered Boosting 지원 모델 수 증가로 Plain 모드 대비 약 1.7배 느린 학습 속도 발생  
- **분산 증가**: 초기 순열 단계에서 샘플이 적은 경우 TS와 예측 분산이 커질 수 있어 여러 순열 사용 필요  
- **메모리 요구**: supporting model 유지로 메모리 소비 증가  

## 5. 일반화 성능 향상 가능성  
Ordered Boosting과 Ordered TS는 **훈련-테스트 분포 차이를 최소화**하여 과적합 위험을 줄인다. 특히 소규모 데이터셋과 고카디널리티 범주형 피처 환경에서 일반화 성능이 크게 개선될 수 있다.

## 6. 향후 연구 영향 및 고려할 점  
- **다른 모델과의 결합**: Transformer, 딥러닝 모델 등과 부스팅 기법 하이브리드 연구 가능  
- **분산 학습 최적화**: supporting model 유지 비용 절감을 위한 효율적 분산/병렬 학습 알고리즘 개발  
- **자동 하이퍼파라미터 최적화**: 순열 수(s), 조합 차수(cmax) 등 민감도 분석 및 자동화  
- **타깃 누수 제거 일반화**: Ordered 원칙을 다른 학습 알고리즘의 타깃 통계, 교차 검증 단계에 적용  

CatBoost의 Ordered Boosting과 Ordered TS는 부스팅 계열 모델의 **타깃 누수 문제**를 근본적으로 해결하며, 범주형 피처 처리 및 일반화 성능 개선에 있어 중요한 연구 방향으로 자리잡았다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/aea9e55a-97d7-4122-8325-dc3e0930fdd9/1706.09516v5.pdf)
