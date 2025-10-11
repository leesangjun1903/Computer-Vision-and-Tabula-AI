# SPOT : Decision Trees under the Predict-then-Optimize Framework

**핵심 주장 및 기여 요약**  
이 논문은 전통적으로 예측 오차(예: MSE)를 최소화하도록 학습된 모델이 downstream 최적화 의사결정에서 최적의 결정을 보장하지 못함을 지적하고, 예측 단계와 최적화 단계를 결합하여 **결정(suboptimality)을 직접 최소화**하는 새로운 손실함수인 **SPO(Smart Predict-then-Optimize) 손실**을 의사결정 트리에 적용하는 **SPOT(SPO Trees)** 방법론을 제안한다.  
주요 기여는 다음과 같다.  
1. **SPO 손실을 트리 학습에 직접 적용**할 수 있는 구조적 성질(각 리프(leaf)에서 비용벡터 평균이 최적 해를 보장)을 활용한 알고리즘 제안.  
2. 기존의 CART 기반 greedy 분할과 최근의 MILP 기반 최적 트리 학습 방식을 **SPO 손실**으로 확장한 두 가지 학습 전략(재귀 분할 및 정수계획법)을 제시.  
3. 해석 가능성 유지하면서도 일반 CART보다 **더 적은 복잡도(리프 수·깊이)**로 **상위 의사결정 품질**(낮은 평균 초과 비용·높은 클릭률)을 달성함을 실험으로 입증.  

***

## 1. 해결하고자 하는 문제  
- 많은 실제 의사결정 문제가 “ $$cᵀw 최소화$$ ” 형태의 최적화로 모델링되나, 비용벡터 c는 불확실하여 ML 모델로 예측한 후 최적화에 투입하는 **Predict-then-Optimize** 프로세스를 사용.  
- 그러나 ML 모델은 MSE 같은 전통적 예측 오차를 최소화하도록 학습되어, downstream에서의 **결정 오류(suboptimality)**를 반영하지 못함.  
- **문제 정의**  
  $$ \min_{w∈S} cᵀw $$  
  진짜 c 대신 예측 $$\hat c$$를 사용했을 때 발생하는 초과 비용을 직접 최소화하는 손실함수 설계 필요.

***

## 2. 제안하는 방법  

### 2.1. SPO 손실 정의  
- **SPO 손실**: 예측 $$\hat c$$가 유도하는 최적 결정 $$w(\hat c)$$이 진짜 비용 c에 대해 얼마나 초과 비용을 내는지 측정  

$$
    \mathrm{SPO}(\hat c, c)
    = \max_{w∈W(\hat c)}\,cᵀw - \underbrace{\min_{w∈S}cᵀw}_{z(c)}
  $$  
  
  여기서 $$W(\hat c)$$는 $$\hat c$$에 대한 최적 해 집합이다.  

### 2.2. 트리 학습 문제 재정의  
- 트리 학습을 “각 리프 ℓ에 속한 관측치 비용벡터 $$c_i$$의 평균 $$\bar c_ℓ$$로 예측했을 때의 SPO 손실 합을 최소화”로 재정의  

$$
    \min_{\text{tree splits}}
    \frac{1}{n}\sum_{ℓ}\sum_{i∈R_ℓ} \bigl(\bar c_ℓᵀw(\bar c_ℓ) - z(c_i)\bigr)
  $$  

- **Theorem 1**: 각 리프에서 비용벡터 평균이 SPO 손실을 최적화하므로, 리프 예측을 평균으로 고정해 분할 구조만 최적화하면 됨.  

### 2.3. 학습 알고리즘  
1. **재귀 분할(SPOT-greedy)**  
   - CART와 유사하게, 가능한 모든 (feature, threshold) 분할에 대해 리프별 SPO 손실을 계산하고 최소 손실 분할 선택.  
   - 리프 크기 및 깊이로 조기 종료 후 SPO 손실 기준 post-pruning 적용.  

2. **정수계획법(MILP-SPOT)**  
   - Theorem 1을 활용해 mixed-integer linear program으로 공식화  
   - 트리 구조·분할 변수·리프 예측을 MILP 제약으로 인코딩 후 Gurobi/CPLEX로 최적화  
   - 사전에 greedy 해로 warm-start, 시간 제한 내에서 근사해 개선 가능.  

3. **앙상블(SPO Forests)**  
   - SPOT 트리를 bootstrap+feature-bagging 방식으로 다수 생성 후 예측 $$\hat c$$ 평균에서 최적 결정 산출.

***

## 3. 모델 구조  
- **단일 SPOT 트리**: 이진 분할 트리  
  - 각 리프에 하나의 비용벡터 예측(평균)  
  - 분할 기준과 임계값이 SPO 손실 최소화 방향으로 결정  
- **SPOT Forest**: 랜덤 포레스트와 동일 구조  
  - B개의 SPOT 트리 앙상블, feature bagging 사용  
  - 리프 예측 평균 후 최적화

***

## 4. 성능 향상 및 한계  

### 4.1. 성능 향상  
- **짧은 트리(depth≤3)**로도 CART(depth≥4) 대비  
  - **Normalized extra travel time** 23–27% 감소[1]
- **뉴스 추천**에서 SPOT(depth=2) 가 CART(unrestricted) 대비  
  - **평균 클릭률** +0.17–4.3% 향상  
- **모델 복잡도**: SPOT 트리 리프 수가 CART의 절반 미만, 해석 가능성↑  

### 4.2. 한계  
- **MILP 계산 비용**: 대규모 데이터·깊은 트리에서 시간·메모리 부담  
- **비용 예측 연속성 가정**: 리프별 비용 평균이 유효하려면 비용 분포 연속성 필요, degenerate case에 소량 노이즈 추가 권장  
- **Surrogate vs. Direct SPO**: 기존 surrogate 손실에 비해 직접 SPO가 최적 보장하지만, nonconvex성·비차별성 관리 필요  
- **앙상블 과적합**: SPO Forest에서 과적합 관측, 개별 트리보다 성능 저하 사례 존재  

***

## 5. 일반화 성능 향상 가능성  
- El Balghiti et al.(2019)에서 SPO 손실 기반 학습의 **일반화 경계**(generalization bounds)를 제시하여, 충분한 데이터 및 적절한 정규화 하에 예측-결정 성능이 안정적으로 유지됨을 이론적으로 뒷받침.  
- **트리 정규화**(깊이 제한·pruning)와 **앙상블**(bagging)으로 분산 및 과적합 제어 가능.  

***

## 6. 향후 연구에 미치는 영향 및 고려사항  
- **확장성**: Gradient-based 모델(신경망)에 SPO 손실 직접 적용 연구  
- **비선형·비선형 제약**: 복잡한 제약 조건·비선형 최적화 문제로 확대  
- **온라인·분산 학습**: 대규모 스트리밍 데이터에서 SPOT 학습  
- **강인성**: 비용 예측 노이즈 및 예측-최적화 mismatch에 대한 안정성 분석  
- **해석 가능성 강화**: 의사결정 규칙 추출 및 시각화 기법 개발  

현실적 의사결정 문제에서 예측과 최적화를 통합하는 방향으로 연구가 진화할 것이며, SPOT은 해석 가능하면서도 결정 품질을 극대화하는 중요한 출발점을 제시한다.  

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f2f3ba4e-05d5-49e2-bd14-d591100db5a8/2003.00360v2.pdf)
