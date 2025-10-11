# Privacy-Preserving Gradient Boosting Decision Trees

**핵심 주장:** 기존 GBDT에 차등 프라이버시를 적용할 때 민감도(sensitivity) 추정치가 느슨하고, 트리별 privacy budget 배분이 비효율적이어서 정확도 손실이 크다.  
**주요 기여:**  
- **Gradient-based Data Filtering (GDF):** 매 반복마다 그라디언트 크기가 기준값을 넘는 소수의 인스턴스만 필터링하여 민감도 상한을 엄밀히 Tighten.  
- **Geometric Leaf Clipping (GLC):** 트리 순서에 따라 기하급수적으로 감소하는 잎 노드 값에 클리핑을 적용해 민감도를 점진적 으로 축소.  
- **Ensemble of Ensembles (EoE):** 각 앙상블 내 병렬 구성, 앙상블 간 순차적 구성으로 privacy budget을 효과적으로 재분배.  
- 제안 기법을 적용한 **DPBoost**는 기존 차등 프라이버시 GBDT 대비 **정확도 손실을 대폭 저감**하고, 순수 GBDT과 유사한 모델 성능을 달성함.[1]

***

# 상세 설명

## 1. 해결하고자 하는 문제  
GBDT는 여러 결정 트리를 순차적(boosting)으로 학습하며 회귀·분류에서 우수한 성능을 보이나, 학습 데이터의 개별 레코드를 보호하려면 **차등 프라이버시(Differential Privacy, DP)** 기법을 도입해야 한다.  
- **민감도(Sensitivity):** 함수 출력 변화량 상한. 기존 GBDT DP 기법은 gain 함수나 리프 값 범위를 추정해 느슨한 상한을 도출, 노이즈 규모가 과도하여 정확도 하락.  
- **Privacy Budget Allocation:** 트리별 budget을 균등 분배하거나 입력 분할 방식을 사용할 경우, 트리 수 증가 시 budget이 작아지거나 각 트리 학습 데이터가 적어져 성능 저하.  

## 2. 제안하는 방법 및 수식  

### 2.1 Gradient-based Data Filtering (GDF)  
- **민감도 이론:** split gain $$G$$와 leaf value $$V$$의 정확한 상한식 유도:  

$$
    G \le \frac{3}{2}g_{\ell}^2,\quad
    V \le g_{\ell}
  $$
  
여기서 $$g_{\ell}$$은 손실 함수 $$\ell$$의 최대 1-노름 그라디언트 상한.[1]
- **데이터 필터링:** 매 반복 t마다 그라디언트 $$\|g_i\|\_1 > g_{\ell}$$ 인 소수의 인스턴스만 제외하고 학습하여 실제 사용 데이터의 그라디언트를 $$g_{\ell}$$ 이하로 제한.  

### 2.2 Geometric Leaf Clipping (GLC)  
- **기하급수적 감소 모형:** 단일 인스턴스 리프 값 $$V_t$$은 shrinkage rate $$\eta$$ 적용 시  

$$
    V_t \le g_{\ell}(1-\eta)^{t-1}.
  $$

- **클리핑 적용:** t번째 트리의 리프 값을 $$g_{\ell}(1-\eta)^{t-1}$$로 상한 처리 후 Laplace 노이즈 추가해 민감도 점진적 감소 도모.[1]

### 2.3 Ensemble of Ensembles (EoE)  
- **내부 병렬 구성:** 앙상블 내 각 트리는 서로 다른 데이터 샘플 사용 → Parallel Composition 적용.  
- **앙상블 간 순차 구성:** 여러 앙상블을 sequential로 학습 → Sequential Composition 적용.  
- **Budget 배분:** 전체 트리 수 $$T$$, 앙상블당 트리 수 $$T_e$$일 때,  

$$
    N_e = \frac{T}{T_e},\quad \text{각 트리 budget} = \frac{\varepsilon}{N_e}.
  $$

## 3. 모델 구조  
DPBoost는 LightGBM 기반 GBDT 학습 루프에 GDF·GLC 삽입, EoE 프레임워크로 privacy budget 관리 기능을 통합한 확장 GBDT 시스템이다.

## 4. 성능 향상 및 한계  
- **정확도 향상:** DPBoost는 기존 DP-GBDT 대비 분류 오류율 10%p 이상 감소, RMSE 대폭 저감.[1]
- **안정성:** 예측 분산이 작아 일관된 성능 보장.  
- **한계:**  
  - 트리 차원(피처 수)이 매우 많으면 노이즈 계산(Exponential Mechanism) 오버헤드 발생.  
  - GDF에서 필터링된 인스턴스는 후속 트리에만 반영 → 특정 도메인에서 outlier 편향 이슈 가능.  

***

# 일반화 성능 향상 가능성에 대한 고찰

DPBoost는 그라디언트 필터링과 기하 클리핑을 통해 모델 학습 단계에서 개별 인스턴스 영향력을 엄격히 제어함으로써, 과적합 위험을 줄이며 **일반화 성능**을 보장한다.  
- **GDF**는 극단치(outlier)에 의한 노이즈 과다 유발을 방지해 모델이 중요한 데이터 분포를 학습하도록 지원.  
- **GLC**는 뒤쪽 트리의 기여도를 서서히 감소시켜 초기 트리에서 주 구조를 학습하게 하고, 후속 트리에서는 미세 조정 수준으로 학습해 과도한 세부적 분할을 억제.  
이로써 DPBoost는 DP 기법 적용 시에도 일반 GBDT와 유사한 과적합 제어 성능을 유지한다.

***

# 향후 연구 영향 및 고려사항

Privacy-Preserving GBDT 분야에서 DPBoost는 **세 가지 축**에서 후속 연구에 기여할 수 있다.  
1. **Federated Learning 통합:** 각 클라이언트 로컬 GDF·GLC 적용 후 중앙 서버 EoE 구성으로 확장 가능.  
2. **Adaptive Budgeting 개선:** 트리별 기여도(예: 정보 이득)에 따라 동적 budget 재분배 알고리즘 연구.  
3. **Outlier 처리 최적화:** GDF 필터링 시 군집 기반 선택이나 샘플 가중치 부여로 편향 최소화.  

향후 연구에서는 **클리핑 기준값 $$(g_{\ell},\eta)$$** 자동 튜닝, 고차원·희소 피처에서의 **노이즈 효율화**, 그리고 **실시간 예측 시스템**에의 적용 가능성 검증을 고려해야 한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cd9dd97d-dbfe-49d3-8160-39f01de883c7/5422-Article-Text-8647-1-10-20200511.pdf)
