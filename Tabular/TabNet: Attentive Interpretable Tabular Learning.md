# TabNet: Attentive Interpretable Tabular Learning

## 주요 주장 및 기여  
TabNet은 **순차적 주의(attention) 메커니즘**을 통해 각 결정 단계(step)에서 인스턴스별로 중요한 피처를 선택하여 학습 효율과 해석 가능성을 동시에 확보한 최초의 딥러닝 아키텍처이다.  
1. **원시 테이블(raw tabular) 데이터**를 전처리 없이 입력으로 사용하며, 엔드투엔드(end-to-end) 학습이 가능하다.  
2. 인스턴스별 **소프트(feature-wise) 선택 마스크**를 학습하여, 모델 용량을 가장 중요한 피처에 집중시켜 **효율적인 학습**과 **해석 가능성**을 제공한다.  
3. 다양한 분류·회귀 벤치마크에서 **트리 기반 모델 및 기존 DNN을 능가**하며, 로컬(local)·글로벌(global) 특성 기여도를 모두 산출한다.  
4. **마스킹(masked) 기반 자기지도 학습(self-supervised pre-training)**을 도입하여, 라벨이 부족한 상황에서 성능과 수렴 속도를 크게 향상시킨다.  

## 문제 정의  
- 기존 DNN은 테이블 데이터의 고차원·희소성·이질적(수치·범주형) 특성에 대한 적절한 귀납적 편향(inductive bias)이 부족해 과적합 및 학습 실패가 잦다.  
- 트리 기반 모델은 전통적으로 강력한 성능과 해석성을 제공하나, 엔드투엔드 표현 학습이나 반지도 학습을 지원하지 못한다.

## 제안 방법  
### 1) 순차적 결정 블록(Decision Step)  
각 단계 $$i$$에서  
1. **주의 변환기(Attentive Transformer)**를 통해 입력 특성 $$f\in \mathbb{R}^{B\times D}$$에 소프트마스크 $$M_i\in \mathbb{R}^{B\times D}$$를 산출:  

$$
     a_i = h_i(a_{i-1}), \quad M_i = \text{sparsemax}(P_{i-1} \odot a_i)
   $$  
   
   여기서 $$P_{i-1}$$은 이전 단계에서 사용된 특성의 스케일 정보, $$\odot$$는 원소별 곱, sparsemax는 희소성 보장(normalization) 함수이다.  
2. **마스크 적용** 후 GLU 기반 **피처 변환기(Feature Transformer)**를 통과시켜,  

$$
     d_i,\, a_i = \text{FeatureTransformer}(M_i \odot f)
   $$  

3. **출력 집계**: 모든 단계의 출력 $$d_i$$를 ReLU 후 선형 결합하여 최종 예측  

$$
     d_{\text{out}} = \sum_{i=1}^{N_{\text{steps}}} \mathrm{ReLU}(d_i), \quad y = W_{\text{final}} d_{\text{out}}.
   $$

### 2) 희소성 및 해석 가능성  
- **엔트로피 기반 희소성 정규화** $$\mathcal{L}_{\text{sparse}} = \lambda_{\text{sparse}}\sum_{i,b,j} M_{b, j}^i\log M_{b, j}^i$$ 를 손실에 추가하여 원하는 희소도 조절.  
- **글로벌 특성 중요도**: 단계별 마스크와 출력세기 $$b_i=\sum_c\mathrm{ReLU}(d_{b,c}^i)$$를 가중합하여  

$$
    M^{\mathrm{agg}}_{b,j}=\sum_{i=1}^{N_{\text{steps}}}b_i\,M_{b,j}^i\,.
  $$

### 3) 자기지도 사전학습  
- 특성 마스크 $$S$$를 랜덤 샘플링하여 **마스킹된 입력** $$ (1-S)\odot f$$ 으로 TabNet 인코더를 학습, 디코더로 마스킹된 원본 $$S\odot f$$ 복원  
- 손실:  

$$
    \mathcal{L}_{\text{recon}}=\sum_{b,j}\frac{(f_{b,j} - \hat{f}_{b,j})^2}{\sigma_j^2}.
  $$

## 모델 구조  
- **인코더**: GLU+배치 정규화(BN) 기반 피처 변환기와, 내부 단계 공유(shared)·단계별(step-dependent) 레이어 결합  
- **디코더**: 유사 구조로 마스킹된 피처 복원  
- **하이퍼파라미터**: 단계 수, 희소 정규화 계수, GLU 레이어 크기 등으로 다양한 데이터셋에 대해 민감도 낮음.[1]

## 성능 향상 및 한계  
- **합성 데이터**(Syn1–Syn6): TabNet은 인스턴스 의존적 특성 선택에서 INVASE와 동등, 글로벌 의존성 데이터에서 기존 방법 상회.[1]
- **실제 데이터**: Forest Cover, Poker Hand, Sarcos 로봇 동역학, Higgs Boson, Rossmann 매출 예측 등 주요 벤치마크에서 XGBoost·LightGBM·CatBoost 및 기존 DNN을 모두 능가.[1]
- **자기지도 효과**: Higgs 데이터에서 10만개 미만 라벨 시, 사전학습으로 수렴 속도 및 정확도 유의미 개선.[1]
- **제한점**: 대용량 희소 벡터나 고차원 희소 카테고리 임베딩에서는 메모리 비용 증가 가능. 희소성 과도 시 정보 손실 우려.  

## 일반화 성능 향상 관점  
- **인스턴스별 희소 선택**은 과적합 방지 및 모델 용량의 집중 할당으로 **소규모 데이터**에서도 강건한 일반화를 유도한다.  
- **공유/단계별 레이어 분리**는 파라미터 효율성을 높이고, 다양한 데이터 분포에 적응 가능성을 강화한다.  
- **자기지도 사전학습**은 도메인 적응 및 라벨 부족 상황에서 표현 학습을 통해 **도메인 일반화**에 기여한다.  

## 향후 연구 영향 및 고려사항  
- **더 복잡한 자기지도 과제**(예: 구조적 마스킹, 컨트라스티브 학습) 도입으로 표현 학습 강화 연구  
- **하이퍼파라미터 자동화**: 단계 수·희소 계수 자동 최적화로 적용 편의성 제고  
- **대규모 카테고리·희소성**: 효율적 임베딩·희소성 조절 메커니즘 연구  
- **공정성·안정성 분석**: 특성 선택 마스크 기반의 편향 진단 및 안정성 검증  
- **멀티모달 통합**: 이미지·텍스트 등 다른 모달리티와 테이블 데이터 동시 학습 가능성 탐색  

TabNet은 테이블 데이터 딥러닝의 새로운 표준을 제시하며, 특히 **효율성·해석가능성·자기지도 학습**을 결합한 점이 향후 다양한 도메인 연구에 중대한 영감을 줄 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7143980c-f8c5-4c86-9f54-384698063e03/16826-Article-Text-20320-1-2-20210518.pdf)

<details>

# TabNet: Attentive Interpretable Tabular Learning

## 1. 핵심 주장 및 주요 기여  
TabNet은 **탐색적 특징 선택(attentive feature selection)** 메커니즘을 통해 입력된 원시 Tabular 데이터를 단계별로 중요한 특징만을 선택·처리함으로써,  
-  고성능 예측  
-  인스턴스별 및 전역적 해석 가능성  
-  적은 파라미터로 효율적 학습  
을 동시에 달성하는 새로운 DNN 아키텍처를 제안한다.

## 2. 문제 정의·제안 방법·모델 구조·성능·한계

### 2.1 해결하고자 하는 문제  
- 기존 딥러닝 모델은 과도한 파라미터로 탭 데이터의 희소·비선형 구조를 제대로 학습하지 못함.  
- 트리 기반 앙상블 모델은 해석성은 높으나, 엔드투엔드 학습과 표현 학습 한계.

### 2.2 제안 방법  
1) **Sequential Attention & Sparse Feature Selection**  
   - 각 결정 단계 $$i$$마다 입력 피처 벡터 $$f\in\mathbb{R}^{B\times D}$$에 대해 주어진 prior scale $$P^{(i-1)}$$와 처리 정보 $$h(a^{(i-1)})$$를 통해 마스크 $$M^{(i)}$$를 계산:  

$$
       M^{(i)} = \mathop{\mathrm{sparsemax}}\bigl(P^{(i-1)} \cdot h(a^{(i-1)})\bigr),\quad \sum_j M^{(i)}_{b,j}=1
     $$
   
   - prior scale 업데이트:  

$$
       P^{(i)} = P^{(i-1)} \odot (\gamma - M^{(i)}),\quad P^{(0)}=\mathbf{1}
     $$
   
   - 희소성 유도를 위한 엔트로피 정규화:  

```math
       L_{\mathrm{sparse}} = \frac{1}{N_{\mathrm{steps}}B}\sum_{i,b,j}-M^{(i)}_{b,j}\log(M^{(i)}_{b,j}+\epsilon)
```

2) **Feature Transformer & Decision Aggregation**  
   - 선택된 피처 $$M^{(i)}\odot f$$를 GLU 기반의 feature transformer $$f^{(i)}$$로 처리하여 결정 임베딩 $$d^{(i)}$$와 다음 단계 정보 $$a^{(i)}$$ 생성.  
   - 최종 결정 임베딩:  

$$\displaystyle d_{\mathrm{out}} = \sum_{i=1}^{N_{\mathrm{steps}}}\mathrm{ReLU}(d^{(i)})$$ → 선형 출력.

3) **Self-Supervised Pre-training**  
   - 입력 피처의 일부를 마스킹하고, TabNet decoder를 통해 복원 학습으로 표현력 강화.

### 2.3 모델 구조  
- **Encoder**:  
  - Nsteps 단계 반복  
  - 각 단계: attentive transformer → sparsemax mask → feature transformer (공유 층 + 단계별 층)  
- **Decoder** (자기지도 학습):  
  - 인코더 출력으로 마스킹된 피처 복원

### 2.4 성능 향상  
- 합성 데이터에서 인스턴스별 선택 과제 성능 동급 또는 상회  
- Forest Cover Type 96.99% vs. LightGBM 89.28%, AutoML Tables 94.95%  
- Poker Hand 99.2% (규칙 기반 100%, 타 DNN 65~71%)  
- Sarcos, Higgs, Rossmann 등 다수 탭 데이터 벤치마크에서 최고 또는 동등 성능 달성  
- 자기지도 사전학습으로 소량 라벨링 시 데이터 효율성·수렴 속도 크게 개선  

### 2.5 한계  
- **하이퍼파라미터 민감도**: Nsteps, γ, λsparse 등 설정에 따라 성능 편차 발생  
- **대규모 모델 크기**: 최적 성능 달성 위해 다단계·다층 GLU 블록 → 연산·메모리 비용  
- **희소성 정규화 trade-off**: 과도한 희소성은 학습 저해 가능

## 3. 일반화 성능 향상 관점  
- **인스턴스별 희소 선택**은 모델이 불필요한 피처에 과적합되지 않게 유도하여 일반화 향상에 기여.  
- **large-batch training + ghost BN**으로 안정적 수렴, 규칙적 배치 노이즈 억제.  
- **사전학습→파인튜닝** 프로토콜로 도메인 적응과 소규모 데이터셋에서도 과적합 최소화.

## 4. 미래 연구 영향 및 고려 사항  
- TabNet의 **단계적 집중(attention) + 희소성** 메커니즘은 다른 도메인(시계열·그래프) 모델에 확장 가능  
- 하이퍼파라미터 자동 최적화(AutoML) 및 **경량화 구조 탐색(Neural Architecture Search)** 결합 필요  
- **희소성 정도 제어**를 위한 적응형 스케줄링 연구로 일반화 성능 추가 향상 여지  
- 자기지도 학습 방식 다양화(Contrastive, Masked Modeling)로 표현력 강화 연구 권장

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9e1762f7-5631-4cdd-a851-bfd235b86d4b/1908.07442v5.pdf)

</details>
