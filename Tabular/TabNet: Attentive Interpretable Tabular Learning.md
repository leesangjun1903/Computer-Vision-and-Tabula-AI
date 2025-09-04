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
