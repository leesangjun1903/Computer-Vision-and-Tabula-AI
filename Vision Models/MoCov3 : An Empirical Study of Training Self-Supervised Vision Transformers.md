# MoCov3 : An Empirical Study of Training Self-Supervised Vision Transformers | Image classification

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
이 논문은 기존의 합성곱 신경망(ConvNet)과 달리 Vision Transformer(ViT)를 자기지도(self-supervised) 학습으로 안정적으로 훈련하기 위한 **필수 조건**과 **불안정성(Instability)** 문제를 규명하고, 이를 완화시키는 **간단한 기법**을 제안한다.  

**주요 기여**  
- ViT 기반 대조학습(contrastive learning) 프레임워크인 **MoCo v3** 제안  
- 배치 크기, 학습률, 옵티마이저 등의 기본 하이퍼파라미터가 ViT 훈련 안정성에 미치는 영향을 체계적 분석  
- **패치 임베딩 층 고정(random patch projection)** 기법을 통해 훈련 불안정성을 완화하고 성능을 1–3% 향상  
- 대형 ViT 모델(Large, Huge) 규모까지 확장하여 자기지도 ViT가 모델 규모 확장에 따른 성능 이점을 확보함을 실증  

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- ViT는 대규모 데이터와 컴퓨팅 파워를 필요로 하는 반면, 자기지도 학습 시에는 **훈련 불안정성**이 발생  
- 불안정성은 발산(divergence)이 아니라, **훈련 곡선 상의 미세한 ‘딥(dip)’ 현상**으로 은폐되어 검증이 어려움  
- 이로 인해 하이퍼파라미터 탐색이나 아키텍처 설계 실험에서 **잠재적 성능 저하**를 초래  

### 2.2 제안 방법  
1. **MoCo v3**  
   - 두 개의 랜덤 크롭(x₁, x₂)을 인코더 fq, fk에 투입하여 쿼리 q와 키 k 생성  
   - 대조 손실(InfoNCE)  

$$  
       L_q = -\log \frac{\exp(q \cdot k^+ / τ)}{\exp(q \cdot k^+ / τ) + \sum_{k^-} \exp(q \cdot k^- / τ)}  
     $$  
   
   - 대칭화 손실: $$L = L_q(x₁,x₂) + L_q(x₂,x₁)$$  
   - 키 인코더 fk를 모멘텀 업데이트(m=0.99)  
2. **불안정성 완화 기법: 패치 임베딩 고정**  
   - ViT의 첫 번째 선형 투영(projection) 층을 무작위 고정(random weights)  
   - 해당 층의 그래디언트를 차단(stop-gradient)  
   - 얕은 층의 급격한 그래디언트 변화로 인한 딥(dip) 현상 완화  
3. **하이퍼파라미터 및 옵티마이저**  
   - 옵티마이저: AdamW (기본), LAMB 대안 검토  
   - 배치 크기: 1k–6k 실험, 4k 이상부터 딥 현상 관찰  
   - 학습률: 선형 스케일링, warmup 40 epochs, cosine decay  

### 2.3 모델 구조  
- **ViT-B/S/L/H**: Transformer 블록 12–32개, 채널 차원 384–1280  
- 패치 크기 16×16 (기본), 7×7 실험으로 시퀀스 길이 증가  
- 사인코사인 포지셔널 임베딩, 클래스 토큰 or 풀링 대체 가능  
- MLP 프로젝션 헤드(3-layer) + 예측 헤드(2-layer) with BatchNorm  

### 2.4 성능 향상  
- **패치 임베딩 고정** 적용 시  
  - MoCo v3 + ViT-B/16: linear probing 72.2% → 73.4% (+1.2%)  
  - SimCLR: 69.3% → 70.1% (+0.8%)  
  - BYOL: 69.7% → 71.0% (+1.3%)  
- **모델 규모 확장**  
  - ViT-B → ViT-L → ViT-H: 76.7% → 77.6% → 78.1% (linear probing)  
  - 작은 패치(7×7) 적용 시 ViT-BN-L/7: 81.0% (최고)  
- **대조 학습 프레임워크 비교**  
  - MoCo v3가 SimCLR, BYOL, SwAV 대비 ViT에 가장 우호적  
- **한계**  
  - 불안정성 해결은 임시방편이며, **근본 원인**(Transformer 전반 최적화 문제)은 미해결  
  - 대규모 데이터(21k, JFT-300M)에서의 자기지도 사전학습 효과 미검증  
  - 포지셔널 정보 활용이 부족하며, 모델이 객체 구조를 충분히 학습하지 못함  

***

## 3. 모델 일반화 성능 향상 관련 고찰  
- **전이 학습(fine-tuning)** 성능  
  - CIFAR-10/100, Flowers-102, Pets 데이터셋에서 ImageNet 자기지도 사전학습 후 fine-tune  
  - ViT-B→ViT-L 간 성능 향상: overfitting 감소 및 일반화 성능 개선  
  - 특히 소규모 데이터셋에서 ResNet 대비 큰 성능 격차 해소  
- **포지셔널 임베딩 제거 실험**  
  - sin-cos vs. learned vs. none: 76.5% → 76.1% → 74.9%  
  - 포지셔널 정보 없이도 높은 표현력 유지 → **강력한 일반화 잠재력**  
- **모델 규모와 overfitting**  
  - 감독 학습 시 ViT-L/H에서 overfitting 심화하나, 자기지도 사전학습으로 완화  
  - 더 많은 unlabeled 데이터 활용 시 **일반화 성능 지속 확장** 가능성  

***

## 4. 논문의 영향 및 향후 연구 고려 사항  
이 연구는 **Vision Transformer를 자기지도 학습** 관점에서 체계적으로 벤치마크하고, **훈련 안정성**이라는 핵심 난제를 부각시켰다.  
- **영향**:  
  - ViT 최적화 연구의 중요성 부각 → Transformer 최적화 이론 및 알고리즘 발전 촉진  
  - 자기지도 비전 모델 설계 지침 제공: 패치 임베딩, 모멘텀 인코더, 하이퍼파라미터  
- **향후 고려 사항**:  
  - **불안정성의 근본 원인** 해명 및 일반화된 해결책 개발  
  - 더 방대한, 다양한 unlabeled 데이터로 자기지도 ViT 확장  
  - **포지셔널 행사 학습** 기법 설계: 구조 정보 활용 강화  
  - Transformer 기반 비전 모델의 **규제 및 정규화** 메커니즘 연구  
  - 최적화 알고리즘(AdamW, LAMB 외) 및 **스케줄링** 전략 심층 탐구  

이 논문이 제공한 실험적 통찰은 ViT 기반 자기지도 학습의 **안정성**, **확장성**, **일반화** 측면 모두에 걸쳐 앞으로의 연구를 견인할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ae9c1847-222f-401c-bd66-e7b547a04c6a/2104.02057v4.pdf)
