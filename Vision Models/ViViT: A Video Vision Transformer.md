# ViViT: A Video Vision Transformer | Video classification, Action recognition

**주요 주장:**  
비디오 분류에 순수(transformer-only) 비전 트랜스포머를 도입하여, 스페이셜–템포럴 토큰화 및 효율적 어텐션 분해(factorisation)를 통해 3D CNN 기반 SOTA 모델을 능가할 수 있음을 보인다.

**주요 기여:**
- 비디오 클립을 3차원 “튜브릿(tubelet)” 또는 균등 프레임 샘플링으로 토큰화하는 두 가지 전략 제시.  
- Transformer 블록 내·외부에서 스페이셜·템포럴 처리를 분해하는 네 가지 아키텍처 설계.  
- 이미지 사전학습(ViT) 가중치의 재활용 및 “중앙 프레임 초기화”로 대규모 비디오 데이터 부족 문제 완화.  
- Epic Kitchens, Kinetics, Something-Something v2 등 5개 벤치마크에서 기존 3D CNN 계열 대비 최대 +5%p 이상의 Top-1 정확도 향상.

***

## 1. 문제 정의 및 제안 방법

### 1.1 해결 과제  
- 3D CNN은 지역적 수용 영역(receptive field) 한계 및 막대한 연산량  
- Transformer는 긴 시퀀스 처리에 유리하나, 비디오 토큰 수가 폭증하며 데이터가 부족할 경우 과적합 위험  

### 1.2 비디오 토큰화 전략  
1) **Uniform Frame Sampling**  
   - $$nt$$ 프레임을 균등 추출 → 각 프레임을 ViT 패치(크기 $$h\times w$$)로 토큰화  
   - 총 토큰 수: $$N = nt \times nh \times nw$$  
2) **Tubelet Embedding**  
   - 3D 컨볼루션처럼 $$t\times h\times w$$ 튜브릿 단위로 직접 임베딩  
   - $$nt = \lfloor T/t\rfloor,\; nh = \lfloor H/h\rfloor,\; nw = \lfloor W/w\rfloor$$  

### 1.3 Transformer 모델 구조  
- 입력 토큰 $$z_0\in\mathbb R^{N\times d}$$에 위치 임베딩 추가  
- 네 가지 아키텍처:
1. **Spatio-Temporal Attention**  
     - 모든 토큰 사이의 전역 어텐션: $$O(N^2)$$  
2. **Factorised Encoder**  
     - 시공간 분리 인코더: 먼저 공간(transformer on $$nh\times nw$$), 이후 시간(transformer on $$nt$$)  
     - FLOPs: $$O((nh\,nw)^2 + nt^2)$$  
3. **Factorised Self-Attention**  
     - 각 블록 내에서 공간→시간 어텐션 분해  

$$
       y_s = \mathrm{MSA}(\mathrm{LN}(z)),\quad
       y_t = \mathrm{MSA}(\mathrm{LN}(y_s))
     $$  
  
4. **Factorised Dot-Product Attention**  
     - 헤드 절반은 공간, 절반은 시간에만 어텐션  

$$
       Y_s = \mathrm{Attention}(Q, K_s, V_s),\quad
       Y_t = \mathrm{Attention}(Q, K_t, V_t)
     $$  

### 1.4 사전학습 및 초기화  
- 이미지 ViT 가중치(ImageNet-21K, JFT) 활용  
- 위치 임베딩은 시간 차원 반복 복제  
- 튜브릿 필터는 “중앙 프레임(initialise only center slice)” 방식으로 초기화하여 시간 정보 학습 유도  

***

## 2. 성능 향상 및 한계

### 2.1 벤치마크 성과  
- **Kinetics-400:** ViViT-L/16×2 FE 단일 뷰 80.6% → 3뷰 81.7% (기존 SOTA 약 80.4%)  
- **Kinetics-600:** 단일 뷰 82.9% → JFT 사전학습 시 84.3%  
- **Epic Kitchens:** Action Top-1 44.0% (기존 대비 +5.5%p)  
- **SSv2:** 65.9% (순수Transformer 중 최고)  

### 2.2 일반화 성능 향상  
- **데이터 부족 대응:** ViT 사전학습 + 튜브릿 중앙 초기화 + 반복적 위치 임베딩  
- **과적합 방지:** Epic Kitchens, SSv2에 대해 stochastic depth, RandAugment, label smoothing, mixup 결합 → +5%p 향상  
- **토큰 분해 구조:** Factorised Encoder가 소규모 데이터셋에서 unfactorised 대비 과적합을 덜 겪으며 더 안정적  

### 2.3 한계  
- **장시간 맥락 포착:** 3D CNN 대비 더 많은 메모리 요구  
- **미세 모션 인식:** SSv2에서 3D CNN 모델 대비 상대적 개선 폭이 작아, 극미세 모션 캡처에는 한계  
- **사전학습 의존:** 대규모 이미지 데이터에 크게 의존하며, 영상 고유 특성만으로 학습한 사례 부족  

***

## 3. 미래 연구에 대한 영향 및 고려사항

- **Transformer 전이학습:** 비디오 작업 전용 대규모 비디오 사전학습 데이터 필요성 대두  
- **효율적 어텐션 구조:** 더욱 경량화된 분해 어텐션·스파스 어텐션 연구 가치  
- **미세 모션 강화:** SSv2와 같은 세밀한 움직임 인식용 토큰화·어텐션 커널 개발  
- **다중 모달 연계:** 오디오·자막 등 부가 정보 통합을 통한 일반화 성능 추가 향상 가능  

비디오 비전 트랜스포머 연구의 토대를 마련한 본 논문은 향후 영상언더스탠딩 분야에서 Transformer 아키텍처를 더욱 확장·최적화하는 계기를 제공할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5d252289-c2ce-4a58-9f5b-281415708642/2103.15691v2.pdf)
