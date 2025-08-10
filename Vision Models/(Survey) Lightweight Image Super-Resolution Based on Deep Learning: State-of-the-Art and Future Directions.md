# Lightweight Image Super-Resolution Based on Deep Learning: State-of-the-Art and Future Directions

## 1. 핵심 주장 및 주요 기여
이 논문은 **경량화된(single-image) 초고해상도(single-image super-resolution, SISR) 모델**의 최신 연구 동향을 종합적으로 검토하고,  
– 네트워크 설계 관점(컨벌루션·잔차·밀집·지식 증류·어텐션·극경량 모델)  
– 손실 함수(𝓁₁/𝓁₂, 찰보니에, 대립 손실, TV 등)  
– 학습 프레임워크(PyTorch·TensorFlow 등)  
– 벤치마크 데이터셋(DIV2K, BSD, Urban100 등)  
네 가지 축으로 분류·비교하였다.  
주요 기여는 다음과 같다:  
- **경량 SISR 모델 분류체계** 제안: 네트워크 설계·손실·프레임워크·데이터셋 네 축으로 체계화  
- **여섯 가지 네트워크 디자인**(Convolution, Residual, Dense, Distillation, Attention, Extremely Lightweight) 상세 분석  
- **성능 지표**(PSNR/SSIM/IFC/LPIPS) 및 **경량화 경진대회**(AIM, NTIRE, Mobile AI) 비교  
- **향후 연구 방향** 제시: 비지역성(long-range) 정보, 실제 degraded 영상, 경량화 일반화, 하드웨어 특화 설계  

## 2. 문제 정의·제안 방법·모델 구조·성능 및 한계

### 2.1 해결하고자 하는 문제
- **목표**: 모바일·임베디드 환경에서 실시간으로 동작 가능한 초고해상도 네트워크  
- **제약**: 메모리·연산량·파라미터 수 제한  
- **도전과제**:  
  1. **네트워크 깊이 증가** 시 기울기 소실  
  2. **작은 필터**의 국소성으로 인한 장거리 의존성 미포착  
  3. **실제 저해상도 영상**의 다양하고 알 수 없는 열화(degradation)  

### 2.2 제안하는 방법
논문 자체는 단일 모델이 아니라 기존 방식을 네 가지 축으로 재분류.  
#### 2.2.1 네트워크 설계
- Convolution: SRCNN, FSRCNN 등 순차적 컨벌루션  
- Residual: ResNet 기반 잔차학습, 예) DRRN, LapSRN (ℓ₁/ℓ₂, Charbonnier 손실)  
- Dense: DenseNet 블록, 예) GLADSR, ESRN  
- Distillation  
  -  Feature distillation: IMDN, RFDN (채널 분할→정보 증류)  
  -  Model distillation: FSRCNN with Privileged Info, VDSR-Distill (teacher→student)  
- Attention:  
  -  Channel attention(SE), Spatial, Multi-scale, Pyramid, Transformer, Pixel  
- Extremely Lightweight: collapsible block, quantization-aware(8-bit)  

#### 2.2.2 수식
- **잔차 학습**: $$ \hat X = F(y; \theta) + y $$  
- **데이터 충실도 & 정규화**:  

$$
  \mathcal{J}(\hat X, k) = \| (X \otimes k)\downarrow_s - y \|^2 + \alpha \Psi(X, \theta),
  $$

$$
  y = (X\otimes k)\downarrow_s + n.
  $$

- **손실 함수 예시**  
  -  픽셀 𝓁₁: $$\mathcal{L}_{\ell1} = \frac1{HWC}\sum|X-\hat X|$$  
  -  Charbonnier: $$\sqrt{(X-\hat X)^2+\varepsilon^2}$$  
  -  GAN 대립 손실: $$\mathcal{L}_{G}=-\log D(\hat X)$$  

#### 2.2.3 모델 구조
- **IMDN**: 정보 다중 증류 블록(IMDB)→대조 인식 채널 어텐션→어댑티브 크롭  
- **RFDN**: 잔차 증류 연결(FDCB)+공간 어텐션  
- **ESRT**: 경량 CNN 백본 + 효율적 Transformer(EMHA)  
- **SESR**: collapsible linear block→NPU 최적화(2×,4× 실시간)

#### 2.2.4 성능 향상
- **PSNR**: 경량 모델들조차 4× Urban100에서 30 dB 전후 달성  
- **실시간**: 1080→4K 2× upscaling NPU 30 FPS 이상 구현  
- **경량화**: 파라미터 100K∼1M, 연산 6∼200 GFLOPs 범위

#### 2.2.5 한계
- **장거리 의존성 부족**: 대부분 컨벌루션 국소 수용역  
- **실제 영상 일반화**: 연구실 degraded 합성 LR에서 벗어나기 어려움  
- **하드웨어·메모리 제약**: Transformer 계열은 메모리 과다

## 3. 모델 일반화 성능 향상 가능성
1. **비지역성(Long-Range) 정보**: Pyramid/Transformer/Non-local 블록 강화  
2. **약한 감독·도메인 적응**: 무쌍(supervisionless)·도메인 갭 해소
3. **다양한 degradation 모델링**: 블러·노이즈·압축 혼합 현실열화 확대  
4. **지식 증류 일반화**: Teacher ensembled hybrid→학생 다양한 조건 적응

## 4. 향후 영향 및 고려사항
- **영향**: 경량화 SISR 표준 분류체계로 연구 방향 정립, 경량화 경진대회 설계 기준 제시  
- **고려사항**:  
  1. **실제 환경 편차**: 도메인 적응·무감독 학습  
  2. **하드웨어 특화**: NPU·모바일 컴파일러 최적화  
  3. **일반화 지표**: PSNR 외 참고적 지각 품질·추론 안정성 평가  
  4. **복합 증감학습**: 강화학습·메타러닝으로 경량 블록 자동 설계

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c1c77d88-6864-4425-9454-8c4e56434f2c/1-s2.0-S1566253523000386-main.pdf

# SISR 경량 모델 분류 체계(Section 4) 

경량(single-image) 초고해상도(super-resolution; SISR) 모델을 **“네트워크 설계”**, **“손실 함수”**, **“학습 프레임워크”**, **“훈련 데이터셋”** 네 가지 관점에서 분류한 것이 이 논문의 핵심 분류체계입니다.  

***

## 1. 네트워크 설계(Network Design)

가장 핵심적인 축으로, 경량 SISR 모델을 크게 아래 여섯 가지 구조 유형으로 나눕니다.

  -  **Convolution-Based**  
    -  순차적 컨벌루션 층만 쌓아 올린 구조  
    -  초기 SRCNN, FSRCNN, VDSR처럼 간단하지만 수용 영역이 작다는 단점  
    -  Sub-pixel Back-Projection, GhostSR 등으로 연산·파라미터 절감 시도  

  -  **Residual-Based**  
    -  입력과 출력의 차(잔차)를 학습하여 수렴 안정성 확보  
    -  ResNet 스타일의 전역·지역 잔차 학습 결합(예: DRRN, LapSRN)  
    -  Cascading, Back-Projection, Fractal, Multi-Receptive-Field 등 변형  

  -  **Dense-Based**  
    -  DenseNet 식으로 모든 이전 층의 출력을 현재 층으로 연결  
    -  Gradient 흐름·특징 재사용 강점  
    -  GLADSR, ESRN처럼 성장률(growth rate)·업샘플링 모듈 결합  

  -  **Distillation-Based**  
    -  **Feature Distillation**: 채널 분할→유용 특징만 다음 계층으로 전달(IMDN, RFDN)  
    -  **Model Distillation**: 대형 Teacher→경량 Student로 지식 이전(FSRCNN-PPI, VDSR-Distill)  

  -  **Attention-Based**  
    -  채널 간·공간적·다중 스케일·피라미드·Transformer·픽셀 단위 어텐션  
    -  SE 모듈, Spatial 어텐션, Multi-scale 어텐션, Pyramid Non-local, Self-Attention 등  
    -  ESRT, MCAN, PAN, LKASR, CFIN 등이 대표적  

  -  **Extremely Lightweight**  
    -  단순·얕은 구조 + 하드웨어(NPU·8-bit 양자화) 최적화  
    -  Collapsible Block(SESR), Channel Mixing(SplitSR, CDFM-Mobile), Quant-Aware(XLSR, ABPN)  

이 여섯 분류는 서로 완전히 배타적이라기보다, 주된 설계 기법에 따라 “주축” 하나로 구분한 것입니다.

***

## 2. 손실 함수(Loss Function)

필요한 목적에 따라 다음 네 가지 주요 계열로 구분됩니다.

  1. **픽셀 손실**(Pixel Loss)  
     -  𝓁₁(MAE)·𝓁₂(MSE)·Charbonnier  
  2. **대립 손실**(Adversarial Loss)  
     -  SRGANㆍPatchGAN으로 질감 복원  
  3. **지각(perceptual) 손실**  
     -  VGG 기반 특징 거리 측정(컨텐츠 손실)  
  4. **정규화/스무딩 손실**  
     -  Total Variation, Sparsity, Multi-scale, Contrastive Loss 등  

각 손실의 조합이 모델 종류에 따라 다르게 쓰이므로, “손실 함수” 축은 “네트워크 설계”와 교차하며 특징을 강화합니다.

***

## 3. 학습 프레임워크(Framework)

– **PyTorch**: 동적 그래프 기반, 디버깅·실험이 용이하여 경량 모델 연구의 주류  
– **TensorFlow**: 정적 그래프 기반, 양자화·모바딜 최적화 강점  
– 과거 Caffe·MatConvNet으로 시작했으나, 최근 거의 모든 모델이 PyTorch/TensorFlow로 구현  

***

## 4. 훈련 데이터셋(Datasets)

– **DIV2K, Flickr1024**: 대규모 고해상도·다양성 확보  
– **BSD, Set5/14, Urban100, Manga109** 등: 테스트용 벤치마크  
– LR-HR 쌍을 미리 제공하지 않는 데이터(OutdoorScene 등)는 bicubic 전처리를 통해 생성  

***

이 네 가지 축을 조합하면, “네트워크 설계”별로 “어떤 손실”, “어떤 프레임워크”, “어떤 데이터”로 학습했는지 일목요연하게 파악할 수 있어, 새로운 경량 SISR 모델 개발·비교·재현에 실질적 가이드가 됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c1c77d88-6864-4425-9454-8c4e56434f2c/1-s2.0-S1566253523000386-main.pdf
