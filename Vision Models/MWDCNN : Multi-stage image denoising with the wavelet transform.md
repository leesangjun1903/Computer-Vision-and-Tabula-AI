# Multi-stage image denoising with the wavelet transform | Image Denoising

## 1. 핵심 주장 및 주요 기여  
본 논문은 **다단계 구조**와 **동적 합성곱(dynamic convolution)**, **웨이블릿 변환(wavelet transform)** 기법을 결합한 경량화된 CNN 기반 이미지 복원 모델(MWDCNN)을 제안한다.  
- **주장**: 네트워크 깊이를 무작정 늘리지 않고, 동적 합성곱 블록(DCB)과 웨이블릿 변환·강화 블록(WEB), 잔차 블록(RB)의 3단계 구조를 통해 우수한 성능과 낮은 계산량을 동시에 달성할 수 있다.  
- **기여**:  
  1. **DCB**: 입력 영상 특성에 따라 여러 합성곱 커널의 가중치를 동적으로 조정해, 성능–연산량 트레이드오프를 최적화.  
  2. **WEB**: DWT/IDWT를 이용해 주파수 정보와 공간 정보를 분리·융합하며, 잔차 밀집 블록(RDB)으로 세부 노이즈를 억제.  
  3. **RB**: 개선된 잔차 밀집 구조로 중간 특징을 정제하고 최종 복원을 수행.  
  4. **경량화**: 전체 파라미터 수 0.5M, FLOPs 2.8G로 유사 성능 모델 대비 연산량 절반 수준.  

## 2. 문제 정의와 제안 방법  
### 2.1 해결 과제  
- 기존 CNN 계열 복원기법은 성능 향상을 위해 네트워크를 지나치게 깊게 설계 ⇒ 훈련 난이도·연산량 급증  
- 수동 파라미터 조정·최적화 알고리즘 의존성  

### 2.2 모델 구조  
입력 $$I_N$$에 대해 세 단계 변환을 거쳐 깨끗한 영상 $$I_C$$를 생성:  

$$
I_C \;=\; f_{\text{RB}}\bigl(f_{\text{WEB}}(f_{\text{WEB}}(f_{\text{DCB}}(I_N)))\bigr)
\;=\;f_{\text{MWDCNN}}(I_N).
$$

1. **DCB (Dynamic Convolutional Block)**  
   - 5층 구성: Conv(5×5) – DynamicConv – Conv(5×5)+ReLU  
   - 동적 합성곱: 입력 피처 맵 $$X$$에 대해 Softmax 기반 가중치 생성기(WG)로 $$\{w_i\}_{i=1}^4$$ 계산, 병렬 커널 $$\{K_i\}$$과 결합
   
$$
     \mathrm{DC}(X)
     =\sum_{i=1}^4 w_i(X)\,\bigl(K_i * X\bigr).
$$

2. **WEB (Wavelet Transform & Enhancement Block)**  
   - 4-layer RDB 기반 FE(feature enhancement) + DWT/IDWT  
   - 주파수 분해:  
     $$\{X_{\mathrm{LL}},X_{\mathrm{LH}},X_{\mathrm{HL}},X_{\mathrm{HH}}\}= \mathrm{DWT}(X)$$  
   - 공간적 잔차 학습: 각 밴드에 RDB 적용 ⇒ 세부 정보 보강  
   - IDWT로 다시 결합  

3. **RB (Residual Block)**  
   - 두 개의 4-layer RDB + ReLU, 중간 잔차 연결, 5×5 Conv, 최종 1×1 Conv + 입력 잔차 결합  

### 2.3 학습 손실  
- MSE 기반 손실:

$$
  \mathcal{L}(\theta)=\frac1{2N}\sum_{i=1}^N \bigl\|f_{\text{MWDCNN}}(I^i_N)-I^i_C\bigr\|_2^2.
  $$
   
- Charbonnier, Pearson 융합 대비 단일 MSE가 PSNR·SSIM에서 최고 성능 확인.

## 3. 성능 향상 및 한계  
### 3.1 성능  
- **PSNR**: BSD68(σ=25)에서 29.28 dB, CBSD68(σ=25)에서 31.45 dB로 기존 DnCNN·ADNet 대비 ≈0.2 dB 향상.  
- **연산 효율**: 파라미터 0.5 M, FLOPs 2.8 G, 실행 시간 0.046 s (1024×1024) — 동급 대비 절반 수준.  
- **실제 노이즈**(CC 데이터셋)에서도 평균 PSNR 35.74 dB로 최고치 기록.  
- **FSIM/SSIM/LPIPS** 지표에서도 최고 성능 달성.  

### 3.2 한계  
- **감독학습 의존**: 깨끗한 레퍼런스 영상이 필요 — 실제 노이즈 데이터 구축 비용 상승  
- **웨이블릿 기법 고정**: Haar 기반 DWT만 사용 ⇒ 다양한 변환군 실험 필요  
- **일반화**: 제한된 노이즈 분포(Gaussian)에 최적화, 비정형 노이즈에 대한 확장성 검증 부족  

## 4. 일반화 성능 향상 가능성  
- **동적 합성곱**: 입력 콘텐츠 의존적 커널 조합으로 다양한 노이즈·영상 도메인에 유연 적용  
- **주파수–공간 융합**: DWT를 통한 노이즈 분리 후 RDB로 구조적 정보 보강 ⇒ 다른 영상 복원(초해상·압축 노이즈 제거)에도 확장 가능  
- **경량 구조**: 낮은 파라미터 수로 모바일·임베디드 환경 전이 학습(transfer learning) 적용 시 과적합 최소화  

## 5. 향후 연구 영향 및 고려 사항  
- **무감독·자기지도 학습**: 실제 노이즈에 대한 클린 레퍼런스 없이 학습하는 Blind/Noise2Void 계열과 결합  
- **다중 웨이블릿 변환**: 다양한 파생 웨이블릿(Daubechies, Symlets 등) 실험으로 주파수 대역 최적화  
- **Cross-domain 일반화**: 의료·위성 영상 등 이질적 도메인에 대한 도메인 어댑테이션 및 동적 커널 재학습 연구  
- **경량화·가속화**: 동적 합성곱과 WEB 구조를 양자화·프루닝하여 실시간 애플리케이션에 통합  

결론적으로, MWDCNN은 **동적 합성곱과 웨이블릿 변환**의 시너지를 통해 경량화된 우수 복원 성능을 달성했으며, 노이즈 유형과 도메인 확장을 위한 다양한 후속 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/75d8cbad-ad03-4027-b9ee-de06309bfc3f/2209.12394v3.pdf
