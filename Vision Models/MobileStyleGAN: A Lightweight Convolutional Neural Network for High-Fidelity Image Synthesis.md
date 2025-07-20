# MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis | Image generation

**핵심 주장**  
MobileStyleGAN은 StyleGAN2의 성능을 거의 유지하면서 파라미터 수와 연산량을 대폭 줄인 경량 스타일-기반 GAN 아키텍처이다.  

**주요 기여**  
1. 주파수 기반 이미지 표현: 출력 이미지를 픽셀 도메인이 아닌 Haar 웨이블릿(DWT/IDWT) 도메인으로 생성.  
2. Depthwise Separable Modulated Convolution: StyleGAN2의 모듈화 합성곱을 깊이별 분리(depthwise-separable) 구조로 경량화.  
3. Demodulation Fusion: 추론 시 스타일별 디모듈레이션을 학습 가능한 상수로 대체하여 연산 혼합(연산 융합) 최적화 가능.  
4. 지식 증류(distillation) 기반 학습 파이프라인: StyleGAN2를 교사 네트워크로 삼아 파라미터를 크게 줄인 학생 네트워크를 학습.  

# 1. 해결하려는 문제  
대다수 스타일-기반 GAN(특히 StyleGAN2)은 수천만 개의 파라미터(≈28M)와 수백 GMAC(≈143GMAC)의 연산을 요구해 모바일·임베디드 환경으로 배포하기 어려움.  

# 2. 제안 방법

## 2.1 웨이블릿 기반 이미지 표현  
- 출력 이미지를 픽셀 대신 DWT(2D Haar 웨이블릿) 계수로 예측하고, IDWT로 재구성  
- 고주파 성분에 직접 정규화 추가로 저·고주파 모두에서 매끄러운 잠재 공간 확보  
$$
\|I - \mathrm{IDWT}(\mathrm{DWT}(I))\|_1 < \epsilon\approx10^{-7}
$$  

## 2.2 구조 변경  
- **합성곱 유닛**:  
  - 모듈화 합성곱(modulated conv) → 깊이별 분리 DW-sep 모듈화 합성곱  
  - $$x'=s\cdot x$$, $$x'' = w_{\mathrm{dw}}\ast x'$$, $$x'''=w_{\mathrm{pw}}\ast x''$$ 후  

    $$\mathrm{demod}\_j = 1/\sqrt{\sum_{i,k}(s_iw'_{i,j,k})^2+\epsilon}$$ 적용  

- **업샘플링**: Conv-transpose → IDWT  
- **예측 헤드**: 중간 해상도마다 auxiliary head 추가, skip-sum 구조 제거  

## 2.3 디모듈레이션 융합  
- 추론 시 디모듈레이션 계수를 학습 가능한 상수 $$p_{\mathrm{demod}}$$로 대체하여  
  $$\mathrm{demod}_j=1/\sqrt{\sum_{i,k}(p_{\mathrm{demod}}w'_{i,j,k})^2+\epsilon}$$  
  → 가중치 융합으로 추론 최적화 가능  

# 3. 아키텍처 및 수식 개요  
- **Mapping**: StyleGAN2와 동일한 8-layer MLP로 latent $$z\to w$$  
- **Synthesis**:  
  - 블록별: IDWT↑ → DW-sep mod conv ×2 → demod fusion → LeakyReLU  
  - Aux head: 각 해상도별 wavelet→RGB 변환  
- **손실 함수**:  

$$
  L_{\mathrm{pix}} = \sum_i \bigl\|F^i_s - \mathrm{DWT}(I^i_t)\bigr\|_1 + \bigl\|\mathrm{IDWT}(F^i_s)-I^i_t\bigr\|_1
  $$  
  
$$
  L_{\mathrm{perc}} = \sum_l \bigl\|\phi_l(I_s)-\phi_l(I_t)\bigr\|_2^2
  $$  
  
$$
  L_{\mathrm{GAN}} = f(-D(G_s)),\quad f(t)=-\ln(1+e^{-t})
  $$  
  
  최종: $$L = \lambda_1L_{\mathrm{pix}}+\lambda_2L_{\mathrm{perc}}+\lambda_3L_{\mathrm{GAN}}$$.  

# 4. 성능 및 한계

| 모델                | 파라미터(M) | 연산량(GMAC) | FID (FFHQ 1024²) | CPU 추론 속도(sec) |
|--------------------|-------------|-------------|------------------|--------------------|
| StyleGAN2          | 28.27       | 143.15      | 2.84             | 4.3                |
| **MobileStyleGAN** | **8.01**    | **15.09**   | **7.75**         | **1.2** (PyTorch)  |
|                    |             |             |                  | **0.16** (OpenVINO)|

- FID 상승(2.84→7.75)라는 품질 저하가 있으나, 연산량·메모리 대폭 감소와 CPU 실시간 생성 가능  
- **한계**: 적은 데이터·다양한 도메인 일반화 실험 미비, 추가 경량화(양자화·가지치기) 필요  

# 5. 일반화 성능 향상 관점  
- 웨이블릿 기반 주파수 분해는 저·고주파 학습을 분리해 노이즈 강건성·표현 다양성 개선 가능  
- 디모듈레이션 융합 덕분에 추론 경량화로 다양한 해상도·도메인에서 실험 확장 용이  
- 그러나 논문에서는 FFHQ(얼굴) 단일 도메인만 평가하므로, 다른 도메인(자연·의료영상)·소량 데이터 상황에서 일반화 검증이 필요  

# 6. 향후 연구 방향 및 고려사항  
- **양자화·가지치기**를 결합해 추가 경량화  
- **도메인 확장**: 자연, 의료, 예술 등 다양한 데이터셋에서 성능·FID 비교  
- **소량 데이터/제한적 데이터** 상황에서 ADA 기법 적용 여부  
- **웨이블릿 종류**(Daubechies, Coiflet 등) 및 멀티레벨 DWT의 영향 검토  
- **학습 안정성**: DW-sep 모듈화 합성곱이 GAN 훈련에 미치는 효과 심층 분석  

---  

이 논문은 스타일 기반 생성 모델의 경량화에 새로운 방향을 제시하며, 주파수 기반 표현과 지식 증류를 활용해 엣지 디바이스 배포 가능성을 크게 확장하였다. 앞으로 다양한 도메인·제한적 데이터 상황에서 일반화 실험과 추가 최적화 기법 연구가 필요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ea2cf92d-1626-4c81-94f9-0ac8ff673087/2104.04767v2.pdf
