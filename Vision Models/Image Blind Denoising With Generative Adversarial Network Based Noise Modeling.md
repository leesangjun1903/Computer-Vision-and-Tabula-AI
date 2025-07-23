# Image Blind Denoising With Generative Adversarial Network Based Noise Modeling | Image denoising

## 1. 핵심 주장 및 주요 기여
이 논문은 **알려지지 않은(no unknown) 잡음을 포함한 이미지(블라인드 노이즈)의 제거**를 위해 다음 두 단계 프레임워크를 제안한다:
1. **GAN 기반 노이즈 모델링**: 입력된 노이즈 이미지에서 노이즈 패치를 추출하고, 이를 통해 WGAN-GP를 학습하여 실제 노이즈 분포를 근사·샘플링  
2. **CNN 기반 복원**: GAN이 생성한 노이즈 샘플과 깨끗한 이미지 패치를 합성해 페어 데이터셋을 구축하고, 이를 DnCNN 유사 구조의 딥 CNN으로 학습하여 최종 이미지 복원  
주요 기여:  
- GAN을 활용해 **페어 학습 데이터가 없는** 블라인드 노이즈 환경에서 노이즈 분포를 **암묵적으로** 학습  
- 생성된 노이즈 샘플로 CNN 복원 모델을 효과적으로 학습시켜 **종래 기법 대비 PSNR 1 dB 이상 향상**  

## 2. 문제 정의 및 제안 방법

### 문제 정의
- 관측 모델: $$y = x + v$$ , 여기서 $$x$$는 깨끗한 이미지, $$v$$는 **알려지지 않은 제로-평균**(image blind) 노이즈  
- 목표: 노이즈 분포를 모르는 상태에서 $$y$$만으로 $$x$$를 복원  

### 제안 방법 개요
1. **노이즈 패치 추출**  
   - 입력 노이즈 이미지에서 크기 $$d\times d$$ 패치 $$\{p_i\}$$ 스캔  
   - 각 패치 내부 $$h\times h$$ 지역 패치 $$\{q_{i,j}\}$$와 평균·분산 차이를 비교해 ‘평탄 영역’ 식별(수식 (1),(2))  
   - 추출한 평탄 패치 $$s_i$$로부터 $$v_i = s_i - \mathrm{Mean}(s_i)$$ 계산  

2. **GAN 기반 노이즈 모델링**  
   - **WGAN-GP** 손실:

$$
       L_{\mathrm{GAN}} = \mathbb{E}\_{\tilde x\sim P_g}[D(\tilde x)]
       - \mathbb{E}\_{x\sim P_r}[D(x)]
       + \lambda\,\mathbb{E}\_{\hat x\sim P_{\hat x}}\bigl(\|\nabla_{\hat x}D(\hat x)\|_2 - 1\bigr)^2
$$
     
  여기서 $$P_r$$은 추출 노이즈 분포, $$P_g$$는 생성 분포  
   - DCGAN 유사 구조의 **생성기**(256–128–64 채널 역컨볼루션)와 **판별기**(64–128–256–512 채널 컨볼루션)  

3. **CNN 기반 노이즈 제거**  
   - GAN이 생성한 노이즈 샘플 $$\{v'_k\}$$와 깨끗한 패치 $$\{x_j\}$$를 합성해 페어 $$\{(y_l=x_j+v'_k, x_j)\}$$ 구축  
   - DnCNN 유사 17-레이어 구조로 **잔차 학습**:

  $$
       L_{\mathrm{CNN}}(\Theta)
       = \frac{1}{2N}\sum_{i=1}^N\bigl\|R(y_i;\Theta)-\bigl(y_i-x_i\bigr)\bigr\|_F^2
     $$
   - 입력 $$y_i$$로부터 잔차 $$R(y_i)$$ 예측 후 $$x_i = y_i - R(y_i)$$ 복원  

## 3. 성능 향상 및 한계

### 성능 향상
- **합성 Gaussian 노이즈** ($$\sigma=25$$): PSNR 29.15 dB (GCBD) vs. 26.77 dB (DnCNN-B), 28.83 dB (WNNM)  
- **혼합 노이즈** ($$s=25$$): PSNR 39.87 dB (GCBD) vs. 35.12 dB (DnCNN-B), 37.63 dB (WNNM)  
- **실제 사진(DND)**: PSNR 35.58 dB (GCBD) vs. 34.61 dB (BM3D) 등[1]  
- **유사 환경 일반화**(NIGHT-B): 훈련에 쓰이지 않은 야간 모바일 사진에서도 또렷한 세부 보존  

### 한계
- **제로-평균 가법 노이즈**만 취급: 실제 저조도 촬영잡음 중 비제로 평균·비가법 성분 비중 미반영  
- GAN 훈련의 **불안정성**: 복잡 분포 학습 시 모드 붕괴(risk of mode collapse) 가능  
- **추출 패치 의존성**: 평탄 영역이 부족한 드문 장면에서는 노이즈 샘플 다양성 한계  

## 4. 모델 일반화 성능 향상 관련 고찰
- GAN으로 학습한 **분포 기반(noise distribution) 샘플링**은 단일 이미지 의존적 “내부 패턴” 기법 대비 **외부 데이터 일반화**에 유리  
- **페어 데이터 랜덤 재조합** 방식으로 에폭마다 새로운 합성 샘플 제공→CNN 과적합 완화  
- **미지 시나리오 확장**: 야간·저조도·복합 센서 노이즈 등 다른 무작위성(noise variability) 환경에서도 GAN만 재학습 후 바로 적용  

## 5. 향후 연구 및 고려 사항
- **비가법·비제로 평준 노이즈 모델링**: Poisson–Gaussian 혼합, 센서 특유 이상치(noise spikes)  
- **안정적 GAN 학습**: Spectral normalization, Progressive training 도입으로 모드 붕괴 완화  
- **Self-supervised 학습 접목**: Noise2Void, Noise2Self 기법과 결합해 노이즈 분리 정확도 향상  
- **다중 스케일 노이즈 패치** 활용: 다양한 크기·텍스처 영역에서 노이즈 통계 반영  

위와 같이, 본 논문은 **GAN 기반으로 페어 학습 데이터가 전무한 블라인드 노이즈 영역에서도 딥 CNN 복원을 가능**하게 함으로써, 실세계 노이즈 제거 연구의 새로운 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/03334a5f-b49f-405f-b208-fe8897ad6107/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf
