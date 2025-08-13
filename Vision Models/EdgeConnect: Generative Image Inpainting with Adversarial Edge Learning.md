# EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning

## 개요  
EdgeConnect는 손상된 이미지 영역을 자연스럽게 채우는 **2단계 적대적 학습(Generative Adversarial Network, GAN)** 기반 모델입니다. 먼저 누락된 부분의 **Edge(윤곽선)** 를 생성하고, 이를 바탕으로 실제 픽셀을 복원하여 세밀한 디테일을 보존합니다[1].

---

## 1. 동기와 기여  
기존 딥러닝 기반 이미지 복원 기법들은 종종 결과물이 과하게 매끄럽거나 흐려지는 문제가 있었습니다.  
EdgeConnect는 **“Line First, Color Next”** 라는 화가의 스케치-채색 과정을 모방하여 불완전한 영역의 구조적 윤곽을 먼저 예측한 뒤 실제 색상을 채우도록 설계했습니다[1].

주요 기여  
- **2단계 구조**: Edge 생성기(Generator)와 이미지 완성 네트워크로 분리  
- **경계 정보 활용**: 손실된 영역 윤곽선을 hallucination 기법으로 생성  
- **End-to-End 학습**: 두 네트워크를 통합하여 최적화  

---

## 2. 모델 구조  

### 2.1. Edge Generator (G₁, D₁)  
- **입력**: 그레이스케일로 변환된 마스크 이미지, 원본 윤곽선 맵, 마스크 정보  
- **아키텍처**:  
  - Down-sampling → 8개의 Residual Block(2-Dilated Convolution) → Up-sampling  
  - Discriminator(D₁): 70×70 PatchGAN[2]  
- **손실 함수**:  
  - Adversarial Loss ($$L_{adv,₁}$$)  
  - Feature Matching Loss ($$L_{FM}$$)  
  - 가중치: $$λ_{adv₁} = 1, λ_{FM} = 10$$

### 2.2. Image Completion Network (G₂, D₂)  
- **입력**: 손상된 컬러 이미지, 결합된 윤곽선 맵  
- **아키텍처**:  
  - 구조는 Edge Generator와 유사하나 색상 및 텍스처 학습에 집중  
  - Discriminator(D₂): 70×70 PatchGAN  
- **손실 함수**(Joint Loss):  
  - L1 Loss (픽셀 차이)  
  - Adversarial Loss ($$L_{adv,₂}$$)  
  - Perceptual Loss (VGG-19의 relu1_1 ~ relu5_1 레이어 이용)  
  - Style Loss (Gram matrix 기반)  
  - 가중치: $$λ_{L1}=1, λ_{adv}=0.1, λ_{perc}=0.1, λ_{style}=250$$[3]

---

## 3. 학습 전략  

- **데이터셋**: CelebA, Places2, Paris StreetView  
- **마스크 종류**:  
  - (1) 정사각형 랜덤 마스크(전체 이미지의 25% 크기)  
  - (2) 불규칙 마스크(Partial Convolutions 논문의 공개 데이터)  
- **Edge 추출**: Canny Detector (Gaussian σ=2)  
- **최적화**: Adam (β₁=0, β₂=0.9)  
- **학습률**:  
  - G₁, G₂ 초기 LR=1e-4 → 수렴 후 1e-5  
  - D₁, D₂: G의 LR × 0.1  
- **End-to-End Fine-tuning**: G₁, G₂, D₂ 통합 후 LR=1e-6 로 미세 조정[3]

---

## 4. 평가 및 결과  

### 4.1. 정성적 결과  
EdgeConnect는 기존 기법 대비 **흐림(Blurriness)** 과 **Checkerboard artifact**가 현저히 줄어든 고품질 복원 결과를 보였습니다[1].

### 4.2. 정량적 지표  
|지표|설명|연산 방식|유리한 방향|
|---|---|---|---|
|L1|픽셀 단위 절대차이|정규화된 L1|낮을수록 좋음|
|SSIM|구조적 유사도|계산식 기반|높을수록 좋음|
|PSNR|신호 대 잡음비|20 log10(MAX/√MSE)|높을수록 좋음|
|FID|Inception-Feature 차이|Wasserstein-2 Distance|낮을수록 좋음|

EdgeConnect는 대부분 지표에서 기존 방법을 앞섰습니다. 특히 FID 개선이 두드러져 전체적 인식률 상승을 확인했습니다[1].

### 4.3. Visual Turing Test  
- 2-Alternative Forced Choice(2AFC), Just Noticeable Differences(JND) 적용  
- 실제 이미지 판별 정확도 약 94.6% (오차 ±0.5%)  
- 사람 지각 기반 평가에서도 우수한 결과를 보였습니다[1].

### 4.4. Ablation Study  
- **Edge Generator 유무**: Edge 모듈 포함 시 모든 지표에서 성능 향상  
- **Canny σ 값 변화**: σ=2일 때 PSNR·FID 최적값 달성  
이로써 경계 정보가 이미지 복원에 핵심적임을 확인했습니다[3].  

---

## 5. 결론  
EdgeConnect는 **“Line First, Color Next”**를 모티브로 윤곽선 예측 후 색상 복원을 수행하여, 세밀한 디테일을 효과적으로 재현합니다. 2단계 GAN 구조와 다양한 손실 함수를 결합하여 기존 딥러닝 기반 인페인팅 기법의 한계를 극복했으며, Restoration, Removal, Synthesis 등 다양한 어플리케이션에 적용 가능합니다[1].

---

References  
[1] Generative Image Inpainting with Adversarial Edge Learning, Kamyar Nazeri et al., arXiv:1901.00212 (2019)  
[2] Image-to-Image Translation with Conditional Adversarial Networks, Phillips et al., arXiv:1611.07004 (2016)  
[3] Big Dream World 블로그, “EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning,” 2021.

[1] https://arxiv.org/abs/1901.00212
[2] https://subinium.github.io/LR007/
[3] https://big-dream-world.tistory.com/80
[4] https://github.com/knazeri/edge-connect
[5] https://github.com/YeshengSu/EdgeConnect
[6] https://huggingface.co/papers/1901.00212
[7] https://github.com/jshi31/edge-connect
[8] https://arxiv.org/pdf/2102.08078.pdf
[9] https://arxiv.org/pdf/1901.00212.pdf
[10] https://github.com/Ma-Dan/edge-connect
[11] https://openaccess.thecvf.com/content_ICCVW_2019/papers/AIM/Nazeri_EdgeConnect_Structure_Guided_Image_Inpainting_using_Edge_Prediction_ICCVW_2019_paper.pdf
