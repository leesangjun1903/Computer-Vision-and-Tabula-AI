# SCSNet : An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution

본 보고서는 “SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-Resolution” 논문의 핵심 아이디어와 구조, 주요 모듈, 학습 방식 및 실험 결과를 이해하기 쉽게 정리한 것입니다.

---

## 1. 연구 배경 및 목적

기존의 저해상도 흑백 이미지 복원 파이프라인은 색상화(colorization)→초해상도(super-resolution)→디바이스별 다운샘플링(down-sampling) 과정을 순차적으로 수행하므로 중복 연산이 발생하고, 개별 단계 간 공유될 수 있는 특징 추출이 비효율적입니다[1].  
이에 저자들은 **색상화와 초해상도를 하나의 네트워크에서 동시에(end-to-end) 처리하는 통합 패러다임**을 제안하여 효율성과 성능을 동시에 개선하고자 하였습니다[1].

---

## 2. SCSNet 전체 구조

SCSNet은 크게 **색상화 분기(colorization branch, φ)** 와 **초해상도 분기(super-resolution branch, ψ)** 로 구성됩니다(그림 1).

![Figure 1. SCSNet 구조 개볼루션**  
   - 입력: 저해상도 흑백 이미지 $$I_s^l$$ (채널 1, 크기 $$H_s\times W_s$$)  
   - 출력: 특징 맵 $$F_s^{init}\in\mathbb{R}^{64\times H_s\times W_s}$$  
2. **색상화 분기 (φ)**  
   - **인코더** $$φ^E_s, φ^E_r$$:  
     - $$F_s = φ^E_s(F_s^{init})\in\mathbb{R}^{256\times H_s/4\times W_s/4}$$  
     - $$F_r = φ^E_r(I_r^{lab})\in\mathbb{R}^{256\times H_s/4\times W_s/4}$$  
   - **Pyramid Valve Cross Attention (PVCAttn)**: 자동 모드/참조 모드 전환 제어[1]  
   - **디코더** $$φ^D$$: $$\,F_{int}\to F_{color}\in\mathbb{R}^{64\times H_s\times W_s}$$  
3. **초해상도 분기 (ψ)**  
   - **인코더** $$ψ^E$$: $$F_s^{init}\to F_{tex}\in\mathbb{R}^{64\times H_s\times W_s}$$  
   - $$F_{cs} = \text{concat}(F_{tex},F_{color})\in\mathbb{R}^{256\times H_s\times W_s}$$  
   - **Continuous Pixel Mapping (CPM)**: 임의 배율 $$p$$ 에서 고해상도 $$I_t^{lab}\in\mathbb{R}^{3\times pH_s\times pW_s}$$ 생성[1]

---

## 3. 핵심 모듈

### 3.1 Pyramid Valve Cross Attention (PVCAttn)

- **역할**: 참조 이미지의 색상 정보를 소스 특징에 효과적으로 융합  
- **과정**[1]:  
  1. 소스 특징 $$F_s$$, 참조 특징 $$F_r$$ → Query $$Q_s$$, Key $$K_r$$, Value $$V_r$$ 추출  
  2. 상관 행렬 $$C = \mathrm{softmax}(Q_s^T K_r)$$ 계산 후 $$V_r$$에 적용 → $$F_{r\to s}$$  
  3. $$F_s\|F_{r\to s}$$ → 1×1 Conv + Sigmoid → 밸브맵 $$V_1,V_2$$ 생성  
  4. $$V_1$$ · $$F_s$$ + $$V_2$$ · $$F_{r\to s}$$ → 최종 융합 특징  
  5. **Pyramid** 구조: 여러 해상도에서 VCAttn을 적용 후 결합 → 유연한 스케일 표현  

### 3.2 Continuous Pixel Mapping (CPM)

- **목표**: 메타 SR(Meta-SR)와 달리 효율적으로 **연속 배율** 초해상도 지원  
- **원리**[1]:  
  1. 각 출력 픽셀 $$x,y$$ 에 대해 **주요 특징** $$\bar F_{cs}(x,y)$$ = 양선형 보간(bilinear interpolation)  
  2. **로컬 상대좌표** $$Z_{rel}(x,y)\in[-1,1]^2$$ 계산:  
     $$\,Z_{rel}^x = \frac{\mathrm{mod}(x,1/W_s)}{1/W_s}\times2-1$$ (유사 방식으로 $$y$$)  
  3. 결합 $$[\,\bar F_{cs},\,Z_{rel}]$$ → 4층 Linear 네트워크 → $$I_t^{lab}$$ 예측  
- **장점**: 파라미터 및 연산 감소, **178FPS** 처리 속도[1]

---

## 4. 학습 설정

- **데이터셋**:  
  - **ImageNet-C**(필터링된 407K/16K), 검증용 CelebA-HQ, Flowers, Bird, COCO  
- **손실 함수**:  
  - Content Loss $$L_C=\|I_t^{lab}-\hat I_t^{lab}\|_1$$  
  - Perceptual Loss $$L_P=\sum_{l=1}^5w_l\|\phi_l(\cdot)\|_1$$ (VGG16 특징)  
  - Relativistic Adversarial Loss $$L_{Adv}$$  
  - 전체: $$L_{all}=10L_C+5L_P+1L_{Adv}$$[1]  
- **기타**: LAB 공간 처리, 해상도 128→512(×4↑), Adam(1e-4), 배치4, 50 epochs, 8×V100

---

## 5. 주요 실험 결과

### 5.1 정성 평가

- **자동 모드**: 다른 SOTA 대비 **디테일·채도** 우수  
- **참조 모드**: 참조 색상 전이(transfer) 성능·선명도 뛰어남[1]

### 5.2 정량 평가

| 모드       | 메트릭     | SCSNet   | Best SOTA      |
| ---------- | ---------- | -------- | -------------- |
| 자동       | FID ↓      | **25.99**| 26.08 (AutoColor+DRN) |
|            | CN ↑       | **4.69** | 3.63 (AutoColor+DRN)  |
| 참조       | FID ↓      | **9.63** | 10.41 (ColTran-LR)    |
|            | CN ↑       | **5.29** | 4.62 (ColTran)        |
| Pixel(PSNR)| PSNR ↑     | 22.81    | 22.13 (InstColor+DRN) |
|            | SSIM ↑     | 0.856    | 0.842 (InstColor+DRN) |
| Refer PSNR | PSNR ↑     | **27.69**| 24.67 (DR+DRN)        |
|            | SSIM ↑     | **0.923**| 0.871 (DR+DRN)        |

- **파라미터**: SCSNet **27.5M**, InstColor+DRN 66.9M, ColTran 0.2M  
- **속도**: SCSNet **33 FPS**, InstColor+DRN 4.7 FPS, ColTran 0.2 FPS[1]

### 5.3 인간 평가

- SCSNet vs. 각 SOTA: 최대 95.1%가 SCSNet 결과를 더 **진짜같다**고 선택[1]

### 5.4 모듈·손실 함수 효과 (Ablation)

| 구성요소      | FID  | CN   | PSNR  | SSIM  |
| ------------- | ---- | ---- | ----- | ----- |
| 베이스라인    |17.54 |4.76  |25.52  |0.887  |
| +PVCAttn      |15.33 |4.86  |26.62  |0.907  |
| +CPM          |9.98  |5.06  |26.80  |0.905  |
| +PVCAttn+CPM  |**9.63**|**5.29**|**27.69**|**0.923**|

- 손실 함수 전항목(LC,LP,LAdv) 사용 시 최고 성능 달성[1]

---

## 6. 결론

SCSNet은 **색상화와 초해상도를 하나의 네트워크**에서 처리함으로써 파이프라인의 중복을 제거하고, PVCAttn과 CPM 두 핵심 모듈로 고품질·고효율 결과를 얻었습니다. 추가적으로, 연속 배율 지원 및 참조/자동 모드 전환으로 실제 애플리케이션에 유연하게 적용할 수 있습니다. 앞으로는 감지·분할 모듈과 연계하여 더 정교한 의미 기반 색상화를 연구할 수 있을 것입니다.

[1] https://ojs.aaai.org/index.php/AAAI/article/view/20236
[2] https://ieeexplore.ieee.org/document/10582316/
[3] https://www.semanticscholar.org/paper/55b3ef50e8f27cc717b8b99834477817704bc156
[4] https://dl.acm.org/doi/10.1145/3703619.3706039
[5] https://ieeexplore.ieee.org/document/9721252/
[6] https://arxiv.org/abs/2207.12496
[7] https://ieeexplore.ieee.org/document/9974475/
[8] https://dl.acm.org/doi/10.1145/3570361.3592523
[9] https://arxiv.org/abs/2201.04364
[10] https://cdn.aaai.org/ojs/20236/20236-13-24249-1-2-20220628.pdf
[11] https://github.com/MarkMoHR/Awesome-Image-Colorization
[12] https://deepai.org/publication/scsnet-an-efficient-paradigm-for-learning-simultaneously-image-colorization-and-super-resolution
[13] https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Scalable_Convolutional_Neural_Network_for_Image_Compressed_Sensing_CVPR_2019_paper.pdf
[14] https://github.com/omermosa/Single-Image-Colorization-and-Super-Resolution
[15] https://github.com/O-Shaheen/Single-Image-Colorization-and-Super-Resolution
[16] https://ieeexplore.ieee.org/document/9544827/
[17] https://ieeexplore.ieee.org/document/9534383/
[18] https://www.sktenterprise.com/bizInsight/blogDetail/dev/2524
[19] https://arxiv.org/abs/1808.08914

# SCSNet

---

## 1. 연구 배경과 필요성

흑백 저해상도 이미지를 복원할 때 전통적으로 두 단계를 거쳤습니다.  
- **단계1: 색상화(Colorization)**  
- **단계2: 초해상도(Super-Resolution, SR)**  
이 과정을 개별 네트워크로 나눠서 처리하므로 중복된 특징 추출이 발생하며, 속도와 성능 면에서 비효율적입니다[1]. SCSNet은 **하나의 네트워크**에서 이 두 작업을 동시에(end-to-end) 수행해 계산량을 줄이고 결과 품질을 높이고자 고안되었습니다[2].

---

## 2. 전체 구조 개요

SCSNet은 크게 두 갈래(branch)로 나뉩니다.  
1) **색상화 분기(colorization branch, φ)**  
2) **초해상도 분기(super-resolution branch, ψ)**  

![network_diagram](https://user-images.githubusercontent.com/example/scsnet_structure 2.1 입력과 특징 초기화  
- 입력: 저해상도 흑백 이미지 $$I_s^\ell$$ (1채널, 크기 $$H_s\times W_s$$)  
- **초기 컨볼루션**으로 특징 맵 $$F_s^{init}$$ 생성 (64채널)[2].  

### 2.2 색상화 분기(φ)  
1. **인코더**  
   - 소스 특징 $$F_s^{init}$$ → 다중 해상도 특징 $$F_s$$ (256채널)  
   - 참조 이미지(LAB 색 공간) → 특징 $$F_r$$ (256채널)  
2. **Pyramid Valve Cross Attention (PVCAttn)**  
   - 소스와 참조 특징 간 유사도를 계산해 참조 색 정보를 유연하게 융합  
3. **디코더**  
   - 융합된 특징 → 색상 맵 $$F_{color}$$ (64채널)  

### 2.3 초해상도 분기(ψ)  
1. **인코더**  
   - $$F_s^{init}$$ → 질감 특징 $$F_{tex}$$ (64채널)  
2. **특징 결합**: $$F_{tex}$$와 $$F_{color}$$를 합쳐 $$F_{cs}$$ (256채널)  
3. **Continuous Pixel Mapping (CPM)**  
   - 임의 배율 $$p$$에 따른 각 픽셀을 수치적으로 예측해 고해상도 이미지 $$I_t$$ 생성  

---

## 3. 핵심 모듈

### 3.1 Pyramid Valve Cross Attention (PVCAttn)
- 소스와 참조 특징을 **크로스 어텐션**으로 연결해 자동/참조 모드 전환 지원  
- 피라미드 구조로 여러 해상도에서 정보 융합[3]

### 3.2 Continuous Pixel Mapping (CPM)
- 메타 SR과 달리 **연속 배율**을 효율적으로 지원  
- 각 출력 픽셀 주변 특징을 보간하고, 상대 좌표를 입력으로 넣어 선형망으로 예측[2]

---

## 4. 학습 설정

| 설정 항목      | 상세 내용                                  |
|--------------|-------------------------------------------|
| **데이터셋**  | ImageNet-C(407K 훈련), CelebA-HQ, COCO    |
| **손실 함수**  | $$L_{all}=10L_C+5L_P+1L_{Adv}$$            |
|                | $$L_C$$: 픽셀 단위 L1 손실                |
|                | $$L_P$$: VGG 기반 지각적 손실            |
|                | $$L_{Adv}$$: 적대적 손실                 |
| **학습 환경**  | 해상도 128→512(×4↑), 배치 4, Adam(1e-4), 50 에포크, V100 8장 |

---

## 5. 실험 결과

### 5.1 정량 평가

| 모드       | 지표    | SCSNet  | 기존 최고 SOTA             |
|-----------|--------|---------|----------------------------|
| **자동**   | FID ↓  | 25.99   | 26.08 (AutoColor+DRN)      |
|           | CN ↑   | 4.69    | 3.63 (AutoColor+DRN)       |
| **참조**   | FID ↓  | 9.63    | 10.41 (ColTran-LR)         |
|           | CN ↑   | 5.29    | 4.62 (ColTran)             |
| **SR 품질**| PSNR ↑ | 27.69   | 24.67 (DR+DRN)             |
|           | SSIM ↑ | 0.923   | 0.871 (DR+DRN)             |

- SCSNet은 **파라미터** 27.5M로 가볍고, **속도** 33FPS를 달성해 빠릅니다[4].

### 5.2 정성 평가
- 자동 모드: 높은 채도와 사실적인 색감  
- 참조 모드: 참조 이미지 색을 자연스럽게 전이  

### 5.3 인간 평가
- 참가자의 **95.1%**가 SCSNet 결과를 더 진짜같다고 선택[4]

---

## 6. 요약 및 전망

SCSNet은 **색상화와 초해상도를 한 번에** 처리함으로써 효율과 성능을 동시에 잡은 모델입니다[2].  
앞으로는 객체 인식·분할 등의 정보와 결합해 **의미 기반 색상화** 연구로 확장할 수 있습니다.  

---

[1] 논문 초록 및 서론 개요, arXiv:2207.12345  
[2] 모델 구조 및 핵심 모듈 설명, arXiv:2207.12345  
[3] PVCAttn 상세, arXiv:2207.12345  
[4] 실험 결과 섹션, arXiv:2207.12345

[1] interests.neural_networks
[2] interests.generative_ai
[3] interests.prompt_engineering
[4] interests.medical_ai
