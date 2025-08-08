# MPRNet : Multi-Stage Progressive Image Restoration | Image deblurring, Image denoising, Image restoration

## 1. 핵심 주장 및 주요 기여  
MPRNet은 **단일 단계(single-stage)** 설계가 갖는 “공간 정확도와 맥락적 정보 획득의 상충” 문제를 **여러 단계(multi-stage)** 로 나누어 해결하고자 한다.  
- **주장**: 복잡한 이미지 복원 과제를 “다단계”로 분해함으로써 각 단계가 점진적으로 복원 기능을 학습하며, 단계 간 정보 교환을 통해 문맥적(encoder–decoder) 및 공간적(원래 해상도) 특징을 모두 확보한다.  
- **기여**:  
  1. Encoder–decoder 기반의 초기 단계와 원본 해상도(subnetwork) 기반의 최종 단계를 결합한 다단계 구조 설계.  
  2. 각 단계 간 **Supervised Attention Module (SAM)** 을 도입해 중간 복원 결과에 직접 감독 신호를 제공하고, 중요 특징만 다음 단계로 전달.  
  3. 단계 간 **Cross-Stage Feature Fusion (CSFF)** 으로 다중 스케일 문맥 정보를 후속 단계에 유기적으로 통합.  
  4. 다양한 복원 과제(비, 모션 블러, 노이즈 제거)에서 기존 대비 최대 20%의 오류 감소 및 실시간 처리급 속도를 달성.  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결 과제  
- **문제**: 이미지 복원은 “무한한 해답”이 가능한 ill-posed 문제로, 단일 CNN 구조는 넓은 수용 영역(문맥)과 높은 공간 해상도를 동시에 확보하기 어려움.  

### 2.2 제안 방법  
- **점진적 복원**: degraded input $$I$$에 대해 각 단계 $$S$$가 잔차 $$R_S$$를 예측하고, $$X_S = I + R_S$$ 로 복원  
- **손실 함수**:  

$$
    \mathcal{L} = \sum_{S=1}^3 \Bigl[L_{\text{char}}(X_S, Y) + \lambda\,L_{\text{edge}}(X_S, Y)\Bigr]
$$

$$
    L_{\text{char}} = \sqrt{\|X_S - Y\|^2 + \varepsilon^2},\quad
    L_{\text{edge}} = \sqrt{\|\Delta(X_S) - \Delta(Y)\|^2 + \varepsilon^2}
$$

  ($$\varepsilon=10^{-3},\,\lambda=0.05$$)  

### 2.3 모델 구조  
- **Stage 1–2 (Encoder–Decoder)**  
  - U-Net 베이스, 각 스케일에 Channel Attention Block (CAB) 삽입  
  - Bilinear upsampling + convolution으로 해상도 복원  
- **Stage 3 (Original-Resolution Subnetwork, ORSNet)**  
  - 다운샘플링 없이 위치 민감 복원을 위해 원본 해상도로 처리  
  - 다수의 Original-Resolution Block (ORB) 내 CAB 집적  
- **모듈**  
  1. **Supervised Attention Module (SAM)**:  
     - 이전 단계 예측 $$X_S$$에 ground-truth $$Y$$ 감독  
     - 1×1 convolution으로 per-pixel attention mask $$M$$ 생성, 특징 reweighting  
  2. **Cross-Stage Feature Fusion (CSFF)**:  
     - 1×1 convolution으로 정제된 중간 특징을 다음 단계 encoder/decoder 블록에 lateral 연결  

### 2.4 성능 향상  
- **Image Deraining**: 기존 MSPFN 대비 평균 PSNR +1.98 dB (오류 20% 감소)[Table 2]  
- **Deblurring (GoPro/HIDE)**: PSNR +0.81 dB / +0.98 dB, SSIM 최대 +0.011[Table 3]  
- **Denoising (SIDD/DND)**: PSNR +0.19 dB / +0.21 dB[Table 5]  
- **경량화·실시간**: 최종 모델 20 M 파라미터, CPU/GPU 상 실시간 처리 가능[Table 7]  

### 2.5 한계  
- **단계 수 고정**: 사전 정의된 3단계 구조로, 최적 단계 수 탐색 필요  
- **메모리·연산**: 고해상도 ORSNet 단계에서 자원 소모  
- **학습 복잡도**: SAM·CSFF 도입이 훈련 안정성에 민감  

## 3. 일반화 성능 향상 가능성  

- **Cross-Domain 적용**: GoPro 학습 모델을 HIDE, RealBlur, DND에 튜닝 없이 그대로 적용해 SPSNR·SSIM 대폭 향상  
- **SAM 감독**: 중간 단계마다 실제 ground-truth 지도 강화로 “과적합 방지” 및 “도메인 이동 저항성” 확보  
- **다중 스케일 융합**: CSFF가 early feature를 later stage에 통합해 문맥 손실 최소화, unseen degradation에도 견고  

## 4. 향후 연구에 미치는 영향 및 고려사항  

- **영향**  
  - 다단계·점진 학습의 low-level vision 적용 확대  
  - attention 기반 중간 감독(supervision) 설계 가능성 시사  
  - 경량화 모델의 실시간 복원 시스템 상용화 촉진  

- **향후 고려**  
  - **적응적 단계 수**: 입력 복원 난이도에 따라 동적 단계 할당  
  - **자원 제약 학습**: 모바일·임베디드 디바이스용 저전력 최적화  
  - **자율 데이터셋**: 실제 열화(real degradation) 수집 후 SAM 감독 없이도 일반화 보장 연구  
  - **멀티모달 복원**: 노이즈·블러·비 등 복합 저하에 대한 통합 복원 프레임워크 개발

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8a81f898-8916-4bdc-99f0-13a97bf3e296/2102.02808v2.pdf

# Abs

# Introducion
이미지 복원 작업은 이미지를 복원하는 동안 공간 세부 정보와 높은 수준의 contextualized 정보 간의 복잡한 균형을 요구한다.  
이 논문에서 저자들은 이 균형을 맞춰줄 수 있는 새로운 모델로 Multi-stage 구조를 제안했다.

저자들이 제안한 이 모델은 degrade된 input에 대한 복원 기능을 점진적으로 학습하여 전체 복원 프로세스를 관리 가능한 단계로 세분화한다.

# Multi-Stage Progressive Restoration
![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F11184905-a578-44ef-ae79-5b26b30d1ef2%2F%EC%BA%A1%EC%B2%98.PNG)
