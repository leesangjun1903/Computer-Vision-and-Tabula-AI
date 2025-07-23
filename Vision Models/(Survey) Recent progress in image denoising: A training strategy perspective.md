# 핵심 주장 및 주요 기여 요약  
Wu et al.(2023)은 **이미지 복원 분야에서 딥러닝 기반 잡음 제거 모델들이 주로 ‘학습 전략(training strategy)’에 의해 분류될 수 있다**는 관점에서 종래 리뷰들을 확장·재구성하였다. 네 가지 학습 전략 카테고리(1) 깨끗한/잡음 이미지 쌍으로 학습, (2) 다중 잡음 이미지로 학습, (3) 단일 잡음 이미지로 학습, (4) 비전 트랜스포머 기반 모델로 학습으로 체계화하여 2016–2022년의 59개 대표 모델을 정리하였다.  
주요 기여  
-  학습 전략 관점의 새로운 분류 체계를 제안  
-  각 카테고리별 대표 모델의 성능·구조 비교 및 분석  
-  실제 노이즈 제거 과제에서의 일반화 한계와 미래 연구 방향 제시  

# 문제 정의 및 접근 방법 상세 설명  
본 논문이 해결하고자 하는 문제  
-  딥러닝 기반 이미지 잡음 제거 모델들(특히 AWGN 제거)은 주로 대규모 “깨끗한/잡음 이미지 쌍”을 요구하나, 실제 환경에서는 구하기 어렵고,  
-  이렇게 학습된 모델은 합성 잡음에 과적합(overfitting)되어 실제 잡음 일반화(generalization)에 취약하다.  

제안된 분류 및 학습 전략  
1. **깨끗한/잡음 이미지 쌍 기반**  
   – 손실함수:  

$$\mathcal{L}(\theta)=\frac{1}{2N}\sum_{i=1}^N\|f(y_i;\theta)-x_i\|^2$$[식 (1)]  
   
   – 잔차 학습(residual learning) 사용 시:  
   
$$\mathcal{L}(\theta)=\frac{1}{2N}\sum_{i=1}^N\|R(y_i;\theta)-(y_i-x_i)\|^2$$[식 (2)]  
   
   – 대표 모델: DnCNN, FFDNet, DRUNet 등  

2. **다중 잡음 이미지 기반(self-/un-)**  
   – Noise2Noise(Lehtinen et al.)  

$$\mathcal{L}(\theta)=\tfrac1N\sum\|f(x_i+n_i;\theta)-(x_i+n_i')\|^2$$[식 (3)]  
   
   – Blind-spot, self-supervised (Noise2Void, Noise2Self 등)  

3. **단일 잡음 이미지 기반**  
   – Deep Image Prior:  

$$\theta^*=\arg\min_\theta\|f_\theta(z)-y\|^2$$, $$z$$는 고정 노이즈 입력[식 ( )]  
   – Self2Self 등 드롭아웃·데이터 증강 이용  

4. **비전 트랜스포머 기반**  
   – IPT, SwinIR, Uformer, Restormer 등 ‘전역 self-attention’으로 장거리 픽셀 상호작용 학습  
   – 손실: L1 또는 Charbonnier 등[식 (16)–(19)]  

모델 구조 및 성능 향상  
-  **DnCNN**: ResNet 구조+배치 정규화→수렴·성능 개선.  
-  **FFDNet**: 노이즈 레벨 맵 입력→공간 가변 노이즈 처리 및 속도↑.  
-  **Noise2Noise**: 깨끗한 레이블 없이 노이즈·노이즈 쌍 만으로 학습 가능.  
-  **Deep Image Prior**: 학습 데이터 없이 파라미터 초기화 만으로 잡음 제거.  
-  **SwinIR**: Swin Transformer 기반→고해상도 이미지 복원 성능 우수.  

한계  
-  “깨끗한/잡음 쌍” 모델들은 실제 잡음 도메인 일반화에 취약  
-  Self-/un-supervised 기법 성능은 쌍 학습 대비 여전히 낮음  
-  DIP 계열은 과적합 방지(early stopping) 필요, 자동 조정 어려움  
-  Transformer 모델은 계산량·메모리 부담 큼  

# 일반화 성능 향상 관련 논의  
-  **도메인 분산 불일치**: 합성 AWGN만 학습한 네트워크가 실제 카메라 잡음의 복잡한 통계에 적응 못 함.  
-  **Blind denoising**: 잡음 분포 추정→Bayesian/Poisson-Gaussian 모델(CBDNet, SDNet)이 일반화 성능 개선  
-  **Self-supervision**: Noise2Void, Noise2Self의 J-invariance 완화(Noise2Same)→추가 정보 활용  
-  **비전 트랜스포머**: 전역 어텐션으로 다양한 국소·비국소 잡음 패턴 학습 가능  
-  **데이터 증강 및 합성**: GAN2GAN, Noisier2Noise로 실제 잡음 분포 근사→강건성↑  

# 향후 연구 영향 및 고려 사항  
-  **실제 잡음 데이터 구축**: 다양한 센서·조명·JPEG 압축 등 잡음 레벨·타입 메타정보 포함 대규모 데이터 필요  
-  **도메인 적응 기법 접목**: 메타러닝·도메인 어댑테이션으로 소량 실제 잡음 레이블만으로 빠른 일반화  
-  **경량화·실시간 처리**: 모바일·임베디드 적용 위한 모델 경량화 집중  
-  **자기 감독 및 합성 기법 발전**: 실제 잡음 근사 능력 높이는 시뮬레이터·GAN 기반 기법  
-  **Transformer 효율화**: 윈도우 어텐션·저용량 어텐션 모듈로 연산·메모리 비용 절감  

이 논문은 **학습 전략별 분류**를 통해 잡음 제거 모델의 현황을 통찰하고, 실제 잡음 일반화라는 핵심 과제를 부각시켜, 이후 **실제 잡음 도메인 적응** 및 **경량·효율적 네트워크 설계** 연구의 방향타 역할을 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/22a46443-4f99-4df1-9d62-9a2f5acff2be/IET-Image-Processing-2023-Wu-Recent-progress-in-image-denoising-A-training-strategy-perspective.pdf

# 2 IMAGE DENOISING METHODS

다음은 “Recent progress in image denoising: A training strategy perspective” 논문의 2장 **‘이미지 잡음 제거 방법(IMAGE DENOISING METHODS)’** 에서 제시한 분류와 각 방식의 주요 아이디어를 쉽게 풀어 정리한 것이다.

## 2.1 깨끗한/잡음 이미지 쌍 기반 학습  
- **개념**:  
  - 실제 깨끗한 영상(x)과 인위적으로 잡음을 입힌 대응 영상(y) 쌍을 대량으로 준비하여, 네트워크가   y→x를 복원하도록 지도학습(supervised learning)  
- **학습 손실 예시**:  
  
1) 직접 복원:  
    
$$\mathcal{L}(\theta)=\frac1{2N}\sum\|\!f(y_i;\theta)-x_i\|^2$$  
  
2) 잔차 학습:  
    
$$\mathcal{L}(\theta)=\frac1{2N}\sum\|\!R(y_i;\theta)-(y_i-x_i)\|^2$$  

- **대표 모델**:  
  - **DnCNN**: 잔차 학습 + 배치정규화 → 빠른 수렴 및 성능 개선  
  - **FFDNet**: 노이즈 레벨 맵 입력 → 다양한 잡음 강도·비균일 잡음 처리  
  - **DRUNet**: U-Net 구조 + ResNet 잔차 학습 → 최첨단 성능 달성  
- **장단점**:  
  - 장점: 합성 잡음(AWGN 등) 제거에 뛰어난 성능  
  - 단점: 실제(Real) 잡음 일반화 어려움, 대량의 깨끗한 데이터 필요, 모델 크기·계산량 큼  

## 2.2 다중 잡음 이미지 기반 학습  
- **개념**:  
  - ‘깨끗한 영상’ 없이, “동일 장면을 잡음이 다른 형태로 촬영한 복수의 잡음 영상 쌍”으로 지도학습  
- **학습 손실** (Noise2Noise 예시):  
  
$$\mathcal{L}(\theta)=\tfrac1N\sum\|f(x_i+n_i;\theta)-(x_i+n_i')\|^2$$  

- **대표 모델**:  
  - **Noise2Noise (N2N)**: 깨끗한 영상 없이도 잡음 간 예측 학습  
  - **GAN2GAN**: 잡음 생성기를 활용해 가상 잡음 쌍 합성  
- **장단점**:  
  - 장점: 깨끗한 데이터 없이도 학습 가능  
  - 단점: “동일 신호 + 서로 독립 잡음” 쌍 수집 제한적, 성능은 깨끗한 쌍 학습 대비 다소 낮음  

## 2.3 단일 잡음 이미지 기반 학습  
- **개념**:  
  - 오직 하나의 잡음 이미지 y만으로 네트워크를 학습  
  - *Deep Image Prior*, *Self2Self* 등 네트워크 구조·드롭아웃·데이터 증강으로 패턴 파악  
- **대표 모델**:  
  - **Deep Image Prior (DIP)**: 합성곱 네트워크 구조 자체가 이미지 통계(prior)를 내장함  
  - **Self2Self (S2S)**: 입력 픽셀 랜덤 마스킹+드롭아웃으로 자기-지도 학습  
- **장단점**:  
  - 장점: 어떠한 외부 데이터도 필요치 않음  
  - 단점: 복잡한 과적합 경향이 있어 ‘초기 중단(early stopping)’ 등 기법 필요, 성능 한계  

## 2.4 비전 트랜스포머 기반 모델  
- **개념**:  
  - *Transformer*의 self-attention으로 영상 내 전역적 픽셀 상호작용 학습  
- **대표 모델**:  
  - **IPT (Image Processing Transformer)**: ImageNet 기반 대규모 합성 데이터로 사전학습 후 다양한 복원 작업 수행  
  - **SwinIR**: Swin Transformer 블록 + 잔차 연결 → 고해상도 복원에 강점  
  - **Uformer/Restormer**: U-Net 형태·Transposed attention, 효율화된 윈도우 단위 self-attention  
- **장단점**:  
  - 장점: 넓은 수용 영역(global context) 포착, 다양한 잡음·복원 태스크에 확장성  
  - 단점: 모델 사전학습 비용 큼, 메모리·계산량 부담  

이와 같이 **학습 전략**에 따라 네 가지로 체계화하여, 각 그룹의 핵심 아이디어와 대표 모델, 장단점을 이해하면 이미지 잡음 제거 기법 전반의 구성을 쉽게 파악할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/22a46443-4f99-4df1-9d62-9a2f5acff2be/IET-Image-Processing-2023-Wu-Recent-progress-in-image-denoising-A-training-strategy-perspective.pdf
