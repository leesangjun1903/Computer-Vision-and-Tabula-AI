# FastGAN : Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis | Image generation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**:  
소량(수십~수천 장)의 고해상도(256², 1024²) 이미지와 제한된 연산 자원(single GPU, 수시간) 환경에서도 안정적으로 고품질 합성 이미지를 생성할 수 있는 경량 GAN 아키텍처를 제안한다.

**주요 기여**:  
1. **Skip-Layer Channel-Wise Excitation (SLE)** 모듈  
   – 저해상도 특성맵($x_{low}$)과 고해상도 특성맵($x_{high}$)를 채널별 게이팅(gating)으로 결합하여(수식 (1))  
   – 긴 범위(skip-connection)에서 정보 흐름을 강화, 스타일/콘텐츠 자동 분리 유도  
2. **Self-Supervised Discriminator (SSD)**  
   – 판별기를 인코더로 간주해 소규모 디코더로 재구성 손실(식 (2))을 추가  
   – 입력 이미지의 전체 영역 특징을 포괄적으로 추출하도록 정규화  
3. **경량화된 전체 GAN 구조**  
   – 고해상도(1024²)에서 단일 conv-layer × 각 레벨, 고해상도 채널=3  
   – 수시간—수십시간(single RTX-2080) 학습 가능  

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제  
- **Few-Shot**: 100장 이하 데이터로도 모드 붕괴 없이 학습  
- **Low VRAM & Fast Training**: 한정된 GPU 메모리(8 GB)·시간(~10 시간)으로 1024² 해상도 학습  
- **High-Fidelity**: StyleGAN2 수준 화질 유지

### 2.2 제안 방법

#### A. Skip-Layer Channel-Wise Excitation (SLE)  
- 수식 (1):  

$$
y \;=\; F(x_{\text{low}}, W) \odot x_{\text{high}}
$$ 

  -  $$F$$: 평균 풀링→1×1 Conv→LeakyReLU→1×1 Conv→Sigmoid  
  -  $$\odot$$: 채널별 곱  
- 긴 범위 skip-connection으로 잔차 흐름 확보와 스타일 콘텐츠 분리  

#### B. Self-Supervised Discriminator (SSD)  
- 판별기 D를 인코더로, 소형 디코더 $$G_{\text{dec}}$$를 추가  
- 재구성 손실(식 (2)):  

$$
\mathcal{L}\_{\text{recons}}
= \mathbb{E}\_{x \sim I_{\text{real}}}\bigl[\|\,G_{\text{dec}}(D_{\text{feat}}(x)) - T(x)\|\bigr]
$$  

  -  두 스케일(16², 8²)의 특징맵→각각 전체·부분 이미지 재구성  
  -  GAN 훈련 시 판별기 손실에 더해 과적합 억제  

#### C. 전체 학습 목표  
- Hinge loss 기반 GAN 학습(식 (3),(4))  
- 판별기: 진위 분류 + 재구성 정규화  
- 생성기: 판별기 판별 결과 최대화

### 2.3 모델 구조  
- **Generator**: Progressive up-sampling, 각 해상도에 단일 conv, 512→256→…→3 채널, SLE 삽입  
- **Discriminator**: Residual down-sampling 블록, 2개 스케일 특징 추출 후 실/가짜 판별, self-supervised 디코더  

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **Few-Shot FID**: 256² 데이터셋 5종 중 4종 최고, 1024² 데이터셋 7종 중 6종 최저 FID 달성  
- **학습 속도**: StyleGAN2 대비 2–4× 빠른 수렴; 1024²에서 10 시간 내 학습 가능  
- **안정성**: 20 시간 학습에도 모드 붕괴 없이 유지  
- **일반화**: SSD로 판별기 과적합 완화, 역추적(inversion) LPIPS 감소  

### 3.2 한계  
- **데이터 규모**: 70k FFHQ 등 대규모 데이터에서 StyleGAN2에 여전히 열세  
- **복잡도**: 단순 구조로 복잡한 도메인(다양한 배경 포함 얼굴)에서 스타일 분리 성능 저하  
- **Self-Supervision 선택**: 대조학습‧회전 예측 결합 시 정규화 성능 저하 관찰  

## 4. 일반화 성능 향상 관점  
- **판별기 재구성 정규화**가 단일 도메인 과적합 억제  
- **SLE**의 스타일 콘텐츠 분리는 적은 데이터 환경에서도 비슷한 패턴 전이 가능  
- **역추적(inversion)**에서 LPIPS 개선: 256² (2.11→1.82), 1024² (2.17→1.92) 감소  
- 여러 도메인(자연, 회화, 애니메이션)에 걸쳐 일관된 이득 확인  

## 5. 향후 연구 영향 및 고려사항  
- **플러그인형 모듈**: SLE·SSD를 다른 GAN에 적용해 “Few-Shot” 특화 확장 가능  
- **조합적 정규화**: 다양한 self-supervision 기법 조합 최적화 필요  
- **대규모 일반화**: 대규모·다양도메인 데이터에 대한 확장성·안정성 평가  
- **고해상도 콘텐츠 편집**: 경량 모델의 실시간 인버전·인터랙티브 편집 응용 연구  

> *이 논문은 최소 데이터·연산 환경 하에서도 고해상도 GAN 학습을 가능케 함으로써, 저자원 환경 애플리케이션(예: 예술가 개인 데이터 기반 이미지 생성, 의료 희귀질환 영상 합성)에 새로운 길을 열었다.*

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/31f3710d-8181-4b0a-a5a4-274f62ee7523/2101.04775v1.pdf
