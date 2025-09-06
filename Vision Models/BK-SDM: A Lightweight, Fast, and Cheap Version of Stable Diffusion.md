# BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion

**핵심 주장 및 주요 기여**  
이 논문은 대규모 Stable Diffusion 모델의 **U-Net 구조를 블록 단위로 제거**하고, **특징 수준 지식 증류(feature-level knowledge distillation)** 기법을 통해 **모델 크기 30∼50% 축소**, **연산량(MACs)·추론 지연 시간 30∼40% 절감**을 달성하면서도, 최소한의 리소스(0.22M 이미지-텍스트 쌍, 13 A100 GPU 일수)로 원본 모델과 경쟁 가능한 **제로샷 텍스트-이미지 생성 성능**을 유지함을 입증한다.

***

## 1. 해결하고자 하는 문제  
대규모 Stable Diffusion 모델은 수십억 개의 파라미터와 다수의 반복적 디노이징 단계로 인해 높은 계산 비용과 긴 추론 시간을 요구한다. 이는 모바일·엣지 디바이스에서의 활용을 제한하고, 연구자들이 제한된 자원으로 신속히 실험하기 어렵게 만든다.

## 2. 제안 방법  
### 2.1 구조적 블록 제거 (Block Pruning)  
- **다운·업 스테이지**의 잔차(residual) 및 어텐션(attention) 블록 쌍 중 두 번째 쌍 제거  
- **Mid-Stage 블록 전부 제거** (모델 크기 약 11% 추가 감소)  
- **가장 내부 스테이지**도 선택적으로 제거하여 최경량화 모델(BK-SDM-Tiny) 구현  
- 블록 제거 시 채널 차원 불일치 블록은 선형 보간 모듈로 대체  

### 2.2 지식 증류 기반 재학습 (Distillation Retraining)  
- **기본 손실**: $$\mathcal{L}_\mathrm{Task} = \mathbb{E}\big[ \|\epsilon - \epsilon_S(z_t,y,t)\|_2^2 \big] $$  
- **출력 수준 KD**:  

$$\mathcal{L}_\mathrm{OutKD} = \mathbb{E}\big[ \|\epsilon_T(z_t,y,t)-\epsilon_S(z_t,y,t)\|_2^2\big]$$  

- **특징 수준 KD**:  

$$\mathcal{L}_\mathrm{FeatKD} = \mathbb{E}\big[\sum_l\|f_T^l(z_t,y,t)-f_S^l(z_t,y,t)\|_2^2\big]$$  

- 최종 목적:  

$$\mathcal{L} = \mathcal{L}_\mathrm{Task} + \mathcal{L}_\mathrm{OutKD} + \mathcal{L}_\mathrm{FeatKD}$$  

### 2.3 모델 구조  
- BK-SDM-Base (0.76B 파라미터), Small (0.66B), Tiny (0.50B)  
- SDM-v1.4 및 SDM-v2.1-base에서 동일 U-Net 설계 적용  
- 제거 위치 및 수는 블록별 민감도 분석을 통해 결정  

## 3. 성능 향상 및 한계  
### 3.1 성능 개선  
- **제로샷 MS-COCO (256×256, 30K 프롬프트)**:  
  - BK-SDM-Base: FID 15.76, IS 33.79, CLIP 0.2878  
  - 원본 SDM-v1.4 대비 파라미터 27% 감소, 지연 30% 감소  
- **소규모 자원(0.22M 데이터, 13 A100 일수)로** 대규모 모델과 유사 성능 확보  
- **엣지 디바이스**: Jetson AGX Orin 4초 이내, iPhone 14에서 4초 미만 추론  
- **하위 작업**: DreamBooth 개인화, 이미지→이미지 변환, 얼굴 생성 등에 성공적 전이  

### 3.2 일반화 성능 향상 가능성  
- **특징 수준 KD**를 통해 내부 표현 학습 강화 ⇒ 텍스트-이미지 정합도 향상  
- 블록 제거 후에도 **교차 어텐션 패턴**이 교사 모델과 유사하게 유지되어, 새로운 도메인·스타일로의 전이 학습에서 일반화 성능 저하 최소화  
- 민감도 분석 기반 제거로 **중요 정보 손실 최소화**, 소량 데이터로도 안정적 학습  

### 3.3 한계  
- **전신 인물 생성**에서 부정확성 관찰  
- 극한 경량화(Tiny) 시 성능 저하 폭 발생  
- 리소스 절감과 품질 사이의 **트레이드오프** 존재  

## 4. 향후 연구 영향 및 고려 사항  
- **대규모 생성 모델 압축 연구**에 구조적 블록 제거와 특징 수준 KD의 시너지를 입증, 후속 연구에서 다양한 기법(양자화, 샘플링 축소)과 결합 가능  
- **엣지 AI** 구현을 위한 경량화 전략 제시  
- 앞으로는  
  - **인물·복잡한 장면** 생성 품질 개선  
  - **자동 블록 선택·동적 경량화** 기법  
  - **더 작은 데이터셋**으로도 뛰어난 일반화 달성 방안 모색이 중요하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a6682df6-ead5-47d3-b71e-6322ed549ebf/2305.15798v4.pdf)
