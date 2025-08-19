# FP-DETR: Detection Transformer Advanced by Fully Pre-training | Object detection

## 1. 핵심 주장 및 주요 기여
FP-DETR는 **객체 검출용 트랜스포머 핵심 모듈(12-layer encoder-decoder)을 ImageNet 분류 과제에서 완전하게 사전학습**하고, 시각적 프롬프트(task adapter)를 통해 객체 검출에 원활히 전이시킴으로써  
- **검출 정확도 유지**  
- **공통 왜곡에 대한 강인성 강화**  
- **소규모 데이터셋에서의 일반화 능력 향상**  
을 동시에 달성함을 보였다.[1]

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제
기존 DETR 계열 모델들은  
1) **백본만** 이미지넷 사전학습하고 트랜스포머 모듈은 무작위 초기화 → 대규모 데이터 필요  
2) 분류와 검출 과제 간 구조·목표 차이(discrepancy) 존재 → 전이 학습 비효율  
라는 한계로 인해, 강인성 및 일반화 성능이 제한됨.

### 2.2 제안 모델 구조
1. **Encoder-only 사전학습**  
   - 디코더 제거 후 멀티스케일 MS-Deformable Self-Attention 기반 12-layer encoder 사용  
   - 백본 대신 간단한 **멀티스케일 토크나이저**(stride 8→2→2→2 convolution)로 피처 추출  
2. **Visual Prompting via Task Adapter**  
   - 객체 쿼리 위치 임베딩 $$p_i$$를 “시각적 프롬프트”로 해석  
   - 프롬프트 간 상호작용 캡처를 위한 **Self-Attention 기반 Task Adaptor** 삽입  
     
   Fine-tuning 입력:

```math
     z_0 = [\,x_i^{obj}+p_i\,]_{i=1}^{N_q}\;\Vert\;[\,x_j\,]_{j=1}^{N}
```
   
   Task Adaptor:

```math
     z_{0:N_q}' = \mathrm{SelfAttn}(z_{0:N_q}),\quad
     z_t = \mathrm{EncoderLayer}_t(z_{0:N_q}',\,z_{>N_q})
```

### 2.3 성능 향상
- COCO val: **Base** 모델이 43.3 mAP 획득(ResNet-50+Deformable DETR 대비 동등)  
- COCO-C(Common Corruptions): 15/15 corruptions 중 14종목 최상위 성능[1]
- Cityscapes(소규모): 29.6 mAP로 동일 파라미터군 중 최고[1]

### 2.4 한계
- 프롬프트(쿼리 위치 임베딩) 및 어댑터만 추가 미세조정 → zero-/few-shot 전이 학습은 미지원  
- Encoder-Decoder 사전학습 시도 시 정확도 저하 → 구조 일반성 부족  
- 사전학습·미세조정 간 추가 최적화 필요성

## 3. 일반화 성능 향상 기전
1. **전 모델 완전 사전학습**: 객체 간 관계 학습 없이 사전학습된 상태에서 Task Adaptor를 통해 관계 모델링을 보완  
2. **간소화된 토크나이저 + MS-Deformable Attention**: 소규모 데이터에서도 효과적 피처 추출  
3. **시각적 프롬프트**: 위치 정보 제공으로 분류 과제 지식의 공간 전이 촉진  
→ **소량 데이터·왜곡 조건에서 일반화 우수**[1]

## 4. 향후 연구 방향 및 고려사항
- **Few-/Zero-Shot 전이**: 프롬프트 튜닝만으로 다양한 검출 과제로 확장(영역 지정 없이 분류만)  
- **Encoder-Decoder 사전학습 기법 개선**: 양방향 토크나이저 결합, 복수 클래스 토큰 활용 연구  
- **프롬프트 구조 최적화**: 비선형 어댑터, 레이어별 특화 어댑터로 효율성·정확도 균형  
- **대규모·다양 도메인 사전학습**: 영상·의료·위성 등 특수 분야 일반화 성능 평가  

이상의 연구 고찰을 바탕으로 **Detection Transformer의 사전학습-미세조정 패러다임**이 더욱 발전할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cd5218ad-5d84-4e08-be82-fa43d391a09d/750_fp_detr_detection_transformer.pdf
