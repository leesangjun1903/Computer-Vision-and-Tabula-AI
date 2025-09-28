# AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model | Anomaly Detection

**주요 메시지:**  
AnomalyDiffusion는 소수의 이상 샘플만으로도 고품질·다양성 높은 이상 이미지와 정확히 정렬된 이상 마스크 쌍을 생성하여 downstream 이상 검사(탐지·위치추정·분류) 성능을 크게 향상시킵니다.[1]

## 1. 간결 요약  
AnomalyDiffusion는 대규모 데이터로 사전학습된 Latent Diffusion Model(LDM)의 강력한 사전지식을 활용해 소수의 이상 샘플(few-shot)만으로도  
-  이상 유형과 위치 정보를 분리해 표현하는 **Spatial Anomaly Embedding**  
-  생성 중 미미한 이상 영역에 더 집중하도록 유도하는 **Adaptive Attention Re-weighting Mechanism**  
을 도입함으로써, 생성 진위(authenticity)와 다양성(diversity)을 모두 개선하고 이상 검사 downstream 과제에서 SOTA를 경신합니다.[1]

## 2. 문제 정의 및 해결 방안  
### 2.1 해결하고자 하는 문제  
산업 생산에서는 이상 샘플이 매우 희소하여 이상 탐지·위치추정·분류 성능이 제한됩니다. 기존의  
-  패치 붙여넣기(crop-paste) 방식: 비현실적 합성  
-  GAN 기반 소수샷: 마스크와 정확히 정렬되지 않음  
문제점을 극복해야 합니다.[1]

### 2.2 제안 방법  
1) **Spatial Anomaly Embedding**  
   - 이상 appearance를 나타내는 **anomaly embedding** $$e_a$$  
   - 이상 위치를 나타내는 **spatial embedding** $$e_s$$  
   - 마스크 $$m$$를 ResNet-50+FPN으로 인코딩해 위치 정보 분리  
2) **Adaptive Attention Re-weighting**  
   - 디노이징 단계별 생성물 $$x_0$$와 정상샘플 $$y$$ 차이에 기반한 가중치 맵  

$$
       w_m = m \odot \mathrm{Softmax}\bigl[f_m(y\odot m - x_0\odot m)^2\bigr]
     $$  
   
  - cross-attention map $$M_c$$ 재가중치:  

$$
       \widetilde M_c = M_c \odot w_m
     $$  
   
이를 통해 채워지지 않은 영역에 더 큰 생성력을 부여합니다.[1]

### 2.3 모델 구조  
LDM 기반 U-Net 디노저에, 텍스트 조건으로 concat된 $$[e_a,e_s]$$를 cross-attention 모듈에 입력하며, 인버스 과정 중 어텐션 가중치를 동적으로 재조정합니다.[1]

## 3. 성능 향상 및 한계  
### 3.1 성능 향상  
-  **생성 진위·다양성**: MVTec 데이터셋에서 Inception Score, IC-LPIPS 모두 SOTA 달성.[1]
-  **이상 위치추정**: U-Net trained on generated data로 픽셀 수준 AUROC 99.1%, AP 81.4%.[1]
-  **분류 정확도**: ResNet18 분류에서 평균 66.1%로 기존 모델 대비 +16.5%p 우위.[1]

### 3.2 한계  
- **해상도 의존**: LDM 기반으로 해상도 확장에 한계  
- **마스크 필요**: 마스크 생성 모듈 있으나, 실제 다양성에 의존  
- **계산 비용**: few-shot 학습에도 상당한 디노이저 업데이트 필요

## 4. 일반화 성능 향상  
AnomalyDiffusion은 소수(≈30%) 이상율에서도 안정적인 localization 성능을 보이며, 더 적은 샘플(10% 이하)에서는 AP 감소가 두드러지지만 30% 이상부터 수렴하는 특성을 보입니다.[1]
이는 **Spatial Anomaly Embedding**이 location 정보 오버피팅을 방지하고, **Adaptive Attention**이 영역별 어려움 차이를 보정하기 때문입니다.

## 5. 향후 연구 방향 및 고려사항  
- **고해상도 확장**: 더 강력한 diffusion backbone 적용  
- **무마스크 생성**: 마스크 없이 위치 제어 가능한 무감독적 접근  
- **실제 공정 데이터 평가**: 복잡한 산업 공정 이상에 대한 일반화 실험  
- **효율성 개선**: 경량화와 학습 단계 계산 비용 절감  

이러한 과제 해결 시, AnomalyDiffusion의 강점을 더욱 극대화할 수 있을 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/378d9d19-9266-47ad-a6b3-7fce7d2450c3/2312.05767v2.pdf)
