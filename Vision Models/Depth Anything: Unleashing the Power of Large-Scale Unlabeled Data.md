# Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data | Depth estimation

## 1. 핵심 주장 및 주요 기여  
Depth Anything은 **단일 이미지로부터의 깊이 추정**(Monocular Depth Estimation, MDE)에서 기존의 소규모 고가 레이블링 데이터 의존을 넘어, 인터넷·공개 데이터에서 무라벨(monocular unlabeled) 이미지를 대규모(62M장)로 자동 주석(pseudo-label)하여 활용함으로써 **모델의 제로샷(Zero-Shot) 일반화 성능**을 획기적으로 개선할 수 있음을 입증했다[1]. 주요 기여는 다음과 같다[1]:  
- 대규모 무라벨 데이터 자동 레이블링 엔진 설계 및 결합  
- 강도 높은 데이터 증강 및 CutMix 기반의 **더 어려운 학습 목표 설정**  
- DINOv2 기반 **연속 특징(feature) 정렬**으로 심층적인 시맨틱(prior) 정보 계승  

## 2. 해결하고자 하는 문제  
- 다양한 환경·도메인(실외·실내·저조도·안개 등)에서 단일 RGB 이미지로부터 깊이 맵을 추정하는 **범용 MDE 모델의 부족**  
- 고가의 LiDAR·스테레오·SfM 기반 레이블 수집의 한계와 소규모 레이블 데이터만으로는 **새로운 장면에 대한 일반화 한계**  

## 3. 제안 방법  
### 3.1 데이터 스케일링 및 자기학습(Self-Training)  
1. 1.5M장 공공 레이블 데이터로 **교사 모델** $$T$$ 학습  
2. $$T$$를 이용해 62M 무라벨 이미지에 **의사 깊이 라벨** $$\hat{\mathcal{D}}^u=\{(u_i,T(u_i))\}$$ 생성  
3. 교사·의사 라벨 데이터를 배치 비율 1:2로 병합하여 **학생 모델** $$S$$ 재학습[1]

### 3.2 더 어려운 최적화 목표  
- 무라벨 이미지에 **강한 색상 왜곡**(color jitter, Gaussian blur) 및 **공간 왜곡(CutMix)** 적용  
- CutMix를 통해 두 이미지 $$u_a,u_s$$를 마스크 $$M$$로 합성 후,  
  $$L_u = \alpha_M\,\rho\bigl(S(U_{ab})\odot M,\,T(u_a)\odot M\bigr) + \alpha_{1-M}\,\rho\bigl(S(U_{ab})\odot(1-M),\,T(u_s)\odot(1-M)\bigr)$$[1]  
- 이를 통해 학생 모델이 **강인한 표현**을 획득  

### 3.3 시맨틱 정합을 통한 정보 상속  
- DINOv2의 **고차원 연속 특징** $$f'$$를 고정(frozen)하고, 학생 특징 $$f$$와의 **코사인 유사도**로 정렬  
- 허용 오차 마진 $$\alpha$$를 두어, $$\cos(f_i,f'_i)>\alpha$$인 픽셀은 손실에서 제외[1]  
- 손실 함수:  
  $$L_{\text{feat}} = 1 - \frac{1}{HW}\sum_{i=1}^{HW}\mathbb{I}\bigl(\cos(f_i,f'_i)<\alpha\bigr)\,\cos(f_i,f'_i)$$

전체 손실: 

$$
L = L_{l} + L_u + L_{\text{feat}}
$$

$L_{l}$ : affine-invariant loss

## 4. 모델 구조  
- **인코더**: DINOv2 사전학습 ViT (S/B/L)  
- **디코더**: DPT 기반 깊이 회귀 구조  
- 교사 모델 학습 후 학생 모델을 **무작위 재초기화**하여 학습  

## 5. 성능 향상  
- **제로샷 상대 깊이 추정**: 6개 공개 데이터셋 평균 AbsRel 0.076 → 0.066, δ₁ 0.947 → 0.984 향상[1]  
- **하위 모델(ViT-S)**도 MiDaS 대형 모델 초월  
- **Metric depth 추정**(NYUv2/KITTI): δ₁ 각각 0.964 → 0.984, 0.978 → 0.982[1]  
- **시맨틱 분할 전이**: Cityscapes mIoU 84.3 → 86.2, ADE20K 58.3 → 59.4[1]  

## 6. 한계  
- 모델 최대 크기 ViT-Large 수준에 그침  
- 학습 해상도 512×512에 제한  

## 7. 향후 연구 영향 및 고려사항  
Depth Anything은 **대규모 무라벨 데이터 활용**과 **학습 난이도 조절**이 MDE 일반화 성능을 획기적으로 높일 수 있음을 보였다. 향후 연구에서는  
- ViT-Giant 등 **더 큰 모델 규모** 적용  
- **해상도 확장(700~1000+)**을 통한 정밀도 향상  
- 시맨틱 정렬 기법의 **다른 연속 표현**(예: CLIP, SAM)으로의 확장 가능성을 고려해야 한다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b606f226-16b2-4e74-81c5-5b8decffada1/2401.10891v2.pdf
