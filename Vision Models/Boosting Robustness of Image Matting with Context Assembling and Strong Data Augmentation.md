# Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation | Image matting

## 1. 핵심 주장 및 주요 기여  
본 논문은 **이미지 매팅(image matting)**의 견고성(robustness)을 크게 향상시키기 위해 두 가지 핵심 요소를 결합한다.  
- **멀티레벨 컨텍스트 어셈블링(Context Assembling)**: 비전 트랜스포머를 활용해 전역(global) 컨텍스트 정보를, 합성곱(convolution) 레이어를 통해 국부(local) 디테일을 모두 포착하는 이중-브랜치(dual-branch) 인코더 설계를 제안.  
- **강력한 데이터 증강(Strong Data Augmentation, SA)**: 합성 데이터와 실제 도메인 간 차이를 줄이기 위해 세 가지 전략(AF, AFB, AC)을 체계적으로 설계·조합.  

이를 통해,  
1) 벤치마크(Composition-1k)에서 기존 대비 SAD 11%·Grad 27% 개선,  
2) 트리맵(trimap) 정밀도 변화에 강건,  
3) 다수의 합성·실제 이미지 벤치마크에서 뛰어난 일반화 성능을 달성하였다.  

## 2. 문제 정의 및 제안 방법  

### 2.1 해결하고자 하는 문제  
- 매팅의 **트리맵 정밀도 의존성**: 사용자가 그린 trimap의 형태·두께 변화에 민감해 정확한 브러싱이 요구됨.  
- **도메인 갭(Domain Gap)**: 합성 데이터로만 학습된 모델이 실제 이미지로는 성능이 크게 저하됨.  

### 2.2 모델 구조  
1) **이중-브랜치 인코더**  
   - Transformer branch: Pyramid Vision Transformer 기반으로 전역 컨텍스트 학습  
   - Convolution branch: 2-stride 합성곱으로 로컬 디테일 보완  
2) **디코더 & 스킵 연결**  
   - 디코더는 MLP 및 합성곱 레이어로 구성하고, 트랜스포머·컨볼루션 브랜치의 특징 맵을 별도 스킵 연결(TSkip, LSkip)  
   - 추가로, 디코더 중간 해상도에 Low-level Feature Assembling Attention(LFA) 블록 삽입  
3) **손실 함수**  
   - L1 예측 손실과 합성 손실에 더해, Laplacian 손실(llap) 및 제안된 gradient penalty 손실(lgp)을 도입  

### 2.3 수식 개요  
- 매팅 기본 방정식:  
  
$$ I_i = \alpha_i F_i + (1 − \alpha_i) B_i $$  

- LFA self-attention:  
  
$$ \text{Attn}(f_{\text{low}}, f_{\text{low}}, f_{\text{dec}}) $$  

- Gradient penalty 손실:  
  
$$ l_{gp} = \|\nabla \alpha - \nabla \hat\alpha\|_1 + \lambda (\|\nabla \alpha\|_1) $$  

## 3. 성능 향상 및 일반화 실험  

### 3.1 벤치마크 성능  
- Composition-1k: SAD 22.87 → 26.4 → 25.0, Grad 7.74 → 10.6 → 9.02 (M7‡ 모델)  
- alphamatting.com 온라인 평가: MSE·Grad 모두 상위권 달성  

### 3.2 일반화 성능  
- Distinction-646, SIMDour, AIM-500 벤치마크에서 모두 기존 대비 전체 지표 개선  
- 특히 **실제 이미지 AIM-500**에서 SA 미적용 대비 Grad 23.68 → 13.06으로 대폭 향상  
- Trimap 정밀도(2px–50px) 변화 실험에서 일관된 안정성 확보  

## 4. 일반화 성능 향상 전략  

1) AF (Augment Foreground Alone): 전경에만 픽셀 단위 선형 증강 적용  
2) AFB (Augment Foreground & Background): 전·배경 개별 선형 증강  
3) AC (Augment Composite): 합성 이미지에 비선형·지역 증강 적용 후 의사 라벨 생성  

이들 전략을 조합(AF+AFB+AC)하여 합성 데이터 정확도 저하 없이 실제 도메인 성능을 획기적으로 개선  

## 5. 한계 및 향후 고려 사항  
- **극단적 조명, 복잡한 얇은 구조** 등 일부 실제 사례에서 매팅 실패 사례 존재  
- 강력한 데이터 증강에도 **진정한 실제 알파 매트 라벨** 부재로 인한 한계  
- 전경 구조 이해를 위한 **메타 지식 학습** 또는 **합성 파이프라인 다양화** 필요  

## 6. 향후 연구에 미치는 영향 및 고려할 점  
- **컨텍스트 어셈블링**과 **도메인 일반화**를 결합한 첫 사례로, 이후 매팅 외 다른 화상 분할 분야에도 확장 가능성  
- 증강 전략 설계 시 합성 방정식 준수 여부를 고려해야 함을 강조  
- 추후 실제 라벨 확보 또는 **자기 지도 학습(self-supervised)** 기법과 결합하여 도메인 갭 최소화 연구 필요

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/bd151639-7914-4762-b301-1dc1d47770ae/2201.06889v1.pdf
