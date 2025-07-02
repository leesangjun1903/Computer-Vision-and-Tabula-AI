
**StyleGAN2** 논문은 기존 StyleGAN의 대표적 이미지 품질 저하 원인을 규명하고, 이를 개선하기 위한 새로운 아키텍처·정규화 기법·학습 방식을 제안하여 무조건적 이미지 생성의 최첨단을 한 단계 끌어올린다[1].

## 1. 문제 정의
기존 StyleGAN(이하 S1)에서 관찰되는 주요 한계는 다음과 같다[1]:
- **물방울 형태(blob) 인공물**: 중간 피처맵에서 수집되는 국소적 통계 왜곡으로 인한 시각적 노이즈  
- **점진적 해상도 성장(progressive growing) 부작용**: 중간 계층에 과도한 주파수 성분이 축적되어 구조적 불안정 및 위치 편향(phase artifact) 발생  
- **잠재 공간 매핑의 불규칙성**: 이미지 품질 지표와 무관하게 경로 길이(perceptual path length, PPL)가 긴 경우(불연속적 변화) 이미지 왜곡  

## 2. 제안 기법
### 2.1. 가중치 **Demodulation**
- 스타일 블록의 Adaptive Instance Normalization(AdaIN)을 제거하고, 스타일 계수 $$s_i$$를 곱한 후  
  convolution 가중치 $$w_{ijk}$$를 조정:  
  1) Modulation: $$w'\_{ijk} = s_i \cdot w_{ijk}$$  
  2) Demodulation: $$w''\_{ijk} = \frac{w'\_{ijk}}{\sqrt{\sum_{i,k}{w'_{ijk}}^2 + \epsilon}}$$  
- 실제로 그룹화된 convolution으로 효율적 구현[1]. 이로써 **blob artifact** 완전 제거 및 FID 유지

### 2.2. **Path Length Regularization (PPL)**
- 생성기 매핑 $$g: W\to Y$$가 고차원 가우시안 노이즈 $$y\sim\mathcal N(0,I)$$에 대해 야코비안 $$J_w$$의 스펙트럼을 균등화하도록 유도  
- 정규화 항:

$$
    L_{\text{pl}} = \mathbb E_{w,y}\Bigl(\lVert J_w^T y\rVert_2 - a\Bigr)^2,
$$

  여기서 $$a$$는 $$\mathbb E_{w,y}\lVert J_w^T y\rVert_2$$의 이동 평균[1].  
- 결과: PPL 대폭 감소(매핑 완만화)로 인한 주관적 화질 개선

### 2.3. **모델 아키텍처 재설계 (No Growing)**
- Skip 연결 기반의 멀티해상도 RGB 합성(generator) + 잔차(residual) 판별기 도입  
- 점진적 해상도 성장 없이도 학습 초기 저해상도에 집중 → 차츰 고해상도 세부묘사 학습  
- 네트워크 용량 확대(고해상도 레이어 채널 수 2배)로 진정한 1024×1024 디테일 활용  

## 3. 성능 개선 및 한계
| 구성          | FFHQ FID↓ | PPL↓  | Precision↑ | Recall↑ |
|--------------|-----------|-------|------------|---------|
| S1 (Baseline)[1]    | 4.40      | 212.1 | 0.721      | 0.399   |
| +Demodulation      | 4.39      | 175.4 | 0.702      | 0.425   |
| +Lazy Reg.         | 4.38      | 158.0 | 0.719      | 0.427   |
| +PPL Reg.          | 4.34      | 122.5 | 0.715      | 0.418   |
| +No Growing (Res)  | **3.31**  | 124.5 | 0.705      | 0.449   |
| +Large Nets (S2)   | **2.84**  | 145.0 | 0.689      | 0.492   |

- **FID**: 기존 4.40 → 2.84 (35% 개선)  
- **PPL**: 212 → 145 (31% 개선)  
- **Recall**: 0.399 → 0.492 (23% 개선)  
- **한계**: PPL 최적화 시 Recall 감소 경향; 데이터셋별 γ(R1) 튜닝 필요[1]

## 4. 일반화 성능 개선 관점
- **매핑 완만화(PPL Reg.)**로 작은 노이즈 변화에도 **구조적 일관성** 유지  
- **Demodulation**이 스타일 믹스 일반화(지나친 채널 증폭 억제) 유도  
- **Skip+Residual 아키텍처**가 shift-invariance 및 다양한 데이터셋(LSUN Car/Cat/Church/Horse)에서 안정적 품질 향상[1]

## 5. 향후 연구 영향 및 고려사항
- **영향**: 데이터 효율적 고품질 GAN 개발, 잠재 공간 해석·편집 기술 고도화 촉진  
- **고려사항**:  
  1. PPL 정규화와 **특정 응용**(예: 소규모 데이터셋) 간 트레이드오프 분석  
  2. 스펙트럼 정규화·대체 노멀라이제이션 조합 연구  
  3. **특징 기반 거리(metrics)** 활용한 정규화(픽셀→피처 공간)로 추가 일반화 가능성 탐색[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/db611b26-d7ef-44b4-8e02-bc34add6c680/1912.04958v2.pdf
