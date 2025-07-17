# HMANet: Hybrid Multi-Axis Aggregation Network for Image Super-Resolution | Super resolution

## 1. 핵심 주장 및 주요 기여  
HMANet은 윈도우 기반 트랜스포머의 제한된 수용 영역을 극복하고, 이미지의 **지역적 (local)** 및 **전역적 (global)** 자기유사성(self-similarity)을 효과적으로 결합하여 초고해상도(single-image super-resolution, SISR) 성능을 크게 향상시킨다.  
주요 기여  
1. **Residual Hybrid Transformer Block (RHTB)**: 채널 어텐션과 윈도우/이동 윈도우(self-attention)를 융합하여 비지역(non-local) 특징 융합 강화.  
2. **Grid Attention Block (GAB)**: 이미지 전체의 유사한 패치들 간 교차 영역(grid) 어텐션을 도입하여 수용 영역 확대.  
3. **맞춤형 사전학습(pre-training) 전략**: ×2, ×3, ×4 배율 모델 간 파라미터 공유 방식으로 초기화를 계단식으로 수행, 소량의 추가 비용으로 표현력 대폭 제고.  
4. 다양한 벤치마크(SET5, SET14, BSD100, Urban100, Manga109) 및 NTIRE 2024 챌린지에서 SOTA 대비 PSNR 0.1∼1.4 dB 개선.

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- 윈도우 기반 Transformer(SwinIR 등)는 고정된 국소 윈도우 내에서만 자기어텐션을 수행하여 **긴 거리 의존성**과 입력 이미지의 **전역적 유사성**을 충분히 활용하지 못함.  
- GAN이나 참조 기반 SR은 추가 입력 시 과도하거나 불일치 시 왜곡 발생.

### 2.2 모델 구조  
HMANet은 입력 $$I_{LR}\in\mathbb{R}^{H\times W\times C_{in}}$$에 대해  
1. **Shallow Feature Extraction**:  

$$
F_0 = H_{\text{Conv}}(I_{LR})
$$

2. **Deep Feature Extraction**:  
   $$F_0$$를 $$M$$개의 RHTB와 3×3 Conv로 처리, 잔차 연결(residual)  
3. **Image Reconstruction**:  

$$
     I_{HR} = H_{\text{Rec}}(F_D + F_0)
$$

#### 2.2.1 Residual Hybrid Transformer Block (RHTB)  
- **Fused Attention Block (FAB)**:  
  - 3×3 컨볼루션 기반 inverted bottleneck + Squeeze-Excitation → 레이어 노멀화 → (윈도우 W-MSA + 이동 윈도우 SW-MSA)×2 → MLP.  
- **Grid Attention Block (GAB)**:  
  - 채널 분할: $$F_{in}\to [F_G,F_W]$$.  
  - $$F_W$$는 W-MSA와 SW-MSA, $$F_G$$는 **Grid-MSA**로 처리.  
  - Grid-MSA: 전체 피처 맵을 $$K\times K$$ 간격의 그리드로 샘플링하여 유사 패치 간 어텐션  
  
$$
    \hat{X} = \mathrm{Softmax}\bigl(\tfrac{GK^\top}{\sqrt{d}}+B\bigr)V,\quad
    \mathrm{Attention}(Q,G,\hat{X})=\mathrm{Softmax}\bigl(\tfrac{QG^\top}{\sqrt{d}}+B\bigr)\hat{X}
$$

  - 포스트‐노멀(post-norm) 설계로 안정화.

### 2.3 사전학습 전략  
1. ImageNet에서 ×2 모델 학습 → 초기 파라미터 획득  
2. 이를 ×3 모델 초기화에 사용하여 학습 →  
3. 다시 ×3 모델 파라미터로 ×2·×4 모델 재초기화→ DF2K로 전체 파인튜닝  
→ PSNR 0.05∼0.09 dB 추가 향상.

## 3. 성능 향상 및 한계  
| 데이터셋   | SwinIR 대비 PSNR 향상 (×4 SR) |
|------------|------------------------------|
| Set5       | +0.23 dB                    |
| BSD100     | +0.17 dB                    |
| Urban100   | +0.97 dB                    |
| Manga109   | +0.37 dB                    |

- **전역 유사성 활용**: Urban100·Manga109의 반복 패턴 복원에서 큰 개선.  
- **채널+그리드 어텐션 시너지**: SSIM도 동반 상승.  
- **계산 비용**: 파라미터 69.9 M, Multi-Adds 170.1 G로 대형 모델.  

한계  
- 대규모 연산량 및 메모리 요구.  
- 매우 낮은 배율(×8 이상) 일반화 미검증.  

## 4. 모델 일반화 성능 향상 가능성  
- 그리드 어텐션은 **이미지 내 자기유사성**을 이용하므로, 저해상도 왜곡 유형(노이즈, 압축 열화)에도 확장 적용 가능.  
- 채널 어텐션＋그리드 어텐션 결합은 도메인 전이(자연→의료영상) 시도 시 과적합 방지에 유리.  
- 단계적 사전학습은 **다중 열화 수준**에 대한 적응을 강화하여, 실제 촬영 환경에서 다양한 저화질 입력에 견고.

## 5. 향후 연구 방향 및 고려사항  
- **효율화**: 대용량 연산 경량화(프루닝, 양자화, 지식 증류)로 실시간/엣지 적용.  
- **더 높은 배율 및 복합 열화**: ×8 SR, 노이즈·블러 복합 사례에서 그리드 어텐션 효과 분석.  
- **도메인 일반화**: 의료·위성·비전텍스트 등 도메인별 자기유사성 특성에 맞춘 그리드 설계 최적화.  
- **비전-텍스트 다중모달**: 이미지-설명 상호참조 참조(super-resolution with textual guidance)에 그리드 어텐션 확장.  
- **표준 벤치마크 외 검증**: 실제 사진 원본(raw) 데이터셋 및 실험실 외 이미지 확대 검증.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/57156575-e53d-43da-a250-35cb7215dc79/2405.05001v1.pdf
