# Focal Self-attention for Local-Global Interactions in Vision Transformers | Image classification, Object detection, Semantic segmentation

**주요 주장:**  
Focal Self-Attention은 Vision Transformer에서 지역적(fine-grained) 및 전역적(coarse-grained) 상호작용을 동시에 효율적으로 포착함으로써, 고해상도 이미지 처리 시 기존의 전 범위 self-attention의 계산·메모리 부담을 크게 줄이면서 모델 성능을 일관되게 향상시킨다.

**주요 기여:**  
1. 지역 영역에는 세밀한 self-attention, 먼 영역에는 요약된 토큰을 활용한 전역 self-attention을 결합한 새로운 메커니즘 제안  
2. Multi-scale 아키텍처에 focal self-attention을 적용한 Focal Transformer 모델 제시  
3. ImageNet 분류(Top-1 83.8%), COCO 객체 검출(box mAP 58.9%), ADE20K 의미 분할(mIoU 55.4%) 등에서 SoTA 성능 경신  
4. 지역·전역 상호작용의 상호 보완적 역할을 실험적으로 규명  

***

## 1. 해결하고자 하는 문제  
Vision Transformer는 전 범위(self-attention)로 짧고 긴 거리의 의존성을 모두 학습하지만, 고해상도 입력에서는 토큰 수 급증에 따른 계산·메모리 비용이 $$O((HW)^2d)$$로 비현실적이다.  
- **지역만 주목(local attention):** 근처 토큰 간 세밀한 상호작용으론 전역 문맥 상실  
- **전역만 주목(global attention):** 전체 문맥 파악 가능하나 세부 정보 희생  

## 2. 제안 방법  
### 2.1 Focal Self-Attention 메커니즘  
각 쿼리 토큰이  
- **Level 1 (fine):** 반경 $$s_w^1$$짜리 가장 가까운 토큰 $$s_r^1 \times s_r^1$$ 블록 주의  
- **Level $$l$$ (coarse):** 더 먼 토큰은 $$s_w^l$$ 서브윈도우로 요약(pooling)하여 $$s_r^l \times s_r^l$$ 블록 주의  
- 레벨별 요약 토큰을 합쳐 self-attention 수행  

수식:  

$$
Q = f_q(x^1),\quad
K_l = f_k(x^l),\quad
V_l = f_v(x^l)
$$  

$$
\text{Attention}(Q_i, K_i, V_i)
= \text{Softmax}\Bigl(\frac{Q_i K_i^\top}{\sqrt{d}} + B\Bigr)V_i,
$$  

$$B$$: 레벨별 상대 위치 bias  

### 2.2 모델 구조  
- 입력 224×224 이미지를 4×4 패치→4단계 multi-scale(해상도 56→7)  
- 각 단계마다 focal self-attention 레이어 $$\times N_i$$  
- 마지막 단계 출력은 분류층 또는 검출·분할 헤드로 전달  

## 3. 성능 향상  
- **ImageNet-1K 분류:** Focal-Base(89.8M) Top-1 83.8% (기존 Swin-Base 83.4%↑0.4p)  
- **COCO 검출:** Focal-Large + HTC++ box mAP 58.7 mask mAP 51.3 (Swin-Large 대비↑0.6p/↑1.1p)  
- **ADE20K 의미 분할:** Focal-Large mIoU 55.4 (Swin-Large 53.5%↑1.9p)  
- **빠른 수렴:** 100 epoch 시 Focal-Tiny 75.7% vs Swin-Tiny 73.9%  

## 4. 한계  
- 전역 풀링·윈도우 주변 토큰 추출 과정의 추가 연산 및 메모리 소모  
- 모델 깊이 경량화 가능성 탐색 미비  
- 단일 multi-scale 구조에만 적용, monolithic ViT나 타 도메인 적용은 향후 과제  

***

## 일반화 성능 향상 관점  
- **단계별 global 강화:** 상위 레이어로 갈수록 전역 토큰 비중 증가, 분류에서 전역 문맥 학습 강화  
- **지역·전역 상호 보완:** 분할·검출·분류 전반에서 다양한 규모 문맥 동시 활용으로 일반화 우수  
- **빠른 수렴:** 전역 정보 조기 학습→적은 epoch 내에서도 우수한 검증 성능 달성  

***

## 연구 영향 및 고려 사항  
- **영향:** Transformers의 효율적 self-attention 설계 가능성 확대, 고해상도 컴퓨터 비전 전 분야 응용 확대  
- **향후 연구 시 고려:**  
  - 윈도우 풀링 비용 절감 기술  
  - focal 메커니즘 단일-스케일 ViT나 언어 모델 적용  
  - 레이어 수·채널 최적화로 경량화 구조 탐색  
  - 학습 데이터 규모·도메인 변화에 따른 focal 설계 적응성 분석

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/046e9e2d-fc65-452d-b1db-3140a9b973f8/2107.00641v1.pdf
