# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions | Image classification, Semantic segmenation, Object detection

## 1. 핵심 주장 및 주요 기여  
Pyramid Vision Transformer(PVT)는 **완전한 Transformer 기반** 백본으로, **다중 해상도(feature pyramid)** 및 **고해상도** 출력을 효율적으로 생성해 CNN 백본을 대체할 수 있음을 보였다.  
- Transformer 단일 스케일 구조(ViT)의 한계를 극복: 고해상도 출력과 계산 비용 문제 해결.  
- **Progressive shrinking pyramid** 구조 도입으로 초기 단계에 세밀한 패치를, 후속 단계에 점진적 축소로 연산량 감소.  
- **Spatial-Reduction Attention(SRA)** 모듈 설계로 고해상도 시퀀스에 대한 메모리·연산 효율성 확보.  
- 다양한 다운스트림(객체 검출·세분화) 과제에서 CNN 대비 유의미한 성능 개선 확인.  

## 2. 문제 정의  
기존 Vision Transformer(ViT)는  
1. **단일 해상도**(columnar) 출력  
2. **높은 연산·메모리 비용**(큰 패치 크기 사용)  
→ 객체 검출·세분화 등 **픽셀 단위 예측**에 부적합  

### 해결 목표  
- **고해상도·다중 해상도** 특징 맵 생성  
- Transformer 완전 비(非)컨볼루셔널 백본 구축  
- 연산 자원 한정하 성능 유지·향상  

## 3. 제안 방법  

### 3.1 구조 개요  
- **4단계(stage)**: 이미지 → {F₁, F₂, F₃, F₄} 스트라이드 {4,8,16,32} 특징 맵 생성  
- 각 단계:  
  1. **Patch Embedding** (Pᵢ×Pᵢ 크기 패치 → 선형 투영)  
  2. **Lᵢ-layer Transformer Encoder** (Self-Attention + FFN)  

### 3.2 Progressive Shrinking Pyramid  
- i단계 입력 크기 Hᵢ₋₁×Wᵢ₋₁를 Pᵢ²로 패치 분할 → (Hᵢ₋₁/Pᵢ)×(Wᵢ₋₁/Pᵢ) tokens  
- 단계별 Pᵢ 값 증가 → 깊은 단계일수록 시퀀스 길이 축소  

### 3.3 Spatial-Reduction Attention (SRA)  
전통적 MHA의 (H·W)² 복잡도 → SRA로 키·값에 공간 축소 적용  

$$
\begin{aligned}
\mathrm{head}_j &= \mathrm{Attention}\bigl(QW_Q^j,\; \mathrm{SR}(K)W_K^j,\; \mathrm{SR}(V)W_V^j\bigr),\\
\mathrm{SR}(X)&=\mathrm{Norm}\bigl(\mathrm{Reshape}(X, R_i)\,W_S\bigr),
\end{aligned}
$$  

- $$R_i$$: i단계 공간축소 비율  
- Reshape 후 선형투영으로 시퀀스 길이·차원 축소  

### 3.4 하이퍼파라미터  
- $$P_i$$: 패치 크기, $$C_i$$: 채널 수, $$L_i$$: Encoder 층 수  
- $$R_i$$: SRA 축소 비율, $$N_i$$: 어텐션 헤드 수, $$E_i$$: FFN 확장비율  

## 4. 성능 향상 및 한계  

| 과제           | 백본              | 주요 수치 비교                        |
|----------------|-------------------|--------------------------------------|
| 이미지 분류    | ResNet50 (21.5% 오류) → PVT-S (20.2%)  
| 객체 검출 (RetinaNet) | R50-1×: 36.3 AP → PVT-S: 40.4 AP  
| 인스턴스 분할 (Mask R-CNN) | R50-1×: 34.4 APₘ → PVT-S: 37.8 APₘ  
| 세그멘테이션 (Semantic FPN) | R50: 36.7 mIoU → PVT-S: 39.8 mIoU  

- **Pure Transformer 검출/분할**: DETR+PVT-S 34.7 AP (+2.4 vs. R50-DETR), Trans2Seg+PVT-S 42.6 mIoU (+2.9)  
- **전이학습의 중요성**: ImageNet 사전학습 시 COCO 1×스케줄 AP +13.8  
- **계산 효율**: 640픽셀 입력 시 PVT-S+RetinaNet은 R50 대비 연산량·지연시간 감소(51.7ms vs. 55.9ms) 및 AP +2.4  

### 한계  
- **초고해상도 입력 시 연산 비용 증가** (ViT보다는 낮지만 CNN보다 증가 속도 큼)  
- **CNN 전용 모듈(SE, dilated conv 등 미탑재)**  
- **추가 경량화·합성곱 활용 연구 필요**  

## 5. 일반화 성능 향상 가능성  
- **글로벌 어텐션 강화**: SRA 계층 누적 시 다양한 스케일에서 장거리 의존성 학습  
- **사전학습 확장**: 대규모 비전·멀티모달 데이터로 사전학습 시 세밀한 표현력·일반화 개선 기대  
- **모듈 결합**: SE 블록, NAS 기반 구조 탐색, 프루닝 기법과 결합 시 더욱 견고한 일반화 가능  

## 6. 향후 연구 영향 및 고려 사항  
- **Transformer 백본 연구 확장**: OCR, 3D, 의료영상 등 다양한 픽셀 단위 과제에 순수 Transformer 적용 촉진  
- **경량화 어텐션 연구**: 더 낮은 복잡도 자가어텐션(SR 비율 최적화, 저비용 토큰 샘플링) 개발  
- **하이브리드 구조 탐색**: CNN·Transformer 이점을 결합한 하이브리드 백본 설계  
- **프레임워크 호환성**: 기존 FPN, RPN 등 모듈과의 효율적 통합 방안 모색  

---  
**결론**: PVT는 CNN 전용 요소 없이 다중 해상도·고해상도 특징 맵을 효과적으로 학습하여, 픽셀 수준 예측 과제 전반에 걸쳐 일반화 성능과 효율성을 크게 향상시킬 수 있는 **순수 Transformer 백본**의 새로운 가능성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9ba3e132-d4a8-4e9c-aaeb-588c0e5cb5a5/2102.12122v2.pdf
