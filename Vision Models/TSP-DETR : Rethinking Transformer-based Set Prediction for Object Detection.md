# TSP-DETR : Rethinking Transformer-based Set Prediction for Object Detection | Object detection

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- DETR(Detection Transformer)의 *느린 수렴 속도*는 주로 Transformer의 cross‐attention 모듈과 Hungarian bipartite matching의 불안정성에 기인한다.  
- cross‐attention을 제거한 **encoder-only** 구조와 전통적 검출기(FCOS, Faster RCNN)의 장점을 결합하면 수렴 속도를 크게 단축하면서도 정확도를 유지·향상시킬 수 있다.

**주요 기여**  
1. **원인 분석**:  
   - Hungarian loss의 매칭 불안정성이 초기 수렴에 일부 영향.  
   - Transformer cross-attention의 희소성(sparsity) 증가가 수렴 지연의 주원인.  
2. **새로운 구조 제안**:  
   - **TSP-FCOS**: FCOS의 multi-level feature pyramid와 encoder-only Transformer 결합 → Feature of Interest(FoI) 선택 메커니즘 도입.  
   - **TSP-RCNN**: Faster RCNN의 RPN과 encoder-only Transformer 결합 → 두 단계(bipartite matching 개선, RoIAlign)로 refinement.  
3. **매칭 개선**:  
   - FCOS/Faster RCNN의 ground-truth assignment 규칙 기반으로 bipartite matching 범위 제한 → Hungarian loss 수렴 가속.  
4. **실험 결과**:  
   - COCO val 기준 TSP-FCOS/TSP-RCNN은 36 epoch 학습 만에 DETR-DC5 500 epoch와 유사하거나 상회하는 AP 달성.  
   - TSP-RCNN+ (96 epoch)로 **45.0% AP** ⇒ 종전 최고 성능 경신.  

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
- **DETR의 긴 학습 시간(≈500 epochs)**: 실용적 확장에 어려움  
- cross-attention 모듈의 학습 불안정성과 Hungarian loss 매칭의 무작위성  

### 2.2 제안 방법 상세  
1. **Encoder-Only DETR 모델**  
   - cross-attention 제거 → self-attention 만으로 feature 조합  
   - 모든 feature point(H/32×W/32) 별로 detection head 적용 → small object 성능↑, large object↓  
2. **TSP-FCOS 구조**  
   - Backbone + FPN으로 multi-level feature 생성  
   - FoI classifier: FCOS ground-truth 규칙으로 top-700 feature 선별  
   - Transformer encoder: 선택된 FoI에 self-attention 수행  
   - Shared FFN → class(객체 vs no-object) + bounding-box regression 예측  
   - **수식**: positional encoding $$\mathrm{PE}(x)\_{2i}=\sin\frac{x}{10000^{2i/d}}$$, $$\mathrm{PE}(x)_{2i+1}=\cos\frac{x}{10000^{2i/d}}$$  
   - bipartite matching 범위 제한: anchor-free 기반 위치 제약  
3. **TSP-RCNN 구조**  
   - RPN으로 top-700 RoI(proposals) 생성 → RoIAlign으로 feature 추출  
   - positional encoding: $$[PE(c_x):PE(c_y):PE(w):PE(h)]$$  
   - Transformer encoder + FFN → set prediction loss  
   - bipartite matching: IoU>0.5 조건 적용  

### 2.3 모델 구조 비교  

| 모델            | 입력 특징      | 매칭 방식                         | refinement 단계 |
|----------------|---------------|----------------------------------|----------------|
| DETR           | queries+cross | Hungarian loss 전통적              | decoder 무반복 |
| encoder-only   | 모든 feature  | Hungarian loss 전통적              | 없음           |
| **TSP-FCOS**   | top-FoI(700)  | FCOS assignment 규칙 기반 제한적 매칭 | 없음           |
| **TSP-RCNN**   | top-RoI(700)  | IoU>0.5 RoI 기반 제한적 매칭      | 두 단계        |

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **수렴 속도**: 36 epochs만에 DETR-DC5 (500 epochs) 대비 동급 성능 달성  
- **정확도**:  
  - TSP-FCOS (R50): 43.1% AP → FCOS 41.0% AP 대비 +2.1%p  
  - TSP-RCNN (R50): 43.8% AP → Faster RCNN 40.2% AP 대비 +3.6%p  
  - TSP-RCNN+ (R50, 96 epochs): 45.0% AP → 종전 최고치 경신  
- **소형 객체** 검출 성능 대폭 강화 (AP\_S +~4%p)  

### 3.2 한계  
- **대형 객체** 검출은 여전히 cross-attention 기반 모델에 소폭 열위  
- **제한된 refinement**: TSP-RCNN의 iterative refinement 단계는 2회 → Deformable DETR(6회) 대비 성능 유지 여지  
- **계산량 증가**: Transformer encoder 추가로 FLOPs·파라미터 증가  

## 4. 일반화 성능 향상 가능성

- **FoI/RoI 기반 선별**: 중요한 feature만 self-attention에 투입 → overfitting 감소, 희소 관계 학습  
- **bipartite matching 제한**: 잡음 레이블 배제 → 안정적 학습, 다양한 데이터셋 전이 학습 시 robust  
- **encoder-only 토대**: cross-attention 제거 후 재구성 설계 → 적은 데이터셋에서도 빠른 수렴 가능 → 저자원 환경 일반화 우수  

## 5. 향후 연구 영향 및 고려 사항

- **Sparse attention 통합**: 더욱 효율적 희소 attention 기법(BigBird, Adaptive Sparse) 적용 → 대형 객체 성능 보완  
- **다층적 refinement**: cascade 구조 심화 및 bounding-box iterative 업데이트 최적화  
- **다중 모달 확장**: 텍스트-이미지 결합 검출, video object detection으로 일반화  
- **데이터 효율 학습**: matching distillation과 limited assignment 병합 → 반지도학습·도메인 적응 성능 강화  

이 논문은 end-to-end set prediction 기반 검출기의 학습 효율성과 예측력을 모두 개선하는 새로운 패러다임을 제시하여, 후속 sparse attention, multi-stage refinement, 적은 데이터 환경에서의 일반화 연구에 중요한 설계 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/33d8aa9d-c4d1-4749-8a44-f3f5cf2c73ec/2011.10881v2.pdf
