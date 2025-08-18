# Deformable DETR: Deformable Transformers for End-to-End Object Detection | Object detection

## 핵심 주장 및 주요 기여  
**Deformable DETR**는 기존 DETR이 겪는 느린 수렴과 낮은 작은 객체 검출 성능 문제를 **극복**하기 위해 제안된 방법이다.  
- **핵심 주장**: Transformer의 전역 어텐션을 제한된 수의 변형 샘플링 지점(deformable sampling points)만을 참조하도록 바꿔, 연산 복잡도와 메모리 비용을 선형 수준으로 낮추면서 빠른 수렴과 향상된 작은 객체 검출 성능을 동시에 달성할 수 있다.  
- **주요 기여**:  
  1. **Deformable Attention Module**: 각 쿼리마다 학습된 오프셋 Δp와 가중치 A를 통해 K개의 위치만 샘플링하도록 하여 어텐션 비용을 $$O(N_qC^2 + N_qKC^2)$$로 감소시킴.  
  2. **Multi-Scale Extension**: 서로 다른 해상도의 feature map들을 L개 레벨로 통합하여, 작은 객체 검출 정확도를 크게 향상.  
  3. **Iterative Bounding Box Refinement**: 디코더 레이어마다 이전 예측을 기반으로 박스 좌표를 반복 정제하여 최종 AP를 추가 개선.  
  4. **Two-Stage Variant**: 엔코더만 사용하여 첫 단계에서 region proposals를 생성하고, 두 번째 단계 디코더로 정제하여 높은 재현율과 정밀도를 동시에 확보.  

## 해결하려는 문제  
1. **느린 수렴**: DETR는 500 에폭 이상 훈련해야 어텐션이 의미 있는 위치에 집중됨.  
2. **작은 객체 검출 미흡**: 전역 어텐션의 $$O(H^2W^2)$$ 복잡도로 고해상도 feature map 처리가 어렵고 작은 객체 정보가 희석됨.  

## 제안 방법  

### 1) Deformable Attention  
- 수식(단일 레벨):  

```math
\text{DeformAttn}(z_q,p_q,X) = \sum_{m=1}^{M} W_m \Bigl(\sum_{k=1}^{K} A_{mqk}\,W'_m\,X\bigl(p_q + \Delta p_{mqk}\bigr)\Bigr)
```
  
  - $$z_q\in\mathbb R^C$$: 쿼리 특징, $$X\in\mathbb R^{C\times H\times W}$$: 키 특징 맵  
  - $$\Delta p_{mqk}\in\mathbb R^2$$: 학습된 샘플링 오프셋, $$A_{mqk}$$: softmax 정규화된 어텐션 가중치  
  - K≪H×W개의 점만 참조 → 연산 복잡도 선형화  

- 멀티스케일 확장:  

```math
\text{MSDeformAttn}(z_q,\hat p_q,\{X_\ell\}) 
= \sum_{m=1}^{M} W_m \Bigl(\sum_{\ell=1}^L\sum_{k=1}^K A_{m\ell qk}\,W'_m\,X_\ell\bigl(\phi_\ell(\hat p_q)+\Delta p_{m\ell qk}\bigr)\Bigr)
```

### 2) 모델 구조  
- **Encoder**: ResNet C3–C5에서 4개 스케일 맵 추출, 각 레벨 256채널로 1×1 컨볼루션 후 Deformable Self-Attention 반복.  
- **Decoder**: 객체 쿼리 300개, 각 쿼리 위치를 참조점으로 삼아 멀티스케일 Deformable Cross-Attention 수행.  
- **헤드**: 디코더 예측을 reference point 기반 상대좌표로 변환하도록 회귀 분기 설계.  

### 3) 성능 향상  
- **수렴 속도**: 50 에폭 학습만으로 DETR-DC5 대비 AP 43.8% → 45.4% (IterRefine 적용 시 45.4% → 46.2%) 달성.  
- **작은 객체(APS)**: +6.2%p 향상 (20.5% → 26.7%).  
- **계산 효율**: FLOPs 173G, V100 기준 FPS 19로 DETR-DC5의 12에서 큰 폭 개선.  

### 4) 한계  
- **메모리 접근 패턴**: unordered sampling으로 인해 GPU 구현 최적화가 어렵고 FPN 대비 여전히 약간 느림.  
- **하이퍼파라미터**: 샘플링 점 수 K, 헤드 수 M, 스케일별 레벨 embedding 등 튜닝 필요.  
- **복잡도 증가**: Two-stage 구조 도입 시 엔코더-디코더 파이프라인이 복잡해지고 추가 메모리 사용  

## 일반화 성능 향상 가능성  
- **적응형 샘플링**: 영상 외 다른 도메인(의료 영상, 원격 탐사)에서도 변형 어텐션으로 희소한 핵심 위치에 집중 가능.  
- **하위 모듈 교체**: 다양한 백본(CSPNet, Swin Transformer)과 결합하여 멀티스케일 정보 융합 성능 극대화.  
- **세미/비지도 학습**: 어텐션 오프셋 예측을 프리트레인 과제로 활용 → 작은 데이터셋 일반화 강화.  
- **도메인 어댑테이션**: 샘플링 지점과 가중치 예측을 도메인별로 미세조정하여 교차-도메인 견고성 확보.  

## 향후 연구에 미치는 영향 및 고려 사항  
- **효율적 어텐션 연구 확대**: ‘Deformable’ 컨셉은 ViT, DETR 계열 외 다른 트랜스포머 모델에도 적용 가능성 제시.  
- **하드웨어 가속기 설계**: sparse/unordered 메모리 접근을 효율화하는 전용 하드웨어 혹은 라이브러리 필요.  
- **학습 안정성**: 초기화 및 gradient 흐름 관리(예: offsets bias 초기화) 중요.  
- **응용 확장**: 비디오 객체 검출, 3D 포인트 클라우드 처리 등 시공간적 변형 어텐션 연구 유도.  

**결론**: Deformable DETR는 리니어 복잡도와 멀티스케일 어텐션으로 DETR의 수렴 및 작은 객체 검출 문제를 효과적으로 해결했으며, 향후 효율적 어텐션과 도메인 일반화 연구의 새로운 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c7ad10b2-dffd-467c-ac28-7bd4fc8cb51a/2010.04159v4.pdf
