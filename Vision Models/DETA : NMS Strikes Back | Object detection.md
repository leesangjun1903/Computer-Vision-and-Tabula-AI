# DETA : NMS Strikes Back | Object detection

## 핵심 주장과 주요 기여

이 논문은 Detection Transformer(DETR) 분야에서 널리 믿어져 온 **"one-to-one matching이 고성능 객체 탐지에 필수적"**이라는 통념을 반박합니다. 연구진은 전통적인 **one-to-many label assignment와 Non-Maximum Suppression(NMS)**이 오히려 더 우수한 성능을 제공함을 실증적으로 증명했습니다.[1]

주요 기여는 다음과 같습니다:

- **DETA(Detection Transformers with Assignment)** 모델 제안: 기존 DETR의 Hungarian matching을 IoU 기반 assignment로 대체[1]
- **2.5 mAP의 성능 향상**: Deformable-DETR 대비 현저한 개선[1]
- **빠른 수렴**: ResNet50 백본으로 12 epoch(1× schedule) 내에 COCO 50.2 mAP 달성[1]
- **아키텍처 동일성**: 기존 transformer 구조를 유지하면서 훈련 방법만 변경[1]

## 해결하고자 하는 문제

### 문제 정의
기존 Detection Transformer들은 Hungarian matching을 통한 one-to-one bipartite matching을 사용하여 end-to-end 객체 탐지를 수행합니다. 하지만 이러한 접근법은 다음과 같은 한계가 있습니다:[1]

1. **느린 수렴 속도**: 기존 NMS 기반 탐지기보다 훨씬 긴 훈련 일정 필요(50-500 epochs vs 12 epochs)[1]
2. **복잡한 matching 과정**: 각 ground truth 객체에 정확히 하나의 예측만 할당하는 제약[1]
3. **작은 객체에 대한 성능 저하**: 특히 challenging한 작은 객체 탐지에서의 한계[1]

## 제안하는 방법

### IoU 기반 Assignment 전략

**First Stage Assignment:**
초기 앵커 박스에 대한 할당은 다음 공식으로 정의됩니다:

```math
\sigma^{init}_i = \begin{cases}
\hat{k} = \arg\max_k \text{IoU}(b^{init}_i, b_k), & \text{if IoU}(b^{init}_i, b_{\hat{k}}) \geq \tau \text{ or } C^{init}_{max}(i, \hat{k}) \\
\emptyset, & \text{otherwise}
\end{cases}
```

여기서 $$\tau = 0.7$$는 첫 번째 단계의 IoU 임계값이며, $$C^{init}_{max}(i, k)$$는 앵커 $$i$$가 객체 $$k$$에 가장 가까운지를 나타냅니다.[1]

**Second Stage Assignment:**
두 번째 단계에서는 제안된 박스를 사용하여 할당을 수행합니다:

```math
\sigma^{prop}_i = \begin{cases}
\hat{k} = \arg\max_k \text{IoU}(b^{prop}_i, b_k), & \text{if IoU}(b^{prop}_i, b_{\hat{k}}) \geq \tau^n_k \text{ or } C^{prop}_{max}(i, \hat{k}) \\
\emptyset, & \text{otherwise}
\end{cases}
```

**Object Balancing 기법:**
각 ground truth 객체에 대해 최대 $$n$$개의 positive assignment만 허용하는 동적 임계값을 적용합니다:

$$
\tau^n_k = \max(\tau, \mu^n_k)
$$

여기서 $$\mu^n_k$$는 객체 $$k$$에 대한 $$n$$번째로 높은 IoU 값입니다.[1]

### 모델 구조

DETA는 기존 Deformable-DETR와 **정확히 동일한 아키텍처**를 유지합니다:[1]

1. **Backbone**: ResNet-50 등의 CNN 백본
2. **Transformer Encoder**: 6-layer encoder로 이미지 특징 추출
3. **Query Selection**: Top-K proposal 선택 메커니즘  
4. **Transformer Decoder**: 6-layer decoder로 최종 탐지 수행
5. **Post-processing**: NMS를 통한 중복 제거

핵심 차이점은 **훈련 손실 함수**에만 있습니다. Hungarian matching 대신 전통적인 IoU 기반 assignment를 사용합니다.[1]

## 성능 향상

### 정량적 성과

**COCO 2017 검증 세트 결과:**
- **DETA (1× schedule)**: 50.5 mAP
- **기존 최고 성능**: 49.8 mAP (Group-DETR)
- **성능 향상**: 0.7 mAP 개선[1]

**세부 성능 분석:**
- 작은 객체(APs): 33.1 mAP
- 중간 객체(APm): 54.7 mAP  
- 큰 객체(APl): 65.2 mAP[1]

### 수렴 속도 개선
- **12 epochs**만으로 50.2 mAP 달성
- 기존 NMS 기반 최고 성능(CenterNet2: 42.9 mAP) 대비 **7.3 mAP 향상**
- 강력한 end-to-end 탐지기(DINO: 49.4 mAP) 대비 **0.8 mAP 향상**[1]

### 일반화 성능

**다양한 데이터셋에서의 성능:**
- **LVIS 데이터셋**: 33.9 mAP (기존 31.5 mAP 대비 2.4 mAP 향상)
- **다양한 백본**: ResNet-50부터 Swin-L까지 일관된 성능 향상[1]

**아키텍처 독립성:**
- Vanilla DETR: 2.1 mAP 향상
- Deformable-DETR: 2.5 mAP 향상
- 다양한 transformer 구조에서 일관된 개선 효과[1]

## 일반화 성능 향상 가능성

### 모델 복잡도 감소
실험 결과, DETA는 **더 적은 transformer layer로도 강건한 성능**을 보입니다:

- **6개 encoder, 1개 decoder layer**: 1.9 mAP만 감소 (Deformable-DETR는 14.3 mAP 감소)
- **더 간단한 디코딩 함수**: 전역 최적화나 self-attention 없이도 효과적[1]

### 훈련 효율성
- **더 많은 positive sample**: one-to-many assignment로 인한 풍부한 훈련 신호
- **안정적인 매칭**: 작은 객체에 대한 object balancing으로 매칭 안정성 향상[1]

### Self-Attention 불필요성
DETA는 decoder에서 **self-attention 없이도 성능 유지**가 가능함을 보였습니다. 이는 더 효율적인 경량 탐지기 개발에 중요한 통찰을 제공합니다.[1]

## 한계

### NMS 의존성
- **후처리 단계 재도입**: end-to-end의 장점 일부 상실
- **임계값 조정 필요**: 태스크별로 NMS 임계값 튜닝 필요
- **군중 장면 한계**: CrowdHuman과 같은 밀집된 객체 환경에서의 어려움[1]

### 박스 표현 제약
- **Two-stage 프레임워크 의존**: 초기 bounding box 표현이 필요
- **Query 유연성 감소**: 박스가 아닌 다른 형태의 query 표현 제한[1]

### 하이퍼파라미터 민감성
- **IoU 임계값 설정**: 데이터셋별로 최적 임계값 탐색 필요
- **Object balancing 파라미터**: $$K$$ 값에 따른 성능 변화[1]

## 향후 연구에 미치는 영향

### 패러다임 전환
이 연구는 **"end-to-end가 반드시 우수하다"**는 고정관념을 깨뜨리고, 전통적인 방법론과 modern architecture의 **하이브리드 접근**이 더 효과적일 수 있음을 입증했습니다.[1]

### 효율적 탐지기 개발 방향
- **경량 transformer 설계**: self-attention 없는 decoder 활용
- **빠른 수렴 기법**: one-to-many assignment의 확장 연구
- **적응적 assignment**: 객체 크기와 밀도에 따른 동적 할당 전략[1]

## 향후 연구 고려사항

### 기술적 개선 방향
1. **적응적 NMS**: 장면별로 최적화된 NMS 임계값 자동 조정
2. **계층적 assignment**: 객체 복잡도에 따른 차별화된 할당 전략
3. **Multi-task 확장**: 인스턴스 분할, 자세 추정 등으로의 확장 가능성

### 이론적 탐구 필요성
- **One-to-many의 이론적 근거**: 왜 더 많은 positive sample이 효과적인지에 대한 깊이 있는 분석
- **Attention mechanism 재평가**: self-attention의 역할과 필요성에 대한 체계적 연구
- **Generalization bound**: IoU assignment의 일반화 이론적 분석[1]

이 연구는 객체 탐지 분야에서 **성능과 효율성의 균형**을 재정의하며, 향후 detection transformer 연구의 새로운 방향을 제시하는 중요한 이정표가 될 것입니다.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/33899f92-630d-4f3e-8911-c4bdb2ff498c/2212.06137v1.pdf
