# DETR : End-to-End Object Detection with Transformers | Object detection

## 1. 핵심 주장 및 주요 기여
- **핵심 주장**  
  객체 검출을 비최대 억제(NMS)나 앵커(anchor) 설계 없이도, 순수한 Transformer 기반의 end-to-end 학습으로 직접적인 “집합(set) 예측” 문제로 접근하여, 기존의 복잡한 검출 파이프라인을 단순화할 수 있음을 보인다.

- **주요 기여**  
  1. ** bipartite matching loss**: 헝가리안 알고리즘을 활용해 예측과 정답 박스 간 1:1 매칭을 수행하는 permutation-invariant 손실 함수 도입  
  2. **Transformer encoder–decoder 구조 적용**: 전역(self-attention) 연산으로 객체 간 관계 및 이미지 전체 문맥을 동시에 고려  
  3. **NMS 제거**: 모델 자체의 set-based loss와 parallel decoding 으로 후처리 과정 불필요  
  4. **Panoptic Segmentation 확장**: 사소한 구조 변경만으로 panoptic segmentation 작업까지 통합 처리  

***

## 2. 문제 정의 및 제안 방법

### 2.1 문제 정의  
기존의 객체 검출은 앵커 생성, 제안(proposal) 생성, NMS 등의 여러 수동 설계 요소에 의존하며, 이로 인한 복잡도 및 하이퍼파라미터 튜닝이 필요하다.

### 2.2 제안 방법  
DETR은 고정된 개수 $$N$$의 객체 쿼리(object queries)를 Transformer decoder에 입력으로 주어, 하나의 병렬적 패스로 전체 객체 집합을 직접 예측한다.  
손실 함수는 예측 세트 $$\hat{Y}=\{\hat{y}\_i\}\_{i=1}^N$$ 와 실제 객체 집합 $$Y=\{y_i\}_{i=1}^M$$ ($$M\le N$$) 간의 1:1 매칭을 다음과 같이 정의하여 최적화한다:

```math
\hat{\sigma} = \arg\min_{\sigma\in S_N}\sum_{i=1}^N \bigl[-\mathbb{1}_{\{c_i\neq\emptyset\}}\log p_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i\neq\emptyset\}}\,L_{\text{box}}(b_i, \hat{b}_{\sigma(i)})\bigr]
```

```math
L_{\text{Hungarian}} = \sum_{i=1}^N \Bigl[-\log p_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i\neq\emptyset\}}\,L_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\Bigr]
```

- **클래스 손실**: 교차 엔트로피  
- **박스 손실**: $$\ell_1$$ + Generalized IoU  

```math
L_{\text{box}} = \lambda_{L1}\|b_i-\hat b_i\|_1 + \lambda_{\text{IoU}}\bigl(1-\text{GIoU}(b_i,\hat b_i)\bigr)
```

### 2.3 모델 구조  
1. **Backbone**: ResNet-50/101으로부터 $$C\times H\times W$$ 피처 맵 추출  
2. **Transformer Encoder**: 1×1 conv로 채널 차원 축소 후, 2D→1D flatten + positional encoding 추가 → 다층 self-attention  
3. **Transformer Decoder**: $$N$$개의 학습된 object queries + decoder self-attention 및 encoder–decoder cross-attention → 각 query별 embedding 출력  
4. **Prediction FFN**: 각 decoder 출력 embedding에 대해 클래스 분류와 박스 회귀 수행  
5. **Auxiliary Losses**: 각 decoder 레이어마다 동일 손실 추가로 학습 안정화  

***

## 3. 성능 향상 및 한계

### 3.1 COCO 객체 검출 성능  
- **AP**: DETR-ResNet50 42.0 vs Faster R-CNN-FPN 42.0 (동일)  
- **큰 객체(AP_L)**: +7.8 AP_L 우위  
- **작은 객체(AP_S)**: –5.5 AP_S 열세  
- **FPS**: 28 vs 26 (유사)  

### 3.2 Panoptic Segmentation  
- **PQ**: DETR-R50 43.4 vs PanopticFPN 42.4  
- **특히 Stuff 클래스 향상**: 전역 self-attention 기반 이미지 문맥 이해로 우수한 성능  

### 3.3 한계  
- 작은 객체 검출 성능 부족  
- Transformer 특유의 긴 학습 스케줄과 높은 계산 비용  
- 고정된 쿼리 수(100)로 인해 극단 다수 객체 이미지는 포화 상태  

***

## 4. 일반화 성능 향상 관점

- **Set Prediction 접근**: 순서 비의존적 매칭으로 클래스별 객체 수 편향 최소화  
- **전역 관계 모델링**: Encoder self-attention이 이미지 전체 맥락과 객체 간 관계를 학습하여, 보기 드문 객체 분포나 out-of-distribution 상황(예: 이상 다량 객체 이미지)에 대한 일반화 잠재력 보유  
- **쿼리 슬롯 분석**: 각 쿼리는 위치·크기별로 전문화되어, 입력 이미지 분포 변화에도 유연 대응  
- **실험**: 학습 시 없던 24마리 기린 동시 검출 성공  

***

## 5. 향후 연구에 미치는 영향 및 고려사항

**영향**  
- 객체 검출 모델의 **end-to-end 통합** 방향 제시  
- NLP 분야 Transformer를 CV set prediction에 성공적으로 이식  
- Panoptic segmentation 등 복합 과제의 **Unified Model** 연구 촉진  

**고려사항**  
- **작은 객체 검출 개선**: 고해상도 피처 활용(FPN 결합 등)  
- **쿼리 수 및 동적 할당**: 이미지당 객체 수 변동에 유연한 쿼리 관리 기법  
- **학습 효율화**: 긴 학습 스케줄, 대용량 self-attention 최적화를 위한 경량화 연구  
- **다중 태스크 확장**: 검출·분할·추적·분류 통합 시스템 설계  

DETR는 객체 검출 분야에 새로운 **단순화된 패러다임**을 제시하며, 향후 CV 및 멀티모달 연구에서 Transformer 기반의 **set prediction** 개념이 활발히 확장될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9ecbda20-dcde-42ed-bf2d-c55ece25d699/2005.12872v3.pdf
