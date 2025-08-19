# MS-DETR: Efficient DETR Training with Mixed Supervision | Object detection

## 1. 핵심 주장 및 주요 기여  
MS-DETR은 DETR(Detection Transformer)의 후보 생성 과정을 명시적으로 개선하기 위해 **one-to-one** 슈퍼비전과 **one-to-many** 슈퍼비전을 혼합한 단일 디코더 학습 방식을 제안한다.  
- **핵심 주장**: 추가적인 one-to-many 슈퍼비전이 디코더 쿼리의 학습을 가속화하고, 더 우수한 객체 후보를 생성하여 학습 효율과 성능을 동시에 향상시킨다.  
- **주요 기여**:  
  1. **단일(primary) 디코더**에 one-to-many 슈퍼비전을 직접 적용하여 추가 디코더나 쿼리 없이 학습 효율을 높임  
  2. 현존하는 DETR 변형(DN-DETR, Group DETR, Hybrid DETR) 대비 빠른 수렴 속도 및 메모리·연산 효율성 확보  
  3. COCO 평가에서 기존 방법 대비 일관된 mAP 개선 및 다른 슈퍼비전 방식(예: IoU-aware loss)과도 상호 보완적임을 검증  

## 2. 문제 정의와 해결 방법

### 2.1 해결하고자 하는 문제  
기존 DETR은 end-to-end 학습 구조에서 self-attention과 one-to-one 슈퍼비전을 통해 후보 복제(duplicate)를 억제하지만, 후보 생성 후보들이 충분히 다양하거나 정확하지 못해 수렴이 느리고 성능 한계가 존재한다.

### 2.2 제안 방법  
MS-DETR은 각 디코더 레이어의 쿼리에 대해 다음 두 가지 손실을 혼합하여 최적화한다:  

1. **One-to-one Loss**  

$$L_{1:1} = \sum_{n=1}^N \big(\ell_{c}^{1:1}(s_{\sigma(n)}, \bar{s}\_n) + \ell_{b}^{1:1}(b_{\sigma(n)}, \bar{b}_n)\big) $$

2. **One-to-many Loss**  
   각 ground-truth $$n$$에 대해 매칭 점수 기반으로 상위 $$K$$개의 쿼리를 선택하여 다음 손실을 계산:  

$$L_{1:m} = \sum_{n=1}^N \sum_{i=1}^{K_n} \big(\ell_{c}^{1:m}(s_{ni}, \bar{s}\_n) + \ell_{b}^{1:m}(b_{ni}, \bar{b}_n)\big) $$  
   
매칭 점수:  
   
$$\text{MatchScore}(s, b, \bar{c}, \bar{b}) = \alpha\,s_{\bar{c}} + (1-\alpha)\,\text{IoU}(b,\bar{b}) $$

최종 학습 목적 함수:  

$$ L = L_{1:1} + L_{1:m} $$

### 2.3 모델 구조  
- **Primary Decoder**: 기존 DETR 디코더 구조 유지  
- **Mixed Supervision Module**: 디코더의 각 레이어 출력(또는 cross-attention 후 내부 쿼리)에 one-to-many 예측 헤드(분류, 박스 회귀)를 추가  
- **Weight Sharing**: one-to-one과 one-to-many 예측기 클래스·박스 헤드 간 가중치를 공유  

### 2.4 성능 향상  
- **수렴 속도**: 12 에폭만에 기존 대비 +3.7 mAP 개선  
- **메모리·연산 효율**: 추가 쿼리·디코더 없이도 Hybrid DETR 대비 메모리 2%↑, 시간 오버헤드 약 3% 수준  
- **다양한 베이스라인**: Deformable DETR, DAB-DETR 등에서 일관된 성능 향상  
- **상호 보완성**: IoU-aware loss 적용 모델에도 +0.5~0.6 mAP 추가 개선  

### 2.5 한계  
- **하이퍼파라미터 민감도**: $$K$$, $$\tau$$, $$\alpha$$ 값 조정 필요  
- **추가 연산 부담**: 디코더 레이어별 one-to-many 예측기 도입 시 소폭 연산 증가  
- **적용 도메인**: 자연 이미지 객체 검출에 최적화, 다른 도메인 적용 시 재검증 필요  

## 3. 일반화 성능 향상 가능성  
- **쿼리 표현 강화**: one-to-many 슈퍼비전이 복수 후보 학습을 통해 쿼리의 표현력을 풍부히 하여, 드물거나 작은 객체 검출에서 일반화 성능 향상 기대  
- **결합 기법**: IoU-aware 손실, DINO denoising query 등 다양한 모듈과 조합할 때 더 강력한 일반화 능력  
- **적응적 매칭**: 데이터셋 편향에 맞춰 $$K$$, $$\tau$$를 동적으로 조정하면 새로운 환경에 빠른 적응 가능  

## 4. 향후 연구 방향 및 고려사항  
- **하이퍼파라미터 자동 최적화**: 매칭 기준 및 쿼리 수 자동 학습 기법 도입  
- **다중 모달 확장**: 비디오, 3D 포인트 클라우드, 멀티스펙트럼 이미지 등 다양한 입력에 대한 mixed supervision 적용  
- **후처리 경량화**: one-to-many 모듈 제거 시에도 성능 유지할 수 있는 지식 증류(Knowledge Distillation) 연구  
- **엔드-투-엔드 일관성**: NMS 없이도 완전 end-to-end 구조 유지하며, 복잡성 최소화를 위한 추가 구조 개선  

MS-DETR은 DETR 계열 모델의 학습 효율과 성능을 동시에 개선하는 실용적인 접근으로, 향후 다양한 검출 연구 및 실제 응용에 폭넓게 활용될 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/43ed538a-251f-4da1-b295-d07dc04ca351/2401.03989v1.pdf
