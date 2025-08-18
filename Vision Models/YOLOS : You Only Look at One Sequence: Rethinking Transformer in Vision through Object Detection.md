# YOLOS : You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection | Object detection

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- 순수한 시퀀스-투-시퀀스(sequence-to-sequence) 관점에서, 최소한의 2D 구조 정보만으로도 Vision Transformer(ViT)를 직접 객체 검출에 적용할 수 있으며, 기존 고성능 검출기와 경쟁력 있는 성능을 얻을 수 있다.  

**주요 기여**  
1. **YOLOS 제안**: ViT의 [CLS] 토큰을 100개의 [DET] 토큰으로 대체하고, 분류 손실을 이진 매칭(bipartite matching) 기반 세트 예측 손실로 교체한 순수 Transformer 기반 객체 검출기.  
2. **최소 유도 편향**: 2D 공간 구조에 관한 사전지식(피라미드 구조, RoI 풀링 등)을 거의 제거하여, 시퀀스 처리 능력만으로 2D 객체 검출 가능성을 입증.  
3. **전이 학습 벤치마크**: 중규모 ImageNet-1k로만 사전학습한 ViT를 COCO 검출에 전이할 때 성능 변화가 크고 포화되지 않음을 보여주어, 전이 학습 전략 평가를 위한 새로운 과제로 제시.  

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- **질문**: 순수한 ViT 아키텍처가 CNN 의존 없이 2D 객체 검출 과제를 해결할 수 있는가?  
- 대부분의 기존 연구는 CNN 백본 또는 지역적 유도 편향을 Transformer에 혼합 적용하였으나, 순수 Transformer가 객체 검출 시퀀스-투-시퀀스 관점만으로 가능한지 불분명.

### 2.2 제안 방법  
- **입력 변환**  

$$
    x_{\text{PATCH}} \in \mathbb{R}^{N \times (P^2 C)},\;
    N = \tfrac{H W}{P^2}
  $$
  
  - $$H\times W$$ 이미지를 $$P\times P$$ 패치로 분할 후 선형 투영  
- **토큰 구성**  

$$
    z_0 = [\,x_1^{\text{PATCHE}};\dots;x_N^{\text{PATCHE}};\;x_1^{\text{DET}};\dots;x_{100}^{\text{DET}}\,] + P
  $$
  
  - [DET] 토큰 100개를 패치 임베딩 뒤에 학습 가능한 위치 임베딩과 함께 추가  
- **Transformer 인코더**  
  - 순수 인코더만 사용, 패치/디텍션 토큰 동등 취급  
  - 층별 연산: LayerNorm → Multi-head Self-Attention → Residual → MLP → Residual  
- **검출 헤드와 손실**  
  - 각 [DET] 토큰에서 분류 & 바운딩 박스 예측(MLP)  
  - Carion et al.의 bipartite matching 기반 set prediction loss 적용  

### 2.3 모델 구조  
| 변형      | 레이어 수 | 임베딩 차원 | [DET] 수 | 파라미터 수 | FLOPs (pre-train) |
|-----------|----------:|-----------:|---------:|------------:|------------------:|
| YOLOS-Ti  |        12 |        192 |      100 |       5.7M  |        1.2G       |
| YOLOS-S   |         6 |        384 |      100 |      22.1M  |        4.5G       |
| YOLOS-B   |        12 |        768 |      100 |      86.4M  |       17.6G       |

### 2.4 성능 향상  
- **COCO 검출(AP)**  
  - YOLOS-Base: 42.0 AP (ViT-Base → BERT-Base 상속)  
  - YOLOS-Small: 36.1 AP (ImageNet-1k 200→300 epochs pre-train)  
  - YOLOS-Tiny: 28.7 AP (with distillation)  
- **사전학습 효과**  
  - 무작위 초기화 대비 ImageNet-1k 사전학습이 전이 학습 FLOPs를 대폭 절감하며 성능 향상  
  - 레이블 감독 vs. self-supervised(DINO) 비교: DINO 800 epochs 전이성능이 레이블 감독 300 epochs와 유사  
- **모델 크기 확대**: 너비·깊이·해상도 복합 스케일링(dwr)이 사전학습 성능에 유리하나, 고해상도 검출 시 공간 어텐션 연산이 병목이 되어 전이 성능 일관성은 미흡  

### 2.5 한계  
1. **연산 비용**: 대규모 토큰 수와 고해상도 처리 시 어텐션 복잡도가 기하급수적으로 증가  
2. **전이 성능 일관성 부족**: CNN 스케일링 전략이 ViT에 그대로 적용되지 않음  
3. **학습 효율**: 150–300 epochs 전이 학습 필요, few-shot 전이 역량 미흡  
4. **공간 정보 활용 부재**: 피라미드 구조나 국소성 편향 부여 없음 → 작은 물체 검출 성능 제한  

***

## 3. 모델의 일반화 성능 향상 가능성  

- **사전학습 민감도**  
  - 전이 성능이 사전학습 방식(레이블-지도 vs. 자체지도), epoch 수, 증류 유무에 크게 의존  
  - *Implication*: 더 풍부한 자체지도(pre-training)와 task-specific prompt 등으로 일반화 능력 제고 가능  
- **토큰 기반 학습 메커니즘**  
  - [DET] 토큰은 위치·크기 전문화(sensitivity)하지만 클래스엔 일정 불문(robust)  
  - 토큰 디자인 및 매칭 전략 개선을 통해 다양한 도메인·물체에 적응력 확대 여지  
- **사전학습→전이 과정 최적화**  
  - PE(interpolation) 전략, 어텐션 레이어 동결/가변화(LoRA 등), 학습률 스케줄링 기법 연구 필요  
- **경량화 및 연산 효율화**  
  - 스파스 어텐션, 저해상도 프리스크리닝, 어댑티브 시퀀스 길이 등으로 실제 응용 일반화  

***

## 4. 향후 연구에 미치는 영향 및 고려 사항  

- **Transformer의 범용 비전 표현 학습**  
  - YOLOS는 “태스크-불문 순수 사전학습→미세조정” 패러다임을 CV에 확산  
  - 대규모 unlabeled 데이터 활용한 자체지도학습과 결합 시 few-shot/zero-shot 객체 검출로 확장 가능  
- **모델 스케일링 재고**  
  - ViT 특유의 어텐션 복잡도 고려한 새로운 스케일링 법칙 제안 필요  
  - 하이브리드 구조(CNN+Transformer)와 순수 구조 간 trade-off 분석  
- **효율적 전이 학습 프로토콜**  
  - PE, 토큰 매칭, 레이어 동결·언프리징 프레임워크 연구를 통해 전이 학습 비용 최소화  
- **실제 응용 고려**  
  - 실시간 검출, 모바일·엣지 환경에서의 경량화  
  - 다양한 도메인(의료·위성·자율주행)에서 일반화 실험 및 벤치마크  

> 결론적으로, YOLOS는 순수 ViT가 객체 검출 과제를 해결할 수 있음을 입증함으로써, 비전&언어 모델링 방법론을 통합하고, 차세대 범용 비전 표현 학습 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/528ca536-4b6c-426a-a906-f4f020192453/2106.00666v3.pdf
