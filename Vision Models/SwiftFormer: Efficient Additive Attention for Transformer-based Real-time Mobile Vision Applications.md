# SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications | Image classification

## 핵심 주장과 주요 기여

SwiftFormer는 자원 제약이 있는 모바일 디바이스에서 실시간 비전 애플리케이션을 위한 **효율적인 가산적 주의 메커니즘(Efficient Additive Attention)**을 제안한 혁신적인 연구입니다. 이 논문의 핵심 주장은 기존 self-attention의 이차 복잡도 문제를 해결하면서도 정확도 손실 없이 모든 네트워크 단계에서 global context를 효과적으로 모델링할 수 있다는 것입니다.[1][2]

### 주요 기여

**1. 효율적인 가산적 주의 메커니즘 도입**
- 기존의 이차 행렬 곱셈 연산을 선형 요소별 곱셈으로 대체[2][1]
- Key-value 상호작용을 단순한 선형 변환으로 대체하여 정확도 손실 없이 계산 복잡도 크게 감소

**2. 일관된 하이브리드 설계**
- 기존 방법들과 달리 모든 네트워크 단계에서 효율적인 주의 메커니즘 사용 가능[1]
- 토큰 길이에 대해 선형 복잡도를 가져 초기 단계부터 활용 가능

**3. 우수한 성능-효율성 트레이드오프**
- SwiftFormer-S: iPhone 14에서 0.8ms 지연시간으로 78.5% ImageNet-1K top-1 정확도 달성[1]
- MobileViT-v2 대비 2배 빠르고 더 정확한 성능

## 해결하고자 하는 문제

### 기존 문제점

**1. Self-attention의 이차 복잡도**
표준 self-attention의 계산 복잡도는 $$O(n^2 \cdot d)$$로, 여기서 $$n$$은 토큰 수, $$d$$는 숨겨진 차원입니다. 이는 다음과 같이 정의됩니다:[1]

$$
\hat{x} = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V
$$

**2. 모바일 디바이스 배포의 어려움**
- 높은 메모리 사용량과 계산 비용으로 인한 실시간 애플리케이션 제약[1]
- 기존 하이브리드 접근법들도 여전히 비효율적인 행렬 곱셈 연산에 의존

**3. 불일치한 네트워크 설계**
- 기존 방법들은 마지막 단계에서만 self-attention 사용[1]
- 초기 단계에서는 이차 복잡도로 인해 사용 불가

## 제안하는 방법과 수식

### 효율적인 가산적 주의 메커니즘

**1. Global Query 생성**
쿼리 행렬 $$Q$$에 학습 가능한 매개변수 벡터 $$w_a \in \mathbb{R}^d$$를 곱하여 global attention query 벡터를 생성합니다:

$$
\alpha = Q \cdot w_a / \sqrt{d}
$$

**2. Global Query 풀링**
학습된 주의 가중치를 기반으로 쿼리 행렬을 풀링하여 단일 global query 벡터를 생성합니다:

$$
q = \sum_{i=1}^{n} \alpha_i * Q_i
$$

**3. Key와의 상호작용**
Global query 벡터 $$q \in \mathbb{R}^d$$와 키 행렬 $$K \in \mathbb{R}^{n \times d}$$ 간의 상호작용을 요소별 곱으로 인코딩합니다:

$$
\hat{x} = \hat{Q} + T(K * q)
$$

여기서 $$\hat{Q}$$는 정규화된 쿼리 행렬, $$T$$는 선형 변환, $$*$$는 요소별 곱셈을 나타냅니다.[1]

### 핵심 혁신점

**Key-Value 상호작용 제거**: 기존의 3단계 처리(Q, K, V 모두 포함)에서 Q-K 상호작용만으로 충분함을 보임[1]

**선형 복잡도**: 토큰 수에 대해 $$O(n \cdot d)$$의 선형 복잡도 달성

## 모델 구조

### SwiftFormer 아키텍처

**1. 계층적 구조**
- 4개 스케일($$1/4, 1/8, 1/16, 1/32$$)에서 특징 추출[1]
- 각 단계마다 Conv. Encoder + SwiftFormer Encoder 구성

**2. Conv. Encoder**
효과적인 지역 표현 학습을 위한 구성:

$$
\hat{X_i} = \text{Conv1}(\text{Conv1,G}(\text{DWConvBN}(X_i))) + X_i
$$

**3. SwiftFormer Encoder**
지역-전역 표현을 효율적으로 인코딩:

$$
\hat{X_i} = \text{Conv1}(\text{DWConvBN}(\hat{X_i}))
$$

$$
\hat{X_i} = \text{QK}(\hat{X_i}) + \hat{X_i}
$$

$$
\hat{X_{i+1}} = \text{Conv1}(\text{ConvBN,1,G}(\hat{X_i})) + \hat{X_i}
$$

## 성능 향상

### ImageNet-1K 분류 성능

| 모델 | 지연시간 (ms) | Top-1 정확도 (%) | 매개변수 (M) |
|------|---------------|------------------|---------------|
| SwiftFormer-XS | 0.7 | 75.7 | 3.5 |
| SwiftFormer-S | 0.8 | 78.5 | 6.1 |
| SwiftFormer-L1 | 1.1 | 80.9 | 12.1 |
| SwiftFormer-L3 | 1.9 | 83.0 | 28.5 |

### 기존 모델 대비 성능

**ConvNets 대비**: SwiftFormer-XS는 MobileNet-v2×1.0보다 0.1ms 빠르면서 3.9% 높은 정확도 달성[1]

**Transformer 대비**: SwiftFormer-S는 DeiT-T보다 4.7배 빠르면서 6.3% 높은 정확도[1]

**하이브리드 모델 대비**: SwiftFormer-L1은 MobileViT-v2×1.5보다 3배 빠르면서 0.5% 높은 정확도[1]

### 다운스트림 태스크 성능

**Object Detection (MS-COCO)**: SwiftFormer-L1 백본으로 41.2 AP_box, 38.1 AP_mask 달성[1]

**Semantic Segmentation (ADE20K)**: SwiftFormer-L1으로 41.4% mIoU 달성[1]

## 일반화 성능 향상 가능성

### 일관된 글로벌 컨텍스트 모델링

SwiftFormer의 가장 중요한 일반화 성능 향상 요소는 **모든 네트워크 단계에서 일관된 글로벌 컨텍스트 학습**입니다. 기존 방법들이 마지막 단계에서만 self-attention을 사용하는 것과 달리, SwiftFormer는:[1]

**1. 스케일별 일관성**: 각 해상도에서 일관된 글로벌 정보 학습으로 모델 성능과 일반화 능력 향상[1]

**2. 고해상도 확장성**: 선형 복잡도로 인해 고해상도 이미지에 대한 확장성과 일반화 성능 보장[1]

**3. 태스크 전이성**: 분류, 검출, 분할 등 다양한 태스크에서 우수한 전이 성능 입증[1]

### 효율적인 표현 학습

**Positional Encoding 불필요**: 위치 인코딩이나 주의 편향 없이도 효과적인 공간 정보 모델링[1]

**다중 도메인 적응성**: 실험 결과 다양한 비전 태스크에서 강건한 성능 유지

## 한계점

### 이론적 한계

**1. 근사적 등가성**: Softmax 정규화에서는 근사적 등가성만 보장되지만, 실험적으로 정확도 영향 없음 확인[1]

**2. 설계 복잡성**: 하이브리드 구조로 인한 설계 복잡성 존재

### 실용적 한계

**1. 메모리 접근 패턴**: 모바일 하드웨어의 메모리 접근 패턴 최적화 필요

**2. 하드웨어 특화**: 특정 하드웨어(iPhone 14)에서의 최적화 결과로 다른 플랫폼에서의 성능 보장 어려움

## 앞으로의 연구에 미치는 영향

### 이론적 기여

**1. 주의 메커니즘 재정의**: Key-value 상호작용이 불필요함을 보여 주의 메커니즘의 본질적 이해 확장[1]

**2. 효율성-정확도 트레이드오프**: 이차 복잡도 장벽을 깨뜨리는 새로운 패러다임 제시[3]

### 실용적 영향

**1. 모바일 AI 민주화**: 자원 제약 환경에서 고성능 비전 모델 활용 가능성 확대[1]

**2. 에지 컴퓨팅**: 실시간 처리가 필요한 응용 분야에서 변환자 모델 활용 촉진[4]

## 향후 연구 시 고려사항

### 알고리즘 측면

**1. 다양한 정규화 기법**: Softmax 외의 정규화 방법론 탐구

**2. 적응적 주의 가중치**: 입력에 따라 동적으로 조정되는 주의 메커니즘 개발

**3. 멀티모달 확장**: 텍스트-이미지 등 멀티모달 데이터에 대한 효율적인 주의 메커니즘

### 하드웨어 최적화

**1. 플랫폼별 최적화**: 다양한 모바일 하드웨어에 대한 최적화 전략 필요

**2. 양자화 기법**: 추가적인 효율성 향상을 위한 양자화 및 프루닝 기법 통합[5]

### 평가 및 벤치마킹

**1. 포괄적 벤치마킹**: 다양한 하드웨어 플랫폼에서의 일관된 평가 프레임워크 필요

**2. 실제 배포 검증**: 실제 모바일 애플리케이션에서의 성능 검증 및 사용자 경험 평가

SwiftFormer는 모바일 비전 AI의 새로운 가능성을 제시한 중요한 연구로, 향후 효율적인 변환자 아키텍처 연구의 중요한 기준점이 될 것입니다.[2][1]

[1] https://ieeexplore.ieee.org/document/10376776/
[2] https://huggingface.co/docs/transformers/main/model_doc/swiftformer
[3] https://proceedings.mlr.press/v201/duman-keles23a/duman-keles23a.pdf
[4] https://ieeexplore.ieee.org/document/10322131/
[5] https://ieeexplore.ieee.org/document/10595969/
[6] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/247ef040-d7c5-409b-9d8a-af5aa1fae6c6/2303.15446v2.pdf
[7] https://arxiv.org/abs/2408.03703
[8] https://ieeexplore.ieee.org/document/10981108/
[9] https://ieeexplore.ieee.org/document/10222812/
[10] https://www.frontiersin.org/articles/10.3389/fpls.2023.1256773/full
[11] https://ieeexplore.ieee.org/document/10470980/
[12] https://dl.acm.org/doi/10.1145/3508396.3512869
[13] https://arxiv.org/abs/2309.01310
[14] http://arxiv.org/pdf/2303.15446.pdf
[15] https://arxiv.org/html/2408.03703
[16] https://arxiv.org/pdf/2207.07268.pdf
[17] https://arxiv.org/pdf/2206.10589.pdf
[18] http://arxiv.org/pdf/2301.13156.pdf
[19] https://arxiv.org/abs/2110.02178v1
[20] http://arxiv.org/pdf/2112.10809.pdf
[21] http://arxiv.org/pdf/2207.05501.pdf
[22] https://arxiv.org/pdf/2501.15369.pdf
[23] https://arxiv.org/pdf/2206.01191.pdf
[24] https://proceedings.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf
[25] https://github.com/Amshaker/SwiftFormer
[26] https://wandb.ai/wandb_fc/tips/reports/The-Problem-with-Quadratic-Attention-in-Transformer-Architectures--Vmlldzo3MDE0Mzcz
[27] https://link.springer.com/article/10.1007/s11263-025-02480-w
[28] https://huggingface.co/MBZUAI/swiftformer-l3
[29] https://openaccess.thecvf.com/content/WACV2021/papers/Shen_Efficient_Attention_Attention_With_Linear_Complexities_WACV_2021_paper.pdf
[30] https://arxiv.org/html/2501.15369v1
[31] https://arxiv.org/abs/2303.15446
[32] https://openreview.net/forum?id=MxGGdhDmv5
[33] https://openaccess.thecvf.com/content/ICCV2023/papers/Shaker_SwiftFormer_Efficient_Additive_Attention_for_Transformer-based_Real-time_Mobile_Vision_Applications_ICCV_2023_paper.pdf
