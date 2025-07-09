# DeepViT: Towards Deeper Vision Transformer | Image classification

## 1. 핵심 주장과 주요 기여

**핵심 주장**: Vision Transformer(ViT)는 CNN과 달리 더 깊게 쌓아도 성능이 향상되지 않으며, 이는 **attention collapse** 현상 때문이다[1]. 깊은 층에서 attention map들이 유사해지면서 모델의 표현 학습 능력이 제한된다[1].

**주요 기여**:
- ViT의 attention collapse 현상을 최초로 체계적으로 분석하고 정의[1]
- **Re-attention** 메커니즘을 제안하여 attention map의 다양성을 증가[1]
- ImageNet-1k에서 32개 블록의 깊은 ViT를 성공적으로 훈련하여 1.6% 성능 향상 달성[1]

## 2. 문제 정의 및 해결 방법

### 해결하고자 하는 문제

**Attention Collapse**: ViT가 깊어질수록 attention map들이 점진적으로 유사해지는 현상[1]. 구체적으로 17번째 블록 이후 attention map 유사도가 90% 이상으로 증가한다[1].

**수식적 정의**: 
층 p와 q 사이의 attention map 유사도는 다음과 같이 측정된다[1]:

$$M^{p,q}\_{h,t} = \frac{(A^p_{h,:,t})^T A^q_{h,:,t}}{||A^p_{h,:,t}|| \cdot ||A^q_{h,:,t}||}$$

여기서 $$A^p_{h,:,t}$$는 p번째 층, h번째 헤드, t번째 토큰의 attention 벡터이다[1].

### 제안하는 방법: Re-attention

**기존 Multi-Head Self-Attention**:
$$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d}})V$$

**제안하는 Re-attention**:
$$Re-Attention(Q, K, V) = Norm(\Theta^T(Softmax(\frac{QK^T}{\sqrt{d}})))V$$

여기서 $$\Theta \in \mathbb{R}^{H \times H}$$는 학습 가능한 변환 행렬이다[1].

**핵심 아이디어**: 같은 층 내의 서로 다른 attention head들은 충분히 다양하므로(유사도 30% 미만), 이들을 학습 가능한 변환 행렬로 재조합하여 새로운 attention map을 생성한다[1].

## 3. 모델 구조

### 기본 구조
- **Patch Embedding**: 입력 이미지를 패치로 분할하여 토큰 임베딩 생성[1]
- **Transformer Blocks**: Self-attention과 Feed-forward 레이어로 구성[1]
- **Classification Head**: 최종 분류를 위한 선형 레이어[1]

### DeepViT 수정사항
- 기존 self-attention 모듈을 Re-attention으로 대체[1]
- **DeepViT-S**: 16개 블록, 396 차원 임베딩[1]
- **DeepViT-L**: 32개 블록, 420 차원 임베딩[1]

## 4. 성능 향상

### 주요 성능 결과
- **32블록 ViT**: 기존 79.3% → DeepViT 80.9% (+1.6% 향상)[1]
- **DeepViT-L**: 55M 파라미터로 82.2% 달성 (DeiT-B는 86M 파라미터로 81.8%)[1]
- **모델 효율성**: 더 적은 파라미터로 더 높은 성능 달성[1]

## 5. 일반화 성능 향상 가능성

### 모델 독립적 특성
- **범용성**: 어떤 transformer 아키텍처에도 적용 가능[1]
- **최소 오버헤드**: 변환 행렬 Θ만 추가되어 계산 비용 최소화[1]
- **End-to-end 학습**: 전체 모델과 함께 최적화 가능[1]

### 확장 가능성
- 다른 비전 태스크(객체 탐지, 세그멘테이션)에 적용 가능[1]
- 더 큰 모델과 데이터셋으로 확장 가능[1]
- 다른 transformer 변형에도 적용 가능[1]

## 6. 한계점

- **하이퍼파라미터 의존성**: 여전히 세심한 하이퍼파라미터 튜닝 필요[1]
- **제한된 평가**: ImageNet 외 다른 데이터셋에서의 평가 부족[1]
- **임계값 설정**: Attention 유사도 임계값(90%)이 경험적으로 결정됨[1]
- **계산 오버헤드**: 변환 행렬로 인한 추가 계산 비용[1]

## 7. 향후 연구에 미치는 영향

### 연구 파급효과
- **새로운 연구 방향**: ViT의 attention collapse 현상에 대한 체계적 연구 시작[1]
- **깊은 transformer 훈련**: 깊은 vision transformer 훈련을 위한 기초 제공[1]
- **효율적 솔루션**: 간단하면서도 효과적인 해결책 제시[1]

### 향후 연구 고려사항
- **적응적 임계값**: Attention 유사도를 위한 적응적 임계값 개발 필요[1]
- **다양한 태스크**: 분류 외 다른 비전 태스크에서의 검증 필요[1]
- **스케일링**: 더 큰 모델과 데이터셋에서의 효과 검증[1]
- **이론적 분석**: Attention collapse의 이론적 근거 및 해결책의 수학적 증명[1]

이 논문은 vision transformer의 깊이 확장성 문제를 해결하는 중요한 첫걸음을 제시하며, 향후 더 깊고 효율적인 vision transformer 개발의 기초를 마련했다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/384df48b-4a99-4e64-ae6e-1e36eef355d8/2103.11886v4.pdf
