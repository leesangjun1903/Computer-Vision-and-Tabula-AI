# RAN : Residual Attention Network for Image Classification | Image classification

## 주요 기여 및 핵심 주장

이 논문의 핵심 기여는 attention mechanism을 state-of-the-art feedforward network 구조에 end-to-end 방식으로 통합한 **Residual Attention Network**를 제안한 것입니다. 주요 기여 사항은 다음과 같습니다:[1]

**1. 스택형 네트워크 구조**: 여러 Attention Module을 쌓아 다양한 유형의 attention을 포착하는 구조를 설계했습니다.[1]

**2. Attention Residual Learning**: 매우 깊은 네트워크에서 수백 개 층까지 확장 가능한 attention residual learning 메커니즘을 제안했습니다.[1]

**3. Bottom-up Top-down Feedforward Attention**: 단일 feedforward 과정에서 bottom-up fast feedforward와 top-down attention feedback을 모사하는 구조를 개발했습니다.[1]

## 해결하고자 하는 문제

### 기존 문제점
- 기존 attention 메커니즘은 주로 순차적 과정으로 공식화되어 feedforward network에서 state-of-the-art 결과를 달성하지 못했습니다[1]
- 단일 attention branch로는 복잡한 배경, 복합 장면, 큰 외관 변화를 가진 이미지를 모델링하기 어려웠습니다[1]
- Attention module을 단순히 쌓으면 성능이 저하되는 문제가 있었습니다[1]

### 논문의 접근법
논문은 **mixed attention mechanism**을 "very deep" 구조에 적용하여 이 문제들을 해결하고자 했습니다.[1]

## 제안하는 방법 및 수식

### 1. 기본 Attention Module 구조

Attention Module은 두 개의 branch로 구성됩니다:
- **Trunk branch**: 특징 처리를 담당 $$T(x)$$
- **Mask branch**: soft attention mask를 학습 $$M(x)$$

기본 출력 공식:

$$H_{i,c}(x) = M_{i,c}(x) \times T_{i,c}(x)$$[1]

여기서 $$i$$는 모든 공간적 위치, $$c \in \{1, ..., C\}$$는 채널 인덱스입니다.

### 2. Attention Residual Learning

핵심 개선사항인 attention residual learning 공식:

$$H_{i,c}(x) = (1 + M_{i,c}(x)) \times F_{i,c}(x)$$[1]

이 방법은 $$M(x)$$가 0에 근사할 때 $$H(x)$$가 원본 특징 $$F(x)$$에 근사하도록 하여 identity mapping을 가능하게 합니다.[1]

### 3. 다양한 Attention 유형

**Mixed Attention (f1)**:

$$f_1(x_{i,c}) = \frac{1}{1 + \exp(-x_{i,c})}$$[1]

**Channel Attention (f2)**:

$$f_2(x_{i,c}) = \frac{x_{i,c}}{\|x_i\|}$$ [1]

**Spatial Attention (f3)**:

$$f_3(x_{i,c}) = \frac{1}{1 + \exp(-(x_{i,c} - \text{mean}_c)/\text{std}_c)}$$[1]

실험 결과 mixed attention이 가장 우수한 성능을 보였습니다.[1]

## 모델 구조

### Soft Mask Branch
- **Bottom-up 과정**: 여러 max pooling을 통해 receptive field를 빠르게 증가시킵니다[1]
- **Top-down 과정**: 대칭적 구조로 전역 정보를 확장하여 각 위치의 입력 특징을 가이드합니다[1]
- **Skip connection**: 서로 다른 스케일의 정보를 포착하기 위해 bottom-up과 top-down 부분 간에 연결을 추가했습니다[1]

### 전체 네트워크 구조
ImageNet용 Attention-56과 Attention-92는 각각 31.9M, 51.3M 파라미터를 가지며, 6.2×10⁹, 10.4×10⁹ FLOPs를 요구합니다.[1]

## 성능 향상

### CIFAR 데이터셋
- **CIFAR-10**: 3.90% 오류율로 state-of-the-art 달성[1]
- **CIFAR-100**: 20.45% 오류율로 최고 성능 기록[1]
- Attention-452는 ResNet-1001보다 훨씬 적은 파라미터로 우수한 성능을 달성했습니다[1]

### ImageNet 데이터셋
- **Top-5 오류율**: 4.8% (single model, single crop)[1]
- ResNet-200 대비 **0.6% top-1 정확도 향상**을 46% trunk depth와 69% forward FLOPs로 달성했습니다[1]
- Attention-56은 ResNet-152 대비 52% 파라미터와 56% FLOPs만으로 더 나은 성능을 보였습니다[1]

### 다양한 기본 유닛 적용
- **ResNeXt**: AttentionNeXt-56이 ResNeXt-101과 동일한 성능을 훨씬 적은 파라미터로 달성[1]
- **Inception**: AttentionInception-56이 Inception-ResNet-v1 대비 0.94% top-1 오류 감소[1]

## 일반화 성능 향상

### 노이즈 저항성
논문의 중요한 발견 중 하나는 **noisy label에 대한 강건성**입니다. Gradient 업데이트 필터로서의 attention mask 특성:[1]

$$\frac{\partial M(x,\theta)T(x,\phi)}{\partial \phi} = M(x,\theta)\frac{\partial T(x,\phi)}{\partial \phi}$$[1]

이 특성으로 인해 mask branch가 잘못된 gradient(noisy label로부터)가 trunk 파라미터를 업데이트하는 것을 방지할 수 있습니다. 실험 결과, 70% 노이즈 레벨에서도 ResNet-164(17.21% 오류) 대비 Attention-92(15.75% 오류)가 우수한 성능을 보였습니다.[1]

### Mixed Attention의 적응성
서로 다른 모듈의 attention-aware feature가 층이 깊어질수록 적응적으로 변화합니다. 예를 들어, 열기구 이미지에서 하위 층의 파란색 특징은 하늘 마스크로 배경을 제거하고, 상위 층의 부분 특징은 풍선 인스턴스 마스크로 정제됩니다.[1]

## 한계

논문에서 명시적으로 언급된 한계는 제한적이지만, 다음과 같은 잠재적 한계들이 있습니다:

1. **계산 복잡성**: Bottom-up top-down 구조로 인한 추가적인 계산 비용
2. **메모리 요구량**: Mask branch 추가로 인한 메모리 사용량 증가
3. **하이퍼파라미터 민감성**: p, t, r 등 여러 하이퍼파라미터 조정의 복잡성[1]

## 향후 연구에 미치는 영향 및 고려사항

### 연구에 미치는 영향

**1. Attention 메커니즘의 패러다임 전환**: 순차적 attention에서 feedforward network 내 통합된 attention으로의 전환점을 제시했습니다.

**2. 효율성과 성능의 동시 달성**: 더 적은 파라미터와 계산량으로 더 나은 성능을 달성할 수 있음을 증명했습니다.

**3. 범용성 입증**: ResNet, ResNeXt, Inception 등 다양한 기본 유닛에 적용 가능함을 보여 광범위한 활용 가능성을 제시했습니다.[1]

### 향후 연구 고려사항

**1. 다른 태스크로의 확장**: 논문에서 언급한 대로 detection, segmentation 등 다른 컴퓨터 비전 태스크에의 적용 연구가 필요합니다.[1]

**2. 더 효율적인 attention 구조**: Bottom-up top-down 구조를 더욱 경량화하면서도 성능을 유지하는 방법 연구.

**3. 자동 하이퍼파라미터 최적화**: p, t, r 등의 하이퍼파라미터를 자동으로 최적화하는 neural architecture search 기법 적용.

**4. 해석 가능성 강화**: Attention map의 시각화 및 해석을 통한 모델의 의사결정 과정 이해 향상.

이 연구는 attention 메커니즘을 convolutional network에 성공적으로 통합한 선구적 연구로, 향후 attention 기반 아키텍처 설계의 중요한 기준점이 될 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4ee9ba1a-7eb8-45e8-a288-dd0399f9b2ae/1704.06904v1.pdf
