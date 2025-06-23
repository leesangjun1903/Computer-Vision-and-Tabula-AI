# Swin Transformer

# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## Abstract
이 논문에서는 컴퓨터 비전의 다양한 목적에 맞게 backbone 역할을 할 수 있는 새로운 비전 트랜스포머인 Swin Transformer를 소개합니다.  
트랜스포머를 언어에서 시각으로 적용하는 데 있어 시각적인 데이터의 규모가 크고 텍스트의 단어에 비해 이미지에서 픽셀의 해상도가 높은 등 두 영역 간의 차이로 인해 어려움이 발생합니다.  
이러한 차이를 해결하기 위해 Shifted Windows(이동되는 윈도우)로 표현이 계산되는 계층적 트랜스포머를 제안합니다.  
이동된 윈도우 방식은 셀프 어텐션 계산을 겹치지 않는 로컬 창으로 제한함으로써 더 큰 효율성을 제공하는 동시에 윈도우 간 연결을 허용합니다.  
기존 ViT는 이미지의 크기에 제곱에 비례했지만, 이 계층적 아키텍처는 다양한 스케일에서 모델링할 수 있는 유연성을 갖추고 있으며 이미지 크기와 관련하여 선형적인 계산량을 가지고 있습니다. (Non-overlapping window내에 있는 patch 간의 self-attention을 수행함으로써 계산복잡도 개선하였음)  
Swin Transformer의 이러한 특성으로 인해 이미지 분류(이미지넷-1K에서 87.3개의 상위 1위 정확도)와 객체 감지(COCO 테스트-디바이스에서 58.7개의 상자 AP 및 51.1개의 마스크 AP) 및 의미론적 분할(ADE20K val에서 53.5mioU)과 같은 광범위한 비전 작업과 호환됩니다.  
성능은 이전 최첨단 기술인 COCO data에서 +2.7개의 상자 AP 및 +2.6개의 마스크 AP를, ADE20K에서 +3.2개의 mIoU를 큰 차이로 능가하여 트랜스포머 기반 모델이 비전 분야에서 잠재력을 입증합니다.  
계층적 설계와 Shifted Window 접근 방식은 모든 MLP 아키텍처에도 유용합니다.  

## Introduction
컴퓨터 비전의 모델링은 오랫동안 CNN에 의해 지배되었습니다.  
AlexNet과 ImageNet image classification 챌린지에 대한 혁신적인 성능을 시작으로 CNN 아키텍처는 더 큰 스케일, 더 광범위한 연결, 더 정교한 convolution 형식을 통해 점점 더 강력해졌습니다.  
다양한 비전 task를 위한 backbone 네트워크 역할을 하는 CNN과 함께 이러한 아키텍처의 발전은 전체 분야를 광범위하게 끌어올린 성능 향상으로 이어졌습니다.

반면에 자연어 처리(NLP)에서 네트워크 아키텍처의 진화는 오늘날 널리 사용되는 아키텍처가 Transformer라는 다른 경로를 택했습니다.  
시퀀스 모델링 및 변환 task를 위해 설계된 Transformer는 데이터의 장거리 의존성 (long-range dependency)을 모델링하는 데 attention을 사용하는 것으로 유명합니다.  
언어 도메인에서의 엄청난 성공으로 연구자들은 컴퓨터 비전에 대한 Transformer를 조사하게 되었으며, 최근 특정 tak, 특히 image classification 분류와 비전-언어 공동 모델링에 대한 유망한 결과를 보여주었습니다.

본 논문은 Transformer가 NLP에서, CNN이 비전에서 하는 것처럼 컴퓨터 비전을 위한 다양한 목적에 맞게 backbone 역할을 할 수 있도록 Transformer의 적용 가능성을 확장하고자 합니다.  
저자들은 언어 도메인에서 비전 영역으로 고성능을 전환하는 데 있어 중요한 문제가 두 modality 간의 차이로 설명될 수 있음을 관찰했습니다.  
이러한 차이점 중 하나는 스케일과 관련이 있습니다.  
언어 Transformer에서 처리의 기본 요소 역할을 하는 단어 토큰과 달리 시각적 요소는 스케일이 상당히 다를 수 있습니다.  
기존 Transformer 기반 모델에서 토큰은 모두 고정된 스케일이며 이러한 컴퓨터 비전 응용화에 적합하지 않은 속성입니다.

또 다른 차이점은 텍스트 구절의 단어에 비해 이미지의 픽셀 해상도가 훨씬 더 높다는 것입니다.  
픽셀 레벨에서 조밀한 예측을 필요로 하는 semantic segmentation과 같은 많은 비전 task가 존재하며, self-attention의 계산 복잡도가 이미지 크기의 제곱에 비례하기 때문에 고해상도 이미지에서 Transformer의 경우 처리하기 어렵습니다.

이러한 문제를 극복하기 위해 본 논문은 계층적 feature map을 구성하고 이미지 크기에 대한 선형 계산 복잡도를 갖는 Swin Transformer라는 범용 Transformer backbone을 제안하였습니다.

- 계층적 구조: 일반적으로 CNN은 동일한 Kernel size를 가져가고 중간에 Maxpooling으로 이미지의 해상도를 줄여가며 수용 필드(receptive field)를 늘려가는 구조입니다. 이렇게 함으로써 더 많은 Representations을 학습할 수 있고, 더 나은 정확도를 가져왔었죠. 이 Pooling 구조를 Transformer에도 도입한 것이 Swin의 첫 번째 제안된 기술입니다. 이렇게 함으로써 다양한 Representations을 학습할 수 있을뿐더러, 줄어든 해상도에서 Attention 연산을 진행하기 때문에 속도에도 이점이 있습니다.

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-fig1.PNG)

위 그림과 같이 Swin Transformer는 작은 크기의 패치 (회색 윤곽선)에서 시작하여 점점 더 깊은 Transformer 레이어에서 인접한 패치를 병합하여 계층적 표현을 구성합니다.  
이러한 계층적 feature map을 통해 Swin Transformer 모델은 feature pyramid network (FPN) 또는 U-Net과 같은 조밀한 예측을 위한 고급 기술을 편리하게 활용할 수 있습니다.

이미지를 분할하는 겹치지 않는 window (빨간색 윤곽선) 내에서 로컬 영역으로 self-attention을 계산하여 선형적인 계산량을 가지게 됩니다.  
각 window의 패치 수는 고정되어 있으므로 복잡도는 이미지 크기에 비례합니다.  
이러한 장점으로 인해 Swin Transformer는 단일 해상도의 feature map을 생성하고 2차 복잡도를 갖는 이전 Transformer 기반 아키텍처와 달리 다양한 비전 task를 위한 범용 backbone으로 적합하다.

이미지가 (b)의 그림같이 패치로 나누어져 이 패치들이 각각의 Query, Key로 작동하여 기존 Transformer 같이 연산이 진행됩니다.  
하지만, 여기서 문제점이 무엇이냐 하면 일반적으로 문장에서 단어의 길이는 많아야 50 정도 될 것입니다.  
그럼 이 단어들이 Query, Key로 각각 연산되겠지요.  
하지만 이미지의 경우를 보면 이미지는 보통 224 * 224 해상도가 되는데 이 모든 픽셀이 Query, Key로 작동한다면 시간이 엄청 걸릴 것입니다.  
ViT의 경우 이미지 패치로 작동되지만 그래도 시간이 오래 걸리는 것에는 변함이 없습니다.

여기서 제안한 기술은 고정된 Window(M * M의 크기)를 이미지 패치에 적용하여, Attention의 연산이 Window 안에서만 연산되게 하는 방식입니다.  
전체가 아닌, 위 빨간색 테두리 안에서만 연산을 진행하자는 것입니다.  
이렇게 하면 당연히 전체 이미지 패치에 대해 연산하는 ViT에 비해서는 당연히 속도가 올라갈 것입니다.

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-fig2.PNG)

Swin Transformer의 핵심 설계 요소는 위 그림과 같이 연속적인 self-attention 레이어 사이의 window 파티션의 이동입니다.  
Shifted window는 이전 레이어의 window를 연결하여 모델링 능력을 크게 향상시키는 연결을 제공합니다.  
이 전략은 또한 실제 지연 시간과 관련하여 효율적입니다.  
Window 내의 모든 쿼리 패치는 하드웨어에서 메모리 액세스를 용이하게 하는 동일한 키 세트를 공유합니다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbUbfJn%2FbtsC0ZqtpMk%2FfiAUKZitL6fQ2HDBKkh4l0%2Fimg.png)

위 그림은 각 Window의 인덱스를 매겨보았습니다.  
[0-15]의 인덱스를 가지는 Windows가 존재하는데, 각 Windows가 독립적으로 Attention 연산을 진행하면 위 그림처럼 서로 다른 인덱스를 가지는 Window 끼리는 Attention 연산이 진행되지 않습니다.  
그럼 장기의존성을 가져 전역적으로 정보를 집계하는 Transformer의 구조를 깨버리게 되는 것이고, 당연히 성능 또한 저하됩니다.

그래서 이 부분을 Shift Windows 방식으로 각각의 윈도끼리의 Attention 연산을 진행하는 것입니다.  
모든 Windows끼리는 아니더라도 일부 Shift Windows 방식을 진행하여 연산 속도와 정확도 둘 다 월등한 성능을 기록하였습니다.

이전의 슬라이딩 window 기반 self-attention 접근 방식은 다른 query 픽셀에 대한 다른 key 세트로 인해 일반 하드웨어에서 낮은 지연 시간으로 어려움을 겪고 있었습니다.  
제안된 shifted window 방식이 sliding window 방식보다 지연 시간이 훨씬 짧지만 모델링 능력은 비슷합니다.  
또한 shfted window 접근 방식은 모든 MLP 아키텍처에도 유익합니다.

## Method
### Overall Architecture
![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-fig3.PNG)

Swin Transformer 아키텍처의 개요는 tiny 버전인 SwinT를 보여주는 위 그림에 나와 있습니다.  
먼저 ViT와 같은 패치 분할 모듈에 의해 입력 RGB 이미지를 겹치지 않는 패치로 분할합니다.  
각 패치는 “토큰”으로 취급되며 해당 feature는 픽셀 RGB 값의 concatenation으로 설정됩니다.  
구현에서는 $4×4$ 의 패치 크기를 사용하므로 각 패치의 feature 차원은 $4 \times 4 \times 3 = 48$ 입니다.  
이 feature에 Linear Embedding 레이어를 적용하여 임의의 차원 $C$ 로 project합니다.

수정된 self-attention 계산(Swin Transformer 블록)이 포함된 여러 Transformer 블록이 이러한 패치 토큰에 적용됩니다.  
Transformer 블록은 토큰 수 $\frac{H}{4} \times \frac{W}{4}$ 를 유지하며 선형 임베딩과 함께 “1단계”라고 부릅니다.  

계층적 표현을 생성하기 위해 네트워크가 깊어짐에 따라 레이어를 패치 병합하여 토큰 수를 줄입니다.  
첫 번째 패치 병합 레이어는 $2×2$ 인접 패치의 각 그룹의 feature를 concat하고 $4C$ 차원의 concat된 feature에 linear layer를 적용합니다.  
이렇게 하면 4의 배수만큼 토큰 수가 줄어들고 출력 차원은 $2C$로 설정됩니다.  
Swin Transformer 블록은 $\frac{H}{8} \times \frac{W}{8}$ 에서 해상도를 유지하면서 feature 변환을 위해 나중에 적용됩니다.  
이 패치 병합 및 feature 변환의 첫 번째 블록은 “2단계”로 부릅니다.  
이 절차는 각각 $\frac{H}{16} \times \frac{W}{16}$ , $\frac{H}{32} \times \frac{W}{32}$ 의 출력 해상도로 “3단계”와 “4단계”로 두 번 반복됩니다.  
이러한 단계는 일반적인 convolution network (ex. VGG, ResNet)와 동일한 feature map 해상도로 계층적 표현을 공동으로 생성합니다.  
결과적으로 제안하는 아키텍처는 다양한 비전 task를 위해 기존 방식의 backbone 네트워크를 편리하게 대체할 수 있습니다.

그럼 Patch Merging을 이용해 해상도를 어떻게 줄일까요?   
Stage 2를 예시로 들면 아래와 같은 그림이 나옵니다.  

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb8RzPc%2FbtsC762oYrt%2FqpqJ2YpoVSQM6BpBvokyUk%2Fimg.png)

Stage 1의 출력인 $\frac {H} {4} \times \frac {W} {4} \times C$ 의 차원을 $2 \times 2$ 그룹들로 나눕니다.  
나눠진 하나의 그룹은 $\frac{H} {8} \times \frac{W} {8} \times C$ 의 차원을 가지고, 4개의 그룹들을 채널을 기준으로 병합합니다(Concat).  
병합된 $\frac{H} {8} \times \frac{W} {8} \times 4C$ 의 차원 축소를 위해 절반인 $2C$ 의 차원으로 축소합니다.  
위 과정들은 모든 Stage에서 동일하게 작용합니다.

이러한 계층적 구조는, 일반적인 Representations보다 더 계층적인 Representations을 학습할 수 있고, 앞선 전술한 것처럼 줄어든 차원만큼 연산속도에도 이점이 있습니다.

#### Swin Transformer block
Swin Transformer는 Transformer 블록의 multi-head self-attention (MSA) 모듈을 shifted window를 기반으로 하는 모듈로 교체하고 다른 레이어는 동일하게 유지함으로써 구축됩니다.  
Swin Transformer 블록은 shifted window 기반 MSA 모듈과 중간에 GELU nonlinearity가 있는 2-layer MLP로 구성됩니다.  
LN (LayerNorm) layer는 각 MSA 모듈과 각 MLP 이전에 적용되고 residual connection은 각 모듈 이후에 적용됩니다.

### Shifted Window based Self-Attention
표준 Transformer 아키텍처와 image classification을 위한 모델은 토큰과 다른 모든 토큰 간의 관계가 계산되는 글로벌 self-attention을 수행합니다.  
글로벌 계산은 토큰 수와 관련하여 2차 복잡도를 초래하여 조밀한 예측이나 고해상도 이미지를 나타내기 위해 막대한 토큰 세트가 필요한 많은 비전 문제에 적합하지 않습니다.  

#### Self-attention in non-overlapped windows
본 논문은 효율적인 모델링을 위해 로컬한 window 내에서 self-attention을 계산할 것을 제안합니다.  
Window는 겹치지 않는 방식으로 이미지를 균등하게 분할하도록 배열됩니다.  
각 window에 $M \times M$ 개의 패치가 포함되어 있다고 가정하면 글로벌 MSA 모듈의 계산 복잡도와 $h \times w$ 패치 이미지를 기반으로 하는 window의 계산 복잡도는 다음과 같습니다.  

```math
\begin{aligned}
\Omega (\textrm{MSA}) = 4hwC^2 + 2(hw)^2 C \newline
\Omega (\textrm{W-MSA}) = 4hw C^2 + 2 M^2 hw C 
\end{aligned}
```

$Ω$ 기호는 연산에 얼마나 시간이 걸리는지 측정한 기준입니다. 위의 식은 ViT, 아래 식은 Swin입니다.  
여기서 $M$은 윈도의 크기인데 보통은 7로 고정합니다.

여기서 MSA는 패치 수 $hw$ 에 대해 2차이고 W-MSA는 M이 고정된 경우 (default는 7) 선형입니다.  
기존 방법은 해상도에 따라 2차원적으로 계산량이 증가합니다.  
다시 말해, 해상도가 올라가면 계산량이 기하급수적으로 증가한다는 의미입니다.  
반면, Swin의 경우 윈도우의 크기는 보통 고정되어 있으니 상수처럼 취급하고, $hw$의 크기에서만 선형적으로 계산량이 증가하기 때문에 계산적인 부분에서는 상당한 이점이 있습니다.
글로벌 self-attention 계산은 일반적으로 큰 $hw$ 에 적합하지 않은 반면 window 기반 self-attention은 확장 가능합니다.

#### Shifted window partitioning in successive blocks
Window 기반 self-attention 모듈은 window 간의 연결이 부족하여 모델링 능력이 제한됩니다.  
저자들은 겹치지 않는 창의 효율적인 계산을 유지하면서 window 사이의 연결을 도입하기 위해 연속되는 Swin Transformer 블록에서 두 개의 파티션 구성을 번갈아 가며 전환하는 shifted window 파티셔닝 방식을 제안하였습니다.  

첫 번째 모듈은 왼쪽 상단 픽셀에서 시작하는 일반적인 window 분할 전략을 사용하고 $8 × 8$ feature map은 크기가 $4 × 4$ $(M = 4)$ 인 $2 × 2$ 창으로 고르게 분할됩니다.  
그러면 다음 모듈은 규칙적으로 분할된 window에서 $(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor)$ 픽셀만큼 window를 대체하여 이전 레이어의 구성에서 shifted window 구성을 채택합니다.  Shifted window 파티셔닝 접근 방식을 사용하면 연속되는 Swin Transformer 블록이 다음과 같이 계산됩니다.  

```math
\begin{aligned}
\hat{z}^l = \textrm{W-MSA} (\textrm{LN} (z^{l-1})) + z^{l-1} \\
z^l = \textrm{MLP} (\textrm{LN} (\hat{z}^l)) + \hat{z}^l \\
\hat{z}^{l+1} = \textrm{SW-MSA} (\textrm{LN} (z^l)) + z^l \\
z^{l+1} = \textrm{MLP} (\textrm{LN} (\hat{z}^{l+1})) + \hat{z}^{l+1} \\
\end{aligned}
```

여기서 $\hat{z}^l$ 와 $\z^l$ 은 각각 블록 $l$ 에 대한 SW-MSA 모듈과 MLP 모듈의 출력 feature를 나타냅니다.  
W-MSA와 SW-MSA는 각각 일반 및 shfted window 파티션 구성을 사용하는 window 기반 MSA를 나타냅니다.

Shifted window 파티셔닝 접근법은 이전 레이어에서 인접한 겹치지 않는 window 사이의 연결을 도입하고 image classification, object detection, semantic segmentation에 효과적입니다.

#### Efficient batch computation for shifted configuration
Shifted window 파티셔닝의 문제는 더 많은 window를 만들며, 일부 창은 $M \times M$보다 작아야 합니다.

```math
\begin{equation}
\lceil \frac{h}{M} \rceil \times \lceil \frac{w}{M} \rceil \rightarrow \bigg( \lceil \frac{h}{M} \rceil + 1 \bigg) \times \bigg( \lceil \frac{w}{M} \rceil + 1 \bigg)
\end{equation}
```

Naive한 해결책은 attention을 계산할 때 더 작은 window를  $M \times M$ 크기로 채우고 패딩된 값을 마스킹하는 것입니다.  
일반 파티셔닝의 window 수가 적은 경우, 이 naive한 솔루션으로 증가된 계산은 상당합니다.  
예를 들어 $2×2$ 의 경우 $3×3$이 되어 window 수가 2.25배 커집니다.  

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-fig4.PNG)

본 논문은 위 그림과 같이 왼쪽 위 방향으로 순환 이동 (cyclic-shifting)하여 보다 효율적인 배치 계산 방식을 제안합니다.  
이 이동 후 일괄 처리된 window는 feature map에서 인접하지 않은 여러 개의 하위 window로 구성될 수 있습니다.  
따라서 마스킹 메커니즘을 사용하여 각 하위 window 내에서 self-attention 계산을 제한합니다.  
Cyclic-shifting를 사용하면 배치된 window의 수가 일반 window 파티션과 동일하게 유지되므로 효율적입니다.  

#### Relative position bias
Self-attention 계산에서 각 head에 대한 Relative position bias $B \in \mathbb{R}^{M^2 \times M^2}$ 를 포함합니다.

```math
\begin{equation}
\textrm{Attention} (Q, K, V) = \textrm{SoftMax} (\frac{QK^\top}{\sqrt{d}} + B) V
\end{equation}
```
Q : Query, K : Key, V : Value 행렬이며 d는 query, key 의 차원입니다.  
$M^2$ 은 window patch 수입니다.  
각 축을 따라 상대적 위치가 $[-M+1, M-1]$ 에 있기 때문에 더 작은 크기의 바이어스 행렬 $\hat{B} \in \mathbb{R}^{(2M−1) \times (2M−1)}$ 을 parameterize하고 $B$ 의 값은 $\hat{B}$ 에서 가져옵니다.  
사전 학습 시 학습된 Relative position bias는 bi-cubic interpolation을 통해 다른 window 크기의 fine-tuning을 위한 모델을 초기화하는 데에도 사용할 수 있습니다.  

Swin Transformer의 성능을 더 이끌 방법으로 Relative position bias를 각 헤드에 더해주었습니다.  
기존에는 Absolute bias를 더해주었는데, 이 부분은 오히려 성능 저하를 나타냈다고 합니다.

### Architecture Variants
저자들은 ViT-B/DeiT-B와 유사한 모델 크기와 계산 복잡도를 갖도록 Swin-B라는 기본 모델을 구축하였습니다.  
또한 모델 크기와 계산 복잡도가 각각 약 0.25배, 0.5배, 2배인 버전인 Swin-T, Swin-S, Swin-L을 도입합니다.  
Swin-T와 Swin-S의 복잡도는 각각 ResNet-50 (DeiT-S)과 ResNet-101의 복잡도와 유사합니다.

## Experiments
- 데이터셋 :
ImageNet-1K image classification, COCO object detection, ADE20K semantic segmentation

### Image Classification on ImageNet-1K
![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table1.PNG)

이미지 분류에서, 왼쪽 결과를 보면 비슷한 파라미터를 가진 모델 중, FLOPs가 가장 낮으며 정확도는 가장 높게 나왔습니다.  
이것은 속도와 정확도 간에 더 나은 Trade-off를 가졌다고 할 수 있고, 모든 구조가 Transformer로 구성되었기 때문에 더 나은 향상을 위한 잠재력이 남아있다고 합니다.

또한, 오른쪽 결과는 더 큰 데이터 세트에 대해 사전 학습 시킨 것인데, 역시 Transformer의 특성에 맞게 많은 데이터에 대한 사전 학습을 시키니 성능이 CNN 모델과 기존 모델의 성능 향상을 볼 수 있습니다.

### Object Detection on COCO
![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table2.PNG)

### Semantic Segmentation on ADE20K
![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table3.PNG)

### Ablation Study
다음은 분류, Object Detection, Semantic Segmentaiton에서 Shifted windows 방식과 Relative position bias를 제거하였을 때 성능에 얼마나 영향을 미치는 가에 대한 결과입니다.

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table4.PNG)

다음은 다양한 self-attention 계산 방법에 대한 실제 속도를 비교한 표입니다.

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table5.PNG)

맨 아래 cyclic 방식이 모든 스테이지에서 가장 빠른 성능을 보였습니다.

다음은 다양한 self-attention 계산 방법을 사용한 Swin Transformer의 정확도를 비교한 표입니다.

![](https://kimjy99.github.io/assets/img/swin-transformer/swin-transformer-table6.PNG)


# Reference
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/swin-transformer
- https://heeya-stupidbutstudying.tistory.com/entry/DL-Swin-Transformer-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://velog.io/@9e0na/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CV-Swin-Transformer2021-Summary
- https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://velog.io/@jus6886/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://deep-learning-study.tistory.com/728
