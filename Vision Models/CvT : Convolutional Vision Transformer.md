# CvT : Introducing Convolutions to Vision Transformers

## Abstract
이 논문에서 Convolution vision Transformer(CvT)라는 새로운 architecture를 소개합니다.  
이는 ViT의 performanc와 efficiency를 개선하기 위해 convolution을 ViT에 도입하여 두 design의 장점을 모두 제공합니다.  
이를 위해 두 가지 주요 수정 사항을 도입했습니다.  
1. 새로운 convolutional token embedding을 포함하는 hierarchy of Transformers 
2. convolution projection을 사용하는 convolution Transformer block

이러한 수정 사항은 ViT architecture에 CNN의 바람직한 properties(i.e. shift, scale, and distortion invariance)을 도입하면서도 Transformer의 장점(i.e. dynamic attention, global context, and better generalization)을 유지합니다.  
우리는 CvT가 ImageNet-1K에서 fewer parameters and lower FLOPs로도 SOTA를 달성하면서, 다른 Vision Transformers와 ResNets을 능가하는 성능을 보여주는 광범위한 실험을 통해 검증했습니다.  
또한 더 큰 Dataset(e.g. ImageNet-22K)에서 Pretrained되고 downstream 작업에 맞추어 finetuning될 때도 성능 이점을 유지합니다.  
ImageNet-22K에서 Pretrained된 CvT-W24는 ImageNet-1K val set에 대해서 87.7% top-1 accuracy를 달성했습니다.

## Introduction
Transformer는 NLP task에서 다양하게 사용되고 있습니다.  
ViT는 large scale에서 경쟁력 있는 image classification 성능을 얻기 위해 Transformer architecture에만 의존하는 최초의 computer vision model입니다.  
ViT design은 language understanding을 위한 Transformer architecture를 최소한의 수정으로 만들어졌습니다.  
먼저, image를 개별적인 non-overlapping한 16×16 patches로 분할합니다.  
그 다음, patch들은 NLP에서의 token으로 취급되며, positional encoding과 함께 합산되어, global relations을 modeling하기 위해 반복되는 standard Transformer layer에 입력됩니다.  

Vision Transformer가 큰 성공을 이뤘음에도 불구하고, 적은 양의 Data로 traing할 때는 유사한 크기의 CNN model(e.g. ResNets)보다 성능이 여전히 낮습니다.  
그 이유 중 하나는 ViT가 vision task를 해결하는 데 CNN architecture에 내재된 특정 바람직한 특성이 부족하기 때문일 수 있습니다.  
예를 들어, image에는 강력한 2D local structure가 있다. (spatially neighboring pixels are usually highly correlated)  
CNN architecture는 local receptive fields, shared weights, and spatial subsampling을 사용하여 이러한 local structure를 강제로 capture하게 합니다.
따라서 어느 정도의 shift, scale, and distiortion invariance를 달성할 수 있습니다.  

또한 convolutional kernels의 hierarchical structure는 단순한 low-level의 edge와 texture에서부터 higher order semantic pattern에 이르기까지 다양한 수준의 complexity를 갖는 local spatial context를 고려한 visual pattern을 학습합니다.  
이 논문에서는 Convolution을 ViT 구조에 도입하여 performance와 robustness를 개선하면서 동시에 높은 수준의 computational and memory efficiency를 유지할 수 있다고 가설을 세웠습니다.
이러한 가설을 검증하기 위해, Convolution을 Transformer에 도입하여 고유의 효율성을 가진 새로운 architecture인 Convolutional Vision Transformer(CvT)를 제안합니다.  

CvT design은 ViT architecture의 두 핵심 부분에 convolution을 도입합니다.  

Transformer를 hierarchical structure를 형성하는 multiple stages로 나눕니다.  
각 stage의 시작 부분은 overlapping(겹치는) convolution operation을 수행하는 token embedding으로 구성되며, 이는 flatten된 token sequence를 다시 spatial grid로 reshape합니다.  
이어서 Layer normalization을 수행합니다.  
이를 통해 model은 local information을 capture할 뿐만 아니라, 단계별로 sequence length를 점진적으로 줄이면서 token feature의 dimension을 증가시킬 수 있습니다.  
이는 spatial downsampling을 수행하면서 Feature map의 수를 증가시키는 CNN의 형태와 비슷합니다.

Transformer module의 각 self-attention block 전에 있는 linear projection을 제안된 convolution projection으로 대체합니다.  
이 projection은 $s×s$ depth-wise separable convolution 연산을 2D로 reshape된 token map에 적용합니다.  
이를 통해 model은 attention mechanism에서 local spatial context를 더 잘 capture하고 semantic ambiguity(의미적 모호성)을 줄일 수 있습니다.  
또한 convolution의 stride를 사용하여 key and value matrices를 subsampling하여 computational complexity를 관리할 수 있으며, 성능 저하를 최소화하면서 efficiency를 4× 이상 향상시킬 수 있습니다.

요약하자면, Convolutional Vision Transformer(CvT)는 CNN의 모든 장점(local receptive fiels, shared weights, and spatial subsampling)과 Transformer의 모든 장점(dynamic attention, global context fusion, and better generalization)을 활용합니다.  
CvT는 CNN-based model과 Transformer-based model에 비해 fewer FLOPS and parameters를 사용하면서도 성능을 향상시킵니다.

## Related Work
self-attention을 사용하여 global dependencies를 전적으로 의존하는 transformer model은 natual language modeling에서 지배적이었습니다.  
최근에는 Transformer based architecture가 visual recognition tasks에서도 CNN의 유효한 대안으로 간주되고 있습니다. (classification, object detection, segmentation, image enhancement, image generation, video processing, and 3D point cloud processing)  

### Vision Transformers.
ViT는 순수 transformer architecture가 image classification에서 SOTA를 달성할 수 있음을 최초로 증명했습니다.  
이는 Data가 충분히 큰 경우에 해당합니다.  
DeiT는 ViT의 data-efficient training 및 distillation을 추가로 탐구했습니다.  

이 연구에서는 image classification에서 local 및 global dependencies를 효율적으로 modeling하기 위해 CNN과 Transformer를 결합하는 방법을 연구했습니다.  

Vision transformer에서 local context를 더 잘 modeling하기 위해, 일부 연구들은 디자인 변환을 시도했습니다.  
예를 들어, Conditional Position encodings Visual Transformer(CPVT)는 ViT에서 사용되는 사전 정의된 positional embdding을 conditional positoin encodings(CPE)로 대체하여 transformer가 임의 크기의 input image를 interpolation 없이 처리할 수 있게 합니다.  
Transformer-iN-Transformer (TNT)는 patch embdding을 처리하는 outer Transformer block과 pixel embdding 간의 relation을 modeling하는 inner transformer block을 모두 사용하여 patch-level 및 pixel-level representation을 modeling합니다.  
Tokens-to-Token(T2T)는 sliding window 내에서 여러 token을 하나의 token으로 연결하여 ViT에서 tokenization을 주로 개선합니다.
그러나 이 작업은 근본적으로 convolution과는 특히 normalization details에서 다르며, 여러 token을 concatenation하는 것은 computation and memory complexity를 크게 증가시킵니다.
PVT(Pyramid vision transformer)는 CNN에서의 multi-scales과 유사하게 Transformer를 위한 multi-stage design(without convolutions)을 통합했습니다.  

이러한 현재의 연구들과는 대조적으로, 본 연구는 image domain specific inductive biases를 가진 convolution을 Transformer에 도입하여 두 architecture의 장점을 모두 달성하는 것을 목표로 합니다.  
Table 1.에서는 위에서 언급한 현재 진행중인 연구들과 우리의 연구 간의 주요 차이점을 positional encodings, type of token embdding, type of projectoin, and Transformer structure in the backbone 측면에서 보여줍니다.

![](https://velog.velcdn.com/images/hseop/post/bbfd7bbf-ec34-4fed-a505-72ac6ceef391/image.png)

### Introducing Self-attentions to CNNs.
Self-attention mechanism은 vision task에서 CNN에 널리 적용되어 왔습니다.  
이러한 연구들 중에서, Non-local networks는 global attention을 통해 long range dependencies를 capture하도록 설계되었습니다.  

Local relation networks는 local window 내에서 pixel/feature 간의 Compositional relation에 기반하여 weight aggregation을 조정합니다.  
이는 공간적으로 인접한 input feature에 대해 고정된 weight aggregation을 사용하는 convolution layer와는 대조적입니다.  
이러한 adaptive weight aggregation은 recognition에 중요한 geometric priors(기하학적 사전)을 network에 도입합니다. 

최근 BoTNet은 ResNet의 마지막 3개의 bottleneck block에서 spatial convolution을 global self-attention으로 대체하는 backbone architecture를 제안하여 image recognition에서 좋은 성능을 달성했습니다.
 
반면, 우리의 연구는 반대 방향으로 진행됩니다 : Transformer에 convolution을 도입하는 것입니다.

### Introducing Convolutions to Transformers.
NLP 및 speech recogntion에서는, convolution이 Tranformer block을 수정하기 위해 사용되었습니다.
Multi-head attention을 conv layer로 대체하거나, 추가적인 conv layer를 병렬로 또는 순차적으로 추가하여 local relationships을 capture합니다.  
다른 선행 연구는 attention map을 residual connection을 통해 후속 layer로 전파하는 방법을 제안했는데, 이 때 attention map은 먼저 convolution으로 변환됩니다.

이러한 연구와 달리, 우리는 vision transformer의 두 주요 부분에 convolution을 도입하는 것을 제안하였습니다.  
Attention operation을 위한 기존의 position-wise linear projection을 우리의 convolutional projection으로 대체하였고 CNN과 유사하게 다양한 Resolution의 2D reshaped toekn map을 가능하게 하는 Multi-stage structure를 사용합니다.  
우리의 unique design은 이전 연구들에 비해 상당한 performance and efficiency benefits 이점을 제공합니다.  

## Convolutional vision Transformer
CvT의 전체 pipeline은 다음과 같습니다.  

![](https://velog.velcdn.com/images/hseop/post/2bfda6dc-2e0a-4a70-8ffc-03642b0609e7/image.png)

우리는 Vision Transformer architecture에 두 가지 convolution-based operation인 Convolutional Token Embedding과 Convolutional Projection을 도입했습니다.  
CNNs에서 사용한 multi-stage hierarchy design이 사용되며, 이 연구에서는 3 stages가 사용됩니다.  

![](https://velog.velcdn.com/images/hseop/post/0fbf092d-770a-4725-b76c-7684057053f3/image.png)

각 stage는 두 부분으로 구성됩니다.  
Input image(or 2D로 reshaped된 token map)는 Convolutional Token Embedding layer를 거치게 되며, 이는 overlapping patch를 사용한 convolution으로 구현됩니다.  
token은 input으로 2D spatial grid로 재구성되고 (overlap 정도는 stride length를 통해 조절할 수 있음) 추가로 layer normalization이 token에 적용됩니다.  

이를 통해 각 stage는 token의 수(i.e. feature resolution)를 점진적으로 줄이는 동시에 token의 width(i.e. feature dimension)을 증가시키게 되어, CNNs의 design과 유사하게 spatial downsampling 및 representation의 richness(풍부함)를 달성할 수 있습니다.  
다른 이전의 Transformer-based architecture와 달리, CvT는 token에 ad-hod position embdding(임시 위치 임베딩)을 더하지 않습니다.  

다음으로 제안된 Convolutional Transformer Block의 스택은 각 단계의 나머지 부분을 구성합니다.

![](https://velog.velcdn.com/images/hseop/post/e8611d37-6ec7-4363-b4fd-89942a9b4448/image.png)

그림 2(b)는 Convolutional Transformer Block의 아키텍처를 보여줍니다.  
여기서 Convolutional Projection이라고 하는 깊이별 분리 가능한 컨볼루션 연산[5]은 표준 위치 방식 대신 쿼리, 키 및 값 임베딩에 각각 적용됩니다.  
또한 classification token은 마지막 단계에서만 추가됩니다.  
마지막으로, 클래스를 예측하기 위해 최종 단계 출력의 classification token에 MLP(multi-layer perceptron) 헤드가 활용됩니다.  

### Convolutional Token Embedding
공식적으로, 이전 단계 $x_{i-1} \in \mathbb {R}^{H_{i-1} * W_{i-1} * C_{i-1}}$ 의 2D 이미지 또는 2D 형상의 출력 토큰 맵이 단계 $i$에 입력으로 주어지면, 우리는 $x_{i-1}$ 을 $C_i$ 를 가진 새로운 토큰 $f(x_{i-1}) \in \mathbb {R}^{H_i * W_i * C_i}$ 에 매핑하는 함수 $f(·)$ 를 학습합니다.  
여기서 $f(·)$는 커널 크기 : $s × s$, 보폭 : $s − o$ 및 $p$ : 패딩(경계 조건을 처리하기 위해)의 2D 컨볼루션 연산입니다.  

![](https://velog.velcdn.com/images/kiolke/post/39389214-0042-423f-9614-c841af85c0c4/image.png)

$f(x_{i-1})$ 은 $H_i * W_i * C_i$ 로 flatten되고 단계 i의 후속 Transformer block에 입력하기 위해 계층 정규화에 의해 정규화됩니다.  

Convolution channel 개수 $C_i$ 를 통해 한 token(feature) dimension을 조절할 수 있습니다. ($C_i$ == kernel 개수 == token dimension)  
kerel size 및 stride를 통해 token 개수($H_i$ × $W_i$) 도 조절 가능합니다.  

컨볼루션 토큰 임베딩 레이어를 사용하면 컨볼루션 작업의 다양한 매개 변수를 통해 각 단계에서 토큰 특징 치수와 토큰 수를 조정할 수 있습니다.  
이러한 방식으로 각 단계에서 토큰 기능 차원을 늘리면서 토큰 시퀀스 길이를 점진적으로 줄입니다.  
이를 통해 토큰은 CNN의 특징 계층과 유사하게 점점 더 큰 공간 풋프린트에 대해 점점 더 복잡한 시각적 패턴을 나타낼 수 있습니다.

### Convolutional Projection for Attention
Convolutional Projection layer의 목표는 local spatial context를 추가로 modeling하고, $K$와 $V$ matrices를 downsampling할 수 있도록 하여 Efficiency를 향상시키는 것입니다.  
기본적으로, Convolutional Projection을 포함한 Transformer block은 original Transformer block의 일반화된 형태입니다.  
우리는 Mult-Head Self-Attention(MHSA)을 위한 기존의 position-wise linear projection을 depth-wise separable convolution으로 대체하여 Convolutional Projection layer를 제안합니다.  

![](https://velog.velcdn.com/images/kiolke/post/186cf8fc-2627-4b2e-96f3-85406111a935/image.png)

#### Implementation Details

![](https://velog.velcdn.com/images/hseop/post/49d207c9-cf0e-4759-9a6c-a8212ee0f6de/image.png)

(a)는 ViT에서 사용된 원래 position-wise linear projection

![](https://velog.velcdn.com/images/hseop/post/e4a3911b-2f2f-477b-bfc7-f5a15b2f6b37/image.png)

(b)는 CvT의 $s×s$ Convolutional Projection이다.
토큰은 먼저 2D 토큰 맵으로 재구성됩니다.  
다음으로, 커널 크기가 $s$인 깊이별 분리 가능한 컨볼루션 레이어를 사용하여 컨볼루션 투영을 구현합니다.  
마지막으로, 투사된 토큰은 후속 프로세스를 위해 1D로 평면화됩니다. 이것은 다음과 같이 공식화될 수 있습니다.  

![](https://velog.velcdn.com/images/kiolke/post/8746f1d0-6d20-460d-a2e0-2c8b06693e0c/image.png)

$x^{q/k/v}_{i}$ 는 계층 $i$에서 $Q/K/V$ 행렬에 대한 토큰 입력이고, xi는 컨볼루션 투영 이전의 방해받지 않은 토큰이며, Conv2d는 깊이별 분리 가능한 컨볼루션으로 구현되며, Depth-wise Conv2d → BatchNorm2d → Point-wise Conv2d에 의해 구현된 커널 크기를 가리킵니다.  
Convolutional Projection 레이어가 있는 새로운 Transformer Block은 원래 Transformer Block 디자인을 일반화한 것입니다.  
원래의 위치별 선형 투영 레이어는 커널 크기가 $1 × 1$인 컨볼루션 레이어를 사용하여 trivial한 방식으로 구현될 수 있습니다.  

#### Efficiency Considerations
첫 번째로 우리는 efficient convolution을 사용합니다.  
컨볼루션 투영에 표준 $sxs$ 컨볼루션을 직접 사용하려면 $s^2C^2$ 매개변수와 $O(s^2C^2T)$ FLOPs(계산 복잡도)가 필요합니다.  
여기서 C는 토큰의 채널 차원이고 T는 처리를 위한 토큰 수입니다.  
대신에, 우리는 standard s×s convolution을 depth-wise separable convolution으로 나눕니다.  
이를 통해 제안된 Convolutional Projection은 원래의 position-wise linear projection과 비교하여 $s^2C$ parameters and $O(s^2C^2T)$ FLOPs만을 추가로 도입하게 되어, 전체 parameter수와 FLOPs 측면에서 무시할 수 있는 수준입니다.  

둘째, 제안된 Convolutional Projection을 활용하여 MHSA 연산에 대한 계산 비용을 줄입니다.  
$s × s$ Convolutional Projection은 1보다 큰 보폭을 사용하여 토큰 수를 줄이는 것을 허용합니다.  

![](https://velog.velcdn.com/images/hseop/post/4259bcbb-e52f-41e9-8692-43ca2900f6e6/image.png)

그림 3(c)는 1보다 큰 stride를 갖는 컨볼루션을 사용하여 키 및 값 프로젝션을 서브샘플링하는 컨볼루션 프로젝션을 보여줍니다.  
키 및 값 프로젝션에 2의 스트라이드를 사용하고 쿼리에 1의 스트라이드를 변경하지 않고 그대로 둡니다.
이러한 방식으로 키와 값에 대한 토큰의 수는 4배 감소하고, 나중 MHSA 작업을 위해 계산 비용은 4배 감소합니다.  

이미지의 인접 픽셀/패치는 모양/의미에서 중복성을 갖는 경향이 있기 때문에 성능 저하가 최소화됩니다.  
또한 제안된 Convolutional Projection의 로컬 컨텍스트 모델링은 해상도 감소로 인한 정보 손실을 보완합니다.  

### Methodological Discussions
#### Removing Positional Embeddings:
모든 Transformer block에 Convolutional Projections을 도입하고 Convolutional Token Embedding을 결합함으로써, network를 통해 local spatial relationships을 modeling할 수 있는 능력을 얻게 되었습니다.  
이러한 built-in 속성 덕분에 network에서 position embedding을 제거해도 성능에 영향을 주지 않으며, 이는 experiments(Section 4.4)에서 입증되었습니다.  

![](https://velog.velcdn.com/images/hseop/post/ac3487d7-4d2a-4706-b05a-8f7910ddf78c/image.png)

![](https://velog.velcdn.com/images/hseop/post/63541a98-69c1-40a7-b865-45bd8450b151/image.png)

## Experiments
이 섹션에서는 대규모 이미지 분류 데이터 세트에서 CvT 모델을 평가하고 다양한 다운스트림 데이터 세트로 전송합니다.  
또한 제안된 아키텍처의 설계를 검증하기 위해 ablation 연구를 수행합니다.

### Setup
평가를 위해 130만 개의 이미지와 1k 개의 클래스가 있는 ImageNet 데이터 세트와 22k 개의 클래스와 14M 개의 이미지를 가진 슈퍼셋 ImageNet-22k를 사용합니다.  
ImageNet-22k에서 사전 훈련된 모델을 [18, 11]에 이어 CIFAR-10/100 [19], Oxford-IIIT-Pet [23], Oxford-IIIT-Flower [22]를 포함한 다운스트림 작업으로 추가로 전송합니다.  

#### Model Variants

![](https://velog.velcdn.com/images/hseop/post/37340575-a9f0-4f21-b4c2-1c9aeeedc154/image.png)

표 2와 같이 각 단계의 Transformer 블록 수와 사용된 hidden feature dimension을 변경하여 다양한 매개변수와 FLOP으로 모델을 인스턴스화합니다.  

#### Training
AdamW [21] 옵티마이저는 CvT-13의 경우 0.05, CvT-21 및 CvT-W24의 경우 0.1의 가중치 감쇠로 사용됩니다.  
코사인 학습률 감쇠 스케줄러를 사용하여 초기 학습률이 0.02이고 총 배치 크기가 300에포크에 대해 2048인 모델을 학습합니다.  
ViT[30]에서와 동일한 데이터 증대 및 정규화 방법을 채택합니다.  
달리 명시되지 않는 한 모든 ImageNet 모델은 224 × 224 입력 크기로 훈련됩니다.

#### Fine-tuning
ImageNet-1k에서 20,000단계, CIFAR-10 및 CIFAR-100에서 10,000단계, Oxford-IIIT Pets 및 Oxford-IIIT Flowers-102에서 500단계에 대해 총 배치 크기 512로 각 모델을 미세 조정합니다.  

### Comparison to state of the art
Transformer 기반 모델과 비교하여 CvT는 더 적은 수의 매개변수와 FLOP로 훨씬 더 높은 정확도를 달성합니다.  
CvT-21은 82.5%의 ImageNet Top-1 정확도를 얻는데, 이는 63% 파라미터와 60%의 FLOP의 감소로 DeiT-B보다 0.5% 더 높습니다.  

![](https://velog.velcdn.com/images/kiolke/post/0f7b0d6c-e4f1-4269-8035-1f6798b9c6ac/image.png)

### Downstream task transfer
우리는 또한 모든 모델이 ImageNet-22k에서 사전 교육을 받은 상태에서 다양한 작업에서 모델을 미세 조정하여 모델을 전송할 수 있는 능력을 조사합니다.  
CvT-W24 모델은 CvT-W24보다 매개 변수 수가 3배 이상인 대형 BitT-R152x4[18] 모델과 비교하더라도 고려된 모든 다운스트림 작업에서 최고의 성능을 얻을 수 있습니다.

![](https://velog.velcdn.com/images/kiolke/post/ab62b8fe-32bf-45b7-b822-7cc0ccaeb7d7/image.png)

### Ablation Study
아키텍처의 제안된 구성 요소의 효과를 조사하기 위해 다양한 절제 실험을 설계합니다.  
첫째, 컨볼루션을 도입하면 위치 임베딩을 모델에서 제거 할 수 있음을 보여줍니다.
그런 다음 제안된 각 컨볼루션 토큰 임베딩 및 컨볼루션 투영 구성 요소의 영향을 연구합니다.

#### Removing Position Embedding
우리는 컨볼루션을 모델에 도입하여 로컬 컨텍스트를 캡처할 수 있도록 했기 때문에 CvT에 위치 임베딩이 여전히 필요한지 연구합니다.
결과는 표 5에 나와 있으며, 모델의 위치 임베딩을 제거해도 성능이 저하되지 않음을 보여줍니다.

![](https://velog.velcdn.com/images/kiolke/post/6bd966f5-ef0d-4926-9cd8-66818902eb1b/image.png)

따라서 위치 임베딩은 기본적으로 CvT에서 제거되었습니다.

#### Convolutional Token Embedding
제안한 Convolutional Token Embedding의 효과를 연구하고 그 결과를 Table 6과 같습니다.

![](https://velog.velcdn.com/images/hseop/post/63541a98-69c1-40a7-b865-45bd8450b151/image.png)

![](https://velog.velcdn.com/images/kiolke/post/b93f3582-3315-451a-ba1f-1d7cd16e059f/image.png)

이러한 결과는 Convolutional Token Embedding의 도입을 검증하여 성능을 향상시킬 뿐만 아니라 CvT가 위치 임베딩 없이 공간 관계를 모델링하는 데 도움이 됩니다.

#### Convolutional Projection
다음 각 단계에 대해 Convolutional Projection 또는 일반 Position-wise Linear Projection을 사용할지 선택하여 제안한 Convolutional Projection이 성능에 어떤 영향을 미치는지 연구합니다.

![](https://velog.velcdn.com/images/kiolke/post/f484e1b0-b86c-4b62-9e12-e08bc9a03047/image.png)

## Conclusion
이 연구에서는 비전 트랜스포머 아키텍처에 컨볼루션을 도입하여 트랜스포머의 이점과 이미지 인식 작업에 대한 CNN의 이점을 결합하는 상세한 연구를 제시했습니다.  
광범위한 실험을 통해 도입된 컨볼루션 토큰 임베딩 및 컨볼루션 프로젝션과 컨볼루션에 의해 활성화된 네트워크의 multi-stage design이 계산 효율성을 유지하면서도 우수한 성능을 달성한다는 것을 입증했습니다.  
또한 컨볼루션에 의해 도입된 내장된 local context structure로 인해 CVT는 더 이상 position embedding이 필요하지 않으므로 변할 수 있는 input resolution에 필요한 광범위한 비전 작업에 적응할 수 있는 잠재적 이점을 제공합니다.



# Reference
- https://velog.io/@hseop/CvT-Introducing-Convolutions-to-Vision-Transformers
- https://deep-learning-study.tistory.com/816
- https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CvT-Introducing-Convolutions-to-Vision-Transformers
- https://velog.io/@kiolke/CvT-Introducing-Convolutions-to-Vision-Transformers-%EC%A0%9C5%EB%B6%80
