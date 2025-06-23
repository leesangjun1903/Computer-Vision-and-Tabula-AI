# What are Vision Transformers?

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

# Abstract
Transformer 구조는 NLP 분야에서 높은 성능을 보이며 표준으로 자리잡는데에 반해, Computer Vision 분야에서는 아직 제한적이었습니다.  
Computer Vision 분야에서 Attention은 CNN에 결합해서 쓰거나, CNN의 구성 요소를 대체하는 식으로 간접적으로만 사용되었습니다.  
이 논문을 발표한 구글팀은 이미지 분야에서 CNN에 대한 의존을 끊고 Transformer를 직접적으로 사용하기 위해 image를 patch로 잘라 Sequence로서 사용하는 방식으로 이미지 분류를 수행해보았습니다.  
거대한 양의 데이터셋 (ImageNet, JFT)으로 pre-train한 후, 중간 사이즈 혹은 적은 양의 이미지 데이터셋에 대해 transfer하는 방식으로 기존의 ResNet 기반의 SOTA 모델들보다 좋은 성능 & 적은 계산량을 보였다고 합니다.

# Introduction
Transformer는 Self-attention을 사용한 구조로, 자연어처리(NLP) 분야에서 높은 성능을 거두며 거의 표준으로 사용되고 있습니다.  
해당 방법론은 거대한 text corpus로부터 pre-train한 후 작은 task의 데이터셋에 대해서 fine-tune하여 처리하는 방식입니다.  
특히 Transformer의 적은 계산량과 높은 확장성으로 인해, 100B가 넘어가는 막대한 parameter를 가지는 모델도 학습이 가능하게 되었습니다.  
현실의 데이터셋이 점점 더 커지는 가운데 이는 현실 상황에 적합한 모델로 평가할 수 있습니다.  
Computer Vision 분야에서는 ViT 논문이 발표되기 전까지는 여전히 CNN 기반 모델들이 지배적이었습니다.  
NLP 분야에서의 성공에 힘입어, CNN 구조의 모델들에 Self-attention을 접목시키려는 노력은 있었습니다.  
게다가 CNN 구조를 통째로 Transformer로 바꾸려는 노력도 있었는데 이는 이론적으로는 괜찮아보이나, specialized된 attention 패턴을 보여서 계산상으로 비효율적이었습니다.  
구글팀은 이러한 상황에서 거의 수정을 거치지 않은 표준 그대로의 Transformer를 이미지에 적용해보고자 했습니다.

아이디어는 이렇습니다.  
이미지를 일정한 크기의 Patch로 나눠서, 이를 단어의 배열처럼 Sequence로 사용합니다.  
그렇게 되면 이미지 패치는 NLP 분야에서 Token (Words)처럼 처리하면 될 것입니다.

구글팀은 먼저 ImageNet과 같은 mid-sized 데이터셋으로 학습시켰는데, 이는 ResNet 기반 모델들에 비해 살짝 안 좋은 성능을 보였습니다.  
이러한 결과는 어떻게 보면 당연한 결과로서 이유는 Inductive bias 때문입니다.

여기서 Inductive bias란 주어지지 않은 입력의 출력을 예측하는 능력입니다.  
즉, 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)으로 볼 수 있습니다.  

- Inductive bias : Relational inductive biases, deep learning, and graph networks
- https://moon-walker.medium.com/transformer%EB%8A%94-inductive-bias%EC%9D%B4-%EB%B6%80%EC%A1%B1%ED%95%98%EB%8B%A4%EB%9D%BC%EB%8A%94-%EC%9D%98%EB%AF%B8%EB%8A%94-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C-4f6005d32558

FC layer의 경우 모든 입력의 요소가 어떤 출력 요소던지 영향을 미칠 수 있습니다.  
반면, CNN은 Convolution Filter가 입력의 요소를 Window Sliding 하게되면서 위치 정보 즉, locality를 다음 layer로 전달 가능합니다.  
CNN과 마찬가지로 RNN도 inductive bias가 있는데 CNN이 공간의 개념을 사용한 것 처럼, RNN은 시간의 개념을 사용합니다.  
그래서 Sequential & Temporal한 Invariance가 존재합니다.

다시 Transformer로 돌아와보면 Transformer는 Attention만을 사용할 뿐, CNN과 RNN 구조에 전혀 의존하지 않습니다.  
Attention은 Query, Key, Value로 나누어서 Attention Score를 계산하고 이를 통해 Sequence가 다른 Sequence의 요소들과 어느 정도의 연관이 있는지를 나타냅니다.  
따라서 FCN 처럼 모든 입력의 요소가 어떤 출력 요소던지 영향을 미칠 수 있다고 생각할 수 있으며 Inductive Bias가 약해집니다.

그래서 구글팀은 mid-sized 데이터셋에 대해서는 높은 Inductive Bias를 지닌 CNN 계열인 ResNet에 비해서 낮은 성능을 보일 수 있다고 지적합니다.   
하지만 14M-300M의 거대한 데이터셋으로 학습을 한다면 large scale이 inductive bias를 이길 수 있다고 주장합니다.  
따라서 ImageNet-21k 혹은 JFT-300M의 데이터셋으로 사전학습해서 전이학습했을 때 SOTA의 성능을 거둡니다.  

# Related Work
먼저 Transformer는 2017년 자연어 처리를 위해서 제안되었고 많은 NLP task에 있어서 SOTA의 성능을 보였습니다.  
앞서 말한대로, 거대한 Corpora로 학습하고 실제 task에 대해서 fine-tune하는 방식이며 이를 활용한 유명한 모델은 BERT와 GPT가 있습니다.  

Self-attention을 이미지에 적용하는 것을 간단하게 생각했을 때 각 픽셀이 각각의 픽셀에게 attend하는 걸 생각해볼 수 있습니다.  
하지만 그렇게 된다면 pixel의 수에 따라 엄청나게 많은 cost가 소요될 수도 있어서 현실적이지 못합니다.  

# Method
구글팀은 최대한 original 형태의 Transformer를 이미지에 사용하고자 했고 해당 노력의 결과로 탄생한 Visual Transformer 구조의 모습은 아래와 같습니다.

![](https://velog.velcdn.com/images/kbm970709/post/ac973a8b-d7e6-4619-9cf4-7f08f58077e7/image.png)

ViT는 original Transformer(Attention is all you need 중)의 구조를 대부분 따릅니다.

물론 완벽히 동일한 아키텍처를 구축할 수는 없겠지만, 최대한 기존 transformer와 가깝게하려고 한 이유는 NLP Transformer의 확장성(scalability)과 효율적인 implementations을 가능하게 하기 위함입니다.

### Vision Transformer (ViT)

ViT의 작동 과정은 5개의 Step으로 설명 가능합니다.

1. 이미지 $x \in R^{H\times W\times C}$가 있을 때 $(P\times P)$ 패치의 크기를 $N(=H\times W /P^2)$ 개로 분할하여 sequence $x_p \in R^{N\times(P^2\times C)}$ 로 구축합니다. 여기서 $(H,W)$ 는 원본 이미지의 해상도, $C$ 는 채널의 수, $(P,P)$ 는 이미지 패치의 해상도입니다.
2. Trainable linear projection을 통해 $x_p$ 의 각 patch를 flatten한 벡터 $D$ 차원으로 변환한 후, 이를 patch 임베딩으로 사용합니다.
3. Learnable class 임베딩과 patch 임베딩에 learnable position 임베딩을 더합니다. 여기서 Learnable class는 BERT 모델의 class 토큰과 같이 classification 역할을 수행합니다.
4. 임베딩을 Transformer encoder에 input으로 넣어 마지막 layer에서 class embedding에 대한 output인 image representation을 도출합니다. 여기서 image representation이란 L번의 encoder를 거친 후의 output 중 learnable class 임베딩과 관련된 부분을 의미합니다.
5. MLP에 image representation을 input으로 넣어 이미지의 class를 분류합니다.

이와 관련한 수식은 다음과 같습니다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoxIJY%2FbtsCZiQmO8U%2F1wXOPgIyHmXOwLQghaJAck%2Fimg.png)

Step 1~3까지는 (1)번 수식, Step 4는 (2), (3)번 수식, Step 5는 (4)번 수식과 연결됩니다.

#### Inductive Bias
ViT에서 MLP는 locality와 translation equivariance가 있습니다.  
왜냐하면 이미지 패치를 sequential하게 잘라서 임베딩했기 때문입니다.  
하지만 MSA는 global하기 때문에 CNN보다 image-specific inductive bias가 낮습니다.
따라서 ViT에서는 모델에 두가지 방법을 사용하여 inductive bias의 주입을 시도합니다.  
이미지 패치들을 잘라내고 대규모 데이터에 대한 Pre-training이 필요하며, 그 후 각각의 Task에 맞게 Fine-tuning이 필요합니다.

- Patch extraction: cutting the image into patches : 패치들로 잘라내는 방법으로 패치 추출
- Resolution adjustment: adjusing the position embeddings for images of diffrent resolution at fine-tuning : 이미지의 해상도에 따라 변하는 시퀀스는 사전 학습된 Position embeddings의 의미를 잃어버릴 수 있기 때문에 position embedding 조정

#### Hybrid Architecture
ViT의 입력으로 raw image가 아닌 CNN으로 추출한 raw image의 feature map을 활용할 수 있습니다.  
Feature map은 이미 raw image의 공간적 정보를 포함하고 있으므로 패치를 자를 때 1x1로 설정해도 됩니다.  
그렇게 한다면 feature map의 공간 차원을 flatten하여 각 벡터에 linear projection을 적용하면 됩니다.  

### Fine-Tuning And Higher Resolution
논문의 저자는 ViT를 large dataset으로 pre-train하고 downstream task에 fine-tune하여 사용합니다.  
이와 같은 경우에는 pre-trained prediction head를 제거하고 $D×K$ zero-initialized feedforward layer로 대체하면 됩니다.  
대체가 이루어진다면 pre-training할 때보다 더 높은 해상도를 fine-tune하는 것에 도움이 됩니다.  
높은 해상도의 이미지를 모델에 적용한다면, patch size는 그대로 가져갈 것이고, 그렇다면 상당히 큰 sequence length를 갖게 됩니다.

물론 ViT는 가변적 길이의 패치를 처리할 수는 있지만, pre-trained position embeddings는 의미를 잃게 됩니다.  
이 경우 pre-trained position embedding을 원본 이미지의 위치에 따라 2D interpolation하면 됩니다.

## Experiments
### Setup
#### Datasets
$$<table><thead><tr><th>Pre-trained Dataset</th><th># of Classes</th><th># of Images</th></tr></thead><tbody><tr><td>ImageNet-1k</td><td>1k</td><td>1.3M</td></tr><tr><td>ImageNet-21k</td><td>21k</td><td>14M</td></tr><tr><td>JFT</td><td>18k</td><td>303M</td></tr><tr><td></td><td></td><td>(High resolution)</td></tr></tbody></table>

ViT는 위와 같이 3개의 데이터셋으로 pre-train 됩니다.  
그 후 이를 몇가지 benchmark tasks에 transfer 합니다. benchmark tasks는 다음과 같습니다.

- ReaL labels, CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102
- 19-task VTAB classification suite

#### Model Variants
![](https://velog.velcdn.com/images/kbm970709/post/08ba0eef-05af-4816-9c36-ea31673c1da7/image.png)

ViT는 총 3개의 volume에 대해서 실험을 진행했으며, 각 볼륨에서도 다양한 패치 크기에 대해 실험을 진행했습니다.  
여기서 Base와 Large 모델은 BERT 모델에서 직접적으로 채택했으며, Huge는 저자들이 추가한 것입니다.  

본 논문의 저자인 구글 팀은 이전 논문으로 transfer learning에 적합한 Big Transformer (BiT) 구조의 ResNet을 발표했습니다.  
이는 batch normalization layer를 group normalization으로 변경하고 standardized convolutional leyer를 사용한 모델입니다.  
해당 모델을 비교군으로 삼아 실험을 진행합니다.

#### Metrics

평가 지표로는 few-shot accuracy와 fine-tuning accuracy를 고려합니다.

- Few-shot accuracy: Training set에 없는 클래스를 맞추는 문제에 대한 정확도
- Fine-tuning accuracy: Fine-tuning 후의 정확도
본 논문의 저자는 fine-tuning의 성능에 집중하고 있기에 fine-tuning accuracy를 사용하지만 fine-tuning의 cost가 너무 크기 때문에 빠른 평가를 위해 때때로는 few-shot accuracies를 사용했습니다.

### Comparison To State Of The Art
![](https://velog.velcdn.com/images/kbm970709/post/2e8051f8-1a6a-430f-a1ea-0460f082eb61/image.png)

거의 모든 데이터셋에서 ViT-H/14 모델이 가장 높은 성능을 보였습니다.  
이는 기존 SOTA 모델인 BiT-L 보다도 높은 성능이며 더 적은 시간이 걸렸습니다.  
또한 주목할 점은 이보다 작은 모델인 ViT-L/16 또한 BiT-L보다 높은 성능을 보였으며 시간은 훨씬 적게 걸렸다는 것입니다.

![](https://velog.velcdn.com/images/kbm970709/post/35deefdc-689a-4a1b-9c0b-5f67a21bc8ab/image.png)

VTAB 데이터셋에서도 ViT-H/14 모델이 가장 좋은 성능을 보였습니다. 해당 실험은 데이터셋을 3개의 그룹으로 나누어 진행한 실험인데 그룹은 다음과 같습니다.

- Natural: tasks like Pets, CIFAR, etc
- Specialized: medical and satellite imagery
- Structured: tasks that require geometric understanding like localization

### Pre-training Data Requirements
![](https://velog.velcdn.com/images/kbm970709/post/91a9fb9a-1db5-4a39-8329-9e1e89d78efb/image.png)

Figure 3 실험을 통해 알 수 있는 것은 크기가 큰 데이터셋으로 pre-training 하는 경우 BiT보다 ViT가 더 높은 성능을 띄고, 반대의 경우는 반대의 성능을 띈다는 것입니다.

Figure 4 실험은 JFT 데이터셋을 각각 다른 크기로 랜덤 샘플링한 데이터셋을 활용하여 진행한 것입니다.  
이를 통해 작은 데이터셋에서는 확실히 inductive bias 효과가 있는 CNN 계열의 BiT가 높은 성능을 보이나, 큰 데이터셋으로 갈수록 ViT 성능이 더 좋아지는 것을 확인할 수 있습니다.

### Scaling Study
![](https://velog.velcdn.com/images/kbm970709/post/e1192157-6f0e-4d91-947f-d69245c75a39/image.png)

Figure 5를 통해 같은 시간이 소모되었을 때 ViT가 더 높은 성능을 거두는 것을 확인할 수 있었습니다.  
따라서 성능과 cost의 trade-off에서 ViT가 BiT보다 우세한 것을 검증해냈습니다.

또한 Cost가 낮을 때는 Hybrid가 ViT보다 유리한 듯 하지만 Cost가 높아지면서 trade-off 차이가 감소합니다.

### Inspecting Vision Transformer
![](https://velog.velcdn.com/images%2Fsjinu%2Fpost%2Fc8ce3b46-5d27-4e6e-82f9-b695770adaff%2Fimage.png)

왼쪽 그림: RGB 이미지를 ViT에 입력하기 전에 이미지를 패치로 나누고, D차원으로 매핑한 후, 학습된 임베딩 필터들의 주요 구성 요소를 보여줍니다.  
이 구성 요소들은 잘 학습된 CNN 필터의 기능과 유사한 패턴을 나타내는 것을 볼 수 있습니다.

가운데 그림: ViT에서 이미지 패치 임베딩에 이어 포지션 임베딩을 주입한 상태를 나타냅니다.  
이 그림은 포지션 임베딩 간의 코사인 유사도를 분석한 것으로, 가까운 패치 간에 높은 유사도가 나타나는 것을 확인할 수 있습니다.  
같은 열이나 같은 행에 있는 패치들 사이에서 높은 유사도가 관찰됩니다.

오른쪽 그림: 각각의 Attention head가 네트워크에서 self-attention 기능을 얼마나 잘 활용하는지 조사한 결과를 보여줍니다.  
이미지 공간에서 attention weights를 기반으로 정보가 통합되는 평균 거리, 즉 "attention distance"를 계산합니다.  
이는 CNN의 수용 필드의 크기와 유사합니다. 224 x 224 크기의 이미지로 실험을 진행했기 때문에, 평균 거리가 대략 112 정도에 이르면 각 픽셀이 전역적으로 정보를 통합했다고 볼 수 있습니다.  
Layer 층이 깊어질수록 모든 Attention head가 이 거리에 근접해 모든 정보를 통합할 수 있는 것을 확인할 수 있습니다.

![](https://velog.velcdn.com/images%2Fsjinu%2Fpost%2Fcaeaa465-a521-441b-9c90-6f57d42494c1%2Fimage.png)

위 그림처럼 classification에 대해 의미적으로 가까운 image regions에 attend하는 모습을 보여줍니다.

## Conclusion
image-specific inductive biases를 특별하게 사용하지 않고, 이미지를 패치로 자른 sequence를 NLP에서 사용하는 Transformer encoder에 넣어서 self-attention을 사용했습니다.  
특히 large datasets으로 pre-train 시킴으로써 기존의 SOTA 모델들을 능가하는 성능과 더 적은 computational cost가 소요됩니다.  
여전히 Challenge는 존재하며 1. Detection과 Segmentation 2. Self-Supervised Learning 에 적용해볼 수 있을 것입니다.

# Refernence
- https://discuss.pytorch.kr/t/vision-transformer-a-visual-guide-to-vision-transformers/4158
- https://velog.io/@kbm970709/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale
- https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0AN-IMAGE-IS-WORTH-16X16-WORDS-TRANSFORMERS-FOR-IMAGE-RECOGNITION-AT-SCALE-Vi-TVision-Transformer
- https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-ViTVision-Transformer%EC%9D%98-%EC%9D%B4%ED%95%B4

# Transformers in Vision: A Survey
# Abstract
자연어 처리에 대한 트랜스포머 모델의 놀라운 결과는 비전 커뮤니티에서 컴퓨터 비전 문제에 대한 트랜스포머의 적용을 연구하는 데 흥미를 불러일으켰습니다.  
트랜스포머는 중요한 이점 중 하나로 입력 시퀀스 요소 간의 긴 의존성을 모델링하고 반복 네트워크(예: Long short-term memory(LSTM)와 비교하여 시퀀스의 병렬 처리를 지원합니다.  
컨볼루션 네트워크와 달리 트랜스포머는 설계에 최소한의 귀납적 편향이 필요하며 자연스럽게 set-function으로 적합합니다.  
또한 트랜스포머의 간단한 설계를 통해 유사한 처리 블록을 사용하여 여러 양식(예: 이미지, 동영상, 텍스트 및 음성)을 처리할 수 있으며, 대용량 네트워크와 대규모 데이터 세트에 대한 뛰어난 확장성을 보여줍니다.  
이러한 강점은 트랜스포머 네트워크를 사용하는 여러 비전 작업에 대한 흥미로운 진전으로 이어졌습니다.  
이 설문조사는 컴퓨터 비전 분야에서 트랜스포머 모델에 대한 포괄적인 개요를 제공하는 것을 목표로 합니다.  
우리는 자기 주의, 대규모 사전 훈련 및 양방향 특징 인코딩과 같은 트랜스포머의 성공에 대한 근본적인 개념에 대한 소개부터 시작합니다.  
그런 다음 인기 있는 recognition tasks(예: 이미지 분류, 객체 감지, 동작 인식 및 분할), generative modeling, multi-modal tasks(예: 시각적 질문 응답, 시각적 추론 및 시각적 접지), video processing(예: activity recognition, visual grounding), low-level vision(예: image super-resolution, image enhancement, and colorization) 및 3D 분석(예: point cloud classification 및 분할)을 포함한 비전 분야의 트랜스포머의 광범위한 응용 분야를 다룹니다.  
우리는 아키텍처 설계와 실험적 가치 측면에서 인기 있는 기법의 각각의 장점과 한계를 비교합니다.  
마지막으로, 개방형 연구 방향과 향후 가능한 작업에 대한 분석을 제공합니다.  
이러한 노력이 컴퓨터 비전 분야에서 트랜스포머 모델의 적용에 대한 현재의 과제를 해결하기 위한 커뮤니티의 관심을 더욱 불러일으킬 수 있기를 바랍니다.

# Introduction
![](https://hoya012.github.io/assets/img/Visual_Transformer/1.PNG)

위의 그림을 보시면 알 수 있듯이 매년 Top-tier 학회, arxiv에 Transformer 관련 연구들이 빠른 속도로 늘어나고 있고 작년(2020년)에는 거의 전년 대비 2배 이상의 논문이 제출이 되었습니다.  
바야흐로 Transformer 시대가 열린 셈이죠.  
근데 주목할만한 점은 Transformer가 자연어 처리 뿐만 아니라 강화 학습, 음성 인식, 컴퓨터 비전 등 다른 task에도 적용하기 위한 연구들이 하나 둘 시작되고 있다는 점입니다.

논문에서는 컴퓨터 비전에 Transformer을 적용시킨 연구들을 크게 10가지 task로 나눠서 정리를 해두었습니다.

1. Image Recognition (Classification)
2. Object Detection
3. Segmentation
4. Image Generation
5. Low-level Vision
6. Multi-modal Tasks
7. Video Understanding
8. Low-shot Learning
9. Clustering
10. 3D Analysis

![](https://hoya012.github.io/assets/img/Visual_Transformer/3.png)

# Foundations
![](https://hoya012.github.io/assets/img/Visual_Transformer/2.PNG)

Transformers의 성공 요소는 크게 Self-Supervision 과 Self-Attention 으로 나눌 수 있습니다.  
세상엔 굉장히 다양한 데이터가 존재하지만, Supervised Learning으로 학습을 시키기 위해선 일일이 annotation을 만들어줘야 하는데, 대신 무수히 많은 unlabeled 데이터들을 가지고 모델을 학습 시키는 Self-Supervised Learning을 통해 모델을 학습 시킬 수 있습니다.  
자연어 처리에서도 Self-Supervised Learning을 통해 주어진 막대한 데이터 셋에서 generalizable representations을 배울 수 있게 되며, 이렇게 pretraining시킨 모델을 downstream task에 fine-tuning 시키면 우수한 성능을 거둘 수 있게 됩니다.  

또 다른 성공 요소인 Self-Attention은 말 그대로 스스로 attention을 계산하는 것을 의미하며 CNN, RNN과 같이 inductive bias가 많이 들어가 있는 모델들과는 다르게 최소한의 inductive bias를 가정합니다.  
Self-Attention Layer를 통해 주어진 sequence에서 각 token set elements(ex, words in language or patches in an image)간의 관계를 학습하면서 광범위한 context를 고려할 수 있게 됩니다.  


# Reference
- https://hoya012.github.io/blog/Vision-Transformer-1/







# A Survey on Visual Transformer

# Reference
- https://jihyeonryu.github.io/2021-04-02-survey-paper1/

