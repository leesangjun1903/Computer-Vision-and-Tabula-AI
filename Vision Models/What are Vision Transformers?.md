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

![](https://velog.velcdn.com/images/kbm970709/post/ac973a8b-d7e6-4619-9cf4-7f08f58077e7/image.png)










# Refernence
- https://discuss.pytorch.kr/t/vision-transformer-a-visual-guide-to-vision-transformers/4158
- https://velog.io/@kbm970709/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale
- https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0AN-IMAGE-IS-WORTH-16X16-WORDS-TRANSFORMERS-FOR-IMAGE-RECOGNITION-AT-SCALE-Vi-TVision-Transformer

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

