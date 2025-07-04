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

