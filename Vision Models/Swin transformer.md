# Swin Transformer

# Abstract
이 논문에서는 컴퓨터 비전의 다양한 목적에 맞게 backbone 역할을 할 수 있는 새로운 비전 트랜스포머인 Swin Transformer를 소개합니다.  
트랜스포머를 언어에서 시각으로 적용하는 데 있어 시각적인 데이터의 규모가 크고 텍스트의 단어에 비해 이미지에서 픽셀의 해상도가 높은 등 두 영역 간의 차이로 인해 어려움이 발생합니다.  
이러한 차이를 해결하기 위해 Shifted Windows(이동되는 윈도우)로 표현이 계산되는 계층적 트랜스포머를 제안합니다.  
이동된 윈도우 방식은 셀프 어텐션 계산을 겹치지 않는 로컬 창으로 제한함으로써 더 큰 효율성을 제공하는 동시에 윈도우 간 연결을 허용합니다.  
기존 ViT는 이미지의 크기에 제곱에 비례했지만, 이 계층적 아키텍처는 다양한 스케일에서 모델링할 수 있는 유연성을 갖추고 있으며 이미지 크기와 관련하여 선형적인 계산량을 가지고 있습니다. (Non-overlapping window내에 있는 patch 간의 self-attention을 수행함으로써 계산복잡도 개선하였음)  
Swin Transformer의 이러한 특성으로 인해 이미지 분류(이미지넷-1K에서 87.3개의 상위 1위 정확도)와 객체 감지(COCO 테스트-디바이스에서 58.7개의 상자 AP 및 51.1개의 마스크 AP) 및 의미론적 분할(ADE20K val에서 53.5mioU)과 같은 광범위한 비전 작업과 호환됩니다.  
성능은 이전 최첨단 기술인 COCO data에서 +2.7개의 상자 AP 및 +2.6개의 마스크 AP를, ADE20K에서 +3.2개의 mIoU를 큰 차이로 능가하여 트랜스포머 기반 모델이 비전 분야에서 잠재력을 입증합니다.  
계층적 설계와 Shifted Window 접근 방식은 모든 MLP 아키텍처에도 유용합니다.  

# Introduction
컴퓨터 비전의 모델링은 오랫동안 CNN에 의해 지배되었습니다.  
AlexNet과 ImageNet image classification 챌린지에 대한 혁신적인 성능을 시작으로 CNN 아키텍처는 더 큰 스케일, 더 광범위한 연결, 더 정교한 convolution 형식을 통해 점점 더 강력해졌습니다.  
다양한 비전 task를 위한 backbone 네트워크 역할을 하는 CNN과 함께 이러한 아키텍처의 발전은 전체 분야를 광범위하게 끌어올린 성능 향상으로 이어졌습니다.

반면에 자연어 처리(NLP)에서 네트워크 아키텍처의 진화는 오늘날 널리 사용되는 아키텍처가 Transformer라는 다른 경로를 택했습니다.  
시퀀스 모델링 및 변환 task를 위해 설계된 Transformer는 데이터의 장거리 의존성 (long-range dependency)을 모델링하는 데 attention을 사용하는 것으로 유명합니다.  
언어 도메인에서의 엄청난 성공으로 연구자들은 컴퓨터 비전에 대한 Transformer를 조사하게 되었으며, 최근 특정 tak, 특히 image classification 분류와 비전-언어 공동 모델링에 대한 유망한 결과를 보여주었습니다.

본 논문은 Transformer가 NLP에서, CNN이 비전에서 하는 것처럼 컴퓨터 비전을 위한 다양한 목적에 맞게 backbone 역할을 할 수 있도록 Transformer의 적용 가능성을 확장하고자 합니다.  
저자들은 언어 도메인에서 비전 영역으로 고성능을 전환하는 데 있어 중요한 문제가 두 modality 간의 차이로 설명될 수 있음을 관찰했습니다.  
이러한 차이점 중 하나는 스케일과 관련이 있습니다.  
언어 Transformer에서 처리의 기본 요소 역할을 하는 단어 토큰과 달리 시각적 요소는 스케일이 상당히 다를 수 있다. 기존 Transformer 기반 모델에서 토큰은 모두 고정된 스케일이며 이러한 비전 애플리케이션에 적합하지 않은 속성이다.

또 다른 차이점은 텍스트 구절의 단어에 비해 이미지의 픽셀 해상도가 훨씬 더 높다는 것이다. 픽셀 레벨에서 조밀한 예측을 필요로 하는 semantic segmentation과 같은 많은 비전 task가 존재하며, self-attention의 계산 복잡도가 이미지 크기의 제곱에 비례하기 때문에 고해상도 이미지에서 Transformer의 경우 처리하기 어렵다.

이러한 문제를 극복하기 위해 본 논문은 계층적 feature map을 구성하고 이미지 크기에 대한 선형 계산 복잡도를 갖는 Swin Transformer라는 범용 Transformer backbone을 제안하였다.



# Reference
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/swin-transformer
- https://heeya-stupidbutstudying.tistory.com/entry/DL-Swin-Transformer-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://velog.io/@9e0na/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CV-Swin-Transformer2021-Summary
- https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://velog.io/@jus6886/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
- https://deep-learning-study.tistory.com/728
