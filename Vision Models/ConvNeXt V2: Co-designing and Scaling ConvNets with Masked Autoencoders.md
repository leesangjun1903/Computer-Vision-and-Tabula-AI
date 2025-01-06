# ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

ConvNet(CNN)은 수동 feature 엔지니어링에 의존하기보다는 다양한 시각적 인식 task에 대한 일반적인 feature 학습 방법을 사용할 수 있도록 함으로써 컴퓨터 비전 연구에 상당한 영향을 미쳤다.  
최근에는 원래 자연어 처리를 위해 개발된 transformer 아키텍처도 모델과 데이터셋 크기에 대한 강력한 확장 가능성으로 인해 인기를 얻었다.  
최근에는 ConvNeXt 아키텍처가 기존 ConvNet을 현대화했으며 순수 convolutional model도 확장 가능한 아키텍처가 될 수 있음을 보여주었다.  
그러나 신경망 아키텍처의 디자인을 탐색하는 가장 일반적인 방법은 여전히 ImageNet에서 supervised learning 성능을 벤치마킹하는 것이다.

본 논문은 ConvNeXt 모델에 마스크 기반 self-supervised learning을 효과적으로 만들고 transformer를 사용하여 얻은 결과와 유사한 결과를 달성하기 위해 동일한 프레임워크에서 네트워크 아키텍처와 MAE를 공동 설계할 것을 제안한다.

# Reference
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/convnext-v2/
