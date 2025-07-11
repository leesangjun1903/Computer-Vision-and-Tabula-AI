# CAMixerSR: Only Details Need More “Attention”

# Abs
CAMixerSR은 이미지 초해상도를 위한 content aware mixer로, 다양한 SR 작업에서 우수한 성능을 보입니다.  
CAMixer는 복잡한 영역에는 self-attention를, 간단한 영역에는 convolution을 할당하여 모델 가속 및 토큰 믹싱 전략을 통합합니다.  
CAMixerSR은 가벼운 SR, 대형 입력 SR 및 전방향 이미지 SR에 대해 우수한 성능을 보이며 SOTA 품질-계산 교환을 달성합니다.  

대규모 이미지(2K-8K) 초해상도(SR)에 대한 수요가 빠르게 증가하고 있는 상황을 충족시키기 위해 기존 방법은 두 가지 독립적인 트랙을 따릅니다:  
1) content-aware routing을 통해 기존 네트워크를 가속화하고  
2) token mixer refining를 통해 더 나은 초해상도 네트워크를 설계합니다.  

이럼에도 불구하고 품질과 복잡성 균형의 추가적인 개선을 제한하는 피할 수 없는 결함(예: 유연하지 않은 경로 또는 비차별적 처리)에 직면합니다.
이러한 단점을 지우기 위해 간단한 컨텍스트에 대해서는 컨볼루션을 할당하고 희소한 텍스처에 대해서는 추가적인 변형 가능한 window-attention를 할당하는 content-aware mixer(CAMIXer)를 제안하여 이러한 체계를 통합합니다.
특히, CAMixer는 학습 가능한 예측기를 사용하여 windows warping을 위한 오프셋, 윈도우를 분류하기 위한 마스크, dynamic property를 가진 convolutional attentions를 포함한 여러 bootstraps을 생성하며, 이는 보다 유용한 텍스처를 스스로 포함하도록 attention를 조절하고 컨볼루션의 표현 기능을 향상시킵니다.
이 모델은 예측기의 정확도를 향상시키기 위해 global classification loss을 도입합니다. 단순히 CAMixer를 적층하여 대규모 이미지 SR, lightweight SR 및 모든 방향을 가지는 이미지 SR에서 우수한 성능을 달성하는 CAMixerSR을 얻습니다.

# Introduction
신경망에 대한 최근 연구는 이미지 초해상도(SR) 품질을 크게 향상시켰습니다[22, 34, 43].  
그러나 기존 방법은 시각적으로 만족스러운 고해상도(HR) 이미지를 생성하지만 특히 2K-8K 대상의 경우 실제 사용에서 철저한 계산을 거칩니다.   
고비용를 완화하기 위해 실제 초해상도 적용을 위해 많은 accelerating frameworks[4, 19]와 lightweight networks[14, 32]가 도입되었습니다.  
그러나 이러한 접근 방식은 협력 없이 완전히 독립적입니다.  
첫 번째 전략인 accelerating frameworks[11, 19, 39]는 이미지 영역마다 서로 다른 네트워크 복잡성이 필요하다는 관찰을 기반으로 하며, 이는 다양한 모델의 (콘텐츠 인식 관점)content-aware routing 관점에서 문제를 해결합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.02.24.png)

그림 1의 중간 이미지에 표시된 것처럼, 큰 이미지를 고정된 패치들로 분해하고 extra classification network를 통해 네트워크에 패치를 할당했습니다.  
ARM[4]은 효율성을 향상시키기 위해 LUT 기반 classifier 및 파라미터 공유 설계 방식을 도입하여 전략을 더욱 발전시켰습니다.  
이러한 전략은 모든 신경망에 일반적이지만 피할 수 없는 두 가지 결함이 남아 있습니다.  
하나는 분류가 제대로 되지 않고 유연하지 못한 partition입니다.  
그림 1은 간단한 모델에 부적절하게 전송된 세부 정보가 거의 없는 창을 표시합니다. (복잡한 이미지 부분과 단순한 이미지 부분이 제대로 분류되지 않는다는 듯)  
다른 하나는 제한된 receptive fields입니다.  
표 2에서 볼 수 있듯이 패치로 이미지를 자르는 것은 receptive fields를 제한하므로 성능에 영향을 미칩니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.12.55.png)

두 번째 전략인 lightweight model design[7, 8, 17, 44]는 제한된 계층 내에서 더 강력한 feature 표현 기능, 즉 이미지를 재구성하기 위해 더 많은 intra-information를 사용할 수 있도록 신경 연산자(self-attention or convolution)와 중심 구조를 개선하는 데 중점을 둡니다.  
예를 들어, NGswin[5]은 self-attention를 위해 N-Gram을 활용하여 계산을 줄이고 receptive field를 확장했습니다.  
IMDN[14]은 효율적인 블록 설계를 위해 information multi-distillation를 도입했습니다.  
이러한 lightweight method은 720p/1080p 이미지에서 인상적인 효율성에 도달했지만, 더 큰 이미지(2K-8K)에서는 사용법이 거의 검토되지 않습니다.  
또한 이러한 접근 방식은 서로 다른 콘텐츠를 식별하고 처리할 수 없습니다.  

먼저 위의 전략을 통합한 이 논문은 서로 다른 feature 영역이 다양한 수준의 token-mixer(토큰이 섞인) 복잡성을 요구한다는 도출된 관찰을 기반으로 합니다.  
표 1에서 볼 수 있듯이 간단한 컨볼루션(Conv)은 간단한 패치에 대해 훨씬 더 복잡한 convolution + self-attention(SA + Conv)로 유사한 성능을 발휘할 수 있습니다.  
따라서 콘텐츠에 따라 서로 다른 복잡성을 가진 토큰 믹서의 루트를 정하는 content-aware mixer(CAMIXer)를 제안합니다.  
그림 1에서 볼 수 있듯이 당사의 CAMixer는 복잡한 window에는 복잡한 self attention(SA)를 사용하고 일반 윈도우에는 간단한 convolution을 사용합니다.  
또한 ClassSR의 한계를 해결하기 위해 보다 정교한 예측기를 소개합니다.  
이 예측기는 여러 조건을 활용하여 추가적인 가치 있는 정보를 생성하여 partition의 정확도를 향상시키고 표현을 개선하여 CAMixer를 향상시킵니다.  
CAMixer를 기반으로 초해상도 작업을 위한 CAMixerSR을 구성합니다.  
CAMixer의 성능을 완전히 검토하기 위해 lightweight SR, 대용량 이미지(2K-8K) SR 및 모든 방향의 이미지 SR에서 실험을 수행합니다.  
그림 2는 CAMixerSR이 lightweight SR과 accelerating framework를 모두 크게 발전시키는 것을 보여줍니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.02.36.png)

다음과 같이 요약됩니다:
- 우리는 convolution and self-attention를 통합한 Content-Aware Mixer(CAMIXer)를 제안하며, 이는 컨볼루션에 간단한 영역을 할당하고 self-attention에 복잡한 영역을 할당하여 추론 계산을 적응적으로 제어할 수 있습니다. 
- 우리는 convolution 적용 후의 갈라진 형태, mask 및 간단한 공간/채널 attention를 생성하기 위한 강력한 예측기를 제안하며, 이는 더 적은 계산으로 넓은 상관 관계를 포착하도록 CAMixer를 변조합니다.
- CAMixer를 기반으로 lightweight SR, 대용량 이미지 SR, 모든 방향의 이미지 SR의 세 가지 까다로운 초해상도 작업에서 최첨단 품질의 계산 절충점을 보여주는 CAMixerSR을 구축합니다.

# Related Work
## Accelerating framework for SR

## Lightweight SR

# Method
## Content-Aware Mixing
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.13.08.png)

## Network Architecture

## Training Loss

# Reference
- https://linnk.ai/insight/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC/camixersr-content-aware-mixer-for-image-super-resolution-3DtqCD_s/
