# MobileNetV2
mobilenetv2는 ReLU 함수를 거치게 되면 정보가 손실된다는 것에 영감을 받아 이를 최소화하기 위해 Inverted Residuals와 Linear Bottlenecks를 제안함

depthwise convolution연산시 채널별로 쪼개서 계산하는데 relu함수 적용시 0으로 처리될때가 많음. 그래서 채널수가 적을때는 리니어하게해야함

linear bottenecks는 레이어에 채널 수가 적다면 linear activation을 사용합니다. 비선형 함수인 relu를 사용하게 되면 정보가 손실되기 때문입니다. 

inverted residuals는 기존의 BottleNeck 구조는 첫 번째 1x1 conv layer에서 채널 수를 감소시키고 3x3 conv로 전달합니다. 채널 수가 감소된 레이어에서 ReLU 함수를 사용하면 정보 손실이 발생하게 됩니다. 따라서 첫 번째 레이어에서 입력값의 채널 수를 증가시키고 3x3conv layer로 전달합니다. 

# MobileNetV2: Inverted Residuals and Linear Bottlenecks 논문 설명

MobileNetV2는 모바일 및 임베디드 기기에서 효율적인 딥러닝 모델을 구현하기 위해 설계된 경량화 신경망 구조로, **Inverted Residual** 구조와 **Linear Bottleneck** 개념을 도입하여 연산량과 메모리 사용량을 줄이면서도 높은 성능을 유지합니다[1].

## 1. 핵심 아이디어

### 1.1 Inverted Residuals (뒤집힌 잔차 구조)
전통적인 Residual Block은 `넓은→좁은→넓은` 채널 구조로, 주요 정보를 넓은 채널에서 처리한 뒤 다시 넓은 채널로 복원한 후 skip connection을 연결합니다.  
MobileNetV2에서는 이를 뒤집어 `좁은→넓은→좁은` 구조로 설계하고, skip connection을 **좁은** 채널 사이에 직접 연결하여 메모리 사용량을 크게 줄입니다[2].

### 1.2 Linear Bottlenecks (선형 병목층)
좁은 채널(bottleneck) 내부에서는 ReLU와 같은 비선형 활성화를 제거하여 정보 손실을 방지합니다.  
비선형 활성화가 좁은 차원에서 정보를 망가뜨리는 것을 막아, 표현력을 유지하면서도 효율적인 경량 모델을 구현할 수 있습니다[2].

## 2. 주요 구성 요소

### 2.1 Depthwise Separable Convolution
- **Depthwise Convolution**: 입력 채널별로 독립적으로 3×3 필터 적용  
- **Pointwise Convolution**: 1×1 필터로 채널 간 선형 결합  
이 조합은 표준 합성곱 대비 연산량을 $$k^2$$ 배 줄이며, MobileNetV2에서는 $$k=3$$을 사용합니다[3].

### 2.2 Bottleneck Residual Block 구조
각 블록은 세 단계로 구성됩니다:  
1. **1×1 확장(Expansion)** + ReLU6  
2. **3×3 Depthwise** + ReLU6  
3. **1×1 축소(Projection)** (선형)  
이후 좁은 채널끼리 skip connection을 통해 출력과 합산합니다[2].

## 3. 전체 아키텍처

MobileNetV2 기본 모델(입력 224×224, width multiplier=1.0)은 다음과 같은 계층으로 구성됩니다[2]:

| 입력 해상도        | 연산 유형            | 확장비 $$t$$ | 출력 채널 $$c$$ | 반복 횟수 $$n$$ | 스트라이드 $$s$$ |
|-----------------|--------------------|------------|--------------|------------|--------------|
| 224²×3          | conv2d              | –          | 32           | 1          | 2            |
| 112²×32         | bottleneck          | 1          | 16           | 1          | 1            |
| 112²×16         | bottleneck          | 6          | 24           | 2          | 2            |
| 56²×24          | bottleneck          | 6          | 32           | 3          | 2            |
| 28²×32          | bottleneck          | 6          | 64           | 4          | 2            |
| 14²×64          | bottleneck          | 6          | 96           | 3          | 1            |
| 14²×96          | bottleneck          | 6          | 160          | 3          | 2            |
| 7²×160          | bottleneck          | 6          | 320          | 1          | 1            |
| 7²×320          | conv2d (1×1)        | –          | 1280         | 1          | 1            |
| 7²×1280         | avgpool (7×7)       | –          | –            | 1          | –            |
| 1×1×1280        | conv2d (1×1, logits)| –          | 클래스 수 $$k$$| 1          | –            |

이 구조는 총 약 3.4M 파라미터와 300M MAdd를 사용하며, 다양한 width multiplier(0.35~1.4)와 입력 해상도(96~224)로 성능-연산량 균형을 조절할 수 있습니다[2].

## 4. 성능 및 응용

### 4.1 이미지 분류
- MobileNetV2(1.0x, 224²): **72.0%** Top-1 정확도, 300M MAdd[2].
- MobileNetV2(1.4x, 224²): **74.7%** Top-1 정확도, 585M MAdd[2].
- 경쟁 모델 대비 연산량 절감 비율 우수(ShuffleNet, NASNet 등)[2].

### 4.2 객체 검출 (SSDLite)
- SSDLite: SSD의 예측 레이어를 depthwise+1×1 conv로 대체  
- MobileNetV2+SSDLite(320²): **22.1 mAP**, 0.8B MAdd, 4.3M 파라미터[2].

### 4.3 의미론적 분할
- DeepLabv3에 적용 시, 출력 stride 16: 75.70% mIOU, 5.8B MAdd[2].
- 좁은 feature map(320채널) 사용 시: 75.32% mIOU, 2.75B MAdd[2].

## 5. 결론

MobileNetV2는 **Inverted Residual**과 **Linear Bottleneck**을 결합하여 모바일 환경에서 뛰어난 메모리·연산 효율과 높은 정확도를 달성한 모델입니다. 이러한 설계 철학은 이후 다양한 경량화 네트워크에도 큰 영감을 주었습니다[1][2].

[1] https://arxiv.org/pdf/1801.04381.pdf
[2] https://arxiv.org/pdf/2208.11011.pdf
[3] https://hackmd.io/@machine-learning/ryaDuxe5L
[4] https://mvcv.tistory.com/23
[5] https://greeksharifa.github.io/computer%20vision/2022/02/10/MobileNetV2/
[6] https://www.33rdsquare.com/what-is-mobilenetv2/
[7] https://paperswithcode.com/method/inverted-residual-block
[8] https://mmclassification.readthedocs.io/en/stable/papers/mobilenet_v2.html
[9] https://iq.opengenus.org/mobilenetv2-architecture/
[10] https://serp.ai/mobilenetv2/
[11] https://serp.ai/inverted-residual-block/
[12] https://arxiv.org/abs/1801.04381
[13] https://ar5iv.labs.arxiv.org/html/1801.04381
[14] https://github.com/aryanasadianuoit/MobileNet_V2
[15] https://pmc.ncbi.nlm.nih.gov/articles/PMC11679759/table/sensors-24-08052-t004/
[16] https://github.com/0jason000/MobileNet_V2
[17] https://www.sciencedirect.com/topics/computer-science/mobilenetv2
[18] https://arxiv.org/pdf/2307.00395.pdf
[19] https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf


# Reference
- 주요 CNN알고리즘 구현 : MobileNet v2 https://velog.io/@tbvjvsladla/26.-%EC%A3%BC%EC%9A%94-CNN%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B5%AC%ED%98%84-MobileNet-v2-1-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EA%B3%A0%EA%B8%89%EC%8B%9C%EA%B0%81-%EA%B0%95%EC%9D%98-%EB%B3%B5%EC%8A%B5
- https://m.blog.naver.com/phj8498/222689054103
- https://gaussian37.github.io/dl-concept-mobilenet_v2/
- What is MobileNetV2? Features, Architecture, Application and More : https://www.analyticsvidhya.com/blog/2023/12/what-is-mobilenetv2/
- https://velog.io/@woojinn8/LightWeight-Deep-Learning-7.-MobileNet-v2
- MobileNetV2(2018) : https://deep-learning-study.tistory.com/541
- 파라미터 용량 : https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html
- https://velog.io/@woojinn8/LightWeight-Deep-Learning-7.-MobileNet-v2
- https://sahiltinky94.medium.com/know-about-mobilenet-v2-implementation-from-scratch-using-pytorch-8e589b55599
- https://hackmd.io/@machine-learning/ryaDuxe5L
