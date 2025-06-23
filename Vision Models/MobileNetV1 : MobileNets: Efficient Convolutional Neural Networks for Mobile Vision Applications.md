# MobileNetV1
- https://ctkim.tistory.com/entry/%EB%AA%A8%EB%B0%94%EC%9D%BC-%EB%84%B7
- https://blog.firstpenguine.school/45

# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

## 개요  
MobileNets는 모바일 및 임베디드 환경에서 실시간 비전 작업을 수행할 수 있도록 경량화된 CNN(C​onvolutional Neural Network) 모델 클래스이다​[1]. 핵심 아이디어는 “Depthwise Separable Convolution”을 도입하여 연산량과 파라미터 수를 크게 줄이면서도 높은 정확도를 유지하는 것이다​[1].

---

## 1. Depthwise Separable Convolution  
전통적인 합성곱(standard convolution)은 입력 채널 수 M, 출력 채널 수 N, 커널 크기 DK×DK인 필터를 사용하여 다음과 같이 연산한다:  
$$\text{Mult-Adds} = D_K \times D_K \times M \times N \times D_F \times D_F$$  
여기서 DF는 피처 맵의 공간적 크기이다​[1].

Depthwise Separable Convolution은 이를 두 단계로 분리한다:  
- **Depthwise Convolution**: 각 입력 채널마다 DK×DK 필터를 적용하여 채널별로 피처를 추출한다.  
  - 연산량: $$D_K^2 \times M \times D_F^2$$​[1].  
- **Pointwise Convolution**: 1×1 필터를 사용하여 채널 간 결합을 수행하며 출력 채널 N을 생성한다.  
  - 연산량: $$M \times N \times D_F^2$$​[1].  

이 두 연산을 합하면  
$$D_K^2 M D_F^2 + M N D_F^2$$  
로, 전체 표준 합성곱 대비 약 8∼9배 연산량을 줄일 수 있다​[1].

---

## 2. MobileNet 네트워크 구조  
MobileNet은 첫 레이어만 표준 합성곱을 사용하고, 나머지는 모두 Depthwise Separable Convolution으로 구성한다. 전체 구조는 다음과 같다:

| Layer Type          | Stride | 필터 형태                    | 출력 크기              |
|---------------------|--------|------------------------------|------------------------|
| Conv                | 2      | 3×3×3→32                     | 112×112×32             |
| Depthwise Conv      | 1      | 3×3 dw                       | 112×112×32             |
| Pointwise Conv      | 1      | 1×1×32→64                    | 112×112×64             |
| ...                 | ...    | ...                          | ...                    |
| Depthwise Conv      | 2      | 3×3 dw                       | 7×7×1024               |
| Pointwise Conv      | 1      | 1×1×1024→1024                | 7×7×1024               |
| Avg Pool            | 1      | 7×7                          | 1×1×1024               |
| Fully Connected     | 1      | 1024→1000                    | 1×1×1000               |
| Softmax Classifier  | 1      |                              | 1×1×1000               |

각 합성곱 레이어 뒤에 Batch Normalization과 ReLU 활성화가 적용된다​[1].

---

## 3. 모델 축소를 위한 하이퍼파라미터  

### 3.1 Width Multiplier (α)  
Width Multiplier α는 각 레이어의 채널 수를 α배로 줄여 모델을 “얇게” 만든다. 예를 들어 α=0.5로 설정하면 입력 채널 M과 출력 채널 N이 각각 0.5M, 0.5N이 된다.  
- 연산량 및 파라미터 수가 대략 α2 비율로 감소한다​[1].

| α 값          | Top-1 정확도 | Mult-Adds (백만) | 파라미터 수 (백만) |
|---------------|--------------|------------------|-------------------|
| 1.0           | 70.6%        | 569              | 4.2               |
| 0.75          | 68.4%        | 325              | 2.6               |
| 0.5           | 63.7%        | 149              | 1.3               |
| 0.25          | 50.6%        | 41               | 0.5               |

---

### 3.2 Resolution Multiplier (ρ)  
Resolution Multiplier ρ는 입력 이미지 및 내부 피처 맵 크기를 ρ배로 줄인다. 일반적으로 입력 해상도를 직접 224, 192, 160, 128 등으로 설정하여 조정한다.  
- 연산량이 ρ2 비율로 감소한다​[1].

| 입력 해상도 | Top-1 정확도 | Mult-Adds (백만) | 파라미터 수 (백만) |
|-------------|--------------|------------------|-------------------|
| 224×224     | 70.6%        | 569              | 4.2               |
| 192×192     | 69.1%        | 418              | 4.2               |
| 160×160     | 67.2%        | 290              | 4.2               |
| 128×128     | 64.4%        | 186              | 4.2               |

---

## 4. 성능 비교  
MobileNet은 동일한 정확도 대비 다수의 기존 모델보다 훨씬 작고 빠르다.

| 모델             | Top-1 정확도 | Mult-Adds (백만) | 파라미터 수 (백만) |
|------------------|--------------|------------------|-------------------|
| MobileNet (224)  | 70.6%        | 569              | 4.2               |
| GoogleNet        | 69.8%        | 1550             | 6.8               |
| VGG16            | 71.5%        | 15300            | 138               |
| SqueezeNet       | 57.5%        | 1700             | 1.25              |
| AlexNet          | 57.2%        | 720              | 60                |

MobileNet은 VGG16보다 32배 작고 27배 적은 연산으로 유사 정확도를 달성하며, GoogleNet보다도 작고 빠르다​[1].

---

## 5. 다양한 응용 사례  
- **Fine-Grained Classification (Stanford Dogs)**: 모바일 환경에서 83.3% 정확도로 Inception V3(84.0%)에 근접한 성능을 달성​[1].  
- **Large-Scale Geolocalization (PlaNet)**: Inception V3 기반 PlaNet 대비 파라미터와 연산량을 크게 줄이면서도 유사 성능 유지​[1].  
- **Face Attribute Classification**: 75M 파라미터 대 모델을 MobileNet으로 증류(distillation)하여 연산량 1% 수준으로 축소하면서도 성능 향상​[1].  
- **Object Detection (COCO)**: SSD 및 Faster-RCNN 프레임워크에 적용 시 VGG나 Inception V2 대비 낮은 연산량으로 경쟁력 있는 mAP 달성​[1].  
- **Face Embedding (FaceNet 증류)**: FaceNet 기반 모델을 MobileNet으로 증류하여 80%대 정확도를 유지하며 경량화​[1].

---

## 결론  
MobileNets는 Depthwise Separable Convolution을 핵심으로 하여 모바일·임베디드 환경에 적합한 경량 모델을 제안한다. Width Multiplier와 Resolution Multiplier를 통해 사용자가 정확도와 리소스 제약 사이에서 유연하게 트레이드오프할 수 있으며, 다양한 비전 응용 분야에서 효율성과 성능을 모두 만족하는 모델임을 입증하였다​[1].

[1] https://arxiv.org/abs/1704.04861
[2] https://arxiv.org/pdf/1704.04861.pdf
[3] https://cumulu-s.tistory.com/46
[4] https://tghim.tistory.com/19
[5] https://www.youtube.com/watch?v=fpauxwMpn7Y
[6] https://arxiv.org/ftp/arxiv/papers/1712/1712.04698.pdf
[7] https://datascience.stackexchange.com/questions/45862/inner-workings-of-mobile-net-resolution-multiplier-what-does-it-do
[8] https://paperswithcode.com/paper/mobilenets-efficient-convolutional-neural
[9] https://iq.opengenus.org/mobilenet/
[10] https://velog.io/@whgurwns2003/mobilenet

