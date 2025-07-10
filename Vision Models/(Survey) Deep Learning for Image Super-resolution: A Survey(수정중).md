# Deep Learning for Image Super-resolution: A Survey

# Abs
이미지 초해상도(SR)는 컴퓨터 비전에서 이미지와 비디오의 해상도를 향상시키기 위한 이미지 처리 기술의 중요한 클래스입니다.  
최근 몇 년 동안 딥 러닝 기술을 사용한 이미지 초해상도의 놀라운 발전을 목격했습니다.  
이 기사는 딥 러닝 접근 방식을 사용한 이미지 초해상도의 최근 발전에 대한 포괄적인 조사를 제공하는 것을 목표로 합니다.  
일반적으로 SR 기술에 대한 기존 연구는 supervised SR, unsupervised SR, domain-specific SR의 세 가지 주요 범주로 그룹화할 수 있습니다.  
또한 공개적으로 사용 가능한 벤치마크 데이터 세트 및 성능 평가 metric과 같은 몇 가지 다른 중요한 문제도 다룹니다.  
마지막으로, 향후 커뮤니티에서 추가로 해결해야 할 몇 가지 미래 방향과 해결되지 않은 문제를 강조하는 것으로 이 조사를 마무리합니다.  

# Introduction
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.37.43.png)

# PROBLEM SETTING AND TERMINOLOGY
## Problem Definitions
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.20.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.34.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.43.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.39.54.png)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.40.03.png)

## Datasets for Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.32.png)

## Image Quality Assessment
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.50.png)

### Peak Signal-to-Noise Ratio
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.17.59.png)

### Structural Similarity
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.22.png)

### Mean Opinion Score
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.42.png)

### Learning-based Perceptual Quality
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.18.52.png)

### Task-based Evaluation

### Other IQA Methods

## Operating Channels
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.06.png)

## Super-resolution Challenges
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.21.png)

# SUPERVISED SUPER-RESOLUTION
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.19.35.png)

## Super-resolution Frameworks

### Pre-upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.20.17.png)

- SRCNN
- VDSR
- MemNet
- DRRN(Residual Network)
- DRCN(Recursive Network)
- Zero Shot SR

### Post-upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.20.26.png)

- FSRCNN(Fast SRCNN)
- ESPCN
- SRGAN
- EDSR
- SRDenseNet
- DSRN 

### Progressive Upsampling Super-resolution
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.21.05.png)

- LapSRN
- MS-LapSRN
- progressive SR (ProSR)

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-25%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%208.21.20.png)

### Iterative Up-and-down Sampling Super-resolution
- DBPN
- SRFBN
- RBPN

## Upsampling Methods
모델의 업샘플링 위치 외에도 업샘플링을 수행하는 방법이 매우 중요합니다.  
다양한 전통적인 업샘플링 방법이 있었지만 [20], [21], [88], [89]를 사용하여 엔드투엔드 업샘플링을 학습하는 것이 점차 추세가 되었습니다.  
이 섹션에서는 몇 가지 전통적인 보간 기반 알고리즘과 딥러닝 기반 업샘플링 레이어를 소개합니다.  

- 

### Interpolation-based Upsampling
이미지 보간, 일명 이미지 스케일링은 디지털 이미지의 크기를 조정하는 것을 말하며 이미지 관련 응용 분야에서 널리 사용됩니다.  
전통적인 보간 방법에는 nearest neighbor  보간, bilinear 및 bicubic 보간, Sinc and Lanczos 리샘플링 등이 있습니다.  
이러한 방법은 해석이 가능하고 구현하기 쉽기 때문에 CNN 기반 SR 모델에서 여전히 널리 사용되는 방법도 있습니다.

#### Nearest-neighbor Interpolation.
nearest-neighbor interpolation은 간단하고 직관적인 알고리즘입니다.  
다른 픽셀에 상관없이 보간할 각 위치에 대해 가장 가까운 픽셀의 값을 선택합니다.  
따라서 이 방법은 매우 빠르지만 일반적으로 낮은 품질의 차단된 결과를 생성합니다.

#### Bilinear Interpolation.
BLI(bilinear interpolation)는 그림 3과 같이 먼저 이미지의 한 축에 대해 선형 보간을 수행한 다음 다른 축에 대해 수행합니다.  
2×2 크기의 receptive field를 갖는 2차 보간을 생성하기 때문에 비교적 빠른 속도를 유지하면서도 nearest neighbor interpolation보다 훨씬 우수한 성능을 보여줍니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.53.51.png)

#### Bicubic Interpolation.
유사하게, bicubic interpolation(BCI)[10]은 그림 3과 같이 두 축 각각에 대해 cubic interpolation을 수행합니다.  
BLI에 비해 BCI는 4 × 4 픽셀을 고려하며, 결함은 적지만 속도는 훨씬 낮아서 더 매끄러운 결과를 가져옵니다.  
사실, anti-aliasing이 적용된 BCI는 SR 데이터 세트를 구축하기 위한 주류 방법이며(즉, HR 이미지를 LR 이미지로 낮추기), SR 프레임워크를 사전에 업샘플링하는 데에도 널리 사용됩니다(섹. 3.1.1).  
사실, interpolation-based upsampling method는 더 이상의 정보를 가져오지 않고 자체 이미지 신호만을 기반으로 이미지 해상도를 향상시킵니다.  

대신, 그들은 종종 계산 복잡성, 잡음 증폭, blurring과 같은 일부 부작용을 도입합니다.  
따라서, 현재 추세는 보간 기반 방법을 learnable upsampling layer로 대체하는 것입니다.

### Learning-based Upsampling
보간 기반 방법의 단점을 극복하고 엔드 투 엔드 방식으로 업샘플링을 학습하기 위해 SR 필드에 transposed convolution layer와 sub-pixel layer가 도입됩니다.  

#### Transposed Convolution Layer.
Transposed convolution layer, 일명 deconvolution layer[90], [91]은 일반 컨볼루션과 반대되는 변환을 수행하려고 시도합니다. (feature map에 기반한 가능한 input들을 예측. convolution output 처럼)
즉, 0을 삽입하고 컨볼루션을 수행하여 이미지를 확장하여 이미지 해상도를 높입니다.  
3 × 3 커널이 있는 2x SR을 예로 들면(그림 4에서 볼 수 있듯이), 입력은 먼저 원래 크기의 두 배로 확장되며, 여기서 추가된 픽셀 값은 0으로 설정됩니다(그림 4b).

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.00.01.png)

그런 다음 커널 크기가 3 × 3, 스트라이드 1 및 패딩 1인 컨볼루션을 적용합니다(도 4c).  
이러한 방식으로 입력은 2배만큼 업샘플링되며, 이 경우 receptive field는 최대 2 × 2입니다.  
transposed convolution은 vanilla convolution과 호환되는 연결 패턴을 유지하면서 end-to-end 방식으로 이미지 크기를 확장하기 때문에 SR 모델[57], [78], [79], [85]에서 upsampling layer로 널리 사용됩니다.  
그러나 이 layer는 각 축에서 쉽게 "불균일한 중첩, uneven overlapping"을 일으킬 수 있으며, 두 축의 곱해진 결과는 다양한 크기의 바둑판과 같은 패턴을 생성하여 SR 성능에 해를 끼칩니다.  

- DBPN
- RED-Net
- SRDenseNet
- DSRN

#### Sub-pixel Layer.
또 다른 end-to-end 학습 가능한 업샘플링 레이어인 sub-pixel layer [84]는 그림 5와 같이 컨벌루션으로 복수의 채널을 생성한 다음 reshaping하여 업샘플링을 수행합니다. 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.00.08.png)

이 레이어 내에서 먼저 s^2배 채널로 출력을 생성하는 컨볼루션이 적용되며, 여기서 s는 scaling factor입니다(그림 5b).  
입력 크기가 h × w × c라고 가정하면 출력 크기는 h × w × s^2c가 됩니다.  
그 후 reshaping operation(일명 shuffle [84])을 수행하여 크기가 sh × sw × c인 출력을 생성합니다(그림 5c).  
이 경우 receptive field는 최대 3 × 3이 될 수 있습니다.  
end-to-end upsampling 방식으로 인해 이 레이어는 SR 모델 [25], [28], [39], [93]에서도 널리 사용됩니다.  
transposed convolution layer에 비해 sub-pixel layer는 더 큰 receptive field를 가지며, 이는 더 현실적인 세부 정보를 생성하는 데 도움이 되는 더 많은 contextual information를 제공합니다.  
그러나 receptive field의 분포가 불균일하고 차단된 영역이 실제로 동일한 receptive field를 공유하기 때문에 다른 블록의 경계 근처에서 일부 결함을 초래할 수 있습니다.  
반면에 차단된 영역에서 인접한 픽셀을 독립적으로 예측하면 출력이 원활하지 않을 수 있습니다.  
따라서 Gao et al. [94]는 독립적인 예측을 상호 의존적인 sequential prediction으로 대체하고 더 부드럽고 일관된 결과를 생성하는 PixelTCL을 제안합니다.

- SRGAN
- CARN
- SRMD
- RDN
- Sub-pixel 개선 : PixelTCL

#### Meta Upscale Module.
이전 방법은 scaling factor를 미리 정의해야 합니다.  
즉, 서로 다른 factor에 대해 서로 다른 업샘플링 모듈을 훈련해야 하는데, 이는 비효율적이고 실제로 필요하지 않습니다.  
따라서 Hu 등[95]은 먼저 meta learning을 기반으로 임의의 스케일링 인자의 SR을 해결하는 meta upscale module(그림 6)을 제안합니다. 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.00.17.png)

구체적으로, HR 이미지의 각 대상 위치에 대해 이 모듈은 LR feature map의 작은 패치(즉, k × k × cin)에 투영하고, 투영 오프셋에 따른 컨볼루션 가중치(즉, k × k × cin × cout)와 조밀한 레이어에 의한 scaling factor를 예측하고 컨볼루션을 수행합니다.  
이러한 방식으로 메타 업스케일 모듈은 단일 모델에 의해 임의의 인자로 지속적으로 확대할 수 있습니다.  
그리고 많은 양의 훈련 데이터(여러 인자가 동시에 훈련됨)로 인해 모듈은 고정된 인자에서 비슷하거나 훨씬 더 나은 성능을 나타낼 수 있습니다.  
이 모듈은 추론 중에 가중치를 예측해야 하지만, 업샘플링 모듈의 실행 시간은 특징 추출 시간의 약 1%만을 차지합니다[95].  
그러나 이 방법은 이미지 내용과 무관한 여러 값을 기반으로 각 대상 픽셀에 대한 많은 수의 컨볼루션 가중치를 예측하므로 더 큰 배율에 직면했을 때 예측 결과가 불안정하고 효율성이 떨어질 수 있습니다.  

오늘날 이러한 학습 기반 계층은 가장 널리 사용되는 업샘플링 방법이 되었습니다.  
특히 업샘플링 후 프레임워크(Post-upsampling Super-resolution)에서 이러한 계층은 일반적으로 저차원 공간에서 추출된 고수준 표현을 기반으로 HR 이미지를 재구성하는 최종 업샘플링 단계에서 사용되므로 고차원 공간에서 압도적인 작업을 피하면서 엔드투엔드 SR을 달성합니다.  

- Meta-SR

## Network Design
오늘날 네트워크 설계는 딥 러닝의 가장 중요한 부분 중 하나였습니다.  
초해상도 분야에서 연구자들은 최종 네트워크를 구성하기 위해 4개의 SR 프레임워크(Super-resolution Frameworks) 위에 모든 종류의 네트워크 설계 전략을 적용합니다.
이 절에서는 이러한 네트워크를 네트워크 설계를 위한 필수 원칙 또는 전략으로 네트워크를 분해하고 소개하며 장점과 한계를 하나씩 분석합니다.

### Residual Learning
He 등[96]이 철저한 매핑 방법 대신 잔차 학습을 위한 ResNet을 제안하기 전에 residual learning은 그림 7a에서 볼 수 있듯이 SR 모델 [48], [88], [97]에서 널리 사용되었습니다.  
이 중 잔차 학습 전략은 크게 global, local residual learning으로 나눌 수 있습니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-27%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.46.33.png)

#### Global Residual Learning
이미지 SR은 입력 이미지가 대상 이미지와 높은 상관관계가 있는 이미지 대 이미지 변환 작업이기 때문에 연구자들은 그들 사이의 잔차, 즉 전역 잔차 학습만 학습하려고 합니다.  
이 경우 완전한 이미지에서 다른 이미지로의 복잡한 변환을 학습하는 것을 피하며, 대신 누락된 고주파 세부 정보를 복원하기 위해 residual map만 학습하면 됩니다.  
대부분의 영역의 잔차가 0에 가깝기 때문에 모델 복잡성과 학습 난이도가 크게 낮아집니다.  
따라서 SR 모델[26], [55], [56], [98]에서 널리 사용됩니다.

- VDSR
- Memnet
- DRRN
- IDN

#### Local Residual Learning
로컬 잔차 학습은 ResNet[96]의 잔차 학습과 유사하며, 계속 증가하는 네트워크 깊이로 인한 degradation 문제[96]를 완화하고 훈련 난이도를 줄이며 학습 능력을 향상시키는 데 사용됩니다.  
SR[70], [78], [85], [99]에도 널리 사용됩니다. 

실제로 위의 방법들은 모두 shortcut connection(종종 작은 상수에 의해 확장됨)과 element-wise addition로 구현되는 반면, 전자는 입력 이미지와 출력 이미지를 직접 연결하는 반면, 후자는 일반적으로 네트워크 내부에서 깊이가 다른 레이어 사이에 여러 개의 shortcut을 추가한다는 차이점이 있습니다.

- RCAN
- RED-Net
- DSRN
- MSRN

### Recursive Learning
많은 파라미터를 도입하지 않고 더 높은 수준의 특징을 학습하기 위해 그림 7b와 같이 동일한 모듈을 여러 번 재귀적으로 적용하는 것을 의미하는 재귀적 학습이 SR 필드에 도입됩니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.39.42.png)

그 중 16-recursive DRCN[82]은 재귀적 단위로 단일 컨볼루션 계층을 사용하여 많은 파라미터 없이 SRCNN[22]의 13 × 13보다 훨씬 큰 41 × 41의 receptive field에 도달합니다.  
DRRN[56]은 25번의 재귀적 단위로 ResBlock[96]을 사용하여 17-ResBlock(기준선)보다 훨씬 더 나은 성능을 얻습니다.  
나중에 Tai et al. [55]는 모든 재귀의 출력이 연결되고 암기와 망각을 위해 추가 1 × 1 컨볼루션을 거치는 6번의 재귀적 ResBlock으로 구성된 메모리 블록을 기반으로 MemNet을 제안합니다.  
cascading residual network(CARN)[28]도 여러 ResBlock을 포함하는 유사한 재귀적 단위를 채택합니다.  
최근 Li et al. [86]은 반복적인 up and-down sampling SR 프레임워크를 사용하고 전체 네트워크의 가중치가 모든 재귀에 걸쳐 공유되는 재귀적 학습 기반의 피드백 네트워크를 제안합니다.  

또한 연구자들은 서로 다른 부분에 서로 다른 재귀 모듈을 사용합니다.  
특히 Han et al. [85]는 LR 상태와 HR 상태 간의 신호를 교환하기 위해 이중 상태 순환 네트워크(DSRN)를 제안합니다.  
각 time step(즉, 재귀)에서 각 세부 항목들의 표현이 업데이트되고 LR-HR 관계를 더 잘 탐색하기 위해 교환됩니다.  
유사하게 Lai et al. [65]는 임베딩 및 업샘플링 모듈을 재귀적 단위로 사용하므로 성능 손실이 거의 없는 대신 모델 크기를 훨씬 줄입니다.  

일반적으로 재귀적 학습은 과도한 파라미터를 도입하지 않고도 더 진보적인 표현을 학습할 수 있지만 여전히 높은 계산 비용을 피할 수 없습니다.  
그리고 본질적으로 vanishing or exploding gradient problems 때문에 잔차 학습(섹 3.3.1) 및 multi-supervision(섹 3.4.4)과 같은 일부 기술은 종종 이러한 문제를 완화하기 위해 재귀적 학습과 통합됩니다[55], [56], [82], [85].  

- DRCN
- DRRN
- MemNet
- CARN
- SRFBN
- DSRN

### Multi-path Learning
다중 경로 학습은 다양한 작업을 수행하는 여러 경로를 통해 feature를 전달하고 더 나은 모델링 기능을 제공하기 위해 이를 다시 융합하는 것을 말합니다.  
구체적으로 다음과 같이 global, local and scale-specific multi-path learning으로 나눌 수 있습니다.

#### Global Multi-path Learning
Global 다중 경로 학습은 이미지의 다양한 측면의 feature를 추출하기 위해 여러 경로를 사용하는 것을 말합니다.  
이러한 경로는 propagation 과정에서 서로 교차할 수 있으므로 학습 능력을 크게 향상시킬 수 있습니다.  
구체적으로 LapSRN[27]에는 다양한 hyperparameter optimization 방식으로 sub-band residual을 예측하는 feature extraction path와 두 경로의 신호를 기반으로 HR 이미지를 재구성하는 다른 path가 포함됩니다.  
유사하게 DSRN[85]은 두 개의 경로를 활용하여 저차원 및 고차원 공간에서 각각 정보를 추출하고 학습 능력을 추가로 향상시키기 위해 지속적으로 정보를 교환합니다.  
그리고 pixel recursive super-resolution[64]는 이미지의 글로벌 구조를 캡처하기 위한 conditioning path와 생성된 픽셀의 연속적 의존성을 캡처하기 위한 이전 경로를 채택합니다.  
이와 대조적으로, Ren et al. [100]은 불균형 구조를 가진 여러 경로를 사용하여 upsampling을 수행하고 모델 끝에서 연합합니다.

- LapSRN
- DSRN
- Pixel recursive super-resolution

#### Local Multi-path Learning
인셉션 모듈[101]에 영감을 받은 MSRN[99]은 그림 7e와 같이 다중 스케일 feature extraction을 위해 새로운 블록을 채택합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.40.05.png)

이 블록에서는 커널 크기가 3 × 3 및 5 × 5인 두 개의 컨볼루션 레이어를 채택하여 feature를 동시에 추출한 다음 출력을 연결하고 동일한 작업을 다시 거쳐 마지막에 1 × 1 컨볼루션을 적용합니다.  
shortcut은 element-wise addition을 적용하여 입력과 출력을 연결합니다.  
이러한 local 다중 경로 학습을 통해 SR 모델은 여러 스케일에서 이미지 feature를 더 잘 추출하고 성능을 더욱 향상시킬 수 있습니다.

- MSRN

#### Scale-specific Multi-path Learning
서로 다른 스케일에 대한 SR 모델은 유사한 feature extraction을 거쳐야 한다는 점을 고려하여 Lim et al. [31]은 단일 네트워크로 다중 스케일 SR에 대처하기 위한 스케일별 multi-path learning을 제안합니다.  
구체적으로 설명하기 위해 모델의 주요 구성 요소(즉, feature extraction을 위한 intermediate layer)를 공유하고 네트워크의 시작과 끝에 스케일별 전처리 경로와 업샘플링 경로를 각각 첨부합니다(그림 7f에서 볼 수 있듯이). 

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.40.15.png)

학습 중에는 선택한 스케일에 해당하는 경로만 활성화되고 업데이트됩니다.  
이러한 방식으로 제안된 MDSR [31]은 서로 다른 스케일에 대해 대부분의 매개 변수를 공유하여 모델 크기를 크게 줄이고 단일 스케일 모델과 유사한 성능을 나타냅니다.  
유사한 scale-specific multi-path learning은 CARN [28] 및 ProSR [32]에서도 채택됩니다.

- EDSR
- MDSR
- CARN
- ProSR

### Dense Connections
Huang 등[102]이 dense block을 기반으로 DenseNet을 제안한 이후, 조밀한 연결은 비전 작업에서 점점 더 인기를 얻고 있습니다.  
Dense block의 각 layer에 대해 이전 모든 레이어의 feature map이 입력으로 사용되고 자체 feature map이 모든 후속 레이어의 입력으로 사용되어 layer dense block $(l ≥ 2)$에서 $l · (l - 1)/2$ 연결로 이어집니다.  
조밀한 연결은 gradient vanishing을 완화하고 신호 전파를 향상시키며 feature 재사용을 장려하는 데 도움이 될 뿐만 아니라 작은 growth rate(dense blocks의 채널 수)을 사용하고 모든 input feature map을 연결한 후 채널을 squeezing해서 모델 크기를 크게 줄일 수 있습니다.  

low-level and high-level feature을 융합하여 고품질의 세부 정보를 재구성하기 위한 풍부한 정보를 제공하기 위해 그림 7d에서 볼 수 있듯이 dense connection이 SR 필드에 도입됩니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.39.57.png)

Tong et al. [79]는 dense blocks을 채택하여 69-SRDenseNet을 구성할 뿐만 아니라 서로 다른 dense block 사이에 dense connection을 삽입합니다.  
즉, 모든 dense block에 대해 이전 블록의 feature map은 input으로 사용되고 자체 feature map은 모든 후속 블록에 input으로 사용됩니다.  
이러한 계층 수준 및 블록 수준의 dense connection은 MemNet [55], CARN [28], RDN [93] 및 ESRGAN [103]에서도 채택됩니다.  
DBPN [57]도 dense connection을 광범위하게 채택하지만 다운 샘플링 유닛과 마찬가지로 dense connection은 모든 업샘플링 유닛 사이에 있습니다.

- SRDenseNet
- MemNet
- CARN
- RDN
- ESRGAN
- DBPN

### Attention Mechanism
#### Channel Attention
서로 다른 채널 간의 feature 표현의 독립성과 상호 작용을 고려하여 Hu et al. [104]는 그림 7c에서 볼 수 있듯이 채널 독립성을 명시적으로 모델링하여 학습 능력을 향상시키는 "squeeze-and-excitation" 블록을 제안합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.39.49.png)

이 블록에서 각 입력 채널은 global average pooling(GAP)을 사용하여 channel descriptor(즉, 상수)로 압축된 다음, 이러한 설명자를 두 개의 dense layer로 공급하여 입력 채널에 대한 채널별 scaling factor를 생성합니다.  
최근 Zhang et al. [70]은 channel attention mechanism을 SR과 통합하고 모델의 표현 능력과 SR 성능을 현저하게 향상시키는 RCAN을 제안합니다.  
feature correlation를 더 잘 학습하기 위해 Dai et al. [105]는 second-order channel attention(SOCA) 모듈을 추가로 제안합니다.  
SOCA는 GAP 대신 second-order feature를 사용하여 channel-wise features를 적응적으로 rescale하고 더 많은 정보와 판별 표현을 추출할 수 있도록 합니다.

- RCAN
- SAN

#### Non-local Attention
대부분의 기존 SR 모델은 local receptive field가 매우 제한되어 있습니다.  
그러나 일부 먼 거리의 물체 또는 텍스처는 로컬 패치 생성에 매우 중요할 수 있습니다.  
따라서 Zhang et al. [106]은 픽셀 간의 넓은 범위의 종속성을 캡처하는 feature를 추출하기 위해 local and nonlocal attention block을 제안합니다.  
특히 feature를 추출하기 위한 trunk branch와 trunk branch의 특징을 적응적으로 스케일링하기 위한 (non)local mask branch를 제안합니다.  
그 중 local branch는 encoder-decoder 구조를 사용하여 local attention을 학습하는 반면, non-local branch는 내장된 Gaussian function을 사용하여 feature map의 두 위치별 인덱스 간의 관계를 평가하여 scaling weight를 예측합니다.  
이 메커니즘을 통해 제안된 방법은 spatial attention을 잘 캡처하고 표현 능력을 더욱 향상시킵니다.  
유사하게 Dai et al. [105]도 non-local attention mechanism을 통합하여 장거리의 공간적 정보를 캡처합니다.

- RNAN
- SAN

### Advanced Convolution
컨볼루션 연산은 심층 신경망의 기본이기 때문에 연구자들은 더 나은 성능 또는 더 큰 효율을 위해 컨볼루션 연산을 개선하려고 시도합니다.

#### Dilated Convolution
contextual 정보가 SR에 대한 현실적인 세부 정보를 생성하는 것을 용이하게 한다는 것은 잘 알려져 있습니다.  
따라서 Zhang et al. [107]은 SR 모델에서 공통된 컨볼루션을 dilated convolution으로 대체하고 receptive field를 두 배 이상 증가시키며 훨씬 더 나은 성능을 달성합니다.

- IRCNN

#### Group Convolution

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.40.25.png)

최근 lightweight CNN[108], [109], Hui et al. [98] 및 An et al. [28]은 vanilla 컨볼루션을 group 컨볼루션으로 대체하여 각각 IDN과 CARN-M을 제안합니다.  
일부 이전 연구에서 입증된 바와 같이, group 컨볼루션은 약간의 성능 손실을 감수하면서 매개 변수와 연산의 수를 훨씬 줄입니다 [28], [98].

- IDN
- CARN

#### Depthwise Separable Convolution
Howard 등[110]이 효율적인 컨볼루션을 위해 depthwise separable convolution을 제안한 이후 다양한 분야로 확장되었습니다.  
구체적으로, factorized된 depthwise separable convolution과 pointwise convolution(즉, 1 × 1 컨볼루션)으로 구성되어 있어 정확도를 조금만 낮추면 많은 매개변수와 연산을 줄일 수 있습니다[110].  
그리고 최근 Nie 등[81]은 depthwise separable convolution을 사용하여 SR 아키텍처를 훨씬 가속화합니다.

### Region-recursive Learning
대부분의 SR 모델은 SR을 픽셀에 독립적인 작업으로 취급하므로 생성된 픽셀 간의 상호 의존성을 제대로 맞출 수 없습니다.  
PixelCNN[111]에서 영감을 받아 Dahl 등[64]은 먼저 두 개의 네트워크를 사용하여 global contextual information과 연속적인 생성 의존성을 각각 캡처하여 픽셀별 생성을 수행하는 pixel recursive learning을 제안합니다.  
이러한 방식으로 제안된 방법은 매우 낮은 얼굴 이미지(예: 8 × 8)에 초해상도화하기 위해 현실적인 모발 및 피부 세부 정보를 합성하며 MOS 테스트[64](Sec. 2.3.3)에 대한 이전 방법을 훨씬 능가합니다.

human attention shifting mechanism[112]에 동기를 부여받은 Attention-FH[113]는 또한 패치를 순차적으로 발견하고 local enhancemen를 수행하기 위해 recurrent policy network에 의존하여 이 전략을 채택합니다.  
이러한 방식으로 각 이미지에 대한 최적의 searching path를 고유의 특성에 따라 적응적으로 personalize할 수 있으므로 이미지의 global intra-dependence를 완전히 활용할 수 있습니다.  

이러한 방법은 어느 정도 더 나은 성능을 보여주지만 긴 propagation path가 필요한 recursive process는 특히 초해상도 HR 이미지의 경우 계산 비용과 훈련 난이도를 크게 증가시킵니다.

- Pixel recursive super resolution
- Attention-FH

### Pyramid Pooling

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.40.35.png)

Spatial pyramid pooling layer[114]에서 영감을 받아 Zhao et al. [115]는 global and local contextual information를 더 잘 활용하기 위해 pyramid pooling module을 제안합니다.  
구체적으로, h x w × c 크기의 feature map에 대해 각 feature map은 M × M bin으로 분할되고 global average pooling을 거치게 되어 M × M × c 출력이 생성됩니다.  
그런 다음 출력을 단일 채널로 압축하기 위해 1 × 1 컨볼루션이 수행됩니다.  
그 후, low-dimensional feature map은 bilinear interpolation을 통해 원래 feature map과 동일한 크기로 upsample됩니다.  
서로 다른 M을 사용하여 모듈은 global 및 local contextual information를 효과적으로 통합합니다.  
이 모듈을 통합함으로써 제안된 EDSR-PP 모델[116]은 기존선 대비 성능을 더욱 향상시킵니다.

- EDSR-PP

### Wavelet Transformation
잘 알려진 바와 같이, wavelet transformation(WT)[117], [118]은 이미지 신호를 텍스처 세부 정보를 나타내는 high-frequency sub-band와 global topological information를 포함하는 low-frequency sub-band로 분해하여 이미지를 매우 효율적으로 표현하는 것입니다.  
Bae et al. [119]는 먼저 WT를 deep learning based SR model과 결합하고 interpolated LR wavelet의 subband를 입력으로 받아 해당 HR sub-band의 잔차를 예측합니다.  
WT와 inverse WT는 각각 LR input을 분해하고 HR output을 재구성하는 데 적용됩니다.  
유사하게, DWSR [120] 및 Wavelet-SRNet [121]도 wavelet domain에서 SR을 수행하지만 더 복잡한 구조로 수행합니다.  
위의 작업들이 각 sub-band를 독립적으로 처리하는 것과 달리, MWCNN [122]는 multi-level WT를 채택하고 연결된 sub-band를 단일 CNN의 입력으로 사용하여 이들 사이의 의존성을 더 잘 캡처합니다.  
wavelet transformation에 의한 효율적인 표현으로 인해 이 전략을 사용하는 모델은 종종 모델 크기와 계산 비용을 훨씬 줄이면서 경쟁 성능을 유지합니다[119], [122].

- Matconvnet
- DWSR
- Wavelet-SRNet
- MWCNN

### Desubpixel
추론 속도를 높이기 위해 Vu 등[123]은 lower-dimensional space에서 시간이 많이 걸리는 feature extraction을 수행할 것을 제안하고, sub-pixel layer(Learning-based Upsampling)의 shuffle operation의 역인 desubpixel을 제안합니다.  
특히, desubpixel operation은 이미지를 공간적으로 분할하고, 추가 채널로 적층하여 정보 손실을 방지합니다.  
이러한 방식으로 모델 초기에는 입력 이미지를 desubpixel별로 downsample하고, 저차원 공간에서 표현을 학습하고, 마지막에는 목표 크기로 upsample합니다.  
제안된 모델은 매우 빠른 추론과 우수한 성능으로 스마트폰의 PIRM 챌린지[81]에서 최고의 점수를 달성합니다.

### xUnit
spatial feature processing와 nonlinear activation를 결합하여 복잡한 feature들을 보다 효율적으로 학습하기 위해 Kligvasser et al. [124]는 spatial activation function를 학습하기 위한 xUnit을 제안합니다.  
특히 ReLU는 입력과의 element-wise multiplication을 수행할 weight map을 결정하는 것으로 간주되며, xUnit은 컨볼루션 및 Gaussian gating을 통해 weight map을 직접 학습합니다.  
xUnit은 성능에 미치는 극적인 영향으로 인해 계산이 더 까다롭지만 ReLU와 성능을 일치시키면서 모델 크기를 크게 줄일 수 있습니다.  
이러한 방식으로 저자는 성능 저하 없이 모델 크기를 거의 50%까지 줄일 수 있습니다.

## Learning Strategies
### Loss Functions
초해상도 분야에서는 손실 함수를 사용하여 재구성 오류를 측정하고 모델 최적화를 안내합니다.  
초기에는 연구자들이 일반적으로 pixelwise L2 loss를 사용하지만 나중에 재구성 품질을 매우 정확하게 측정할 수 없다는 것을 발견합니다.  
따라서 재구성 오류를 더 잘 측정하고 보다 현실적이고 고품질의 결과를 생성하기 위해 다양한 손실 함수(예: content loss[29], adversarial loss[25])가 채택됩니다.  
오늘날 이러한 손실 함수는 중요한 역할을 하고 있습니다.  
이 섹션에서는 널리 사용되는 손실 함수를 자세히 살펴보도록 하겠습니다.  
이 섹션의 표기법은 간결성을 위해 대상 HR 이미지 ˆIy, 생성된 HR 이미지 Iy의 첨자 y를 무시하고 2.1절, Problem Definitions을 따릅니다.

#### Pixel Loss
Pixel loss는 두 이미지 간의 픽셀 단위 차이를 측정하며 주로 L1 손실(즉, 평균 절대 오차) 및 L2 손실(즉, 평균 제곱 오차)을 포함합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.40.53.png)

여기서 h, w 및 c는 각각 평가된 이미지의 높이, 너비 및 채널 수입니다.  
또한, 다음과 같이 주어진 픽셀 L1 손실의 변형, 즉 Charbonnier loss[27], [125]가 있습니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.41.04.png)

여기서 엡실론은 수치 안정성을 위한 상수(예: 10-3)입니다.

pixel loss는 생성된 HR 이미지 ˆI이 픽셀 값의 ground truth I에 충분히 가깝도록 제한합니다.  
L1 loss와 비교할 때, L2 loss는 더 큰 오류에 불이익을 주지만 작은 오류에 더 관대하기 때문에 종종 너무 부드러운 결과를 초래합니다.  
실제로 L1 손실은 L2 손실[28], [31], [126]에 비해 향상된 성능과 수렴을 보여줍니다.  
PSNR(Peak Signal-to-Noise Ratio)의 정의는 pixel-wise difference와 높은 상관 관계가 있으며 pixel loss을 최소화하면 PSNR이 직접 최대화되기 때문에 점진적으로 pixel loss loss function이 가장 널리 사용되는 loss function가 됩니다.  
그러나 pixel loss는 실제로 image quality(예: perceptual quality[29], textures[8])을 고려하지 않기 때문에 결과는 고주파 세부 정보가 부족한 경우가 많으며 지나치게 부드러운 텍스처[25], [29], [58], [74]에 대해 시각적으로 만족하지 않습니다.

#### Content Loss
이미지의 perceptual quality을 평가하기 위해 content loss을 SR[29], [127]에 도입합니다.  
특히 사전 학습된 이미지 분류 네트워크를 사용하여 이미지 간의 의미론적 차이를 측정합니다.  
이 네트워크를 φ로 나타내고 l번째 layer에서 추출된 고수준의 표현을 φ(l)(I)로 표시하면 content loss는 다음과 같이 두 이미지의 고수준의 표현 사이의 유클리드 거리로 표시됩니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.41.21.png)

여기서 hl, wl 및 cl은 각각 레이어 l 상의 표현의 높이, 너비 및 채널 수를 나타냅니다.  
본질적으로 content loss는 계층적 이미지 feature에 대한 학습된 지식을 분류 네트워크 φ에서 SR 네트워크로 전달합니다.  
pixel loss과 대조적으로 content loss는 출력 이미지 ˆI이 픽셀을 정확하게 일치시키도록 강요하는 대신 목표 이미지 I와 지각적으로 유사하도록 장려합니다.  
따라서 시각적으로 더 인지 가능한 결과를 생성하고 이 필드 [8], [25], [29], [30], [46], [103]에서도 널리 사용되며, 여기서 VGG [128] 및 ResNet [96]은 가장 일반적으로 사용되는 사전 학습된 CNN입니다.

#### Texture Loss
재구성된 이미지가 대상 이미지와 동일한 스타일(예: 색상, 텍스처, 대비)을 가져야 하고 (Gatys 등에 의한 스타일 표현에 의해 동기 부여되어야 한다는 점을 고려하여 [129], [130]), texture loss(일명 style reconstruction loss)은 SR에 도입됩니다.  
[129], [130]에 이어 이미지 텍스처는 서로 다른 feature 채널 간의 상관 관계로 간주되고 그램 매트릭스 G(l) ∈ R^cl×cl로 정의되며, 여기서 G(l) ij는 레이어 l에서 벡터화된 피처 맵 i와 j 사이의 내적입니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.41.32.png)

여기서 vec(·)는 벡터화 연산을 나타내고, φ(l) i(I)는 이미지 I의 레이어 l 상의 피쳐맵들의 i번째 채널을 나타냅니다.  
그런 다음 texture loss은 다음에 의해 주어집니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.41.41.png)

Sajadi 등이 제안한 EnhancedNet[8]은 텍스처 손실을 사용하여 훨씬 더 사실적인 텍스처를 생성하고 시각적으로 더 만족스러운 결과를 생성합니다.  
그럼에도 불구하고 텍스처에 맞게 패치 크기를 결정하는 것은 여전히 경험적입니다.  
너무 작은 패치는 텍스처 영역에서 결함을 초래하는 반면 너무 큰 패치는 텍스처 통계가 다양한 텍스처의 영역에서 평균을 내기 때문에 전체 이미지에서 결함을 초래합니다.

#### Adversarial Loss
최근 몇 년 동안 강력한 학습 능력으로 인해 GAN[24]은 점점 더 많은 관심을 받고 다양한 비전 작업에 도입되고 있습니다.  
구체적으로, GAN은 생성(예: 텍스트 생성, 이미지 변환)을 수행하는 생성기와 대상의 분포에서 샘플링된 생성된 결과물을 입력으로 받아 각 입력이 대상의 분포에서 나오는지 여부를 판별하는 판별기로 구성됩니다.  
훈련 중에는 두 가지 단계가 교대로 수행됩니다. (a) 생성기를 고정하고 판별기를 더 잘 식별하도록 훈련시키거나 (b) 판별기를 고정하고 생성기를 훈련시켜 판별기를 속입니다.  
적절한 반복적인 적대적 훈련을 통한 결과로 생성기는 실제 데이터의 분포와 일치하는 출력을 생성하는 동안, 판별기는 생성된 데이터와 실제 데이터를 구별할 수 없습니다.

초해상도 측면에서 적대적 학습을 채택하는 것은 간단하며, 이 경우 SR 모델을 생성기로 취급하고 입력 이미지의 생성 여부를 판단하기 위해 추가 판별기를 정의하면 됩니다.  
따라서 Ledig et al. [25]는 먼저 cross entropy에 기반한 adversarial loss을 사용하는 SRGAN을 다음과 같이 제안합니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.41.49.png)

여기서 Lgan_ce_g 및 Lgan_ce_d는 각각 생성기(즉, SR 모델)와 판별기 D(즉, 이진 분류기)의 적대적 손실을 나타내고, Is는 ground truth에서 무작위로 샘플링된 이미지를 나타냅니다.  
또한 Enhancenet [8]도 유사한 적대적 손실을 채택합니다.  
또한 Wang et al. [32] 및 Yuan et al. [131]은 보다 안정적인 훈련 프로세스와 더 높은 품질 결과를 위해 least square error에 기반한 적대적 손실을 사용합니다 [132]:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.42.02.png)

특정 형태의 적대적 손실에 초점을 맞춘 위의 작업과 달리 Park et al. [133]은 pixel-level 판별기가 무의미한 고주파 노이즈를 생성한다고 주장하고, 실제 HR 이미지의 더 의미 있는 속성을 캡처하는 사전 학습된 CNN에 의해 추출된 고수준 표현에서 작동하도록 또 다른 feature-level 판별기를 부착합니다.  
Xu et al. [63]은 생성기와 multi-class 판별기로 구성된 다중 클래스 GAN을 통합합니다.  
그리고 ESRGAN [103]은 입력 이미지가 실제 또는 가짜일 확률 대신 실제 이미지가 가짜보다 상대적으로 더 사실적일 확률을 예측하여 더 자세한 텍스처를 복구하도록 안내하는 relativistic GAN [134]을 사용합니다.

광범위한 MOS 테스트(섹. 2.3.3)는 adversarial loss and content loss로 훈련된 SR 모델이 pixel loss로 훈련된 모델에 비해 낮은 PSNR을 달성하더라도 지각 품질에서 상당한 향상을 가져온다는 것을 보여줍니다[8], [25].  
실제로 판별기는 실제 HR 이미지의 학습하기 어려운 latent pattern을 추출하고 생성된 HR 이미지가 일치하도록 밀어 넣어 더 현실적인 이미지를 생성하는 데 도움이 됩니다.  
그러나 현재 GAN의 훈련 프로세스는 여전히 어렵고 불안정합니다.  
GAN 훈련을 안정화하는 방법에 대한 일부 연구가 있었지만[135], [136], [137], SR 모델에 통합된 GAN이 올바르게 훈련되고 활성적인 역할을 수행하도록 하는 방법은 여전히 문제로 남아 있습니다.

#### Cycle Consistency Loss
Zhu et al. [138]이 제안한 CycleGAN에서 영감을 받아 Yuan et al. [131]은 초해상도를 위한 cycle-in-cycle 접근 방식을 제시합니다.  
구체적으로, 그들은 LR 이미지 I을 HR 이미지 ˆI로 초해상도 할 뿐만 아니라 다른 CNN을 통해 ˆI을 다른 LR 이미지 I'로 다운샘플링합니다.  
재생성된 I'는 입력 I과 동일해야 하므로 픽셀 수준의 일관성을 제한하기 위해 cycle consistency loss이 도입됩니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.40.16.png)

#### Total Variation Loss
생성된 이미지의 노이즈를 억제하기 위해 Aly et al. [140]에 의해 total variation (TV) loss[139]이 SR에 도입됩니다.  
이는 인접 픽셀 간의 absolute difference의 합으로 정의되며 다음과 같이 이미지에 노이즈가 얼마나 있는지 측정합니다:

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.40.28.png)

Lai et al. [25] 와 Yuan et al. [131] 또한 spatial smoothness을 지우기 위해 TV 손실을 채택합니다.

#### Prior-Based Loss
위의 손실 함수 외에도 생성을 제한하기 위해 외부 사전 지식도 도입됩니다.  
특히, Bulat et al. [30]은 얼굴 이미지 SR에 초점을 맞추고 얼굴 경계의 일관성을 제한하기 위해 face alignment network(FAN)를 도입합니다.  
FAN은 face alignment prior을 도입하기 위해 사전 학습되고 통합된 다음 SR과 공동으로 학습됩니다.  
이러한 방식으로 제안된 Super-FAN은 LR face alignment과 얼굴 이미지 SR 모두에서 성능을 향상시킵니다.  

사실, content loss과 texture loss은 기본적으로 분류 네트워크를 도입하여 SR에 대한 계층적 이미지 feature에 대한 사전 지식을 제공합니다.  
더 많은 사전 지식을 도입함으로써 SR 성능을 더욱 향상시킬 수 있습니다.  

이 절에서는 SR에 대한 다양한 손실 함수를 소개합니다.  
실제로 연구자들은 특히 distortion-perception tradeoff[25], [103], [142], [143], [144]에 대해 생성 프로세스의 다양한 측면을 제한하기 위해 weighted average [8], [25], [27], [46], [141]에 의해 여러 손실 함수를 결합하는 경우가 많습니다.  
그러나 다양한 손실 함수의 가중치는 많은 경험적 탐색이 필요하며, 합리적이고 효과적으로 결합하는 방법은 여전히 문제로 남아 있습니다.

### Batch Normalization
심층 CNN의 훈련을 가속화하고 안정화하기 위해 Sergey et al. [145]는 네트워크의 내부의 covariate shift을 줄이기 위한 batch normalization(BN)를 제안합니다.  
특히, 각 미니 배치에 대해 정규화를 수행하고 표현 능력을 유지하기 위해 각 채널에 대해 두 개의 extra transformation parameter를 훈련합니다.  
BN은 중간 feature distribution를 보정하고 vanishing gradient를 완화하기 때문에 더 높은 학습 속도를 사용하고 initialization에 덜 주의할 수 있습니다.  
따라서 이 기술은 SR 모델 [25], [39], [55], [56], [122], [146]에서 널리 사용됩니다.  
그러나 Lim et al. [31]은 BN이 각 이미지의 scale 정보를 잃고 네트워크 범위의 유연성을 제거한다고 주장합니다.  
따라서 BN을 제거하고 절약된 메모리 비용(최대 40%)을 사용하여 훨씬 더 큰 모델을 개발하여 성능을 크게 향상시킵니다.  
일부 다른 모델 [32], [103], [147]도 이 경험을 채택하고 성능을 향상시킵니다.

- SRGAN
- SRMD
- Memnet
- DRRN
- MWCNN

- Removing BN
- EDSR
- ProSR
- ESRGAN

### Curriculum Learning
Curriculum learning[148]은 더 쉬운 작업에서 시작하여 점진적으로 난이도를 높이는 것을 의미합니다.  
초해상도는 ill-posed problem(여러 개의 정답이 있을 수 있는 문제)이며 항상 큰 scaling factor, noise and blurring과 같은 불리한 조건을 겪기 때문에 학습 난이도를 줄이기 위해 Curriculum learning이 통합됩니다.  
큰 스케일링 인자를 가진 SR의 난이도를 줄이기 위해 Wang et al. [32], Bei et al. [149] 및 An et al. [150]은 모델 구조(섹션 3.1.3)뿐만 아니라 훈련 절차에서도 점진적인 ProSR, ADRSR 및 progressive CARN을 각각 제안합니다.  
훈련은 2x upsampling으로 시작하여 훈련을 마친 후 4x 이상의 scaling factor를 가진 부분을 점차 장착하고 이전 부분과 섞습니다.  
특히 ProSR은 처음 수준의 출력과 [151]에 이어 이전 레벨의 업샘플링된 출력을 선형으로 결합하여 블렌딩하고, ADRSR은 이들을 연결하고 다른 컨볼루션 레이어를 부착하는 반면, progressive CARN은 이전 재구성 블록에서 2배 해상도로 이미지를 생성하는 재구성 블록으로 대체합니다.

또한 Park et al. [116]은 8x SR 문제를 3개의 하위 문제(즉, 1x~2x, 2x~4x, 4x~8x)로 나누고 각 문제에 대해 독립적인 네트워크를 훈련합니다.  
그런 다음 그 중 2개를 연결하고 미세 조정한 다음 세 개와 함께합니다.  
또한 어려운 조건에서 4x SR을 1x~2x, 2x~4x 및 노이즈 제거 또는 디블러링 하위 문제로 분해합니다.  
대조적으로 SRFBN[86]은 반대의 조건, 즉 쉬운 degradation에서 시작하여 점진적으로 증가하는 degradation 복잡성에서 SR을 위해 이 전략을 사용합니다.  
일반적인 훈련 절차에 비해 curriculum learning은 특히 큰 요인의 경우 훈련 난이도를 크게 줄이고 총 훈련 시간을 단축합니다.  

- ProSR
- ADRSR
- Progressive CARN
- SRFBN

### Multi-supervision
Multi-supervision은 gradient propagation를 향상시키고 vanishing and exploding gradient을 피하기 위해 모델 내에 여러 감독 신호를 추가하는 것을 말합니다.  
recursive learning에 의해 도입된 gradient problem를 방지하기 위해 DRCN[82]은 recursive unit와 함께 multi-supervision을 통합합니다.  
특히, recursive unit의 각 출력을 reconstruction module에 공급하여 HR 이미지를 생성하고 모든 중간 재구성을 통합하여 최종 예측을 구축합니다.  
또한 recursive learning을 기반으로 하는 MemNet[55] 및 DSRN[85]에서도 유사한 전략을 취합니다.  
또한, progressive upsampling framework(Progressive Upsampling Super-resolution)에 따른 LapSRN[27], [65]은 propagation 중에 서로 다른 스케일의 중간 결과를 생성하기 때문에 multi-supervision 전략을 채택하는 것은 간단합니다.  
특히, 중간 결과는 실측 HR 이미지에서 다운샘플링된 중간 이미지와 동일해야 합니다.  
실제로 이 multi-supervision 기술은 손실 함수에서 일부 용어를 추가하여 구현되는 경우가 많으며, 이러한 방식으로 supervision signal가 더 효과적으로 back-propagated되어 훈련 난이도를 낮추고 모델 훈련을 향상시킵니다.  

- DRCN
- MemNet
- DSRN
- LapSRN

## Other Improvements
네트워크 설계 및 학습 전략 외에도 SR 모델을 더욱 개선하는 다른 기술이 있습니다.

### Context-wise Network Fusion
Context-wise network fusion(CNF)[100]은 여러 SR 네트워크의 예측을 융합한 stacking 기술을 의미합니다(즉, 3.3.3절의 다중 경로 학습의 특수한 경우).  
구체적으로 설명하자면, 서로 다른 모델 구조로 개별 SR 모델을 개별적으로 훈련하고 각 모델의 예측을 개별 컨볼루션 레이어에 공급한 다음 마지막으로 최종 예측 결과가 될 출력을 합산합니다.  
이 CNF 프레임워크 내에서 세 개의 lightweight SRCNN[22], [23]으로 구성된 최종 모델은 허용 가능한 효율성으로 SOTA 모델과 비슷한 성능을 달성합니다[100].  

### Data Augmentation
Data augmentation은 딥 러닝으로 성능을 향상시키는 데 가장 널리 사용되는 기술 중 하나입니다.  
이미지 초해상도의 경우 cropping, flipping, scaling, rotation, color jittering 등이 유용한 증강 옵션입니다 [27], [31], [44], [56], [85], [98].  
또한 Bei et al. [149]는 RGB 채널을 무작위로 셔플하여 데이터를 증강할 뿐만 아니라 색상 불균형으로 데이터 세트로 인한 color bias을 완화합니다.

### Multi-task Learning
Multi-task learning[152]은 object detection 및 semantic segmentation[153], head pose estimation 및 facial attribute inference[154]과 같은 관련 작업의 훈련 신호에 포함된 도메인별 정보를 활용하여 일반화 능력을 향상시키는 것을 의미합니다.  
SR 분야에서 Wang et al. [46]은 semantic 지식을 제공하고 semantic별 세부 정보를 생성하기 위한 semantic segmentation network를 통합합니다.  
특히, semantic map을 입력으로 사용하고 중간 feature map에서 수행되는 affine transformation의 spatial-wise 매개변수를 예측하기 위한 spatial feature transformation을 제안합니다.  
따라서 제안된 SFT-GAN은 풍부한 semantic 영역을 가진 이미지에서 더 현실적이고 시각적으로 만족스러운 텍스처를 생성합니다.  
또한, 노이즈 이미지를 직접 초해상도하면 노이즈 증폭이 발생할 수 있음을 고려하여 DNSR[149]은 노이즈 제거 네트워크와 SR 네트워크를 별도로 훈련한 다음 이들을 연결하고 미세 조정을 함께 수행할 것을 제안합니다.  
유사하게, cycle-in-cycle GAN(CinCGAN)[131]은 cycle-in-cycle 노이즈 제거 프레임워크와 cycle-in-cycle SR 모델을 결합하여 노이즈 감소 및 초해상도를 공동 수행합니다.  
서로 다른 작업은 데이터의 서로 다른 측면에 초점을 맞추는 경향이 있기 때문에 관련 작업을 SR 모델과 결합하면 일반적으로 추가 정보와 지식을 제공하여 SR 성능이 향상됩니다.

- SRF-GAN
- DNSR
- CinCGAN

### Network Interpolation
PSNR 기반 모델은 이미지를 ground truth에 더 가깝게 생성하지만 blurring 문제를 유발하는 반면, GAN 기반 모델은 더 나은 지각 품질을 제공하지만 불쾌한 결함(예: 무의미한 노이즈로 인해 이미지가 더 "현실적"임)를 보입니다.  
왜곡과 인식의 균형을 더 잘 맞추기 위해 Wang et al. [103], [155]는 네트워크 보간 전략을 제안합니다.  
특히, PSNR 기반 모델을 훈련하고 미세 조정을 통해 GAN 기반 모델을 훈련한 다음 두 네트워크의 모든 해당 매개 변수를 보간하여 중간 모델을 도출합니다.  
네트워크를 재교육하지 않고 보간 가중치를 조정하여 훨씬 적은 결함으로 의미 있는 결과를 생성합니다.

### Self-Ensemble
Self-ensemble, 일명 향상된 예측[44]은 SR 모델에 의해 일반적으로 사용되는 추론 기법입니다.  
구체적으로, 서로 다른 각도(0 ◦, 90 ◦, 180 ◦, 270 ◦)의 회전과 수평 뒤집기가 LR 이미지에 적용되어 8개의 이미지 세트를 얻습니다.  
그런 다음 이러한 이미지가 SR 모델에 공급되고 해당 inverse transformation이 재구성된 HR 이미지에 적용되어 출력을 얻습니다.  
최종 예측 결과는 이러한 출력의 평균 [31], [32], [44], [70], [78], [93] 또는 중간값[83]에 의해 수행됩니다.  
이러한 방식으로 이러한 모델은 성능을 더욱 향상시킵니다.

- EDSR
- ProSR
- RCAN
- RED-Net
- RDN

## State-of-the-art Super-resolution Models
최근 몇 년 동안 딥 러닝에 기반한 이미지 초해상도 모델이 점점 더 많은 관심을 받고 있으며 최첨단 성능을 달성했습니다.  
이전 섹션에서는 모델 프레임워크(섹 3.1), 업샘플링 방법(섹 3.2), 네트워크 설계(섹 3.3) 및 학습 전략(섹 3.4)을 포함하여 SR 모델을 특정 구성 요소로 분해하고 이러한 구성 요소를 계층적으로 분석하고 장점과 한계를 식별합니다.  
사실, 오늘날 대부분의 최첨단 SR 모델은 기본적으로 위에서 요약한 여러 전략의 조합에 기인할 수 있습니다.  
예를 들어, RCAN[70]의 가장 큰 기여는 channel attention mechanism(섹 3.3.5)에서 비롯되며, subpixel upsampling(섹 3.2.2), residual learning(섹 3.3.1), pixel L1 loss(섹 3.4.1) 및 self-ensemble(섹 3.5.5)과 같은 다른 전략도 사용합니다.  
유사한 방식으로 표 2에서 볼 수 있듯이 몇 가지 대표 모델과 주요 전략을 요약합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.40.50.png)

SR 정확도 외에도 효율성은 또 다른 매우 중요한 측면이며 다양한 전략은 효율성에 어느 정도 영향을 미칩니다.  
따라서 이전 섹션에서는 제시된 전략의 정확도를 분석할 뿐만 아니라 post-upsampling(섹 3.1.2), recursive learning(섹 3.3.2), dense connections(섹 3.3.4), xUnit(섹 3.3.11)과 같이 효율성에 더 큰 영향을 미치는 전략에 대한 구체적인 영향을 나타냅니다.  
또한 그림 8과 같이 SR 정확도(즉, PSNR), 모델 크기(즉, 매개 변수 수) 및 계산 비용(즉, number of Multi-Adds)에 대한 몇 가지 대표적인 SR 모델을 벤치마크합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-28%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.41.08.png)

정확도는 4개의 벤치마크 데이터 세트(예: 세트5[48], 세트14[49], B100[40] 및 Urban100[50])에서 PSNR의 평균으로 측정됩니다.  
그리고 모델 크기와 계산 비용은 출력 해상도가 720p(즉, 1080 × 720)인 PyTorch-OpCounter[157]를 사용하여 계산됩니다.  
모든 통계는 원본 논문에서 파생되거나 공식 모델에서 계산되며 scaling factor는 2입니다.  

# UNSUPERVISED SUPER-RESOLUTION
기존의 초해상도 작업은 대부분 supervised learning, 즉 일치하는 LR-HR 이미지 쌍으로 학습하는 데 중점을 둡니다.  
그러나 동일한 장면의 이미지를 수집하지만 해상도가 다르기 때문에 SR 데이터 세트의 LR 이미지는 HR 이미지에 대해 미리 정의된 degradation를 수행하여 얻는 경우가 많습니다.  
따라서 훈련된 SR 모델은 실제로 미리 정의된 degradation의 역 과정을 학습합니다.  
사전에 manual degradation을 도입하지 않고 실제 LR-HR 매핑을 학습하기 위해 연구자들은 unsupervised SR에 점점 더 많은 관심을 기울입니다.  
이 경우 훈련을 위해 쌍을 이루지 않은 LR-HR 이미지만 제공되므로 결과적으로 나오는 모델은 실제 시나리오에서 SR 문제를 처리할 가능성이 더 높습니다.  
다음으로 딥 러닝이 포함된 기존의 여러 unsupervised SR 모델을 간략하게 소개하고 더 많은 방법은 아직 탐구되지 않았습니다.  

## Zero-shot Super-resolution
단일 이미지 내부에 들어있는 내부 이미지 통계 정보가 SR에 충분한 정보를 제공했다는 점을 고려하여 Shocher et al. [83]은 대규모 외부 데이터 세트에 대한 일반 모델을 훈련하는 대신 테스트하는 중에 image-specific SR 네트워크를 훈련하여 unsupervised SR에 대처할 수 있는 zero-shot super-resolution(ZSSR)를 제안합니다.  
특히, [158]을 사용하여 단일 이미지에서 degradation kernel을 추정하고 이 커널을 사용하여 이 이미지에 대해 다양한 scaling factor와 augmentation으로 degradation를 수행하여 작은 데이터 세트를 구축합니다.  
그런 다음 SR을 위한 작은 CNN이 이 데이터 세트에서 훈련되어 최종 예측에 사용됩니다.  
이러한 방식으로 ZSSR은 모든 이미지 내부의 가장자리(1 dB for estimated kernels and 2 dB for known kernels)에 cross-scale internal recurrence을 활용하므로 이상적이지 않은 조건(즉, non-bicubic degradation에 의해 얻고 실제 장면에 더 가까운 블러링, 노이즈, 압축 결함과 같은 영향을 겪은 이미지)에서 이전 접근 방식을 크게 능가하는 동시에 이상적인 조건(즉, bicubic degradation에 의해 얻은 이미지)에서 경쟁한 결과를 제공합니다.  
그러나 테스트 중에 다른 이미지에 대해 서로 다른 네트워크를 훈련해야 하기 때문에 추론 시간이 다른 네트워크보다 훨씬 깁니다.

- ZSSR

## Weakly-supervised Super-resolution
사전 정의된 degradation를 도입하지 않고 초해상도에 대처하기 위해 연구자들은 weakly-supervised learning, 즉 페어링되지 않은 LR-HR 이미지를 사용하여 SR 모델을 학습하려고 시도합니다.  
그 중 일부 연구자는 먼저 HR-to-LR degradation를 학습하고 이를 사용하여 SR 모델을 학습하기 위한 데이터 세트를 구성하는 데 사용하는 반면, 다른 연구자는 cycle-in-cycle network를 설계하여 LR-to-HR and HR-to-LR mapping을 동시에 학습합니다.  
다음에는 이러한 모델에 대해 자세히 설명합니다.

### Learned Degradation
사전 정의된 degradation는 차선책이기 때문에 페어링되지 않은 LRHR 데이터 세트에서 degradation를 학습하는 것이 실현 가능한 방향입니다.  
Bulat et al. [159]는 먼저 HR-to-LR GAN을 훈련하여 페어링되지 않은 LR-HR 이미지를 사용하여 degradation를 학습한 다음 첫 번째 GAN을 기반으로 수행된 페어링된 LR-HR 이미지를 사용하여 SR용 LR-to-HR GAN을 훈련하는 2단계 프로세스를 제안합니다.  
특히, HR-to-LR GAN의 경우, HR 이미지가 생성기에 공급되어 LR 출력을 생성하는데, 이는 HR 이미지를 downscaling하여 얻은 LR 이미지(average pooling으로)뿐만 아니라 실제 LR 이미지의 분포와도 일치시키는 데 필요합니다.  
훈련을 마친 후 생성기는 LR-HR 이미지 쌍을 생성하는 degradation model로 사용됩니다.  
그런 다음 LR-to-HR GAN의 경우 생성기(즉, SR 모델)가 생성된 LR 이미지를 입력으로 받아 HR 출력을 예측하며, 이는 해당 HR 이미지뿐만 아니라 HR 이미지의 분포와도 일치시키는 데 필요합니다.  
이 2단계 프로세스를 적용함으로써 제안된 unsupervised model은 초해상도 실제 LR 이미지의 품질을 효과적으로 높이고 이전의 SOTA 작업에 비해 크게 향상되었습니다.  

### Cycle-in-cycle Super-resolution
unsupervised 초해상도에 대한 또 다른 접근 방식은 LR 공간과 HR 공간을 두 개의 도메인으로 처리하고 cycle-in-cycle 구조를 사용하여 서로 간의 매핑을 학습하는 것입니다.  
이 경우, 훈련 목표에는 매핑된 결과를 대상 도메인 분포와 일치하도록 푸시하고 round-trip mapping을 통해 이미지를 복구할 수 있도록 하는 것이 포함됩니다.  
CycleGAN [138]에서 영감을 얻은 Yuan et al. [131]은 4개의 생성기와 2개의 판별기로 구성된 cycle-in-cycle SR 네트워크(CinCGAN)를 제안하여 각각 노이즈가 있는 LR <--> 깨끗한 LR 및 깨끗한 LR <--> 깨끗한 HR 매핑을 위한 두 개의 CycleGAN을 구성합니다.  
구체적으로, 첫 번째 CycleGAN에서 노이즈가 있는 LR 이미지는 생성기에 입력되고 출력은 실제 깨끗한 LR 이미지의 분포와 일치해야 합니다.  
그런 다음 다른 생성기에 입력되어 원래 입력을 복구해야 합니다.  
여러 손실 함수(예: adversarial loss, cycle consistency loss, identity loss)가 cycle consistency, distribution consistency, and mapping validity을 보장하기 위해 사용됩니다.  
다른 CycleGAN은 매핑 도메인이 다르다는 점을 제외하면 유사하게 설계되었습니다.  
미리 정의된 degradation를 피하기 때문에 unsupervised CinCGAN은 supervised 방법과 비슷한 성능을 달성할 뿐만 아니라 매우 가혹한 조건에서도 다양한 사례에 적용할 수 있습니다.  
그러나 SR 문제의 ill-posed essence과 CinCGAN의 복잡한 구조로 인해 훈련 ​​난이도와 불안정성을 줄이기 위한 몇 가지 고급 전략이 필요합니다.  

- CinCGAN

## Deep Image Prior
CNN 구조가 inverse problem에 대한 많은 낮은 수준의 이미지 통계 정보를 사전에 포착하기에 충분하다는 점을 고려하여 Ulyanov et al. [160]은 SR을 수행하기 위해 수작업으로 사전 제작된 무작위로 초기화된 CNN을 사용합니다.  
구체적으로, 그들은 랜덤 벡터 z를 입력으로 사용하고 target HR 이미지 Iy를 생성하려고 하는 생성기 네트워크를 정의합니다.  
목표는 다운샘플링된 ˆIy가 LR 이미지 Ix와 동일한 ˆIy를 찾도록 네트워크를 훈련하는 것입니다.  
네트워크는 무작위로 초기화되고 훈련되지 않으므로 유일한 prior는 CNN 구조 자체입니다.  
이 방법의 성능은 여전히 ​​supervised method(2dB)보다 나쁘지만 기존의 bicubic upsampling(1dB)보다 상당히 우수합니다.  
게다가 CNN 구조 자체의 합리성을 보여주고 CNN 구조나 self-similarity와 같은 수작업으로 사전 제작된 prior과 딥 러닝 방법론을 결합하여 SR을 개선하도록 촉구합니다.

- DIP

# DOMAIN-SPECIFIC APPLICATIONS

## Depth Map Super-resolution

## Face Image Super-resolution

## Hyperspectral Image Super-resolution

## Real-world Image Super-resolution

## Video Super-resolution

## Other Applications

# CONCLUSION AND FUTURE DIRECTIONS
이 논문에서는 딥 러닝을 통한 이미지 초해상도의 최근 발전에 대한 광범위한 survey를 제공했습니다.  
주로 supervised and unsupervised SR의 개선에 대해 논의했으며 몇 가지 도메인별 응용 프로그램도 소개했습니다.  
큰 성공에도 불구하고 여전히 해결되지 않은 문제가 많이 있습니다.  
따라서 이 섹션에서는 이러한 문제를 명시적으로 지적하고 향후 진화에 대한 몇 가지 유망한 동향을 소개합니다.  
이번 survey를 통해 연구자에게 이미지 SR을 더 잘 이해할 수 있을 뿐만 아니라 이 분야의 향후 연구 활동과 응용 프로그램 개발을 용이하게 할 수 있기를 바랍니다.

