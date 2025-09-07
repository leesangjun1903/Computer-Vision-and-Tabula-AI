
# 다양한 Convolution 기법 상세 분석

딥러닝 네트워크 설계에서 Convolution은 **표현력**과 **효율성**을 동시에 고민해야 하는 핵심 연산입니다. 아래에서는 각 기법의 수학적 정의와 특징을 심도 있게 살펴보겠습니다.

***

## 1. Standard Convolution  
**정의**  
입력 텐서 $$X\in\mathbb{R}^{H\times W\times C_{\text{in}}}$$에 대해, 커널 텐서  

$$\;W\in\mathbb{R}^{K\times K\times C_{\text{in}}\times C_{\text{out}}}$$를 적용하면 출력 $$Y\in\mathbb{R}^{H'\times W'\times C_{\text{out}}}$$를 얻습니다.  

수식으로는  

$$
Y_{h,w,o}
=\sum_{c=1}^{C_{\text{in}}}\sum_{i=1}^{K}\sum_{j=1}^{K}
W_{i,j,c,o}\,\cdot\,X_{h+i',\,w+j',\,c}
+b_{o}
$$  

($$i',j'$$는 패딩 및 스트라이드 적용 인덱스)  
**파라미터 수**: $$C_{\text{in}}\times C_{\text{out}}\times K^2$$  
**연산량**: $$H'W'\times C_{\text{in}}C_{\text{out}}K^2$$  
**특징**  
- 모든 채널을 동시에 처리하므로 공간·채널 특징을 한 번에 학습합니다.  
- 연산량과 파라미터가 많아, 대형 모델에서 병목이 발생할 수 있습니다.

***

## 2. Dilated Convolution  
**정의**  
커널 내부에 **구멍(dilation rate $$d$$)**을 두어 수용 영역을 확장합니다.  

$$
Y_{h,w,o}
=\sum_{c=1}^{C_{\text{in}}}\sum_{i=1}^{K}\sum_{j=1}^{K}
W_{i,j,c,o}\,
\cdot\,X_{h+d(i-\lceil\frac{K}{2}\rceil),\,w+d(j-\lceil\frac{K}{2}\rceil),\,c}
$$  

**수용 영역**: $$((K-1)d+1)\times((K-1)d+1)$$  
**파라미터 수**: 기존 Convolution과 동일  
**연산량**: 패딩·스트라이드 조건 동일 시 표준 Convolution과 동일  
**특징**  
- **넓은 문맥(Context)**을 추출하면서 연산량 증가는 없습니다.  
- Semantic Segmentation, 객체 검출에서 경계 정보 손실 없이 피처를 보존할 때 유리합니다.

# Dilated Convolution 완전 가이드

딥러닝에서 **Dilated Convolution**(또는 Atrous Convolution)은 **수용 영역(Receptive Field)**을 넓히면서도 파라미터 수를 늘리지 않는 기법입니다. 주로 **Semantic Segmentation**과 같은 픽셀 단위 예측에 활용됩니다.

## 1. Dilated Convolution 개념

Dilated Convolution은 **커널 내부에 간격(dilation rate)**을 추가합니다.  
- **dilation = 1**: 일반적인 3×3 컨볼루션과 동일  
- **dilation = 2**: 3×3 필터가 7×7 영역을 커버  
- **dilation = 4**: 3×3 필터가 15×15 영역을 커버  

즉, 패딩된 입력에서 **간격만 뛰워서** 연산함으로써 넓은 영역의 문맥 정보를 한 번에 바라볼 수 있습니다.  

## 2. 왜 Dilated Convolution을 사용할까?

1. **넓은 Receptive Field**  
   - 큰 커널(예: 7×7)을 쓰지 않고도 동일한 영역을 바라봅니다.  
   - 파라미터는 3×3 필터 기준(9개)만 사용합니다.

2. **세그멘테이션 성능 향상**  
   - Pooling 없이 원본 해상도를 유지하며 문맥을 캡처합니다.  
   - 풀링 경로는 디테일이 손실되기 쉬워 세그멘테이션에 부적합합니다.

3. **다양한 스케일 학습**  
   - 서로 다른 dilation을 조합해 멀티스케일 정보를 학습할 수 있습니다.  

## 3. 수학적 정의

입력 $$X\in\mathbb{R}^{H\times W\times C}$$, 커널 $$W\in\mathbb{R}^{K\times K\times C\times N}$$일 때, 출력 $$Y\in\mathbb{R}^{H'\times W'\times N}$$ 성분은 다음과 같습니다:

$$
Y_{h,w,o}
=\sum_{c=1}^{C}\sum_{i=1}^{K}\sum_{j=1}^{K}
W_{i,j,c,o}\,\cdot\,X_{\,h + d\,(i-\lceil\frac{K}{2}\rceil),\;w + d\,(j-\lceil\frac{K}{2}\rceil),\,c}
$$

여기서 $$d$$는 **dilation rate**입니다.

## 4. 적용 시나리오

- **Semantic Segmentation**: FCN, DeepLab, DRN 등  
- **Object Detection**: 작은 물체의 문맥 포착  

## 5. Dilated Residual Network (DRN)

**ResNet**에 Dilated Convolution을 접목한 구조입니다.  
- 기존 ResNet의 **일부 스트라이드** 대신 dilation을 사용합니다.  
- 해상도를 유지하면서 깊은 레이어의 Receptive Field를 크게 합니다.

### PyTorch 구현 예시

```python
import torch.nn as nn

def conv3x3(in_ch, out_ch, stride=1, dilation=1):
    return nn.Conv2d(in_ch, out_ch,
                     kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch,
                 stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch,
                             stride, dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch,
                             stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)
```

- `padding=dilation`으로 출력 크기를 유지합니다.  
- `dilation>1`인 블록을 쌓아 다양한 Receptive Field를 확보합니다.

## 6. 결론

Dilated Convolution은 **파라미터 효율성**과 **넓은 문맥 캡처**를 동시에 달성하는 핵심 기술입니다.  
특히 **픽셀 단위 예측**이 중요한 세그멘테이션 분야에서 큰 성능 향상을 기대할 수 있습니다.  
ResNet과 결합한 DRN 구조로 더욱 깊고 효율적인 네트워크를 구현해 보세요.

[1](https://gaussian37.github.io/dl-concept-dilated_residual_network/)

***

## 3. Transposed Convolution  
**정의**  
입력 $$X\in\mathbb{R}^{H\times W\times C_{\text{in}}}$$에서 **업샘플링**된 출력 $$Y\in\mathbb{R}^{sH\times sW\times C_{\text{out}}}$$를 만듭니다.  
수식 관점에서는 Convolution 연산의 선형 연산 행렬을 전치한 형태로 볼 수 있습니다.  
**파라미터 수**: $$C_{\text{in}}\times C_{\text{out}}\times K^2$$  
**연산량**: $$sH\cdot sW\times C_{\text{in}}C_{\text{out}}K^2$$  
**특징**  
- 업샘플링 과정에서 **학습 가능한 파라미터**로 해상도를 복원합니다.  
- Checkerboard 아티팩트(격자무늬)가 발생할 수 있어, 주의해야 합니다.

***

## 4. Separable Convolution  
**정의**  
공간 방향(2D)을 두 개의 1D Convolution으로 분리합니다.  

$$
k(x,y)=k_x(x)\,k_y(y)
$$  

이론적으로 $$K^2$$ 곱셈을 $$2K$$ 곱셈으로 줄일 수 있습니다.  
**파라미터 수**: $$C_{\text{in}}\times C_{\text{out}}\times2K$$  
**연산량**: $$H'W'\times C_{\text{in}}C_{\text{out}}\times2K$$  
**특징**  
- 모든 커널이 분해 가능할 때만 적용할 수 있습니다.  
- Sobel 필터 등 전통적 이미지 프로세싱에서 유래했습니다.

***

## 5. Depthwise Convolution  
**정의**  
각 입력 채널마다 **하나의 2D 필터**를 적용합니다.  

$$
Y_{h,w,c}
=\sum_{i=1}^{K}\sum_{j=1}^{K}
W_{i,j,c}\,\cdot\,X_{h+i',\,w+j',\,c}
$$  

**파라미터 수**: $$C_{\text{in}}\times K^2$$  
**연산량**: $$H'W'\times C_{\text{in}}\times K^2$$  
**특징**  
- 채널 간 상호작용이 없습니다.  
- Grouped Convolution의 특수한 경우($$g=C_{\text{in}}$$)입니다.

***

## 6. Pointwise Convolution  
**정의**  
$$1\times1$$ 커널로 **채널 결합**만 수행합니다.  

$$
Y_{h,w,o}
=\sum_{c=1}^{C_{\text{in}}}
W_{1,1,c,o}\,\cdot\,X_{h,w,c}
$$  

**파라미터 수**: $$C_{\text{in}}\times C_{\text{out}}$$  
**연산량**: $$H'W'\times C_{\text{in}}C_{\text{out}}$$  
**특징**  
- 채널 수 변환, 차원 축소·확장에 사용됩니다.  
- 선형 결합 방식으로 채널 정보를 압축합니다.

***

## 7. Depthwise Separable Convolution  
**정의**  
Depthwise → Pointwise 순으로 연산을 분리합니다.  
**연산량**:  

$$
H'W'\bigl(C_{\text{in}}K^2 + C_{\text{in}}C_{\text{out}}\bigr)
$$  

표준 Convolution 대비 **약 $$\tfrac{1}{C_{\text{out}}}+\tfrac{1}{K^2}$$** 비율만 필요합니다.  
**파라미터 수**: $$C_{\text{in}}K^2 + C_{\text{in}}C_{\text{out}}$$  
**특징**  
- MobileNet, Xception, EfficientNet 등에서 널리 사용됩니다.  
- 공간/채널 학습을 분리하여 효율성을 극대화합니다.

***

# Depthwise Separable Convolution 완전 가이드

딥러닝이 모바일과 엣지 디바이스로 확장되면서 **효율적인 네트워크 설계**는 필수가 되었습니다. 그 중심에 있는 기술이 바로 **Depthwise Separable Convolution**입니다.[^1][^2][^3]

## 등장 배경: 왜 필요했을까요?

기존의 Standard Convolution은 강력하지만 **연산량이 많다**는 치명적인 단점이 있었습니다. CNN이 발전하면서 Bottleneck Layer나 Global Average Pooling 같은 기법들이 나왔지만, 모바일 기기에서 실시간으로 돌릴 수 있는 수준까지는 부족했습니다.[^2]

특히 모바일 환경이 늘어나면서 '**어떻게 하면 네트워크 크기와 연산량을 줄일 수 있을까?**'라는 고민에서 Depthwise Separable Convolution이 탄생했습니다.[^2]

<img width="809" height="577" alt="image" src="https://github.com/user-attachments/assets/09ae9b64-1dbe-46fe-af37-b5cb4c75175d" />

Depthwise Separable Convolution 구조도

## Standard Convolution의 작동 원리

먼저 기존 방식을 이해해봅시다. Standard Convolution에서는:[^1]

- **입력**: $(D_F, D_F, M)$ 크기의 피처맵 (높이, 너비, 채널)
- **필터**: $(D_K, D_K, M)$ 크기의 커널이 $N$개
- **출력**: $(D_G, D_G, N)$ 크기의 피처맵

하나의 필터가 **모든 채널을 동시에 처리**하는 것이 핵심입니다. 이때 연산량은 $N \times D_G^2 \times D_K^2 \times M$이 됩니다.[^1]

## Depthwise Separable Convolution: 2단계 분할 전략

Depthwise Separable Convolution은 기존 연산을 **두 단계로 분할**합니다:[^1][^2]

### 1단계: Depthwise Convolution (Filtering Stage)

**각 채널을 독립적으로 처리**하는 단계입니다:[^1][^3]

- 한 개의 필터가 **한 개의 채널에만** 연산을 수행
- 입력 채널 수만큼 필터가 존재 ($M$개)
- 출력: $(D_G, D_G, M)$ - 채널 수가 유지됨

Standard Convolution과 달리 **채널 간 혼합이 일어나지 않습니다**. 각 채널의 **공간적(Spatial) 정보만을 추출**하는 역할을 합니다.[^2][^3]

### 2단계: Pointwise Convolution (Combination Stage)

**1×1 Convolution을 통한 채널 조합** 단계입니다:[^1]

- $(1, 1, M)$ 크기의 필터를 $N$개 사용
- 이전 단계 출력 $(D_G, D_G, M)$을 입력으로 받음
- 출력: $(D_G, D_G, N)$ - 원하는 출력 채널 수 달성

이 과정에서 **채널 간 정보가 혼합**되어 최종적으로 Standard Convolution과 **동일한 입출력 차원**을 얻습니다.[^1]

## 연산량 분석: 얼마나 효율적일까요?


Standard Convolution과 Depthwise Separable Convolution의 연산량 비교

### Depthwise Convolution 연산량

- 한 위치당: $D_K^2$ 곱셈
- 전체: $M \times D_G^2 \times D_K^2$[^1]


### Pointwise Convolution 연산량

- 한 위치당: $M$ 곱셈
- 전체: $N \times D_G^2 \times M$[^1]


### 총 연산량

**Depthwise Separable**: $M \times D_G^2 \times (D_K^2 + N)$[^1]

**효율성 비율**: $\frac{1}{N} + \frac{1}{D_K^2}$[^1]

### 실제 예시

$N = 1024, K = 3$일 때:

$\frac{1}{1024} + \frac{1}{9} = 0.112$ ≈ **약 9배 효율적**[^1]

필터 크기가 3×3일 때 대부분의 효율성 향상은 **$\frac{1}{D_K^2} = \frac{1}{9}$ 항목**에서 나옵니다.[^1]

## 실제 활용 사례

이 기법은 현재 다음 모델들에서 **활발히 사용**되고 있습니다:[^1]

- **MobileNet V1, V2**: 모바일 최적화의 대표 모델
- **Xception**: 구글의 효율적인 CNN 아키텍처
- **EfficientNet**: 복합 스케일링과 함께 사용[^2]


## 성능 vs 효율성: 트레이드오프는 어떨까요?

놀랍게도 **연산량 감소 대비 성능 저하가 크지 않습니다**. 일반적으로 연산량이 줄면 성능도 따라 떨어지는 것이 상식이지만, Depthwise Separable Convolution은 이 공식을 깨뜨렸습니다.[^2]

그 이유는 **두 단계 설계**에 있습니다:

1. **Spatial Feature 추출**과 **Channel-wise Feature 조합**을 분리
2. 각 단계 사이에 **활성화 함수를 거쳐 표현력 향상**[^2]

## 구현 시 주의사항

Depthwise Conv Layer와 Pointwise Conv Layer는 **엄밀하게는 따로 연산**됩니다. 두 레이어를 단순히 붙이는 것보다는 **중간에 활성화 함수**를 넣는 것이 표현력 면에서 더 효과적입니다.[^2]

## 결론: 왜 중요한 기술인가?

Depthwise Separable Convolution은 **모바일 딥러닝의 게임 체인저**입니다. 약 9배의 연산량 감소를 이루면서도 성능 저하를 최소화했기 때문입니다.[^1][^2]

특히 딥러닝이 클라우드에서 엣지로 이동하는 현 시점에서, 이 기술의 중요성은 더욱 커지고 있습니다. **효율성과 성능의 균형**을 찾는 모든 연구자와 개발자에게 필수적인 지식이라고 할 수 있습니다.

앞으로 더 많은 모델에서 이 기법을 보게 될 것이며, 변형된 형태로도 계속 발전할 것으로 예상됩니다. 따라서 딥러닝을 공부하는 여러분에게는 **반드시 정확히 이해해야 할 핵심 개념**입니다.

<div style="text-align: center">⁂</div>

[^1]: https://gaussian37.github.io/dl-concept-dwsconv/

[^2]: https://velog.io/@iissaacc/Depthwise-Separable-Convolution

[^3]: https://ankle96.tistory.com/59

## 8. Grouped Convolution  
**정의**  
입력 채널을 $$g$$그룹으로 나누고, 각 그룹에 독립적 필터를 적용합니다.  
**파라미터 수**: $$\tfrac{C_{\text{in}}C_{\text{out}}K^2}{g}$$  
**연산량**: $$\tfrac{H'W'C_{\text{in}}C_{\text{out}}K^2}{g}$$  
**특징**  
- ResNeXt, ShuffleNet에서 성능·효율 균형을 위해 활용됩니다.  
- 그룹 수 $$g$$는 하이퍼파라미터입니다.

***

## 9. Deformable Convolution  
**정의**  
기존 샘플링 격자에 **학습 가능한 오프셋** $$\Delta p$$를 더해 유연한 receptive field를 만듭니다.  

$$
Y_{h,w,o}
=\sum_{c=1}^{C_{\text{in}}}\sum_{i,j}
W_{i,j,c,o}\,
\cdot\,X(h+i'+\Delta p^h_{i,j},\,w+j'+\Delta p^w_{i,j},\,c)
$$  

**파라미터 수**: 표준 Convolution + 오프셋 학습 파라미터  
**특징**  
- Bilinear Interpolation으로 비정수 좌표를 처리합니다.  
- 객체 검출·세그멘테이션에서 복잡한 형태 대응에 탁월합니다.

***

## 10. Lightweight Convolution  
**정의**  
채널을 $$H$$그룹으로 묶고, 각 그룹마다 **하나의 1D 커널**을 공유합니다.  
가중치 행렬 $$\mathbf{W}\in\mathbb{R}^{H\times K}$$를 **Softmax**로 정규화하여 사용합니다.  

$$
\tilde{W}_{h,i}=\frac{\exp(W_{h,i})}{\sum_{j}\exp(W_{h,j})}
$$  

그룹당 채널 수 $$d/H$$이고, 입력 $$(B,d,T)$$ 시퀀스에 대해  

$$
Y^{(g)}_{b,c,t}
=\sum_{i=1}^{K}\tilde{W}_{g,i}\,X_{b,\,g\cdot\frac{d}{H}+c,\,t+i'}
$$  

**파라미터 수**: $$H\times K$$  
**연산량**: $$B\times d\times T\times K$$  
**특징**  
- 연산량·파라미터를 크게 줄이면서도 **컨텍스트** 정보를 확보합니다.  
- NLP의 시퀀스 모델, 실시간 세그멘테이션 등에서 사용됩니다.

## 11. Dynamic Convolution
# Dynamic Convolution 완전 가이드

Dynamic Convolution은 **Lightweight Convolution**을 확장한 기법입니다. 입력 시퀀스의 각 타임스텝마다 **커널 가중치**를 **동적으로 생성**해, 시퀀스 특성에 맞춘 유연한 필터링을 수행합니다. NLP와 시계열 모델에 특히 유용합니다.

## 1. 개념 정리

1. **Lightweight Convolution 복습**  
   - 채널을 $$H$$그룹으로 나누고, 각 그룹마다 하나의 1D 커널을 공유합니다.  
   - 커널 가중치는 Softmax 정규화로 안정적 학습을 돕습니다.

2. **Dynamic Convolution 확장**  
   - 각 타임스텝의 **중심 단어**나 피처를 **선형 변환**해 커널 가중치를 생성합니다.  
   - 생성된 커널을 그룹별로 적용해, 시퀀스의 문맥 정보를 동적으로 반영합니다.

## 2. 수학적 정의

입력 $$\mathbf{X}\in\mathbb{R}^{B\times C\times T}$$에서,  
타임스텝 $$t$$의 중심 피처 $$\mathbf{x}_t$$를 선형 투영해 그룹별 커널 $$\mathbf{w}_t\in\mathbb{R}^{H\times K}$$을 만듭니다:

$$
\mathbf{w}_t = \text{Softmax}\bigl(\mathbf{W}_\text{proj}\,\mathbf{x}_t + \mathbf{b}\bigr)
$$

여기서 $$\mathbf{W}_\text{proj}\in\mathbb{R}^{(H\times K)\times C}$$는 학습 가능한 투영 행렬입니다.  
각 그룹 $$g$$의 출력은:

$$
Y_{b,\,g,\,t}
=\sum_{i=1}^{K}
w_{t,\,g,\,i} \times
X_{b,\,g\cdot\frac{C}{H}+c,\,t+i-\lceil\frac{K}{2}\rceil}
$$

로 계산합니다.

## 3. 특징과 장점

- **동적 문맥 반영**: 시퀀스 위치마다 커널이 달라져, 문맥 변화에 민감하게 대응합니다.  
- **파라미터 효율성**: Lightweight Convolution 대비 소폭 증가한 파라미터로 더 큰 표현력을 얻습니다.  
- **실시간 처리**: 1D 연산만 사용해 연산 부담이 적습니다.

## 4. 활용 분야

- **자연어 처리**: RNN·Transformer 대체 또는 보조 모듈로 사용해 어텐션 비용 절감  
- **음성·오디오 처리**: 시계열 패턴에 따라 필터를 동적으로 조정  
- **실시간 시그널 처리**: 입력 특성에 민감한 필터링이 필요한 경우

## 5. PyTorch 예시 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv1D(nn.Module):
    def __init__(self, channels, kernel_size, groups):
        super().__init__()
        assert channels % groups == 0
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.proj = nn.Linear(channels, groups * kernel_size)
    
    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.size()
        # 중심 피처: 시간 차원을 평균으로 요약
        context = x.mean(dim=2)            # (B, C)
        # 동적 커널 생성
        w = self.proj(context)             # (B, G*K)
        w = w.view(B, self.groups, self.kernel_size)
        w = F.softmax(w, dim=-1)           # (B, G, K)
        
        # 패딩
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad), mode='reflect')
        
        # 그룹별 컨볼루션
        x_grp = x_pad.view(B, self.groups, -1, T + 2*pad)
        # (B, G, C/G, T+2p)
        
        out = []
        for i in range(self.kernel_size):
            out.append(x_grp[:, :, :, i:i+T] * w[:, :, i:i+1])
        out = sum(out)                      # (B, G, C/G, T)
        out = out.view(B, C, T)
        return out

# 사용 예시
x = torch.randn(8, 64, 100)  # (batch, channels, seq_len)
dyn_conv = DynamicConv1D(channels=64, kernel_size=5, groups=8)
y = dyn_conv(x)                # (8, 64, 100)
```

- `proj` 레이어가 **중심 피처 → 커널 가중치**를 만듭니다.  
- 그룹별로 잘라서 연산한 뒤, 다시 합쳐 최종 출력을 얻습니다.

***

Dynamic Convolution은 **가변적 문맥 처리**가 필요한 모든 시퀀스 모델에 강력한 옵션이 됩니다. 어텐션보다 가볍고, 상황에 맞춰 필터를 생성하는 능력을 활용해 보세요.

[1](https://ankle96.tistory.com/59)

***

이상으로 **열 가지** 대표적인 Convolution 기법을 수학적 정의와 함께 상세히 살펴보았습니다. 각 기법이 **어떤 상황**에서 **어떤 트레이드오프**를 가지는지 이해하고, 적절히 조합하여 **효율적이고 강력한** 네트워크를 설계하세요.

# 다양한 Convolution 기법 완전 정복 가이드

딥러닝 모델을 설계할 때, **Convolution**은 핵심 중의 핵심 레이어입니다. 하지만 표준 Convolution만으로는 성능과 효율성 면에서 한계가 있기에, **다양한 변형 기법**들이 등장했습니다. 이 글에서는 대표적인 Convolution 기법들을 살펴보고, PyTorch 예시 코드를 통해 실제 모델에 적용하는 방법까지 익혀보겠습니다.

## 1. Standard Convolution  
기본 중의 기본입니다. 입력 채널 수 $$C_{\text{in}}$$, 출력 채널 수 $$C_{\text{out}}$$, 커널 크기 $$K$$를 가질 때 파라미터는 $$C_{\text{in}} \times C_{\text{out}} \times K^2$$입니다.  
PyTorch 예시:
```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
```
- 특징: 모든 채널 정보를 한 번에 섞어서 처리합니다.  
- 단점: 연산량·파라미터가 많습니다.

## 2. Dilated Convolution (Atrous Convolution)  
커널 내부에 **간격(dilation rate)**을 두어 넓은 수용 영역(Receptive Field)을 확보합니다.  
```python
dilated = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
```
- 장점: 큰 컨텍스트 정보를 취득하면서도 연산량 증가는 최소화합니다.  
- 활용 분야: 실시간 세그멘테이션, 객체 검출.

## 3. Transposed Convolution (Deconvolution)  
업샘플링을 위한 기법입니다.  
```python
deconv = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
```
- 특징: 다운샘플된 피처맵을 원래 크기로 복원할 때 사용합니다.  
- 주의: ‘Deconvolution’이 실제 수학적 역연산은 아닙니다.

## 4. Separable Convolution  
커널을 두 개의 작은 커널로 분리해 연산량을 줄입니다.  
- 수직 + 수평 1D 커널 분리  
- 파라미터 절약 효과가 크지만, 모든 커널이 분리 가능한 것은 아닙니다.

## 5. Depthwise Convolution  
각 채널을 **독립적으로** 처리합니다. 입력 채널 수 $$M$$와 동일한 수의 필터를 사용해, 채널 간 정보 교환 없이 공간 정보만 추출합니다.  
```python
depthwise = nn.Conv2d(128, 128, kernel_size=3, groups=128, padding=1)
```
- 연산량: $$M \times H \times W \times K^2$$

## 6. Pointwise Convolution  
1×1 커널로 **채널 조합**만 수행합니다.  
```python
pointwise = nn.Conv2d(128, 256, kernel_size=1)
```
- 연산량: $$H \times W \times M \times N$$  
- 주로 채널 수를 줄이거나 늘릴 때 사용합니다.

## 7. Depthwise Separable Convolution  
Depthwise + Pointwise를 결합한 기법입니다.  
```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3,
                                   groups=in_ch, padding=1)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

dsc = DepthwiseSeparableConv(64, 128)
```
- 연산량: $$H\!W\bigl(MK^2 + MN\bigr)$$  
- 표준 대비 **약 8–9배 효율적**입니다.

## 8. Grouped Convolution  
채널을 여러 그룹으로 나누어 독립적으로 Convolution을 수행합니다.  
```python
grouped = nn.Conv2d(64, 64, kernel_size=3, groups=4, padding=1)
```
- 그룹 수 $$g$$만큼 파라미터·연산량이 $$1/g$$로 감소합니다.  
- ResNeXt, ShuffleNet 등에 활용됩니다.

## 9. Deformable Convolution  
컨볼루션 샘플링 위치에 **학습 가능한 오프셋**을 추가해 유연한 수용 영역을 만듭니다.  
```python
# torchvision.ops.deform_conv import DeformConv2d 필요
from torchvision.ops import DeformConv2d

deform = DeformConv2d(64, 128, kernel_size=3, padding=1)
```
- 장점: 복잡한 기하학적 변형에 적응 가능합니다.  
- 주로 객체 검출·세그멘테이션 모델에서 사용합니다.

***

위 기법들을 잘 조합하면, **모델 성능과 효율성**을 모두 잡을 수 있습니다. 특히 모바일/엣지 디바이스용 네트워크를 설계할 때 Depthwise Separable Convolution과 Grouped Convolution은 필수로 익혀두세요. Convolution의 다양한 변종을 이해하고 활용하면, **딥러닝 모델의 확장성과 실용성**을 한층 높일 수 있습니다.

[1](https://ankle96.tistory.com/59)
[2](https://eehoeskrap.tistory.com/431)

## 10. Lightweight Convolution  

Lightweight Convolution은 Depthwise Convolution과 비슷하지만, **커널을 채널 그룹별로 공유**하고 **가중치에 Softmax 정규화**를 적용합니다.  

- **그룹 공유**: 연속된 $$d/H$$개 채널마다 하나의 커널을 공유합니다.  
- **Softmax 정규화**: 타임스텝 방향으로 가중치를 Softmax로 정규화해 안정적인 학습을 돕습니다.  

예를 들어, $$d=6, H=3$$이면 채널 (1,2), (3,4), (5,6)마다 동일한 커널을 사용합니다. 이로써 커널 수를 $$d\,k$$에서 $$d\,H$$로 크게 줄일 수 있습니다.  

Lightweight Convolution은 **컨텍스트 정보**가 중요하면서도 **연산량을 줄여야 할 때** 유용합니다. 특히 자연어 처리의 시퀀스 모델이나 실시간 세그멘테이션에서 활용됩니다.  

```python
# PyTorch 예시 (의사 코드)
class LightweightConv(nn.Module):
    def __init__(self, channels, kernel_size, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(groups, kernel_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, channels, width)
        w = self.softmax(self.weight)  # (groups, kernel_size)
        # 그룹별 컨볼루션 연산 (Pseudo-implementation)
        return grouped_conv1d(x, w, groups=self.groups)
```

Lightweight Convolution은 **연산량을 줄이면서**도 **충분한 정보**를 추출할 수 있도록 설계되었습니다. 디바이스 환경에 제약이 있을 때 고려해 보세요.
