# Image Gradients와 Edge Detection 가이드

결론부터 말하면, 에지는 픽셀 밝기의 큰 변화로 정의되며, 이를 계산하는 가장 기본 도구는 이미지의 **gradient**이며, 노이즈에 강인하게 검출하려면 가우시안 스무딩과 **DoG/LoG** 또는 **Canny** 파이프라인을 활용하는 것이 효과적입니다.[1][2][3][4][5][6]

## 개요
이미지를 함수 $$f(x,y)$$로 보면, 에지는 밝기 변화가 큰 지점의 연속입니다. 이를 계산하기 위해 1차 도함수(gradient)와 2차 도함수(Laplacian)를 이산적으로 근사하고, 노이즈 억제를 위해 가우시안 필터를 결합합니다. 에지 방향은 항상 gradient 방향과 수직이며, 실제 구현에서는 Sobel/Scharr, DoG/LoG, Canny 같은 고전적 연산자를 많이 사용합니다.[7][8][2][3][1]

## Gradient 기본기
- Gradient는 $$\nabla f = [\partial f/\partial x, \partial f/\partial y] $$로 정의되며, 이산 영상에서는 finite difference로 근사합니다.[7][1]
- Forward, Backward, Central difference로 1차 도함수를 근사하고, convolution 형태의 필터로 구현합니다.[1][7]
- Gradient 크기와 방향은 $$\|\nabla f\| = \sqrt{G_x^2+G_y^2} $$, $$\theta = \arctan2(G_y,G_x) $$로 계산합니다. 에지의 방향은 gradient와 수직입니다 [7][4].

## Sobel/Prewitt/Scharr
- Sobel은 3×3 마스크로 $$G_x, G_y$$를 계산하며, 인접 픽셀에 가중치를 주어 잡음에 다소 강합니다.[5][6][7]
- Scharr는 회전 불변성과 방향 응답을 개선한 파생 연산자로, Sobel 대비 방향 민감도가 좋습니다.[2][5]
- Horizontal Sobel은 수직 에지에, Vertical Sobel은 수평 에지에 민감하게 반응합니다.[7][1]

```
Sobel 필터는 이미지 처리에서 경계(에지)를 검출하기 위해 사용되는 3×3 크기의 미분 마스크입니다.
수평과 수직 방향의 경계 강도를 각각 계산하는 두 개의 커널로 구성되어 있으며, 영상 내 밝기 변화가 급격한 부분을 강조합니다.

주요 특징은 다음과 같습니다:

수직 경계 감지를 위해 3×3 마스크는 중앙 열의 값에 2와 -2를 포함해 주변 픽셀에 가중치를 부여합니다.
수평 및 수직 방향으로 미분을 수행해 이미지의 기울기((G_x), (G_y))를 구하고, 이들의 조합으로 엣지의 크기와 방향도 계산할 수 있습니다.
경계 검출에 있어 노이즈에 다소 민감하지만 빠르고 간단해 기본적인 엣지 검출에 적합합니다.
```

```
Scharr 필터 과정은 이미지의 엣지를 잘 검출하기 위해 1차 미분을 강화한 마스크를 사용하는 방법입니다.
Sobel 필터의 단점인 중심에서 멀어진 커널의 정확도 저하 문제를 개선하여 방향과 크기 모두를 더 정확히 계산합니다.

구체 과정은 다음과 같습니다:

이미지에 Scharr 마스크를 컨볼루션하여 x축(dx=1, dy=0)과 y축(dx=0, dy=1)의 그래디언트를 구함.

(커널 마스크:

X축: ([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
Y축: ([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
)

x, y 방향 미분 결과를 조합해 엣지의 크기(gradient magnitude)를 계산함. 보통 ( \sqrt{G_x^2 + G_y^2} ) 형태로 산출.
경계 처리(mode), 스케일 조정(scale), 출력 타입(ddepth) 등을 설정할 수 있음.
OpenCV에서는 cv2.Scharr() 함수 또는 cv2.Sobel() 함수의 ksize 파라미터에 -1 또는 FILTER_SCHARR를 넣어 사용 가능.
즉, Scharr 필터는 Sobel 필터보다 미분 계산이 정밀하여 가장자리 검출 시 더 정확한 결과를 제공합니다.
```

```
Prewitt 필터는 영상에서 수평 및 수직 방향의 가장자리(에지)를 감지하는 미분 기반 필터입니다.
3x3 마스크(커널)를 각각 수평과 수직 엣지 검출용으로 사용하며, 픽셀 강도의 급격한 변화를 통해 경계선을 찾아냅니다.

핵심 특징은 다음과 같습니다:

미분 마스크로 작동하며, 이미지 신호의 1차 미분을 계산합니다.

수평 엣지용과 수직 엣지용 두 가지 마스크를 가집니다.

마스크의 합은 0이며, 양수와 음수 값을 포함하여 가장자리 차이를 강조합니다.

예를 들어, 수직 방향 마스크는 다음과 같습니다:

[ \begin{bmatrix}
-1 & 0 & 1 \
-1 & 0 & 1 \
-1 & 0 & 1
\end{bmatrix}
]

이 마스크는 왼쪽과 오른쪽 픽셀 강도의 차이를 감지하여 수직 에지를 검출합니다.

이미지에 마스크 컨벌루션을 수행하여 각 방향의 경사도를 계산하고, 두 방향의 결과를 결합하여 전체 에지를 감지합니다.

SciPy와 같은 라이브러리에서는 scipy.ndimage.prewitt 함수로 축을 지정해 필터를 적용할 수 있습니다.

Prewitt 필터는 Sobel 필터와 유사하나 더 단순하며, 컴퓨터 비전, 의료 영상 처리 등 다양한 분야에서 활용됩니다.
```

## 노이즈와 스무딩
- 미분은 고주파(노이즈)를 증폭합니다. 따라서 미분 전 가우시안 스무딩이 필수적입니다.[2][1][7]
- DoG(Derivative of Gaussian) 혹은 LoG(Laplacian of Gaussian)를 사용하면 “스무딩+미분”을 결합할 수 있어 계산과 잡음 억제 측면에서 유리합니다.[3][7]
- 스케일(가우시안 $$\sigma$$)이 크면 굵은 구조, 작으면 미세 구조를 강조합니다.[4][9]

### DoG(Derivative of Gaussian)
**Derivative of Gaussian (DoG)** 은 가우시안 함수의 미분으로, 주로 영상 처리와 컴퓨터 비전에서 경계와 특징을 검출하는 데 사용되는 필터입니다. 이 필터는 입력 신호에 가우시안 함수의 미분을 컨볼루션하여 신호의 변화율, 즉 에지(edge)를 부드럽게 추출합니다.

1차 미분 가우시안 필터는 신호의 변화가 급격한 부분을 강조하여 에지를 검출합니다. 수학적으로, 1차 미분 가우시안은 ( $y = \frac{d}{dx} \left(\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}\right)$ ) 형태이고, 실제 계산식은 ( $-\frac{x}{\sigma^2}$ ) 곱하기 가우시안 함수로 표현됩니다.

#### 용도와 특징: 

DoG 필터는 Sobel 필터 등과 같이 간단한 미분 근사 필터를 가우시안 블러와 조합한 형태로, 특정 방향과 스케일에서의 에지 검출에 사용됩니다. 또한, 여러 방향으로 돌려가며 적용하면 방향성 에지 검출이 가능하며, 고차 미분 가우시안까지 확장 가능합니다.

DoG는 차이의 의미에서 ‘Difference of Gaussians’와 헷갈리기 쉬우나, 엄밀히 말해 DoG는 가우시안 함수의 미분(또는 미분값을 근사하는 필터)이며, Difference of Gaussians는 서로 다른 스케일의 가우시안 블러 이미지 차이를 이용하는 필터입니다.

요약하면, Derivative of Gaussian(DoG)는 가우시안 함수의 미분 필터로, 신호의 변화, 즉 에지 등을 부드럽게 검출하기 위해 영상 신호와 컨볼루션하여 사용하는 중요한 필터입니다.

### Laplacian of Gaussian (LoG)
**Laplacian of Gaussian (LoG)** 는 가우시안 블러를 적용한 뒤에 라플라시안 연산자를 취하는 엣지 검출 기법입니다. 즉, 먼저 이미지에 노이즈 감소를 위해 가우시안 필터를 적용하고, 그 결과에 라플라시안(2차 미분) 연산을 수행하여 경계(엣지)를 검출합니다.

LoG 함수 수식은 다음과 같습니다:

```math
[LoG(x, y) = \nabla^2 G(x, y) = \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2}
]
```

여기서 ( G(x,y) )는 2차원 가우시안 함수로,

```math
[G(x,y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
]
```

이고, ($\sigma$)는 표준편차입니다.

이를 풀어 쓰면 LoG 함수의 수식은 다음과 같습니다:

```math
[LoG(x,y) = -\frac{1}{\pi \sigma^4} \left(1 - \frac{x^2 + y^2}{2\sigma^2}\right) \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
]
```

이 수식은 중심에서 주변으로 동심원이 형성되는 형태이며, ‘멕시코 모자’(Mexican hat) 모양으로 불립니다.

LoG는 노이즈에 강한 엣지 검출이 가능하며, 이미지에서 영 교차점(zero-crossing)을 통해 경계 검출을 수행합니다. 가우시안 블러 덕분에 일반 라플라시안보다 노이즈 영향이 적고 부드러운 엣지를 얻을 수 있습니다.

## DoG vs LoG
- DoG는 서로 다른 $$\sigma$$의 가우시안을 차분하거나, 가우시안의 도함수를 직접 사용하는 방식으로 구현합니다. 실제로 LoG를 근사하는 데 널리 쓰입니다.[9][3]
- LoG는 Zero-crossing을 이용해 에지를 검출하며, 잡음에 민감한 Laplacian에 가우시안을 결합해 안정화합니다.[3][7]
- LoG는 이중선(edge doubling) 현상이 나타날 수 있으나, 에지 위치를 제로 크로싱으로 정밀하게 찾는 장점이 있습니다. 상황에 따라 DoG와 LoG를 선택적으로 사용합니다.[3][7]

## Canny 파이프라인
Canny edge detection은 가우시안 필터로 노이즈를 줄인 후, 소벨 연산자로 이미지의 그라디언트를 계산해 에지를 찾고, 비최대 억제로 에지를 얇게 만든 뒤, 이중 임계값과 히스테리시스 과정을 통해 최종 에지를 결정하는 다단계 알고리즘입니다.

주요 단계는 다음과 같습니다:

- 가우시안 스무딩: 노이즈 제거를 위해 이미지를 부드럽게 함
- 그라디언트 계산: 소벨 필터로 엣지 강도와 방향 추출
- 비최대 억제(Non-maximum suppression): 에지를 얇게 하여 불필요한 픽셀 제거
- 이중 임계값(Double thresholding): 강한 에지, 약한 에지, 비에지 구분
- 히스테리시스 에지 추적(Track edge by hysteresis): 약한 에지 중 강한 에지와 연결된 부분만 최종 에지로 포함
이 방식은 정확한 에지 위치를 찾고 오류율이 낮으며, 노이즈에 견고한 장점이 있습니다. OpenCV에서도 쉽게 구현 가능하며, 이미지 처리 분야에서 널리 쓰입니다.

- Canny는 가우시안 스무딩 → Sobel로 gradient 계산 → 비최대 억제(NMS) → 이중 임곗값과 hysteresis로 에지 연결까지 수행합니다.[1][2]
- 설계 목표는 좋은 검출(잡음에 강함), 정확한 위치, 하나의 반응(얇은 에지)이며, 실무에서 가장 안정적인 고전적 에지 검출기입니다.[2][1]

임곗값 설정 시 $$T_{low}$$ – $$T_{high}$$ 비율을 적절히 잡아 약한 에지를 강한 에지와 연결할 때만 남깁니다.[1][2]

## 경계 처리
- Convolution 시 외곽 픽셀 처리가 필요합니다: zero padding(clip), wrap, copy edge, reflection, 또는 가장자리 근처에서 필터 크기 조정 등입니다.[7]
- 일반적으로 reflection이 구조 보존에 유리하며, 단순 zero padding은 테두리 아티팩트를 유발할 수 있습니다.[7]

## 파이토치/오픈CV 예시 코드
아래 코드는 학습 없이 고전적 필터를 바로 적용하는 실습 예시입니다. 연구/과제 보고서에 그대로 활용할 수 있도록 간결하게 구성했습니다.[5][2][1]

### OpenCV: Sobel, Scharr, LoG, Canny
```python
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# 1) Gaussian smoothing
blur = cv2.GaussianBlur(img, (5,5), 1.0)  # sigma=1.0
# 2) Sobel gradients
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
ang = cv2.phase(gx, gy, angleInDegrees=True)

# 3) Scharr (더 정교한 방향 응답)
gx_s = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
gy_s = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
mag_s = cv2.magnitude(gx_s, gy_s)

# 4) LoG (Gaussian + Laplacian)
log = cv2.GaussianBlur(img, (0,0), 1.0)
log = cv2.Laplacian(log, cv2.CV_32F, ksize=3)

# 5) DoG (sigma1, sigma2 차분)
g1 = cv2.GaussianBlur(img, (0,0), 1.0)
g2 = cv2.GaussianBlur(img, (0,0), 2.0)
dog = g1 - g2

# 6) Canny
edges = cv2.Canny(img, threshold1=50, threshold2=150)  # T_low=50, T_high=150
```
이 코드는 가우시안 스무딩 후 Sobel/Scharr로 gradient를 계산하고, LoG/DoG로 대안적 검출을 비교하며, Canny로 안정적인 에지 맵을 생성합니다.[5][2][3][1]

### PyTorch: 커널로 Sobel/LoG 구현
```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Load gray image [1,1,H,W]
img = Image.open("input.jpg").convert("L")
to_tensor = transforms.ToTensor()
x = to_tensor(img).unsqueeze(0)  # [1,1,H,W], float32 in [0,1]

# Reflection padding for robust borders
x_pad = F.pad(x, (1,1,1,1), mode="reflect")

# Sobel kernels
kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)

gx = F.conv2d(x_pad, kx)
gy = F.conv2d(x_pad, ky)
mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
ang = torch.atan2(gy, gx)  # [-pi, pi]

# LoG kernel (예: 5x5 근사)
log_kernel = torch.tensor([
    [0, 0, -1,  0,  0],
    [0, -1, -2, -1, 0],
    [-1,-2, 16, -2,-1],
    [0, -1, -2, -1, 0],
    [0, 0, -1,  0,  0]
], dtype=torch.float32).view(1,1,5,5)

x_pad2 = F.pad(x, (2,2,2,2), mode="reflect")
log_resp = F.conv2d(x_pad2, log_kernel)
```
이 구현은 딥러닝 프레임워크 내에서 고전적 필터를 합성곱으로 통합하고, 패딩을 reflection으로 설정해 외곽 안정성을 확보합니다.[1][7]

## 딥러닝과의 연결
- 고전적 에지 맵은 데이터 전처리, 손실 가중(edge-aware loss), 경계 증강 등에 유용합니다.[10][2]
- 최신 연구는 SAM 같은 대규모 비전 모델의 경계 표현력을 활용해 비지도 에지 검출을 향상하기도 합니다. 라벨 비용을 줄이면서 경계 성능을 확보하는 추세입니다.[11]
- 하이브리드 접근(가이드 필터·주의 모듈 결합)은 복잡 텍스처나 열영상 등 특수 도메인에서 경계 품질을 크게 끌어올립니다.[12][10]

## 실무 팁
- 경계 두께를 1픽셀로 만들려면 NMS 또는 morphological thinning을 적용합니다. Canny는 NMS를 내장합니다.[2]
- 스케일 선택은 데이터 특성에 맞춥니다. $$\sigma$$가 클수록 굵은 구조, 작을수록 디테일이 살아납니다. 멀티스케일 결합으로 강건성을 높일 수 있습니다.[4][3]
- 외곽 패딩은 reflect를 우선 고려하고, 임곗값은 히스토그램과 Otsu/비율기반으로 자동화하면 재현성이 좋아집니다.[13][7]

```
morphological thinning은 이진 이미지에서 객체의 중심선(skeleton)을 추출하는 기법으로, 객체의 형태를 유지하면서 외곽 픽셀을 반복적으로 제거해 두께를 얇게 만드는 과정입니다. 이 작업은 객체의 연결성과 모양을 변경하지 않는 한도 내에서 픽셀을 제거하며, 주로 스켈레톤화에 활용됩니다.

구체적으로, thinning은 hit-or-miss 변환과 밀접하게 관련되어 있으며, 특정 구조 요소(structuring element)를 이용해 이미지의 픽셀을 선택적으로 감소시킵니다. 이 과정은 여러 방향으로 회전한 구조 요소들을 순차적으로 적용하면서 반복되며, 더 이상 삭제할 픽셀이 없을 때까지 계속됩니다.

요약하면, morphological thinning은 다음과 같습니다:

이진 이미지에 적용되는 형태학적 연산
객체의 두께를 줄이면서 골격(skeleton)을 보존
hit-or-miss 변환 기반의 구조 요소 사용
반복적이고 점진적으로 픽셀을 제거
스켈레톤화, 경계 검출 등 다양한 응용 가능
파이썬의 scikit-image 라이브러리에서는 morphology.thin() 함수로 쉽게 수행할 수 있습니다.
```

```
Otsu algorithm은 이미지 이진화 방법으로, 픽셀의 명암 분포(히스토그램)를 분석하여 배경과 전경의 두 클래스 간 분산을 최대화하거나 클래스 내 분산을 최소화하는 최적의 임계값을 자동으로 찾습니다. 이 임계값으로 이미지를 두 그룹으로 구분해 효과적인 객체 분할을 수행합니다.

구체적으로, Otsu 방법은 다음 과정을 거칩니다:

이미지 히스토그램을 구해 픽셀 명암값 분포를 파악합니다.

가능한 모든 임계값에 대해 두 그룹으로 나누고, 각 그룹의 분산을 계산합니다.

이때 두 그룹 내 분산의 가중합이 최소가 되는 임계값을 선택합니다. 이는 같은 의미로 두 그룹 간 분산을 최대화하는 것과 같습니다.

선택된 임계값으로 픽셀을 흑백(보통 배경-전경)으로 이진화합니다.

이 알고리즘은 bimodal(두 산출점 뚜렷한) 히스토그램에 특히 효과적이며, 비지도적이며 추가 매개변수 없이 자동으로 임계값을 찾아내는 장점이 있습니다. OpenCV 등에서도 쉽게 구현되어 널리 사용됩니다.
```

## 추가 학습자료
- Gradient–Edge 기초와 finite difference, Sobel/DoG/LoG 비교, 경계 처리 요약은 링크의 노트가 간결하게 정리합니다.[7]
- 이미지 그라디언트의 정의, 크기/각도 계산, 에지 방향-그라디언트 수직 관계의 직관은 보충 자료에서 쉽게 복습할 수 있습니다.[14][15]
- Canny 전체 파이프라인과 NMS, 이중 임계값의 동작 원리는 실무에 바로 적용 가능한 설명으로 정리되어 있습니다.[2][1]

## 체크리스트
- 입력 노이즈가 크면 가우시안 스무딩부터 적용했는가 ?[2][7]
- 에지 목적이 경계 위치인지/윤곽 단순화인지에 따라 Sobel/LoG/Canny를 구분했는가 ?[3][1][2]
- 경계 처리 방식(reflect 등)으로 테두리 아티팩트를 최소화했는가 ?[7]
- 임계값 선택을 자동화하거나 데이터셋에 맞게 튜닝했는가 ?[13][2]

## 한 줄 정리
에지는 밝기의 큰 변화이며, 스무딩과 적절한 미분 연산자(특히 **Canny** 또는 **LoG/DoG**)를 조합하면 잡음에 강하고 위치 정확한 에지 맵을 얻을 수 있습니다.[3][1][2][7]

[1](https://blog.everdu.com/397)
[2](https://blog.roboflow.com/edge-detection/)
[3](https://www.uomustansiriyah.edu.iq/media/lectures/5/5_2024_11_20!08_05_18_PM.pdf)
[4](https://reminder-by-kwan.tistory.com/157)
[5](https://www.geeksforgeeks.org/software-engineering/edge-detection-using-prewitt-scharr-and-sobel-operator/)
[6](https://www.tutorialspoint.com/edge-detection-using-prewitt-scharr-and-sobel-operator)
[7](https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Image-Filtering-Image-Gradients-and-Edge-Detection)
[8](https://jinwoo-jung.com/62)
[9](https://hannibunny.github.io/orbook/preprocessing/04gaussianDerivatives.html)
[10](https://www.mdpi.com/2076-3417/15/7/3551)
[11](https://ieeexplore.ieee.org/document/10490131/)
[12](https://jmstt.ntou.edu.tw/journal/vol33/iss1/3)
[13](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13163/3030576/Optimization-application-of-Canny-algorithm-in-gear-image-edge-detection/10.1117/12.3030576.full)
[14](https://dsbook.tistory.com/205)
[15](https://hygenie-studynote.tistory.com/40)
[16](https://link.springer.com/10.1007/s11277-023-10628-5)
[17](https://ir.uitm.edu.my/id/eprint/102633/1/102633.pdf)
[18](https://link.springer.com/10.1007/s41870-022-01059-9)
[19](https://ieeexplore.ieee.org/document/10096818/)
[20](https://ieeexplore.ieee.org/document/9848927/)
[21](https://ieeexplore.ieee.org/document/10962536/)
[22](https://zenodo.org/record/4301154/files/44%2021Dec17%2010Nov17%203Nov17%209638-11943-1-ED.docx%20(Edit%20I).pdf)
[23](https://gjeta.com/sites/default/files/GJETA-2024-0054.pdf)
[24](http://arxiv.org/pdf/2409.01609.pdf)
[25](https://www.mdpi.com/1424-8220/23/15/6883)
[26](http://section.iaesonline.com/index.php/IJEEI/article/download/2597/605)
[27](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2022/185/2022/isprs-annals-V-2-2022-185-2022.pdf)
[28](https://arxiv.org/pdf/2501.18929.pdf)
[29](https://arxiv.org/html/2405.02508)
[30](http://arxiv.org/pdf/2308.14084.pdf)
[31](https://zenodo.org/record/4070962/files/33%2021Apr17%2021Dec16%2013796-31018-1-Zly%20Ed.pdf)
[32](https://faceyourfear.tistory.com/78)
[33](https://velog.io/@lcy1221/%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-15-Edge-Detection)
[34](https://www.ultralytics.com/blog/edge-detection-in-image-processing-explained)
[35](https://sarah0518.tistory.com/122)
[36](https://velog.io/@hseop/Edges)
[37](https://www.youtube.com/watch?v=Z00QZo4Tqug)
[38](https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_edge_filter.html)
[39](https://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf)
[40](https://en.wikipedia.org/wiki/Canny_edge_detector)
[41](http://www.cs.toronto.edu/~fidler/slides/2015/CSC420/lecture3.pdf)
[42](https://velog.io/@yeontachi/CV-Edge-Detection-Edge-detection-and-Image-gradient)

# Image Gradients and Edge Detection

딥러닝에서 이미지의 특징을 뽑아내는 첫걸음은 **엣지(edge)**, 즉 픽셀 강도가 크게 변하는 지점을 찾는 것입니다. 이 글에서는 이미지에서 엣지를 검출하기 위해 사용하는 그래디언트(gradient) 개념부터 대표 필터인 소벨(Sobel), DoG(Derivative of Gaussian), LoG(Laplacian of Gaussian) 필터를 차례로 살펴보겠습니다. 마지막으로 필터를 이미지 경계에 적용할 때 생기는 문제와 해결 방법까지 정리합니다.

***

## 1. 이미지 그래디언트란 무엇인가요?  
이미지를 2차원 함수 $$f(x,y)$$로 보면, 픽셀 값은 함수값입니다. 이 함수의 1차 미분값을 모아 놓은 것이 **그래디언트**입니다.  
- $$\frac{\partial f}{\partial x}$$: 가로 방향 변화량  
- $$\frac{\partial f}{\partial y}$$: 세로 방향 변화량  

연속 함수에서는 미분의 극한(limit)을 쓰지만, 디지털 이미지는 불연속 값입니다. 따라서 **유한 차분(finite difference)** 으로 근사합니다.  
1. 전진 차분(Forward)  
2. 후진 차분(Backward)  
3. 중앙 차분(Central)  

이 차분 연산을 **컨볼루션(convolution)** 으로 구현하면, 이미지에 필터를 적용해 한 번에 미분값을 계산할 수 있습니다.

***

## 2. 소벨(Sobel) 필터로 엣지 검출하기  
엣지는 그래디언트 방향의 **수직 방향**에서 나타납니다.  
- **수평 소벨 필터**: 세로 엣지 검출  
- **수직 소벨 필터**: 가로 엣지 검출  

소벨 필터는 노이즈 억제용 가중치가 포함되어 있어, 단순 차분보다 덜 민감하게 변화량을 포착합니다.  
1. 원하는 방향의 소벨 커널 선택  
2. 이미지에 컨볼루션 수행  
3. $$G_x, G_y$$를 종합해 엣지 방향과 크기 계산  

***

## 3. 노이즈와 블러: DoG 필터  
미분 연산은 **고주파 노이즈**를 증폭시킵니다. 따라서 엣지 검출 전에는 노이즈를 줄여야 합니다.  
- 먼저 **가우시안 블러**로 이미지를 부드럽게 한 뒤 소벨 같은 미분 필터를 적용  
- 이를 하나로 합친 것이 **DoG(Derivative of Gaussian)** 필터로, 연산을 한번에 처리해 효율적입니다.

***

## 4. 2차 미분과 LoG 필터  
1차 미분 대신 **2차 미분(Laplacian)** 을 사용하면, 엣지 근처에서 값이 **제로 교차(zero crossing)** 합니다.  
- **Laplacian of Gaussian(LoG)**: 가우시안 블러 → 라플라시안 연산 순서  
- 엣지 위치에서 0으로 바뀌는 지점을 찾아 검출  
- DoG와 비교 시, 더 정밀하지만 이중 선(double line) 검출이 발생할 수 있습니다.  

두 방법 중 어느 것이 무조건 좋다고 할 수 없으므로, 상황에 맞춰 선택합니다.

***

## 5. 이미지 경계 처리 방법  
필터를 적용할 때, 이미지 가장자리 픽셀도 처리해야 합니다. 대표 방법은 다음과 같습니다.  
- **검은색 테두리(Clip)**: 외곽을 0으로 패딩  
- **래핑(Wrap around)**: 반대쪽 픽셀을 이어 붙임  
- **에지 복사(Copy edge)**: 가장자리 픽셀을 복사해 패딩  
- **반사(Reflect across edge)**: 경계를 기준으로 대칭 복사  
- **가변 필터(Vary filter)**: 경계 근처 필터 크기 조정  

각 방법마다 장단점이 있으므로, 데이터 특성에 따라 선택하세요.

***

이제 이미지 그래디언트와 엣지 검출 필터의 기본 원리와 대표적인 구현 방식을 이해하셨습니다. 딥러닝 전처리 단계에서 엣지 기반 특징을 추출할 때 이 필터들을 적극 활용해 보세요.

[1](https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Image-Filtering-Image-Gradients-and-Edge-Detection)

https://velog.io/@claude_ssim/%EA%B3%84%EC%82%B0%EC%82%AC%EC%A7%84%ED%95%99-Image-Filtering-Image-Gradients-and-Edge-Detection
