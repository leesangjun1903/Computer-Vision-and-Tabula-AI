# Image resampling : Lanczos resampling 타당성

## Lanczos resampling
Lanczos resampling은 주어진 이산 신호를 새로운 샘플링 비율로 재구성(upsampling 또는 downsampling)하는 기법으로, 원본 신호에 Lanczos 커널을 합성곱(convolution)하여 인터폴레이션 값을 산출합니다. 이 방법은 고품질의 신호 재샘플링에 많이 사용됩니다.

구체적으로, 보간값 S(x)는 원본 샘플 $s_i$ 와 Lanczos 커널 $L(x-i)$를 윈도우 크기 a 범위 내에서 합산하여 계산하며, 다음 식으로 표현됩니다:

```math
[S(x) = \sum_{i=\lfloor x \rfloor - a + 1}^{\lfloor x \rfloor + a} s_i \cdot L(x - i)
]
```

여기서 a는 필터 크기 조절 인자, ($\lfloor x \rfloor$)는 내림 함수입니다. 이 커널은 중심에서 1의 값을 가지며, 정수 위치에서는 0이 되므로 주어진 샘플을 정확히 보간합니다.

또한, 다운샘플링 시에는 반사(aliasing) 현상을 줄이기 위해 필터 스케일을 조정하며, 업샘플링은 커널을 그대로 사용합니다.

Lanczos 커널은 sinc 함수와 유사하게 사인 함수로 정의되며(즉, sinc 함수에 윈도우(절단)를 적용한 형태), 주로 이미지 확대·축소 시 픽셀 값 간 보간(interpolation)에 활용되어 연속적이고 부드러운 결과를 냅니다. 이 방식은 Gibbs 현상을 완화하며, 특히 급격한 신호 변동 구간에서 뛰어난 성능을 발휘합니다.

요약하면, Lanczos resampling은 신호나 이미지를 고품질로 확대 축소할 때 sinc 함수 기반의 커널을 활용하여 샘플링을 보간하는 방법입니다.

## 타당성 :
요약: “Lanczos 리샘플링은 단조(monotone) 함수/데이터를 보간할 때 ‘출렁거림(overshoot/ringing)’으로 단조성이 쉽게 깨지고, 수치적 관점에서 바람직하지 않을 수 있다”는 것은 타당한 지적입니다.  
Lanczos(windowed sinc)는 음의 로브를 갖는 커널이므로 계단/급경사 주변에서 Gibbs형 진동과 과(저)슈트를 유발하며(Gibbs 진동은 신호를 제한된 길이의 함수로 근사할 때 나타나는 불필요한 진동(overshoot와 undershoot)을 의미하며, Lanczos 윈도우드 sinc 필터는 이러한 문제를 줄여 보다 부드럽고 정확한 보간 결과를 제공합니다.), 이는 단조성 상실과 직접 연결됩니다. 

반대로, 단조 보존(monotonicity-preserving) 큐빅 에르미트/스플라인 계열(Fritsch–Carlson/Hyman 류)이나 제약된 cubic convolution은 단조 데이터를 단조로 보간하도록 설계되어 이러한 출렁임을 억제합니다. 

다만, “보간 간격이 줄수록 오차가 늘어난다”는 일반론은 Lanczos 자체의 이론적 수렴을 부정한다기보다, 비단조 커널이 강한 경계/discretization 조합에서 국소 오차나 인공 진동이 두드러질 수 있음을 현상적으로 지적한 것으로 해석하는 것이 적절합니다.[1][2][3][4][5][6][7]

## 핵심 평가
- Lanczos 커널은 $$L(x)=\mathrm{sinc}(\pi x) \mathrm{sinc}(\pi x/a) \mathbf{1}_{|x|<a}$$ 형태의 윈도우를 적용한 sinc function 으로, 음의 로브와 유한 지지(support) 때문에 계단/급경사 주변에서 링잉과 오버슈트를 생성합니다. 이는 “단조가 깨지는” 현상과 동일 계열의 문제이며, a가 커지면 로브 주기가 짧아지고 진동 패턴이 바뀌지만, 현상 자체가 사라지지는 않습니다. [3][2][6][8]
- 이미지/신호 처리에서 알려진 사실: Lanczos-3 등은 디테일 보존과 에일리어싱 억제에는 강점이 있으나, 텍스트 에지나 고대비 경계에서 링잉이 생기며 단조성이 요구되는 수치형 보간에는 부적절할 수 있습니다. 이는 실무 문서와 평가지표 글에서도 반복적으로 언급됩니다.[2][9][6]
- “단조 보존” 보간은 별도의 이론과 기법이 존재합니다. Fritsch–Carlson, Hyman 조건을 가한 큐빅 에르미트/스플라인은 입력 데이터가 단조이면 출력도 단조가 되도록 미분값을 제한합니다. NASA/OSTI 및 고전 논문들이 그 충분/필요 조건과 구현법을 제시합니다.[4][10][7][11][1]
- Cubic convolution(Keys 1989)는 계산량 대비 정확도를 최적화하려는 커널이지만, 매개변수에 따라 음의 로브가 생겨 오버슈트를 일으킬 수 있습니다. 따라서 “제약 없는” 일반 bicubic/Keys 커널이 자동으로 단조를 보장하는 것은 아닙니다. 다만, 단조 제약 또는 클램핑을 결합하면 단조 보존을 달성할 수 있습니다.[5][9][6]

```
에일리어싱(Aliasing)은 연속적인 아날로그 신호를 디지털로 샘플링하는 과정에서 샘플링 주파수가 원래 신호의 최대 주파수의 2배보다 낮거나 필터링이 적절하지 않을 때, 서로 다른 주파수 신호가 구별되지 않고 왜곡되어 나타나는 현상입니다.
이로 인해 신호가 실제와 다르게 복원되거나 이미지에서 계단 현상이 발생합니다.

특히 이미지 처리에서는 선이나 도형의 가장자리가 우둘투둘해지는 계단 현상으로 나타나며, 이를 완화하기 위한 기술을 안티에일리어싱(Anti-Aliasing)이라 합니다.
```

#### Monotonicity-preserving cubic Hermite/spline method
Monotonicity-preserving cubic Hermite/spline 방법은 주어진 데이터 포인트 사이에서 원래 데이터가 증가하거나 감소하는 성질(단조성, monotonicity)을 유지하면서 부드러운 3차 다항식(Hermite 다항식 또는 스플라인)을 이용해 보간하는 기법입니다.

이 방법의 핵심은 보간 다항식의 기울기(도함수) 값들을 조정하여 보간 곡선이 데이터의 단조성(증가/감소 형태)을 유지하도록 하는 것입니다. 대표적인 연구로 Fritsch와 Carlson의 논문에서, 각 구간의 기울기와 데이터 차분 사이의 관계를 통해 단조성 유지에 필요한 수학적 조건을 제시하고, 이를 기반으로 기울기를 제한하거나 재계산하는 절차를 설명합니다.

주요 조건 및 방식은 다음과 같습니다:

- 기울기의 부호가 데이터 구간의 차분 부호와 같아야 한다: 즉, 증가하는 구간에서는 기울기가 양수, 감소하는 구간에서는 음수여야 단조성을 유지합니다.
- 만약 원래의 3차 Hermite 보간에서 파생되는 기울기가 단조성을 깨는 경우, α, β(기울기에 대한 비율 변수)를 계산해 이들이 단조성 보존 영역 내에 있도록 조정(클리핑 또는 제한)합니다.
- 구체적으로, α와 β가 특정 반경을 넘지 않도록 하여 기울기가 급격히 변해 곡선의 변곡이나 진동이 발생하지 않도록 합니다.

이러한 제한을 통해 Gibbs 현상(고주파 진동)이나 불필요한 극값 생성을 막아 데이터의 근본 형태를 보존할 수 있습니다.
이외에도, 비선형적으로 도함수를 계산하거나, 합리적 비-3차 스플라인(rational bi-cubic spline) 등의 확장된 형태로 단조성 보존을 강화하는 연구들이 있습니다.

즉, monotonicity-preserving cubic Hermite/spline은 데이터의 단조성 특성을 보존하기 위해 스플라인의 도함수를 조절함으로써, 곡선이 불필요한 진동 없이 부드럽고 현실적인 형태를 갖도록 만드는 보간 기법입니다.

#### Fritsch–Carlson / Hyman
Fritsch–Carlson 방법은 1980년에 발표된 단조 조각별 3차 보간법으로, 데이터가 단조일 때 보간 곡선도 단조성을 유지하도록 슬로프를 조절하는 기법입니다. 주로 함수의 극단점에서 과도한 진동 없이 부드러운 보간을 제공합니다.

Hyman 방법은 1983년에 발표된, Fritsch–Carlson과 유사한 목적으로 만들어진 단조성 보존 3차 보간법으로, 스플라인을 필터링하여 단조성을 유지하도록 합니다. 특히, Hyman 필터링을 사용해 더욱 정확한 단조성 보존 보간을 수행합니다.

두 방법은 모두 데이터의 단조 구간을 보존하면서 자연스러운 3차 스플라인을 생성한다는 점에서 공통점이 있지만, 알고리즘적 접근과 수학적 처리 방식에 차이가 있습니다. Fritsch–Carlson 알고리즘은 구현이 상대적으로 간단하면서 효율적이며, Hyman 알고리즘은 보다 엄밀한 단조성 보존 측면에서 강점을 가집니다.

요약하면, Fritsch–Carlson과 Hyman 알고리즘은 모두 단조 조건을 만족하는 3차 보간법이며, 각각 독자적인 수학적 방법과 필터링 절차를 통해 단조성을 보존하도록 설계된 대표적인 기법들입니다.

#### Cubic convolution
Cubic convolution은 디지털 이미지나 신호 처리에서 사용되는 보간(Interpolation) 기법 중 하나로, 4개의 인접한 샘플값을 기준으로 3차 다항식을 이용해 새로운 값(중간값)을 부드럽게 계산하는 방법입니다. 이 방법은 2차 다항식을 정확히 재현할 수 있어 선형 보간보다 더 높은 정확도를 가진다는 특징이 있습니다.

주요 특징은 다음과 같습니다.

- 보간 커널은 -2와 2 사이에서만 정의된 조각별 3차 다항식으로, 이 범위 밖에서는 값이 0입니다. 따라서 계산에 4개의 인접한 샘플만 사용합니다.
- 커널은 대칭이고 중심점 0에서는 값이 1, 다른 정수 위치에서는 0이 되도록 설계되어 있습니다.
- 보간 함수는 연속적이고 한 번 미분 가능하며, 일반적으로 smoothing(저역통과) 특성이 강해 이미지가 더 자연스럽고 부드럽게 보입니다.
- 2차 다항식을 완벽히 재현하기 때문에 선형 보간보다 신호 왜곡이 적습니다.
- 1차원은 인접 4점, 2차원은 인접 16점(4x4) 데이터를 사용합니다.

네이버 위키와 관련 연구에서는 가중치 계산에 b, c라는 파라미터 값을 조정해 커널의 곡선 모양과 부드러움을 조절한다고 설명합니다.

간단히 말해, cubic convolution은 주변 4개의 데이터 샘플을 이용하여 3차 다항식을 통해 중간값을 계산하는 고급 보간법으로, 선형 보간보다 부드럽고 정확한 결과를 제공합니다.

## 어디까지가 합리적인가
- 합리적 주장
  - Lanczos는 단조를 보장하지 않으며, 단조 데이터/함수의 2D 보간에서 “출렁이는” 비단조 결과를 낼 수 있습니다. 이는 커널의 음의 로브와 Gibbs형 진동 특성상 자연스러운 결론입니다.[3][6][2]
  - a(로브 수) 변경으로 진동 빈도/패턴이 바뀌지만, 단조성 위반의 근본 원인이 사라지지 않으므로 연구 목적(단조/형상 보존)의 보간에는 부적합할 수 있습니다.[2][3]
  - 단조 보존이 중요한 과제라면 Fritsch–Carlson/Hyman 류의 제약된 큐빅 에르미트/스플라인을 사용하는 것이 정합적입니다.[10][7][1][4]
- 주의가 필요한 표현
  - “보간 간격이 줄어들면 오차가 늘어난다”는 일반화는, 이상화된 밴드리미티드 모형에서는 sinc 기반 보간이 수렴한다는 사실과 배치됩니다. 보다 정확히는, 유한 지지(finite support)/윈도우화(windowing)와 실제 신호의 비밴드리미티드성, 경계/격자 상호작용으로 인해 국소 오버슈트가 감소하지 않거나 드러날 수 있다는 식으로 기술하는 것이 학술적으로 안전합니다.[6][3][2]
  - 또한 “Lanczos가 반드시 나쁘다”기보다는, “디테일 보존/에일리어싱 억제에는 강점, 단조/형상 보존이 중요한 수치 보간에는 부적합”이라는 트레이드오프 관점이 더 정확합니다.[9][3][6]

```
밴드리미티드(band-limited)란 신호 또는 함수가 특정 주파수 범위 내에만 에너지가 집중되어 있고, 그 범위 밖에서는 에너지가 거의 또는 전혀 존재하지 않는 상태를 의미합니다.

유한 지지(finite support)라는 용어는 일반적으로 수학이나 신호처리 분야에서 사용되며, 함수나 신호가 일정 구간(유한한 범위) 내에서만 값이 0이 아니고 그 밖에서는 0인 경우를 말합니다.
즉, 신호나 함수가 유한한 범위에만 '지지'(support)를 가지고 있다라는 뜻입니다.

윈도우화(windowing)는 주로 신호처리에서 신호의 일부분을 선택하여 곱하는 과정으로, 보통 무한하거나 매우 긴 신호를 유한한 길이의 신호로 만들어서 분석하거나 처리할 때 사용합니다.
윈도우 함수(window function)를 곱하면 그 신호는 유한 지지를 가지게 됩니다.

따라서, 윈도우화는 신호 등에 유한 지지를 부여하는 과정이라고 이해하시면 됩니다.
윈도우 함수를 적용하여 신호가 특정 구간 내에서만 유효하게 만들어 그 이후는 0으로 만들어 분석이나 처리가 용이하도록 하는 기법입니다.
이는 푸리에 변환 등의 연산에서 매우 중요한 역할을 합니다.
```

## 수식으로 보는 단조성 위반의 원인
- 1D Lanczos 재구성은

$$
  S(x)=\sum_{i=\lfloor x\rfloor-a+1}^{\lfloor x\rfloor+a}s_i\,L(x-i),\quad
  L(t)=\mathrm{sinc}(\pi t)\,\mathrm{sinc}\left(\frac{\pi t}{a}\right)\mathbf{1}_{|t| < a}
  $$
  
  이며, $$L(t)$$가 음의 값을 갖는 구간을 포함합니다. 음의 로브는 단조 증가 데이터라도 국소적으로 가중합이 감소/반전되게 하여 $$S'(x)\ge0$$를 보장하지 못합니다. 이는 계단 응답의 오버슈트/링잉과 동일한 메커니즘입니다.[8][3][6]
- 반면 단조 보존 큐빅 에르미트는 구간 $$[x_i,x_{i+1}]$$에서

$$
  p(x)=h_{00}(\tau)f_i+h_{10}(\tau)h\,m_i+h_{01}(\tau)f_{i+1}+h_{11}(\tau)h\,m_{i+1},\quad \tau=\frac{x-x_i}{h}
  $$
  
  에 대해 기울기 $$m_i,m_{i+1}$$에 Fritsch–Carlson/Hyman류 제약을 부여하여 $$f_i\le f_{i+1}$$이면 $$p'(x)\ge0$$를 보장합니다(충분조건).[7][1][4]

## 실험 가이드: 단조 함수에서 Lanczos와 단조-보존 스플라인 비교
- 목표: $$z=x^2+y^2$$ 같은 단조(radially monotone) 함수의 희소 샘플에서 2D 보간 시 Lanczos의 링잉/비단조와 단조-보존 보간의 차이를 수치적으로 관찰합니다.[3][4][6]
- 핵심 포인트
  - 1D 단면에서 Lanczos 커널의 음의 로브로 인한 오버슈트가 나타나며, 그 결과 단조성이 깨질 수 있습니다.[6][3]
  - Monotone cubic Hermite(Fritsch–Carlson/Hyman)는 동일 데이터에서 단조를 보존합니다.[1][4][7]

### 파이썬 예시 코드(딥러닝/과학컴퓨팅 환경 가정)
- 내용: 1D와 2D에서 비교. 1D는 신뢰성 높은 관측이 쉬워 핵심 현상을 명확히 보여줍니다. 2D는 이미지 업스케일 상황과 유사합니다.[4][5][7][3][6]

```python
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt

# 1) 테스트 함수: 단조 증가 함수
# f(x) = x^2 on [0, 1], sparse samples
x_coarse = np.linspace(0, 1, 9)
y_coarse = x_coarse**2

# Query grid
x_fine = np.linspace(0, 1, 1000)

# A. Lanczos 1D 구현: 윈도우드 sinc 커널(표준화)
def lanczos_kernel(t, a):
    # sinc normalized as sin(pi t)/(pi t)
    def sinc(u):
        out = np.ones_like(u)
        nz = u != 0
        out[nz] = np.sin(np.pi*u[nz])/(np.pi*u[nz])
        return out
    w = np.abs(t) < a
    k = sinc(t) * sinc(t / a)
    return k * w

def lanczos_interpolate_1d(x, y, xq, a=3):
    # uniform grid assumed
    h = x[1]-x
    # For each query, sum over neighbors within a lobes
    yq = np.zeros_like(xq)
    for j, x0 in enumerate(xq):
        # nearest integer index
        i0 = int(np.floor((x0 - x) / h))
        idxs = np.arange(i0 - a + 1, i0 + a + 1)
        idxs = idxs[(idxs >= 0) & (idxs < len(x))]
        t = (x[idxs] - x0) / h
        w = lanczos_kernel(t, a)
        # Optional: normalization to preserve DC (flux)
        # w_sum = w.sum()
        # if w_sum != 0: w = w / w_sum
        yq[j] = np.sum(y[idxs] * w)
    return yq

y_lanczos = lanczos_interpolate_1d(x_coarse, y_coarse, x_fine, a=3)

# B. Monotone cubic Hermite (PCHIP) — 단조 보존
pchip = interpolate.PchipInterpolator(x_coarse, y_coarse)  # monotone-preserving
y_pchip = pchip(x_fine)

# 진단: 단조성 위반 검출
def monotone_violations(y):
    dy = np.diff(y)
    return np.sum(dy < -1e-12)

viol_lanczos = monotone_violations(y_lanczos)
viol_pchip = monotone_violations(y_pchip)

print("Lanczos violations:", viol_lanczos)
print("PCHIP violations:", viol_pchip)

plt.figure(figsize=(6,4))
plt.plot(x_fine, x_fine**2, 'k--', label='truth')
plt.plot(x_fine, y_lanczos, 'r', label='Lanczos a=3')
plt.plot(x_fine, y_pchip, 'b', label='Monotone PCHIP')
plt.scatter(x_coarse, y_coarse, c='k', s=20)
plt.legend(); plt.tight_layout()
plt.show()
```

- 설명
  - Lanczos 구현은 $$L(t)=\mathrm{sinc}(\pi t)\mathrm{sinc}(\pi t/a)$$ 커널을 사용합니다. 음의 로브 때문에 단조 함수에서도 국소 감쇠가 생겨 미분이 음수가 되는 구간이 발생할 수 있습니다.[8][3][6]
  - PCHIP은 Fritsch–Carlson 계열의 단조 보존 기법으로, 데이터가 증가이면 보간도 증가가 되도록 구간 기울기를 제한합니다.[7][1][4]
  - 옵션으로 Lanczos에 flux 보존(normalization)을 넣을 수 있으나, 단조성 문제 자체를 해결하지는 못합니다.[12][3]

### 2D 실험(이미지 업스케일)
```python
# 2) 2D 스칼라장: z = x^2 + y^2
nx, ny = 17, 17
xs = np.linspace(-2, 2, nx)
ys = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(xs, ys, indexing='xy')
Z = X**2 + Y**2

# 업스케일 그리드
fx, fy = 4, 4
x_hi = np.linspace(xs, xs[-1], fx*(nx-1)+1)
y_hi = np.linspace(ys, ys[-1], fy*(ny-1)+1)
Xh, Yh = np.meshgrid(x_hi, y_hi, indexing='xy')

# 2D Lanczos: separable 적용
def lanczos_interp2d(xs, ys, Z, xq, yq, a=3):
    # separable: first x, then y
    # 1) along x
    Zx = np.zeros((Z.shape, len(xq)))
    for r in range(Z.shape):
        Zx[r,:] = lanczos_interpolate_1d(xs, Z[r,:], xq, a=a)
    # 2) along y
    Zxy = np.zeros((len(yq), len(xq)))
    for c in range(Zx.shape[1]):
        Zxy[:,c] = lanczos_interpolate_1d(ys, Zx[:,c], yq, a=a)
    return Zxy

Zh_lanc = lanczos_interp2d(xs, ys, Z, x_hi, y_hi, a=3)

# 2D 단조 보존 대안: 행/열별 PCHIP(separable). 완전한 2D 단조 보존은 아님(실무 근사)
def pchip_interp1d_grid(xs, arr, xq):
    out = np.zeros((arr.shape, len(xq)))
    for r in range(arr.shape):
        p = interpolate.PchipInterpolator(xs, arr[r,:])
        out[r,:] = p(xq)
    return out

Zx_p = pchip_interp1d_grid(xs, Z, x_hi)
Zhp = np.zeros((len(y_hi), len(x_hi)))
for c in range(Zx_p.shape[1]):
    p = interpolate.PchipInterpolator(ys, Zx_p[:,c])
    Zhp[:,c] = p(y_hi)

# 단조성 진단: 원점에서 방사 방향 단면 추출
def radial_profile(Zimg, xs, ys):
    # y=0 단면
    y0_idx = np.argmin(np.abs(ys - 0))
    return Zimg[y0_idx,:], xs

line_l, xs_line = radial_profile(Zh_lanc, x_hi, y_hi)
line_p, _ = radial_profile(Zhp, x_hi, y_hi)

print("Lanczos 2D line violations:", monotone_violations(line_l))
print("PCHIP 2D line violations:", monotone_violations(line_p))

plt.figure(figsize=(6,4))
plt.plot(x_hi, line_l, 'r', label='Lanczos a=3 (y=0)')
plt.plot(x_hi, line_p, 'b', label='PCHIP separable (y=0)')
plt.legend(); plt.tight_layout()
plt.show()
```

- 설명
  - 2D는 분리 가정으로 Lanczos와 PCHIP을 적용하여 대표 단면에서 단조 위반 여부를 확인합니다. 이상적 2D 단조 보존은 전용 알고리즘이 필요하지만, 분리형 PCHIP만으로도 1D 단면 단조 위반을 크게 줄일 수 있습니다.[3][4][7]
  - 텍스처/에지 인접 영역에서 Lanczos의 링잉이 시각적으로 확인될 수 있습니다(경계 주변 헤일로/어두운 링).[9][2][6]

## 실무 권장안
- 단조/형상 보존이 중요한 수치 보간(예: 누적 분포, 에너지 밀도, 확률/누적량 등)에는 단조 보존 큐빅(예: PCHIP, Fritsch–Carlson/Hyman 제약 스플라인)을 사용합니다.[11][1][4][7]
- 시각 품질과 디테일 보존이 중요한 업스케일(텍스처 유지)에는 Lanczos-3/2를 고려하되, 고대비 경계 링잉 완화를 위해 샤프닝 강도 조절(Sharpening Strength Adjustment), 클램핑(Clamping), 프리필터링(pre-filtering)을 병행합니다.[9][6][3]
- Keys(1989) cubic convolution은 계산 효율이 좋고 널리 쓰이지만, 매개변수 선택에 따라 음의 로브로 오버슈트가 발생할 수 있으므로, 단조 보존이 필요하면 파생 제약/클램핑을 결합합니다.[5][6][9]

#### PCHIP
PCHIP은 Piecewise Cubic Hermite Interpolating Polynomial의 약자로, 구간별 3차 헤르미트 다항식을 이용해 데이터 사이를 보간하는 방법입니다. 주로 데이터의 모양(특히 단조성)을 보존하는 보간에 사용됩니다.

주요 특징은 다음과 같습니다:

- 모노토닉성 유지: 데이터가 단조로울 경우 그 성질을 보존함으로써 불필요한 진동이나 과대경향(overshoot)을 방지합니다.
- 4차 미분 가능성: 연속적 1차 도함수(C1 연속성)를 가지므로 부드러운 보간 결과를 제공합니다.
- 적용 범위: 여러 프로그래밍 언어 및 수치해석 라이브러리에서 제공되며 (예: SciPy, MATLAB, Boost, R 등의 라이브러리).

사용 예: 주어진 데이터 점들에 대해 새로운 x 값에서 함수 값을 추정할 때 사용하며, 쿼리 좌표에 따라 보간된 값을 반환합니다.

즉, PCHIP은 데이터의 본질적 모양을 해치지 않고 부드럽게 중간값을 예측해야 하는 경우에 적합한 보간법입니다.

## 정리 :
- 핵심 메시지(2문장): Lanczos 리샘플링은 윈도우드 sinc 커널의 특성상 오버슈트/링잉을 유발하며, 단조 데이터 보간에서 단조성이 쉽게 깨집니다. 단조 보존이 중요한 작업에는 Fritsch–Carlson/Hyman 제약의 단조-보존 큐빅 보간을 사용하는 것이 타당합니다.[4][6][7][3]
- 기술적 근거(요약): Lanczos 커널의 음의 로브가 계단 응답에 과(저)슈트와 진동을 만들며, 이는 단조 위반으로 나타납니다. 반면 단조-보존 큐빅은 기울기 제약으로 $$p'(x)\ge0$$을 보장합니다.[6][7][3][4]

```
샤프닝 강도 조절은 이미지나 영상에서 선명도를 적절히 조절하는 것으로, 과도하면 인위적이거나 노이즈가 강조될 수 있어 적절한 강도 설정이 중요합니다.
클램핑은 필터 적용 시 값이 특정 범위를 벗어나지 않도록 제한하는 기법이며, 프리필터링은 샤프닝 전 노이즈를 줄이기 위해 미리 처리하는 과정입니다.

구체적으로:

샤프닝 강도 조절은 언샤프 마스크(USM) 같은 필터에서 매개변수 조절로 이뤄지며, 강도가 높아질수록 경계가 더 날카로워지지만 노이즈도 강조됩니다.
클램핑은 샤프닝 후 픽셀 값이 너무 밝거나 어두워지는 것을 방지해 부자연스러운 결과를 줄이는 역할을 합니다.
프리필터링은 샤프닝 전에 노이즈 감소 필터를 적용해 노이즈 강조 문제를 완화합니다. 특히 저조도나 고감도 촬영 이미지에서 필수적입니다.
이 기술들은 엔비디아의 이미지 샤프닝 기능처럼 하드웨어나 소프트웨어에서 적용되기도 하며, 적절한 스케일링 모드 변경으로 기능 활성화가 가능합니다. 또한 이미지 차분에 기반한 필터 설계 시 일차 미분을 사용해 계산을 단순화하기도 합니다.
```

참고문헌
- Lanczos resampling 개요와 수식, 커널 특성(윈도우드 sinc): 위키 및 설명 자료.[8][3]
- Gibbs/링잉과 Lanczos의 주파수-영역 해석, 윈도우 효과: 해설 글.[2]
- 링잉/오버슈트의 원인과 로브 구조, bicubic과의 비교: 위키.[6]
- 단조-보존 큐빅 보간의 조건과 구현: NASA/OSTI 보고, Fritsch–Carlson/Hyman 계열.[10][1][7][4]
- Cubic convolution(Keys 1989)의 설계와 한계(음의 로브 가능성): 원 논문.[5]
- 도구/실무 관찰(디테일 vs 링잉 트레이드오프): 소프트웨어/가이드 문서.[9]

[1](https://en.wikipedia.org/wiki/Monotone_cubic_interpolation)
[2](https://mazzo.li/posts/lanczos.html)
[3](https://en.wikipedia.org/wiki/Lanczos_resampling)
[4](https://ntrs.nasa.gov/api/citations/19910011517/downloads/19910011517.pdf)
[5](http://www.ncorr.com/download/publications/keysbicubic.pdf)
[6](https://en.wikipedia.org/wiki/Ringing_artifacts)
[7](https://www.osti.gov/servlets/purl/5328033)
[8](https://stackoverflow.com/questions/1854146/what-is-the-idea-behind-scaling-an-image-using-lanczos)
[9](https://pixinsight.com/doc/docs/InterpolationAlgorithms/InterpolationAlgorithms.html)
[10](https://arxiv.org/pdf/2102.11564.pdf)
[11](https://www.sciencedirect.com/science/article/pii/S0377042701005064)
[12](https://github.com/jeffboody/Lanczos)
[13](https://awintersky.tistory.com/24)
[14](https://www.mdpi.com/1099-4300/23/5/493/pdf)
[15](https://arxiv.org/pdf/1904.06012.pdf)
[16](https://arxiv.org/pdf/1909.03437.pdf)
[17](https://arxiv.org/pdf/2410.11090.pdf)
[18](http://downloads.hindawi.com/journals/jam/2015/908924.pdf)
[19](https://arxiv.org/pdf/2312.03848.pdf)
[20](http://arxiv.org/pdf/2207.05275.pdf)
[21](https://arxiv.org/pdf/2409.15053.pdf)
[22](https://www2.ia-engineers.org/conference/index.php/icisip/icisip2016/paper/download/1124/806)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC8142998/)
[24](https://arxiv.org/html/2411.18436v3)
[25](https://www.ams.org/journals/mcom/1980-35-152/S0025-5718-1980-0583502-2/S0025-5718-1980-0583502-2.pdf)
[26](http://arxiv.org/pdf/2106.00284.pdf)
[27](https://quantum-journal.org/papers/q-2023-05-23-1018/pdf/)
[28](http://arxiv.org/pdf/2407.21777.pdf)
[29](http://arxiv.org/pdf/2411.04212.pdf)
[30](http://arxiv.org/pdf/2401.15339.pdf)
[31](http://arxiv.org/pdf/2306.07435.pdf)
[32](http://arxiv.org/pdf/2308.15683.pdf)
[33](http://arxiv.org/pdf/2411.00367.pdf)
[34](https://arxiv.org/abs/2102.11564)
[35](https://www.cloudynights.com/topic/507142-lanczos-resampling-issue/)
[36](https://github.com/chartjs/Chart.js/issues/3086)
[37](https://proceedings.mlr.press/v178/buchholz22a/buchholz22a.pdf)
[38](https://www.jb101.co.uk/2020/12/27/monotone-cubic-interpolation.html)
[39](http://pfeifer.phas.ubc.ca/refbase/files/Meijering-MedImAnal-2001-5-111.pdf)
[40](https://arxiv.org/pdf/2302.02849.pdf)
[41](https://arxiv.org/pdf/2203.12532.pdf)

https://awintersky.tistory.com/24
