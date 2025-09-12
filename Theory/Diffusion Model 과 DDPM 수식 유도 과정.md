
# Diffusion Model과 DDPM 수식 유도 가이드

딥러닝을 공부하는 대학생을 위해 **Diffusion Model**과 **DDPM(Denoising Diffusion Probabilistic Model)**의 수식 유도 과정을 쉽게 풀어 설명합니다. 이 글을 통해 개념을 이해하고, 실제로 논문 수준의 모델 구성 예시까지 익힐 수 있습니다.  

***

## 1. Diffusion Model 개요  
Diffusion Model은 두 개의 확률 분포를 사용합니다.  
1) **Forward Process**: 데이터에 점진적으로 노이즈를 추가하는 분포 $$q$$.  
2) **Reverse Process**: 노이즈를 제거하며 데이터를 복원하는 분포 $$p_\theta$$.  

모델 학습은 $$p_\theta$$가 주어진 데이터에 가장 잘 맞도록 **Maximum Likelihood Estimation**을 수행하는 과정입니다. 하지만 직접 Likelihood를 최적화하기 어려워 **ELBO(Evidence Lower Bound)**를 구해 최대화합니다.

***

## 2. ELBO로 변환하기  
우선 로그 우도 $$\log p_\theta(x)$$를 ELBO 형태로 분해합니다.  

$$
\log p_\theta(x) = \mathbb{E}_{q(z|x)}\bigl[\log p_\theta(x,z) - \log q(z|x)\bigr] + \mathrm{KL}\bigl(q(z|x)\|p_\theta(z|x)\bigr)
$$  

이 수식은 데이터 $\(x\)$의 로그 우도 $\(\log p_{\theta }(x)\)$를 변분 하한과 KL 발산 항으로 분해한 것입니다.

$\(p_{\theta }(x)\)$는 모델이 생성하고자 하는 실제 데이터 분포를 나타내며, $\(\log p_{\theta }(x)\)$는 이 분포에 대한 로그 우도입니다. 이 값은 직접 계산하기 어렵기 때문에, 간접적인 방법으로 최적화됩니다.            
$\(q(z|x)\)$는 데이터 $\(x\)$가 주어졌을 때 잠재 변수 $\(z\)$의 분포를 나타내는 인코더 또는 추론 분포입니다. DDPM에서는 이 분포가 노이즈를 추가하는 포워드 프로세스(forward process)에 해당합니다.

$\(p_{\theta }(x,z)\)$는 데이터 $\(x\)$와 잠재 변수 $\(z\)$의 결합 분포를 나타내며, $\(p_{\theta }(x,z)=p_{\theta }(x|z)p_{\theta }(z)\)$로 표현될 수 있습니다.

$(\mathrm{KL}\bigl(q(z|x)\|p_{\theta }(z|x)\bigr)\)$는 $\(q(z|x)\)$와 $\(p_{\theta }(z|x)\)$ 사이의 KL 발산입니다.  
$\(p_{\theta }(z|x)\)$는 데이터 $\(x\)$가 주어졌을 때 잠재 변수 $\(z\)$의 사후 분포를 나타내며, DDPM에서는 이 분포가 노이즈를 제거하는 리버스 프로세스(reverse process)에 해당합니다.

# 유도 :
$\(\log p_{\theta }(x)\)$는 잠재 변수 $\(z\)$에 대한 주변화(marginalization)를 통해 $\(\log \int p_{\theta }(x,z)dz\)$로 표현될 수 있습니다.  
이 식은 $\(q(z|x)\)$를 곱하고 나누어 $\(\log \int q(z|x)\frac{p_{\theta }(x,z)}{q(z|x)}dz\)$로 변형됩니다.  

Jensen의 부등식을 적용하면 $\(\log p_{\theta }(x)\ge \mathbb{E}q(z|x)\bigl[\log \frac{p\theta (x,z)}{q(z|x)}\bigr]\)$가 됩니다. 이 우변이 바로 ELBO입니다.  

#### Jensen's Inequality
함수 (f)가 볼록(convex)일 때 임의의 확률변수 (X)에 대해 다음이 성립합니다:

```math
[f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
]
```

여기서 $(\mathbb{E}[\cdot])$는 기댓값을 의미합니다.

이 부등식의 의미는, 볼록 함수에 대한 함수값의 평균(우변)이 평균값에 대한 함수값(좌변)보다 크거나 같다는 것으로, 함수가 아래로 볼록할 때 적용됩니다. 또한 함수가 오목(concave)하면 부등식 방향이 반대가 됩니다.

- $\(\log \)$ 함수의 위로 볼록성 활용: $\(\log \)$ 함수는 위로 볼록하므로, 어떤 확률변수 $A$와 $B$에 대해 $E[\log A] \leq \log E[A]$가 성립합니다.
- 부등식 적용: 주어진 부등식에서 $A$에 해당하는 부분을 $p_{\theta}(x,z)/q(z|x)$라고 생각하면, $z$에 대한 기대값을 취하기 전의 $A$에 젠센의 부등식을 적용합니다.

$\(E_{q(z|x)}[\log (\frac{p_{\theta }(x,z)}{q(z|x)})]\le \log (E_{q(z|x)}[\frac{p_{\theta }(x,z)}{q(z|x)}])\)$

ELBO는 $\(\mathbb{E}\_{q(z|x)}\bigl[\log p\theta (x,z)-\log q(z|x)\bigr]\)$로 다시 쓰여질 수 있습니다.  

$$
\(\log p_{\theta }(x)=\mathbb{E}\_{q(z|x)}\bigl[\log \frac{p\theta (x,z)}{q(z|x)}\bigr]+\mathrm{KL}\bigl(q(z|x)\|p_{\theta }(z|x)\bigr)\)
$$ 

관계가 성립합니다.  

이는 $\(\mathrm{KL}\bigl(q(z|x)\|p_{\theta }(z|x)\bigr)=\mathbb{E}_{q(z|x)}\bigl[\log \frac{q(z|x)}{p\theta (z|x)}\bigr]\)$ 를 이용하여 증명될 수 있습니다.

여기서 KL 항은 늘 음이 아니므로,  

$$
\mathcal{L}_\text{ELBO} = \mathbb{E}_{q(z|x)}\bigl[\log p_\theta(x,z) - \log q(z|x)\bigr]
$$  

를 최대화하면 됩니다.  

Diffusion Model에서는 시계열 단계 $$t=0,\dots,T$$를 도입해 아래와 같이 ELBO를 확장합니다.  

$$
\mathcal{L}_\text{ELBO}
= \sum_{t=1}^T \mathbb{E}_{q(x_{t},x_0)}\left[ 
   -\log \frac{p_\theta(x_{t-1}\mid x_t)}{q(x_t\mid x_{t-1})}
\right].
$$  

각 단계에서 노이즈가 더해진 데이터 $(x_t)$로부터 이전 단계 $(x_{t-1})$를 복원하는 확률 모델 $(p_\theta)$와, 실제 노이즈 추가 과정의 분포 $(q)$ 간의 차이를 측정하며, 이 값을 전체 단계에 대해 합산한 것입니다.

즉, 이 ELBO는 reverse process 모델 $(p_\theta)$가 forward noising 과정을 잘 근사하는 정도를 나타내며, 이를 최대화(또는 해당 음의 로그 비율을 최소화)할 때 실제 데이터 분포의 likelihood를 효율적으로 근사하고 나아가 좋은 생성 모델이 됩니다.

#### 유도 과정 :
원본 데이터 $(x_0)$에서 점차 노이즈를 추가하는 forward diffusion process $(q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1}))$를 정의합니다.  
반대로 노이즈에서 점차 깨끗한 데이터로 복원하는 reverse generative process $(p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t))$를 학습합니다.

데이터의 우도를 최대화하는 대신, 잠재 변수 (중간 상태)에서 변분 근사를 사용해 ELBO를 최대화합니다.  
ELBO는 다음과 같이 변환할 수 있습니다:

```math
[ \log p_\theta(x_0) \geq \mathbb{E}{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right].
]
```

우도 (p_\theta(x_{0:T}))를 사슬 법칙(chain rule)을 따라 다음과 같이 인수 분해할 수 있습니다.

```math
[ p_\theta(x_{0:T}) = p_\theta(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t)
]
```

또 근사분포 $(q(x_{1:T} \mid x_0))$도 마찬가지로

```math
[ q(x_{1:T} \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1})
]
```

위 우도 비율을 단계별로 나누어 로그를 적용하면:

```math
[ \mathbb{E}_{q(x{1:T} \mid x_0)} \left[ \log p(x_T) + \sum_{t=1}^T \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})} \right].
]
```

이 식은 최대화되어야 하는 값입니다. 이를 최소화되는 손실 함수로 바꾸기 위해 전체 식에 음수 부호가 적용됩니다.

음수 부호가 적용된 후, $\(\log p(x_{T})\)$ 항은 일반적으로 무시되거나 다른 방식으로 처리됩니다.  
이는 $\(p(x_{T})\)$가 모델 학습 과정에서 직접적으로 최적화하기 어려운 항이거나, 다른 항들에 비해 상대적으로 중요도가 낮다고 간주될 수 있기 때문입니다.  

결과적으로, 시계열 단계별 합으로 표현된 항들만 남게 되며, 이는 각 시간 단계에서의 재구성 오차와 관련된 항으로 해석될 수 있습니다.

또한, 기댓값은 전체 시퀀스 $\(q(x_{1:T}\mid x_{0})\)$ 대신 각 시점의 $\(x_{t}\)$와 초기 상태 $\(x_{0}\)$의 결합 분포 $\(q(x_{t},x_{0})\)$에 대해 계산되어, 개별 시점의 기여도를 강조하고 수학적 편의성을 높이며 확률적 미분 가능성을 확보합니다.  
( $\(q(x_{t},x_{0})\)$는 $\(q(x_{t}\mid x_{0})p(x_{0})\)$로 분해될 수 있으며, 이는 특정 시점의 조건부 분포와 초기 분포를 사용하여 기댓값을 계산하는 것을 가능하게 합니다. 이는 전체 시퀀스에 대한 결합 분포보다 다루기 쉬울 수 있습니다.)

최종적으로 최대화 대신, 음의 로그 우도 (negative log likelihood)를 최소화하는 손실함수로 바꾸기 위해 부호를 바꾸고 시계열 단계별 합으로 나타냅니다:

```math
[ \mathcal{L}_\text{ELBO} = \sum_{t=1}^T \mathbb{E}_{q(x_t, x_0)} \left[- \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right].
]
```

***

## 3. DDPM Loss 유도  
DDPM은 위 ELBO를 실용적으로 단순화한 버전입니다.  
1) **불필요한 항 제거**: 초기와 최종 단계 손실($$L_0, L_T$$)은 학습에 영향이 적어 무시합니다.  

```math
[\mathcal{L}' = \sum_{t=1}^{T-1} \mathbb{E}_{q(x_t,x_0)} \left[- \log p_\theta(x_{t-1} \mid x_t) \right] + \text{const.}
]
```

2) **Fixed 노이즈 파라미터**: 노이즈 수준 $$\beta_t$$는 학습 대상이 아니므로 KL 항에서 제외합니다.  

$(\beta_t$) 는 확산(노이즈 추가) 과정에서 각 단계의 노이즈 양을 조절하는 하이퍼파라미터로, $(q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I))$ 와 같이 정의됩니다. 즉, 각 단계의 노이즈 분포는 $(\beta_t)$ 를 평균과 분산에 반영하여 노이즈를 주입합니다.

- 정방향 확산: 원본 데이터 $(x_0)$ 에 점진적으로 노이즈 $(\beta_t)$를 추가하여 $(x_t)$를 만듭니다.
- 후방 복원: 모델 $(p_\theta(x_{t-1}|x_t))$는 이러한 노이즈가 추가된 $(x_t)$로부터 이전 상태를 역으로 복원하는 확률 분포를 학습합니다.

$(q(x_{t-1} \mid x_t, x_0))$는 사실상 노이즈가 적용된 역방향 분포이며, 이를 이용해 아래 KL divergence로 표현 가능하기 때문에, 결국 모든 단계에서 최소화해야 할 핵심 손실은  

$$
L_{t-1} 
= \mathrm{KL}\bigl(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)\bigr).
$$  

### 3.1. $$q(x_t\mid x_0)$$ 재매개화  
Forward 과정에서  

$$
q(x_t\mid x_0) = \mathcal{N}\bigl(x_t;\,\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)\mathbf{I}\bigr)
$$  

이고, reparameterization trick을 통해  

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$  

로 표현합니다.

### 3.2. $$q(x_{t-1}\mid x_t,x_0)$$ 평균과 분산  
Bayes 정리와 가우시안 분포 성질을 이용해  

$$
q(x_{t-1}\mid x_t,x_0) 
= \mathcal{N}\bigl(x_{t-1};\,\tilde\mu_t(x_t,x_0),\,\tilde\beta_t I\bigr)
$$  

으로 유도합니다.  
- 평균 $$\tilde\mu_t$$  
- 분산 $$\tilde\beta_t$$  

### 3.3. $$p_\theta(x_{t-1}\mid x_t)$$ 정의  
DDPM은 $$q$$의 평균을 근사하기 위해  

$$
p_\theta(x_{t-1}\mid x_t) 
= \mathcal{N}\bigl(x_{t-1};\,\mu_\theta(x_t,t),\,\beta_t I\bigr)
$$  

로 설계합니다. 여기서 $$\mu_\theta$$는 네트워크가 예측하는 값입니다.

### 3.4. 최종 Loss 식  
KL Divergence 공식을 적용한 뒤 상수 항을 제거하면,  

$$
L_t = \mathbb{E}_{x_0,\epsilon}\bigl[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\bigr]
$$  

를 최소화하는 방식으로 최종 Loss를 정의합니다.  

***

## 4. 모델 구성 예시  
PyTorch로 DDPM 구성의 핵심 코드를 간략히 소개합니다.

```python
import torch
import torch.nn as nn

class SimpleDDPM(nn.Module):
    def __init__(self, betas, model):
        super().__init__()
        self.betas = betas
        self.alphas = 1 - betas
        self.alphas_cum = torch.cumprod(self.alphas, dim=0)
        self.model = model  # UNet 같은 네트워크

    def forward(self, x0):
        T = len(self.betas)
        t = torch.randint(0, T, (x0.size(0),), device=x0.device)
        alpha_t = self.alphas_cum[t].view(-1,1,1,1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_t)*x0 + torch.sqrt(1-alpha_t)*noise
        pred_noise = self.model(xt, t)
        return nn.MSELoss()(pred_noise, noise)
```

- **betas**: 노이즈 스케줄  
- **model**: UNet 구조 등 시간 정보를 활용해 노이즈를 예측  

***

## 5. 마무리  
Diffusion Model과 DDPM의 수식 유도 과정을 따라가며 **ELBO 변환**, **KL Divergence 단순화**, **네트워크 학습 Loss**로 이어지는 흐름을 익혔습니다. 마지막으로 제시한 예시 코드를 바탕으로 자신만의 DDPM을 구현해 보세요.  
이해하기 어려운 부분은 차근차근 수식 유도 과정을 직접 펜과 종이에 써보며 확인하는 것을 추천합니다.

[1](https://xoft.tistory.com/33)

https://xoft.tistory.com/33
