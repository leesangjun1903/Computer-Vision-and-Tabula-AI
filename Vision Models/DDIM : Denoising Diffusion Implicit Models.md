# DENOISING DIFFUSION IMPLICIT MODELS

DDPM을 통해 noise를 걷어내며 sample을 생성해가는 형태의 generative Markov Chain Process를 배울 수 있었다.  
그러나 이 Process는 일일히 sampling을 해야하는 형태로 매우 많은 step이 필요하고 one step에 끝나는 GAN에 비해 크게 느렸다.  
따라서 본 논문에서는 process를 non-markovian으로 일반화한 DDIM을 제시한다.

# Abstract
DDPM은 adversarial training 없이도 고품질 image generation이 가능함을 보였다.  
그러나 Sample을 생성해내기 위해 많은 step(시간이 오래걸림)의 Markov Chain Process가 필요하다.  
Sampling을 가속화하기 위해, Denoising Diffusion Implicit Models(DDIMs)을 제시하는데, 좀 더 빠르게 sample을 생성하기 위해 non-markovian diffusion process로 DDPM을 일반화한다.  
non-Markovian process를 통해 좀더 deterministic한 generative process를 학습시킬 수 있으며, high quality의 sample을 보다 빠르게 생성할 수 있게 되었다.  
DDPM 대비 속도가 10배에서 50배 빠름을 입증했다.

# Introduction
Deep Generation 모델은 high quality 이미지 생성의 여러 도메인에서 능력을 입증했다.  
GAN은 likelihood-based methods인 VAE와 flow 모델 대비 더 좋은 퀄리티를 보였으나, 학습의 안정화를 위해 매우 소수의 optimization과 구조를 사용할 수 밖에 없고, data distribution의 modes를 cover하는 데에 실패했다.  
DDPM과 NCSN(Noise Conditional Score Network)는 GAN에 상응할만한 high quality sample들을 adversarial training 없이도 보였는데, Generative Markov Chain Process를 사용하여 Noise로 부터 Sample을 생성하는 방식이다.  
다만, 이 방식은 high quality sample을 생성하는 데에 많은 iteration이 필요하고, 이로인해 one pass로 생성해내는 GAN보다 훨씬 느리다.  
DDPM과 GAN의 효율성 차이를 줄이기 위해, 우리는 DDPM과 유사한(same objective function으로 학습하는) DDIM을 제시한다.  
DDPM에서 사용된 forward diffusion process를 Markovian에서 계속되는 reverse generative Markov chain에 적합한 non-Markovian으로 일반화하였다.  
이는 sample quality를 조금만 손해보면서 sample efficiency(샘플링 속도)를 비약적으로(10x에서 100x로) 향상시킬 수 있다.  
(샘플링을 10배에서 100배까지 가속화해도 DDPM보다 우수한 생성 품질을 갖는다.)
또한 DDIM은 “consistency” property에서도 DDPM보다 우수한데, 만약 같은 initial latent variable에서 생성을 시작했다면 sample들은 비슷한 high-level feature를 갖게 된다.  
(동일한 초기 latent 변수로 시작하여 다양한 길이의 Markov chain으로 여러 샘플을 생성하면 샘플들이 높은 일관성을 가진다.)  
또한 이러한 DDIM의 “consistency” 때문에, sematically meaningful image interpolation이 가능하다.  
(DDPM은 이미지 space 근처에서 보간해야 하지만 DDIM은 일관성이 높아 초기 latent 변수를 조작하여 의미적으로 유의미한 보간이 가능하다.)  

# Background - DDPM
데이터의 분포 $q(x_0)$
가 주어질 때 모델 분포 $p_\theta (x_0)$ 가 $q(x_0)$ 
를 근사하도록 학습한다.

$$ \begin{equation}
p_\theta (x_0) = \int p_\theta (x_{0:T}) dx_{1:T}, \quad \quad p_\theta (x_{0:T}) := p_\theta (x_T) \prod_{t=1}^T p_\theta^{(t)} (x_{t-1} | x_t)
\end{equation} $$

파라미터 $θ$
는 variational lower bound

```math
\begin{equation}
\max_{\theta} \mathbb{E}_{q(x_0)} [\log p_\theta (x_0)] \le \max_{\theta} \mathbb{E}_{q(x_0, x_1, \cdots, x_T)} [\log p_\theta (x_{0:T}) - \log q(x_{1:T} | x_0)]
\end{equation}
```

을 최대화시키는 방향으로 학습된다. 

$q(x_{1:T} \vert x_0)$ 는 잠재 변수에 대한 inference distribution이며, DDPM은 $q(x_{1:T} \vert x_0)$
을 고정시키고 학습을 진행한다. 또한 감소 수열 $\alpha_{1:T} \in (0,1]^T$
로 매개변수화된 Gaussian transition이 있는 다음의 Markov chain을 사용하였다.

```math
\begin{equation}
q (x_{1:T} | x_0) := \prod_{t=1}^T q (x_t | x_{t-1}), \quad q (x_t | x_{t-1}) := \mathcal{N} \bigg(\sqrt{\frac{\alpha_t}{\alpha_{t-1}}} x_{t-1}, \bigg( 1 - \frac{\alpha_t}{\alpha_{t-1}} \bigg) I \bigg)
\end{equation}
```
(실제 DDPM 논문에서는 $α_t$
대신 
$\bar{\alpha}_t$ 로 표기)

$q (x_t \vert x_{t-1})$ 를 forward process라 한다. 또한, 
$x_T$ 에서 $x_0$ 로 샘플링하는 Markov chain $p_\theta (x_{0:T})$를 generative process라 하며 
이는 reverse process $q(x_{t-1} \vert x_t)$
로 근사된다.  
Forward process에 대하여 
```math
\begin{equation}
q (x_t | x_0) := \int q(x_{1:t} | x_0) dx_{1:(t-1)} = \mathcal{N} (x_t ; \sqrt{\alpha_t} x_0, (1-\alpha_t)I)
\end{equation}
```
가 성립하기 때문에 
$x_t$ 를 
$x_0$ 와 noise 변수 
$ϵ$ 의 선형 결합
```math
\begin{equation}
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N} (\textbf{0}, I)
\end{equation}
```
으로 표현할 수 있다. 
$α_T$ 를 0에 충분히 가깝게 설정하면 임의의 
$x_0$ 에 대하여 $q (x_T \vert x_0)$ 는 표준 가우시안 분포로 수렴한다.  
따라서 $p_\theta (x_T) := \mathcal{N} (\textbf{0},I)$ 로 설정하는 것은 자연스럽다.  
모든 조건문이 학습 가능한 평균 함수와 고정된 분산을 갖는 가우시안으로 모델링되면 다음 식으로 단순화할 수 있다.  

```math
\begin{equation}
L_\gamma (\epsilon_\theta) := \sum_{t=1}^T \gamma_t \mathbb{E}_{x_0 \sim q(x_0), \epsilon_t \sim \mathcal{N} (\textbf{0}, I)}
\bigg[ \| \epsilon_\theta^{(t)} (\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon_t) - \epsilon_t \|_2^2 \bigg], \quad
\epsilon_\theta := \{\epsilon_\theta^{(t)}\}_{t=1}^T
\end{equation}
```
$\gamma := [\gamma_1, \cdots, \gamma_T]$ 는 $\alpha_{1:T}$ 에 의존하는 양의 계수이다.  
DDPM은 생성 성능을 최대화하기 위해 $\gamma = \textbf{1}$ 로 두었다.

# VARIATIONAL INFERENCE FOR NON-MARKOVIAN FORWARD PROCESSES

![image](https://github.com/user-attachments/assets/bf3debf6-e21b-45d4-8588-f7ba4c687564)

생성 모델이 inference process의 역과정으로 근사되므로 생성 모델에 필요한 iteration의 수를 줄이기 위해 inference process를 다시 생각해야 한다.  
여기서 중요하는 것은 DDPM 목적 함수 $L_\gamma$가 
주변 분포(Marginal Distribution) $q(x_t \vert x_0)$ 에만 의존하며 
결합 분포 $q(x_{1:T}\vert x_0)$에는 직접적으로 의존하지 않는다는 것이다.  
같은 주변 분포에 대해서 수 많은 결합 분포가 존재하기 때문에 non-Markovian인 새로운 inference process가 필요하며 이에 대응되는 새로운 generative process가 필요하다.  
또한 이 non-Markovian inference process는 DDPM의 목적 함수와 같은 목적 함수를 가진다는 것을 보일 수 있다.

## Non-Markovian forward processes
실수 벡터 $\sigma \in \mathbb{R} _{\ge 0}^T$
에 대한 

inference distribution $q_\sigma (x_{1:T} \vert x_0)$ 은 다음과 같다.
```math
\begin{equation}
q_\sigma (x_{1:T} | x_0) := q_\sigma (x_T | x_0) \prod_{t=2}^T q_\sigma (x_{t-1} | x_t, x_0) \\
\textrm{where} \quad q_\sigma (x_T | x_0) = \mathcal{N} (\sqrt{\alpha_t} x_0, (1-\alpha_t)I)
\end{equation}
```

모든 
$t > 1$
에 대하여
```math
\begin{equation}
q_\sigma (x_{t-1} | x_t, x_0) = \mathcal{N} \bigg( \sqrt{\alpha_{t-1}} x_0  + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 I \bigg) 
\end{equation}
```

이다.  
모든 $t$ 에 대하여 
$q_\sigma (x_t \vert x_0) = \mathcal{N} (\sqrt{\alpha_t} x_0, (1-\alpha_t)I)$ 를 보장하기 위하여 평균 함수가 위와 같이 선택되었다.  
따라서 평균 함수는 의도한대로 주변 분포와 일치하는 결합 분포를 정의한다.

베이즈 정리에 의해 forward process는
```math
\begin{equation}
q_\sigma (x_t | x_{t-1}, x_0) = \frac{q_\sigma (x_{t-1} | x_t, x_0) q_\sigma (x_t | x_0)}{q_\sigma (x_{t-1} | x_0)}
\end{equation}
```
이며 이 또한 가우시안 분포이다.  
각각의 $x_t$ 가 $x_{t−1}$ 과 $x_0$ 모두에 의존하므로 DDIM의 forward process는 더 이상 Markovian이 아니다.  
$σ$ (분산) 의 크기로 얼마나 forward process가 확률적인지를 조절할 수 있으며 
$σ → 0$ 일 때 어떤 $t$ 에 대해 
$x_0$ 와 
$x_t$ 를 알면 고정된 
$x_{t−1}$ 를 알 수 있는 극단적인 경우에 도달한다.

## Generative Process and Unified Variational Inference Objective
다음으로 각 $p_\theta^{(t)} (x_{t-1} \vert x_t)$ 가 
$q_\sigma (x_{t-1} \vert x_t, x_0)$ 에 대한 지식을 활용하는 
학습 가능한 generative process $p_\theta (x_{0:T})$
를 정의한다.  
$x_t$ 가 주어지면 먼저 대응되는 
$x_0$ 를 예측하고 이를 이용하여 
$q_\sigma (x_{t-1} \vert x_t, x_0)$ 로 
$x_{t-1}$ 을 샘플링한다.

$x_0 \sim q(x_0)$ 와 $\epsilon_t \sim \mathcal{N} (\textbf{0},I)$ 에 대하여 $x_t$ 는 $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon_t$ 로 계산할 수 있다.  
그 다음 모델 $\epsilon_\theta^{(t)} (x_t)$ 가
$x_0$ 에 대한 정보 없이 
$x_t$ 로부터 
$\epsilon_t$ 를 예측한다.  
식을 다음과 같이 다시 세우면 주어진 
$x_t$ 에 대한 
$x_0$ 의 예측인 denoised observation을 예측할 수 있다.  

```math
\begin{equation}
f_\theta^{(t)} (x_t) := \frac{1}{\sqrt{\alpha_t}} (x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)})
\end{equation}
```

그런 다음 고정된 prior(T단계에서의 노이즈 상태) $p_\theta (x_T) = \mathcal{N} (\textbf{0}, I)$ 에 대한 generative process를 다음과 같이 정의할 수 있다.  

```math
\begin{equation}
p_\theta^{(t)} (x_{t-1} | x_t) = \cases{\mathcal{N}(f_\theta^{(t)} (x_1), \sigma_1^2 I) & t = 1 \\ q_\sigma (x_{t-1} | x_t, f_\theta^{(t)} (x_t)) & t > 1}
\end{equation}
```
$q_\sigma (x_{t-1} \vert x_t, f_\theta^{(t)} (x_t))$ 는 위에서 정의한 $q_\sigma (x_{t-1} \vert x_t, x_0)$ 의 $x_0$ 대신 $f_\theta^{(t)} (x_t)$ 를 대입하여 사용할 수 있다.  
Generative process가 
모든 $t$ 에서 성립하도록
$t = 1$ 인 경우에 약간의 Gaussian noise를 추가한다.  

파라미터 
$θ$
는 다음 목적 함수로 최적화된다.

```math
\begin{aligned}
J_\sigma (\epsilon_\theta) & := \mathbb{E}_{x_{0:T} \sim q_\sigma (x_{0:T})} [\log q_\sigma (x_{1:T} | x_0) - \log p_\theta (x_{0:T})] \\
&= \mathbb{E}_{x_{0:T} \sim q_\sigma (x_{0:T})} \bigg[ \log \bigg( q_\sigma (x_T | x_0) \prod_{t=2}^T q_\sigma (x_{t-1} | x_t, x_0) \bigg)
- \log \bigg( p_\theta (x_T) \prod_{t=1}^T p_\theta^{(t)} (x_{t-1} | x_t) \bigg) \bigg] \\
&= \mathbb{E}_{x_{0:T} \sim q_\sigma (x_{0:T})} \bigg[ \log q_\sigma (x_T | x_0) + \sum_{t=2}^T \log q_\sigma (x_{t-1} | x_t, x_0)
- \sum_{t=1}^T \log p_\theta^{(t)} (x_{t-1} | x_t) - \log p_\theta (x_T) \bigg]
\end{aligned}
```
$J_σ$ 의 정의를 보면 
$σ$ 에 따라 목적 함수가 다르기 때문에 다른 모델이 필요하다는 것을 알 수 있다.  

이 목적 함수를 정리하면 아래와 같이 정리할 수 있다.  
모든 $\sigma > 0$ 에 대하여 $J_\sigma = L_\gamma + C$ 인 $\gamma \in \mathbb{R}_{\ge 0}^T$ 와 $C \in \mathbb{R}$ 가 존재한다.  

Variational objective $L_γ$ 의 특별한 점은 $\epsilon_\theta^{(t)}$ 가 
다른 $t$ 에서 공유되지 않는 경우 
$\epsilon_\theta^{(t)}$ 에 대한 최적해가 가중치 $γ$ 에 의존하지 않는다는 것이다.  

이러한 성질은 두 가지 의미를 갖는다.
- DDPM의 variational lower bound에 대한 목적 함수로 
$L_1$ 을 사용하는 것이 가능하다.
$J_σ$ 가 일부 
$L_γ$ 와 같이 때문에 
$J_σ$ 의 최적해는 
$L_1$ 의 해와 동일하다.

# Sampling from Generalized Generative Processes
Markovian process를 위한 generative process뿐만 아니라 non-Markovian process를 위한 generative process도 
$L_1$ 으로 학습할 수 있다.  
따라서 pre-trained DDPM을 새로운 목적 함수에 대한 해로 사용할 수 있으며 
$σ$ 를 변경하여 필요에 따라 샘플을 더 잘 생성하는 generative process를 찾는 데 집중할 수 있다.

## Denoising Diffusion Implicit Models
다음 식에서 $x_t$ 로부터 
$x_{t−1}$를 생성할 수 있다.

```math
\begin{aligned}
x_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\bigg( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta^{(t)} (x_t)}{\sqrt{\alpha_t}} \bigg)}_{\textrm{predicted } x_0}
+ \underbrace{\sqrt{1-\alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)} (x_t)}_{\textrm{direction pointing to } x_t}
+ \underbrace{\sigma_t \epsilon_t}_{\textrm{random noise}}
\end{aligned}
```
$\epsilon_t \sim \mathcal{N} (0, I)$ 는 $x_t$ 에 독립적인 가우시안 노이즈이며 $\alpha_0 := 1$ 로 정한다.  
$σ$ 를 변경하면 같은 모델 $ϵ_θ$ 를 사용하여도 generative process가 달라지기 때문에 모델을 다시 학습하지 않아도 된다.  

모든 $t$ 에 대하여
```math
\begin{equation}
\sigma_t = \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}} \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}}
\end{equation}
```
로 두면 forward process가 Markovian이 되며 generative process가 DDPM이 된다.

모든 $t$ 에 대하여 $\sigma_t = 0$ 으로 두면, $x_{t-1}$ 와 $x_0$ 에 대하여 forward process가 deterministic해진다.  
이 경우 $x_T$ 부터 $x_0$ 까지 모두 고정되어 샘플링되기 때문에 모델이 implicit probabilistic model이 된다.  
이를 DDPM 목적 함수로 학습된 implicit probabilistic model이기 때문에 Denoising Diffusion Implicit Model (DDIM)이라 부른다.

## Accelerated generation processes
