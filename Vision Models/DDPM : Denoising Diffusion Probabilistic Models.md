# Denoising Diffusion Probabilistic Models

# DDPM

# Abs
저희는 비평형 열역학의 고려 사항에서 영감을 얻은 잠재 변수 모델 클래스인 확산 확률 모델을 사용하여 고품질 이미지 합성 결과를 제시합니다.  
저희의 최상의 결과는 Langevin dynamics을 이용한 diffusion probabilistic model과 denoising score matching 사이의 연결에 따라 설계된 가중치 변동 경계에 대한 훈련을 통해 얻어지며, 저희 모델은 자연스럽게 autoregressive decoding의 일반화로 해석될 수 있는 점진적 손실 압축 방식을 선보입니다.  
unconditional CIFAR10 데이터 세트에서 저희는 9.46의 Inception 점수와 3.17의 최첨단 FID 점수를 얻었습니다.  
256x256 LSUN에서 ProgressiveGAN과 유사한 샘플 품질을 얻었습니다.  

# Introduction
간단히 결론부터 설명하면 DDPM은 주어진 이미지에 time에 따른 상수의 파라미터를 갖는 작은 가우시안 노이즈를 time에 대해 더해나가는데, image가 destroy하게 되면 결국 noise의 형태로 남을것이다. (normal distribution을 따른다.)  
이런 상황에서 normal distribution 에 대한 noise가 주어졌을때 어떻게 복원할 것인가에 대한 문제이다.  
그래서 주어진 Noise를 통해서 완전히 이미지를 복구가 된다면 image generation하는 것이 된다.  
이 논문에서는 diffusion probabilistic models의 과정을 보여준다.  
diffusion model은 유한한 시간 뒤에 이미지를 생성하는 variational inference을 통해 훈련된 Markov chain을 parameterized한 형태이다.  
Markov Chain은 이전의 샘플링이 현재 샘플링에 영향을 미치는 p(x_t-1|x_t) 형식을 의미한다.  
그래서 이 diffusion model에서의 한 방향에 대해서는 주어진 이미지에 작은 gaussian noise를 점진적으로 계속 더해서 완전히 image가 destroy 되게하는 과정을 의미한다.

이 논문은 diffusion model(리뷰)을 발전시킨 논문이다. Diffusion model은 parameterized Markov chain을 학습시켜 유한 시간 후에 원하는 데이터에 맞는 샘플을 만드는 모델이다.  
Forward process에서는 Markov chain이 점진적으로 noise를 추가하여 최종적으로 가우시안 noise로 만든다.  
반대로 reverse process는 가우시안 noise로부터 점진적으로 noise를 제거하여 최종적으로 원하는 데이터에 맞는 샘플을 만든다.  
Diffusion이 작은 양의 가우시안 noise로 구성되어 있기 때문에 샘플링 chain을 조건부 가우시안으로 설정하는 것으로 충분하고,  
간단한 신경먕으로 parameterize할 수 있다.

기존 diffusion model은 정의하기 쉽고 학습시키기 효율적이지만 고품질의 샘플을 만들지 못하였다.  
반면, DDPM은 고품질의 샘플을 만들 수 있을 뿐만 아니라 다른 생성 모델 (ex. GAN)보다 더 우수한 결과를 보였다.  
또한, diffusion model의 특정 parameterization이 학습 중 여러 noise 레벨에서의 denoising score matching과 비슷하며, 샘플링 중 Langevin dynamics 문제를 푸는 것과 동등하다는 것을 보였다.

그 밖의 특징으로는

1. DDPM은 고품질의 샘플을 생성하지만 다른 likelihood 기반의 모델보다 경쟁력 있는 log likelihood가 없다.
2. DDPM의 lossless codelength가 대부분 인지할 수 없는 이미지 세부 정보를 설명하는 데 사용되었다.
3. Diffusion model의 샘플링이 autoregressive model의 디코딩과 유사한 점진적 디코딩이라는 것을 보였다.

# Background

![](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0e46190c-ca06-48f9-b201-7377c8b31f18%2F%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-02-20_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_4.28.54.png?table=block&id=b6e28b39-c04c-4041-a44c-c201e417f5f3&cache=v2)

Diffusion에서 중요한 개념은 바로 'Stochastic Process' 이다.

이것은 time-dependent variables을 통해서 이루어진다는 것이다.

Diffusion을 간략하게 살펴보면, Backward, Forward process가 있다.
Backward process는 noise에서 이미지로 가는 것이고,
Forward process는 이미지에서 noise로 가도록 하는 것 이다.

여기서 Backward process를 training하는 것이 바로 Diffusion model인 것이다!

Diffusion model : $p_\theta (x_0) := \int p_\theta (x_{0:T}) dx_{1:T}$  
$x_{0:T}$ size = $x_0 \sim q(x_0 )$ size  
$p_\theta (x_{0:T})$ : Reverse Process  
$p(x_T ) = \mathcal{N} (x_T ; 0, I )$ 에서 시작하는 Gaussian transition으로 이루어진 Markov chain으로 정의된다.

$p_\theta (x_{0:T}) := p(x_T) \prod_{t=1}^T p_\theta (x_{t-1}|x_{t})$  
$p_\theta (x_{t-1}|x_{t}) := \mathcal{N} (x_{t-1} ; \mu_\theta (x_t , t), \Sigma_\theta (x_t , t))$

Diffusion model이 다른 latent variable model과 다른 점은 forward process 혹은 diffusion process라 불리는 approximate posterior $q(x_{1:T}|x_0)$ 가 $\beta_1, \cdots, \beta_T$ 에 따라 가우시안 noise를 점진적으로 추가하는 Markov chain이라는 것이다.  

$q (x_{1:T}|x_0) := \prod_{t=1}^T q (x_{t}|x_{t-1})$    
$q (x_{t}|x_{t-1}) := \mathcal{N} (x_{t} ; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$

학습은 negative log likelihood에 대한 일반적인 variational bound을 최적화하는 것으로 진행된다.

$L:= \mathbb{E} [-\log p_\theta (x_0)] \le \mathbb{E_q} \bigg[ -\log \frac{p_\theta (x_{0:T})}{q(x_{1:T}|x_0)} \bigg] \le \mathbb{E_q} \bigg[ -\log p(x_T) - \sum_{t \ge 1} \log \frac{p_\theta (x_{t-1}|x_t)}{q(x_t|x_{t-1})} \bigg]$

$beta_t$ 는 reparameterization으로 학습하거나 hyper-parameter로 상수로 둘 수 있다.  
또한 $beta_t$가 충분히 작으면 forward process와 reverse process가 같은 함수 형태이므로 reverse process의 표현력은 $p_\theta (x_{t-1}|x_t)$에서 가우시안 conditional의 선택에 따라 부분적으로 보장된다.

Forward process에서 주목할만한 것은 closed form으로 임의의 시간 t에서 샘플링 $xt$가 가능하다는 것이다.

$\alpha_t := 1-\beta_t, \quad \bar{\alpha_t} := \prod_{s=1}^t \alpha_s$  
$q(x_t | x_0) = \mathcal{N} (x_t ; \sqrt{ \bar{\alpha_t}} x_0 , (1-\bar{\alpha_t})I)$

한 번에 샘플링이 가능하므로 stochastic gradient descent을 이용하여 효율적인 학습이 가능하다.  
L을 다음과 같이 다시 쓰면 분산 감소로 인해 추가 개선이 가능하다.

```math
\begin{equation} L = \mathbb{E}_q \bigg[ \underbrace{D_{KL} (q(x_T | x_0) \; || \; p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{KL} (q(x_{t-1} | x_t , x_0) \; || \; p_\theta (x_{t-1} | x_t))}_{L_{t-1}} \underbrace{- \log p_\theta (x_0 | x_1)}_{L_0} \bigg] \end{equation}
```

위 식은 KL divergence으로 forward process posterior (ground truth)와 $p_\theta (x_{t-1} \vert x_t)$ 를 직접 비교하며, 이는 tractable하다. 두 가우시안 분포에 대한 KL Divergence는 closed form으로 된 Rao-Blackwellized 방식으로 계산할 수 있기 때문에 L을 쉽게 계산할 수 있다.

$q(x_{t-1} \vert x_t, x_0)$는 다음 식으로 계산할 수 있다.

```math
\begin{aligned}
q (x_{t-1} | x_t, x_0) &= \mathcal{N} (x_{t-1} ; \tilde{\mu_t} (x_t, x_0), \tilde{\beta_t} I), \\
\rm{where} \quad \tilde{\mu_t} (x_t, x_0) &:= \frac{\sqrt{{1} \bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
\quad \rm{and} \quad \tilde{\beta_t} := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
\end{aligned}
```
$q(x_{t-1} \vert x_t)$ 는 계산하기 어렵지만 $q(x_{t-1} \vert x_t, x_0)$는 계산할 수 있다.  
즉, $x_t$에서 $x_{t-1}$을 바로 구하는 것은 어렵지만 $x_0$를 조건으로 주면 쉽게 구할 수 있다.

# Diffusion models and denoising autoencoders
마지막 목표인 DDPM까지 왔다!  
사실 요즘에는 DDPM보다 DDIM을 더 자주 사용하는데, 추후 논문리뷰로 남겨놓겠다.  
DDPM은 기존의 diffusion 방식을 보다 빠르고 적은 cost로 training시킬 수 있는 trick?을 소개한다.

## Forward Process and LT
기존의 DIffusion model을 다시 생각해보자.  
Forward process와 backward process가 있었다.  
여기서 backward process는 당연히 trainable하다. 또한 Forward process도 trainable 하다.  
Forward process에서 중요한 parameter인 β는 trainable하게 훈련되었다. 

그러나 DDPM에서는 β를 constant하게 유지하여서 Forward process를 훈련할 필요가 없게 만들었고, 이 결과 위에서 마지막으로 봤던 First loss term(LT)를 무시할 수 있게 되었다.

## Reverse Process and L1:T−1
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbIveKV%2Fbtsp7U1i99v%2F6EDE0SoGRB1JrV0zVNUxek%2Fimg.png)

Reverse process (1<t<T)인 구간에서 DDPM은 2가지 방법을 제시한다.  
첫번째는 분산과 관련된 조건이다.  
$\Sigma_\theta (x_t, t) = \sigma_t^2 I$로 두었으며, $\sigma_t$는 학습하지 않는 $t$에 의존하는 상수이다. 
Reverse process에서 sampling을 할 때, gaussian distribution의 분산을 위의 2가지 방식으로 설정하는 것이다.  
첫번째 경우에는 x0가 N(0,I)로 최적화되고, 두번째 경우에는 one point로 최적화된다고 언급하고 있다.  

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbc7fFv%2Fbtsp3mEtbOM%2FeXKbQmvvwZ3UC5uEeFoIq0%2Fimg.png)

두번째는 평균과 reparameterizing을 이용하여 loss function을 design하는 것이다.

$\mu_\theta (x_t, t)$ 를 나타내기 위해 특정 parameterization을 제안한다.  
$p_\theta (x_{t-1}\vert x_t) = \mathcal{N} (x_{t-1} ; \mu_\theta (x_t, t), \sigma_t^2 I)$에 대하여 다음과 같이 쓸 수 있다.  

```math
\begin{equation}
L_{t-1} = \mathbb{E}_q \bigg[ \frac{1}{2\sigma_t^2} \| \tilde{\mu_t} (x_t, x_0) - \mu_\theta (x_t, t) \|^2 \bigg] + C
\end{equation}
```

Equation (4)에서 소개되었던(위에서도 언급함) αt=(1-βt)를 이용하여서 reparameterizing을 하는 것이다.

Equation (10)의 경우 Equation (7)에 근거하여서 변환할 수 있다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcGOz6X%2Fbtsp7pgmxYG%2Ff6EwgbA5VjyJKQlJn38z7k%2Fimg.png)

$\epsilon_\theta$는 $x_t$로부터 $\epsilon$을 예측하는 function approximator이다.  

$x_{t-1} \sim p_\theta (x_{t-1} \vert x_t)$의 샘플링은 다음과 같이 진행된다.  

```math
\begin{equation}
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \bigg( x_t - \frac{\beta_t}{\sqrt{1- \bar{\alpha}_t}} \epsilon_\theta (x_t, t) \bigg) + \sigma_t z, \quad z \sim \mathcal{N} (0, I)
\end{equation}
```
정리하면, 학습과 샘플링 과정은 다음 알고리즘과 같이 진행된다.

![](https://kimjy99.github.io/assets/img/ddpm/ddpm-algorithm.PNG)

샘플링 과정 (Algorithm 2)은 데이터 밀도의 학습된 기울기로 $\epsilon_\theta$을 사용하는 Langevin 역학과 유사하다.

추가로, parameterization을 한 $\mu_\theta$를 objective function 식에 대입하면

```math
\begin{equation}
L_{t-1} - C = \mathbb{E}_{x_0, \epsilon} \bigg[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta (\sqrt{{1} \bar{\alpha}_t} + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \bigg]
\end{equation}
```

인데, 이는 여러 noise 레벨에서의 denoising score matching과 유사하며 Langevin-like reverse process의 variational bound과 같다.

## Data scaling, reverse process decoder, and L0
이미지 데이터는 0부터 255까지의 정수로 되어있고 -1부터 1까지의 실수로 선형적으로 스케일링되어 주어진다.  
이를 통해 신경망(reverse process)이 표준 정규 prior $p(x_T)$에서 시작하여 언제나 스케일링된 이미지로 갈 수 있게 한다.  
이산적인(discrete) log likelihood를 얻기 위하여 reverse process의 마지막 항 $L_0$를 가우시안 분포 $\mathcal{N} (x_0; \mu_\theta (x_1, 1), \sigma_1^2 I)$에서 나온 독립적인 discrete decoder로 설정하였다.

```math
\begin{aligned}
p_\theta (x_0 | x_1) &= \prod_{i=1}^D \int_{\delta_{-} (x_0^i)}^{\delta_{+} (x_0^i)} \mathcal{N} (x; \mu_\theta^i (x_1, 1), \sigma_1^2) dx \\
\delta_{+} (x) &= \begin{cases}
  \infty & (x = 1) \\
  x + \frac{1}{255} & (x < 1)
\end{cases}
\quad &\delta_{-} (x) = \begin{cases}
  -\infty & (x = -1) \\
  x - \frac{1}{255} & (x > -1)
\end{cases}
\end{aligned}
```
$D$는 데이터의 dimensionality이며 $i$는 각 좌표를 나타낸다.

## Simplified training objective
저자들은 training objective를 다음과 같이 simplification하였다.

```math
\begin{equation}
L_{\rm{simple}} := \mathbb{E}_{t, x_0, \epsilon} \bigg[ \| \epsilon - \epsilon_\theta (\sqrt{ \bar{\alpha}_t} + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \bigg]
\end{equation}
```

여기서 $t$는 1과 T 사이에서 uniform하다.  
Simplified objective는 기존의 training objective에서 가중치를 제거한 형태이다.  
이 가중치항은 $t$에 대한 함수로, $t$가 작을수록 큰 값을 가지기 때문에 $t$가 작을 때 더 큰 가중치가 부여되어 학습된다.  
즉, 매우 작은 양의 noise가 있는 데이터에서 noise를 제거하는데 집중되어 학습된다.  
따라서 매우 작은 $t$에서는 학습이 잘 진행되지만 큰 $t$에서는 학습이 잘 되지 않기 때문에 가중치항을 제거하여 큰 $t$에서도 학습이 잘 진행되도록 한다.

실험을 통하여 가중치항을 제거한 $L_{\rm{simple}}$이 더 좋은 샘플을 생성하는 것을 확인했다고 한다.

# Experiments
- 모든 실험에서 $T = 1000$
- $\beta_t$는 $\beta_1 = 10^{-4}$에서 $\beta_T = 0.02$로 선형적으로 증가
- $x_T$에서 signal-to-noise-ratio는 최대한 작게 $(L_T = D_{KL}(q(x_T\vert x_0) \; | \; \mathcal{N}(0,I)) \approx 10^{-5})$
- 신경망은 group normalization을 사용하는 U-Net backbone (unmasked PixelCNN++과 비슷한 구조)
- Transformer sinusoidal position embedding으로 모델에게 시간 $t$를 입력
- 16x16 feature map에서 self-attention 사용

# Results
![](https://kimjy99.github.io/assets/img/ddpm/ddpm-table1.PNG)

# Conclusion

