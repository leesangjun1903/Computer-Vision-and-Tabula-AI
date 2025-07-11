# Deep Unsupervised Learning using Nonequilibrium Thermodynamics

# DPM

# Abs
반복적인 forward diffusion process로 data 분포를 파괴시키며, reverse diffusion process로 data 분포를 복원한다.  
이렇게 함으로써, flexible하고 tractable한 생성 모델 학습이 가능하다. (학습이 빠르고, sampling이 용이하다.)

# Introduction
Probabilistic 모델은 tractability와 flexibility가 trade-off 관계를 갖는다.

Tractable: Gaussian, Laplace 분포와 같이 해석 측면에서 유용하지만 custom data 분포를 표현하기 어렵다.  
Flexible: custom data 분포에 fitting시키기는 용이하지만 tractable하지 못한 경우가 많다.  
이 경우 cost가 expensive한 Monte Carlo process를 통해 intractable 문제를 해결해왔음  

생성 모델은 tractability와 flexibility 사이에서 trade-off가 있다.  
Tractable한 모델은 가우시안 분포처럼 수치적으로 계산이 되며 데이터에 쉽게 맞출 수 있다.  
하지만, 이러한 모델은 복잡한 데이터셋을 적절하게 설명하기 어렵다.  
반대로, flexible한 모델은 임의의 데이터에 대해서도 적절하게 설명할 수 있지만 학습하고 평가하고 샘플을 생성하는데 일반적으로 매우 복잡한 Monte Carlo process가 필요하다.

## Diffusion probabilistic models
저자는 다음 4가지가 가능한 probabilistic model을 정의하는 새로운 방법을 제안한다.

1. 굉장히 유연한 모델 구조
2. 정확한 샘플링
3. 사후 계산 등을 위해 다른 분포들과의 쉬운 곱셈
4. Model log likelihood와 각 state의 확률 계산이 쉽다.
저자는 diffusion process를 사용하여 가우시안 분포처럼 잘 알려진 분포에서 목표로 하는 데이터의 분포로 점진적으로 변환하는 generative Marcov chain을 사용한다.
Marcov chain이기 때문에 각 상태는 이전 상태에 대해 독립적이다.
Diffusion chain의 각 step의 확률을 수치적으로 계산할 수 있으므로 전체 chain도 수치적으로 계산할 수 있다.

확산 과정에 대한 작은 변화를 추정하는 것이 학습 과정에 포함된다.  
이는 수치적으로 정규화할 수 없는 하나의 potential function을 사용하여 전체 분포를 설명하는 것보다 작은 변화를 추정하는 것이 더 tractable하기 때문이다.  
또한 임의의 데이터 분포에 대해서 diffusion process가 존재하기 때문에 이 방법은 임의의 형태의 데이터 분포를 표현할 수 있다.

다시 설명하면, 기존 방법 대비 DPM이 갖는 장점은 다음과 같다.

- Flexible한 모델 구조
- 정확한 sampling
- 다른 분포와의 multiplication이 쉽다. (posterior 계산에 용이)
- Log likelibood, probability estimation cost가 expensive하지 않다. (Monte Carlo에 비해)
DPM은 generative Markov chain이다.(gaussian/binomial -> target data distribution)

Gaussian 등 잘 알려진 분포로부터 sampling함으로써 tractability를 확보한다.  
매 step에서 small perturbation을 estimate하기 때문에, 한 번에 전체를 예측하는 것보다 tractable하다.  
Iterative diffusion process로 target data 분포에 fitting시킴으로써 flexibility도 확보한다.

## Relationship to other work
기존 연구에서는 variational learning/inference로 flexibility를 챙기고, approximate posterior로 tractability를 확보했다.  
이러한 연구들과 DPM의 차별점은 다음과 같다.

Variational method보다 sampling의 중요성이 낮다.  
Posterior 연산이 쉽다.  
Inference/generation process가 same functional form이다.  
(예를 들어 VAE에서는 variational inference로 학습하고 reparameterization trick으로 sampling을 활용한 generation이 가능하게 하는데, 이러한 비대칭성으로 인해 challenging하게 된다는 관점?)
각 time stamp마다 layer를 두어 1000개의 layer를 가지며, 각각의 time stamp마다 upper/lower bound 정의 가능  
Probability model을 학습했던 연구들은 다음과 같다.  

- Wake-sleep
- Generative stochastic networks
- Neural autoregressive distribution estimator
- Adversarial networks (GAN)
- Mixtures of conditional gaussian scale mixtures (MCGSM)

Physics idea 관련 연구들은 다음과 같다.
- Annealed Importance Sampling (AIS)
- Langevin dynamics
- Kolmogorov forward and backward equation

# Algorithm
![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/fig1.webp)

첫 번째 줄은 forward diffusion process를 의미한다.  
2-d swiss roll data가 시간이 증가됨에 따라서 gaussian 분포로 바꾸어가는 것을 볼 수 있다.  
두 번째 줄은 reverse diffusion process이다.  
Gaussian 분포로부터 원본 데이터 분포를 나름 잘 복원하는 것을 볼 수 있다.  
마지막 줄은 각 time step에서 데이터 분포의 평균(Expectation값들)의 변화 방향을 시각화한 것이다.  
t=T에서 t=0에 가까워질수록, 더 강하게 원본 2-d data 분포를 복원해내려는 모습을 볼 수 있다.

## Forward Trajectory
학습할 데이터의 분포 : $q(x^{(0)})$  
반복적으로 적용되는 Markov diffusion kernel : $T_{\pi}(y{\mid}y{\prime};{\beta})$  
최종적으로 변화될 데이터의 분포 : $\pi(y)$

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq1.webp), $\beta$ : Diffusion rate

그리고 Markov diffusion kernel에서 y를 x와 t로 표현하면, forward process q와 의미상 동일해진다.

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq2.webp)

즉, markov 성질을 활용하여 forward trajectory는 수식 3과 같이 표현할 수 있다.

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq3.webp)

## Reverse Trajectory
Forward trajectory와 같은 방식으로 수식 1을 reverse process로 표현하면 수식 5와 같다.

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq5.webp)

참고로 continuous diffusion에서 diffusion rate가 매우 작을 경우, forward process의 역연산이 identical functional form을 갖는 것이 증명되었다고 한다.
(On the Theory of Stochastic Processes, with Particular Reference to Applications (Feller, 1949))

즉, forward process가 가우시안/이항 분포를 따르는 경우, reverse process도 가우시안/이항 분포를 따른다.  
이는 forward process q를 모방하는 reverse process p를 모델링할 수 있는 근거가 된다.  

그리고 (아직 objectvie에 대한 언급은 없지만, forward를 모방하는) reverse trajectory는 gaussian 분포의 경우 mean/covariance를 예측하고 binomial 분포의 경우 bit flip probability를 예측하는 방식으로 학습한다.  
즉, 상기한 값을 예측하는 cost * time이 training cost에 해당한다.

## Model Probability
DPM의 목표는 원본 데이터 분포를 복원하는 것으로 $p(x^{(0)})$는 수식 6과 같이 표현할 수 있다.

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq6.webp)

수식 6은 intractable하기 때문에(계산 불가능하기 때문에), 수식 3과 수식 5에서 정의한 forward/reverse trajectory를 대입하여 수식 7~9까지 전개한다.  

그러므로 대신 forward에 대해 평균을 낸 forward와 reverse의 상대적 확률을 계산한다. (Annealed importance sampling과 Jarzynski 등식에서 힌트를 얻었다고 함)  

```math
\begin{aligned}
p(x^{(0)}) &= \int dx^{(1 \cdots T)} p(x^{(0 \cdots T)})
\frac{q(x^{(1 \cdots T)} | x^{(0)})}{q(x^{(1 \cdots T)} | x^{(0)})} \\
&= \int dx^{(1 \cdots T)} q(x^{(1 \cdots T)} | x^{(0)})
\frac{p(x^{(0 \cdots T)})}{q(x^{(1 \cdots T)} | x^{(0)})} \\
&= \int dx^{(1 \cdots T)} q(x^{(1 \cdots T)} | x^{(0)}) p(x^{(T)})
\prod_{t=1}^T \frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)} | x^{(t-1)})}
\end{aligned}
```

최종적으로 forward trajectory $q(x^{(1 \cdot \cdot \cdot T)}|x^{(0)})$ 로부터 sampling 한 번이면 된다.

$p(x^{(T)})$ : 초기값  
p / q 꼴은 2.2에서 언급했던 것처럼 diffusion rate가 매우 작은 경우 forward/reverse 모두 gaussian 분포를 따르기 때문에 쉽게 연산이 가능하다.  

저자들은 이러한 증명 과정을 두고 statistical physics 분야의 quasi-static process에 해당한다고 언급한다.

- Quasi-static process는 복잡한 문제를 단순화시켜 푸는 것을 의미하는데, 다음의 과정을 강조한 표현으로 생각된다.  
- Annealed importance sampling와 유사하게, 현재 data 분포에서 시작하여 intermediate 분포를 거쳐 target 분포로 나아간다는 점  
- Forward process와 reverse process가 동일한 분포를 따른다는 점  

## Training
![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq10.webp)

DPM은 log likeligood를 maximize한다.  
(따로 언급되어 있지는 않지만 cross-entropy 꼴과 유사하기에, forward process q를 target으로 reverse process p를 모델링하겠다는 접근으로도 보인다.)

이후 2.3의 수식 9를 대입한 다음,  

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq11.webp)

Jensen’s inequality에 의해 lower bound를 구할 수 있게 된다.

- $E[log(X)] \le log[E(X)]$ (log 함수는 concave)
- $E = E_{x^{(1 \cdot \cdot \cdot T)} \sim q}$
- $X = p(x^{(T)}) \Pi_{t=1}^{T} \frac{p(x^{(t-1)}{\mid}x^{(t)})}{q(x^{(t)}{\mid}x^{(t-1)})}$

```math
\begin{aligned}
L &\ge \int dx^{(0 \cdots T)} q(x^{(0 \cdots T)}) \log \bigg[
    p(x^{(T)}) \prod_{t=1}^T \frac{p(x^{(t-1)}|x^{(t)})}{q(x^{(t)} | x^{(t-1)})} \bigg] = K
\end{aligned}
```

이후 Appendix B에 의해 ELBO는 수식 13~14로 정리된다.  

```math
\begin{aligned}
L \ge K =& -\sum_{t=2}^T \int dx^{(0)} dx^{(t)} q(x^{(0)}, x^{(t)}) D_{KL} \bigg(
    q(x^{(t-1)} | x^{(t)}, x^{(0)}) \; || \; p(x^{(t-1)} | x^{(t)}) \bigg) \\
&+ H_q (X^{(T)} | X^{(0)}) - H_q (X^{(1)} | X^{(0)}) - H_p (X^{(T)})
\end{aligned}
```

Forward/reverse trajectory가 동일하기에 수식 13에서의 equality가 만족한다. (KLD term이 0이 되면서 entropy term만 남기 때문에 equality로 보는 것으로 추측)

또한, (entropy term은 diffusion step과 무관하기에) ELBO를 maximize하는 것이 reverse process p를 모델링하는 것과 같아진다.

엔트로피들과 KL divergence는 계산이 가능하므로 K는 계산이 가능하다.  
등호는 forward와 reverse가 같을 때 성립하므로 $\beta_t$가 충분히 작으면 L이 K와 거의 같다고 볼 수 있다.  

Reverse Markov transition을 찾는 학습은 lower bound를 최대화하는 것과 같다.

```math
\begin{aligned}
\hat{p} (x^{(t-1)} | x^{(t)}) = \underset{p(x^{(t-1)} | x^{(t)})}{{argmax}}  K
\end{aligned}
```

### Setting the Diffusion rate $\beta_t$
Forward trajectory에서 diffusion rate $\beta_t$ 값은 중요하다.   
가우시안 분포의 경우 먼저 lower bound K에 gradient ascent 알고리즘을 적용하여 diffusion schedule을 학습한다.  
이는 VAE와 마찬가지로 explicit한 방식이다. (참고로 first step인 β1의 경우 overfitting 방지를 위해 작은 상수 값으로 설정했고, lower bound K의 미분 계산 과정에서는 diffusion rate를 상수로 설정했다.)

다음으로 binomial 분포의 경우, 매 step마다 1/T 만큼 diffusion한다.($\beta_t = (T-t+1)^{-1}$)

## Multiplying Distributions and Computing Posteriors
단순 inference는 $p(x^{(0)})$로 표현할 수 있지만, denoising같이 second distribution을 활용하는 경우에 posterior 계산이 필요하고 그러려면 분포끼리 곱할 수 있어야 한다.  
이를 위해 기존 생성모델에서는 여러 테크닉을 적용했어야 한다.

반면 DPM에서는 second distribution을 단순 small perturbation으로 간주하거나, 아예 각 diffusion step에 곱하는 방식으로 손쉽게 해낼 수 있다.

- second distribution은 일종의 condition으로 생각됨
- $\tilde{p}(x^{(0)}) \propto p(x^{(0)}) r(x^{(0)})$

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/fig3.webp)

(a): example holdout data  
(b): (a) + gaussian noise (var=1)  
(c): generated by sampling from posterior over denoised images conditioned on (b)  
(b)의 noise를 reverse diffusion으로 제거하였더니 (a) 이미지와 유사하게 복원한다.  
(d): generated by diffusion model, (c)와 다른 이미지를 생성하는 결과로, init condition이 달라지면 다른 이미지가 생성됨을 보여준다.  

### Modified Marginal Distributions
$\tilde{Z}_t$ 를 추가하여 modified reverse process를 정의한다.
```math
\begin{equation}
\tilde{p} (x^{(t)}) = \frac{1}{\tilde{Z}_t} p(x^{(t)}) r(x^{(t)})
\end{equation}
```

$\tilde{p}(x^{(0 \cdots T)})$ : 수정된 reverse trajactory
$\tilde{Z}_t$ : normalizing constant 

### Modified Diffusion Steps
Reverse process의 markov kernel이 equilibrium condition을 따르기에, 수식 17로 표현할 수 있다.

```math
\begin{equation}
p (x^{(t)}) = \int dx^{(t+1)} p(x^{(t)} | x^{(t+1)}) p(x^{(t+1)}) 
\end{equation}
```

더 나아가, 저자들은 modified reverse process에서도 equilibrium condition을 따를 것이라고 가정한다.  
따라서 이렇게 전개할 수 있다.

```math
\begin{aligned}
\tilde{p} (x^{(t)}) &= \int dx^{(t+1)} \tilde{p} (x^{(t)} | x^{(t+1)}) \tilde{p} (x^{(t+1)}) \\
\frac{p(x^{(t)}) r(x^{(t)})} {\tilde{Z}_t} &= \int dx^{(t+1)} \tilde{p} (x^{(t)} | x^{(t+1)}) 
\frac{p(x^{(t+1)}) r(x^{(t+1)})} {\tilde{Z}_{t+1}} \\
p(x^{(t)}) &= \int dx^{(t+1)} \tilde{p} (x^{(t)} | x^{(t+1)})
\frac{\tilde{Z}_t r(x^{(t+1)})} {\tilde{Z}_{t+1} r(x^{(t)})} p(x^{(t+1)}) \\
\end{aligned}
```

만약 아래 수식 21이 성립한다면 수식 20을 만족할 수 있다.

```math
\begin{equation}
\tilde{p} (x^{(t)} | x^{(t+1)}) = p (x^{(t)} | x^{(t+1)}) \frac{\tilde{Z}_{t+1} r(x^{(t)})}{\tilde{Z}_t r(x^{(t+1)})}
\end{equation}
```

위 식이 정규화된 확률 분포가 아닐 수 있기 때문에 다음과 같이 정의한다.

```math
\begin{equation}
\tilde{p} (x^{(t)} | x^{(t+1)}) = \frac{1}{\tilde{Z}_t (x^{(t+1)})} p (x^{(t)} | x^{(t+1)}) r(x^{(t)})
\end{equation}
```

이 수식이 가능한 이유는 다음과 같다.

- $r({x^{(t)}})$ 는 small variance를 갖기 때문에, $\frac{r(x^{(t)})}{r(x^{(t+1)})}$ 을 small perturbation으로 간주할 수 있게 된다.
- 가우시안 분포에 대한 작은 pertubation은 평균에 영향을 주지만 normalization 상수에는 영향을 주지 않기 때문에 위 식과 같이 정의할 수 있다.

### Applying $r({x^{(t)}})$

만약 $r({x^{(t)}})$ 가 충분히 smooth 하다면 reverse diffusion kernel에 대한 small perturbance로 간주할 수 있고, 이는 $\tilde{p}$와 $p$가 거의 같아진다고 설명하고 있다.  

$r({x^{(t)}})$ 가 가우시안/이항 분포와 곱셈이 가능하다면, 그냥 reverse diffusion kernel과 곱해버리면 된다.  

### Choosing $r({x^{(t)}})$
![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq23.webp)

저자들이 step에 따른 scheuled r(x)도 실험해봤으나, constant로 두는 것 대비 이점이 없다고 판단했다.

## Entropy of Reverse Process
Forward를 알고 있기 때문에 reverse에서 각 step의 조건부 엔트로피에 대한 upper bound와 lower bound를 구할 수 있으므로 log likelihood에 대해 구할 수 있다.  

```math
\begin{equation}
H_q (X^{(t)}|X^{(t-1)}) + H_q (X^{(t-1)}|X^{(0)}) - H_q (X^{(t)}|X^{(0)}) \le H_q (X^{(t-1) }|X^{(t)}) \le H_q (X^{(t)}|X^{(t-1)})
\end{equation}
```

# Experiments
- Dataset: Toy data, MNIST, CIFAR10, Dead Leaf Images, Bark Texture Images
- 각 데이터셋에 대한 생성 및 impainting

Forward diffusion kernel과 reverse diffusion kernel은 다음과 같다.

```math
\begin{equation}
q(x^{(t)}|x^{(t-1)}) = \mathcal{N} (x^{(t)}; x^{(t-1)} \sqrt{1-\beta_t}, I \beta_t) \\,
p(x^{(t-1)}|x^{(t)}) = \mathcal{N} (x^{(t-1)}; f_\mu (x^{(t)}, t), f_\Sigma (x^{(t)}, t))
\end{equation}
```

$f_\mu$, $f_\sigma$ : MLP 로 학습  
$f_\mu, f_\sigma, \beta_{1 \cdots T}$ : 학습 대상

![](https://kimjy99.github.io/assets/img/dul-nt/dul-nt-table.PNG)

초기 데이터 분포의 영향을 제거한 K_L_null도 기록한 것이 인상적이다.

# Appendix
## Experimental Details
![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/figd.webp)

- Input image에 mean pooling을 통해 downsample한다.
- Multi-scale conv layer를 거친 feature map들을 다시 원본 사이즈로 upsample하고, 전부 더한다.
- 1x1 conv로 표현된 dense transform을 통해 temporal coefficient를 얻는다.
- 최종적으로 mean/covariance image를 얻는다.

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq62.webp)

convolution output y 에서 평균, 분산에 대한 convolution output 을 계산한다.  
$z^{\mu}_i$ : 1x1 conv를 거쳐 2 branch(temporal coefficient)로 나뉘는 부분  

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq63.webp)

$g_j(t)$는 bump function이며 softmax 함수 꼴이고, bump center인 τj는 (0, T) 범위를 갖는다.  

![](https://dongwoo-im.github.io/assets/img/posts/2023-10-02-DPM/eq64.webp)

위에서 구한 $z^{\mu}$, $z^{\Sigma}$ 를 바탕으로 각 pixel i에 대한 $\mu_i$, $\Sigma_{ii}$를 예측할 수 있게 된다.
