
# WGAN : Wasserstein GAN | Image generation

## 1. 배경 및 동기
전통적인 GAN(Generative Adversarial Network)은 생성자(Generator)와 구분자(Discriminator)가 서로 경쟁하며 학습하는 구조로, 데이터 분포를 근사하는 데 뛰어난 성능을 보인다[1].  
그러나 JS(Jensen–Shannon) 발산을 최적화 목표로 사용할 때 다음과 같은 문제가 발생한다:  
- **학습 불안정성:** Discriminator가 너무 빨리 학습하면 Generator에게 전달되는 gradient가 소실(vanishing)된다[1].  
- **모드 붕괴(mode collapse):** Generator가 일부 패턴만 생성하고 다양한 샘플을 생성하지 못하는 현상이 자주 발생한다[1].

WGAN은 이러한 한계를 극복하기 위해 **Wasserstein 거리(EM 거리, Earth Mover's Distance)**를 최적화 목표로 제안되었다[2].

## 2. 이론적 개념

### 2.1. Wasserstein 거리(EM Distance)
Wasserstein 거리 $$W_1(P, Q)$$는 확률 분포 $$P$$와 $$Q$$ 간의 **Optimal Transport(최적 수송)** 개념을 활용하여 정의된다. 직관적으로 두 분포를 일치시키기 위해 움직여야 하는 “질량”의 최소 비용을 의미한다[3].  

수학적으로는 다음과 같이 쓸 수 있다:  

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P,Q)} \mathbb{E}_\{(x,y)\sim \gamma}[\|x - y\|]$$  

여기서 $$\Pi(P,Q)$$는 $$P$$와 $$Q$$의 모든 결합 분포(joint distribution) 집합이다[3].

### 2.2. Kantorovich–Rubinstein 쌍대성
Wasserstein 거리의 계산은 위 정의로는 직접 계산이 어렵다. 쌍대성 이론에 따라 1-Lipschitz 함수 $$f$$를 이용해 다음과 같이 변형된다[3]:  

$$W_1(P, Q) = \sup\_\{\|f\|_L \le 1} \mathbb{E}\_{x\sim P}[f(x)] - \mathbb{E}\_{x\sim Q}[f(x)].$$  

여기서 $$\|f\|_L \le 1$$는 $$f$$의 Lipschitz 상수(Lipschitz constant)가 1 이하임을 의미한다[3].

## 3. WGAN 알고리즘

### 3.1. Critic(판별자)과 Generator(생성자)
- **Critic**: 전통적인 Discriminator 대신 “Critic”이라 칭하며, 1-Lipschitz 제약을 만족하는 신경망 $$f_w$$를 학습한다[1].  
- **Generator**: 잠재 변수 $$z\sim p(z)$$를 입력으로 받아 $$G_\theta(z)$$를 생성하며, Critic이 평가한 값 차이를 최대화하도록 학습한다[1].  

### 3.2. 학습 목표
Critic과 Generator의 목적함수는 다음과 같다[1]:

- Critic 학습(내부 반복 $$n_{\text{critic}}$$회):
$$\max_{w:\|f_w\|\_L \le 1} \mathbb{E}\_{x\sim P_r}[f_w(x)] - \mathbb{E}\_{z\sim p(z)}[f_w(G_\theta(z))].$$
  
- Generator 학습:
$$\min_\theta  -\mathbb{E}\_{z\sim p(z)}[f_w(G_\theta(z))].$$

### 3.3. Lipschitz 조건 보장
원 논문에서는 **weight clipping** 방식을 사용하여 모든 파라미터 $$w$$를 $$[-c, c]$$ 구간으로 제한함으로써 1-Lipschitz 조건을 근사적으로 만족시켰다[1].  
이후 gradient penalty 등을 활용한 **WGAN-GP** 방식이 등장하여 보다 안정적인 학습을 제공한다[4].

## 4. WGAN의 장점
- **Gradient 안정성**: Critic이 고정된 상태에서 충분히 학습 가능하므로 Generator에 전달되는 gradient가 소실되지 않는다[1].  
- **모드 붕괴 완화**: Wasserstein 거리 특성상 분포 전체를 고려하므로 일부 샘플에 치우치는 현상이 줄어든다[1].  
- **학습 지표 제공**: Wasserstein loss가 실제 거리에 비례하므로 학습 진행을 직관적으로 모니터링할 수 있다[2].

## 5. 결론
WGAN은 기존 GAN의 학습 불안정성과 모드 붕괴 문제를 효과적으로 해결한 방법이다. Optimal Transport 기반의 Wasserstein 거리를 활용하여 이론적 안정성과 실험적 성능을 모두 개선하였으며, 이후 다양한 변형(WGAN-GP, SN-WGAN 등)이 제안되어 GAN 학습의 표준 기법으로 자리잡았다[2][4].

[1] http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
[2] https://arxiv.org/abs/1701.07875
[3] https://ddangchani.github.io/Wasserstein-GAN/
[4] https://arxiv.org/pdf/1705.02438.pdf
[5] https://arxiv.org/abs/2204.00741
[6] https://www.semanticscholar.org/paper/2f85b7376769473d2bed56f855f115e23d727094
[7] https://www.mdpi.com/2073-8994/16/3/285
[8] https://iajit.org/upload/files/Enhanced-Soccer-Training-Simulation-Using-Progressive-Wasserstein-GAN-and-Termite-Life-Cycle-Optimization-in-Virtual-Reality.pdf
[9] https://link.springer.com/10.1007/s10489-024-05313-4
[10] https://arxiv.org/abs/2401.16947
[11] https://ieeexplore.ieee.org/document/10831026/
[12] https://arxiv.org/abs/2411.06397
[13] https://ieeexplore.ieee.org/document/10122519/
[14] https://www.semanticscholar.org/paper/fdd72abcfab06548e862ec45015427dbcf78e3ae
[15] http://biorxiv.org/lookup/doi/10.1101/2023.08.25.554841
[16] https://arxiv.org/html/2405.16351v1
[17] https://arxiv.org/pdf/2109.05652.pdf
[18] https://arxiv.org/pdf/2204.00387.pdf
[19] http://arxiv.org/abs/1803.01541
[20] http://arxiv.org/pdf/1701.07875.pdf
[21] https://arxiv.org/abs/1701.07875v2
[22] https://code731.tistory.com/86
[23] https://cumulu-s.tistory.com/31
[24] https://pages.cs.wisc.edu/~sharonli/courses/cs839_fall2020/slides/presentation15.pdf
[25] https://linkinghub.elsevier.com/retrieve/pii/S1568494624012298
[26] https://link.springer.com/10.1007/s00521-022-07968-x
[27] https://www.semanticscholar.org/paper/5e51b7bc40d9e65cc53f736b80c5af716b3a6683
[28] https://www.semanticscholar.org/paper/2587ad39afaf8a452f77ddba9c843701add431a7
[29] https://blog.outta.ai/221
[30] http://papers.neurips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf

https://ahjeong.tistory.com/7
