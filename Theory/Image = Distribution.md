# Image = Distribution (High-Dimensional Probability Distribution)

핵심 요약: 이미지를 “확률분포”로 보는 관점은 합리적이며, 현대 생성모델의 수학적·알고리즘적 틀과 일치합니다. 특히 score-based diffusion은 밀도 그 자체가 아닌 로그밀도의 기울기(스코어) $$ \nabla_x \log p(x) $$를 학습하여 고차원 분포를 모델링하고 샘플링하는 정교한 방법으로 이 제약을 우회합니다.[1][2][3][4]

## 무엇이 “이미지는 확률분포”인가
- 데이터셋의 각 이미지는 어떤 확률공간에서 샘플링된 관측치이며, 생성모델은 이 샘플들을 낳는 미지의 분포 $$p_{\text{data}}(x)$$를 추정하고 샘플링하는 것을 목표로 합니다.[4][5]
- 이미지가 고차원이라는 말은, 예를 들어 $$H \times W \times C$$ 픽셀의 경우 $$x \in \mathbb{R}^{HWC}$$의 벡터로 놓이며, 분포 $$p(x)$$는 이 거대한 공간에서의 확률밀도(또는 질량)로 정의된다는 뜻입니다.[4]
- 따라서 “이미지는 확률분포다”라는 표현은 “이미지 데이터셋은 하나의 고차원 확률분포에서 나왔다”는 의미의 준말로 받아들이는 것이 정확합니다. 이 해석은 생성모델 전반에 깔린 표준 가정과 합치됩니다.[6][4]

## 픽셀 단위 직관은 어디까지 타당한가
- 픽셀 마진 분포(히스토그램)와 조건부 분포를 직관적으로 설명하는 것은 유익하지만, 실제 이미지는 강한 장거리 상관과 구조(에지, 질감, 물체 수준 요인)를 가지므로 단순한 독립 픽셀 가정으로는 충분하지 않습니다.[4]
- “첫 픽셀의 분포 distribution $p(x_1)$ 을, 그에 조건부인 두 번째 픽셀은 $p(x_1|x_2)$”이라는 전개는 체인룰 $$p(x)=\prod_i p(x_i \mid x_{<i})$$의 직관과 맞습니다. 다만 실제 모델은 CNN/Transformer 등의 표현학습으로 이러한 복잡한 조건구조를 암묵적으로 포착합니다.[4]
- Cifar-10과 같은 실제 데이터셋은 정규화 방식과 특징공간에 따라 특정한 통계 패턴을 보일 수 있으나, 개별 픽셀 히스토그램만으로는 데이터 분포의 본질(고차원 구조)을 포착하기 어렵습니다.[7][4]

## 왜 명시적 밀도모델은 어려운가
- 고차원 이미지의 우도 $$p_\theta(x)$$를 명시적으로 모델링하는 것은 난해합니다. 복잡한 멀티모달 분포, 강한 상관구조, 고차원에서의 희소성 때문에 폐형식 밀도나 단순 가우시안 혼합으로는 한계가 큽니다.[6][4]
- 이 때문에 VAE/Flow/Autoregressive/Score-based/Diffusion 등 다양한 접근이 등장했고, 특히 스코어 추정은 밀도 정규화 제약(적분이 1)을 직접 맞추지 않고도 분포의 “형상”을 학습하게 해줍니다.[3][1][6]

## 스코어의 아이디어: 제약을 우회하는 법
- 스코어 함수는 $$s(x)=\nabla_x \log p(x)$$입니다. 정규화상수에 독립이므로, $$p(x)$$를 직접 정규화하지 않고도 분포의 기울기장을 학습할 수 있습니다.[2][3]
- Song–Ermon(2019)은 소음이 다양한 수준에서 섞인 분포의 스코어를 점진적으로 학습하고, Langevin dynamics로 샘플을 생성하는 방법을 제시하였습니다(Noise-Conditional Score Networks).[8][2][3]
- 이후 SDE 틀에서 전방 과정이 데이터를 점차 가우시안으로 보내는 확산을 정의하고, 시간역행 SDE와 스코어 추정을 결합해 노이즈에서 데이터로 복원하는 강력한 샘플러가 확립되었습니다(Score-SDE).[9][1][8]

### Noise-Conditional Score Networks
Noise-Conditional Score Networks (NCSN)은 데이터에 서로 다른 크기의 노이즈를 점진적으로 추가하여 각 노이즈 단계에서 데이터 분포의 그래디언트(Score function)를 학습하는 모델입니다. 이렇게 여러 노이즈 레벨에 조건화하여 score를 추정함으로써 더 안정적이고 세밀한 데이터 생성이 가능해집니다.

구체적으로, NCSN의 입력은 노이즈가 추가된 데이터와 그 노이즈 정도를 나타내는 스케일(σ)이며, 출력은 해당 노이즈 수준에서의 로그 확률 밀도 함수의 그래디언트(Score)입니다. 이 조건화 덕분에 모델은 노이즈 크기에 따라 스케일이 변하는 score를 효과적으로 학습합니다.

학습된 score를 통해 Langevin dynamics 같은 확률적 샘플링 기법으로 원래 데이터 분포에서 새로운 데이터를 생성할 수 있습니다. NCSN은 score matching과 확산 모델(diffusion model)의 기반을 이루며, 점차 노이즈를 줄여가며 고해상도의 실제와 유사한 샘플을 만들어내는 강력한 generative model입니다.

## 수식으로 보는 핵심
- 스코어 매칭의 목적: $$\min_\theta \mathbb{E}\_{p_{\text{data}}}\left[\frac{1}{2}\lVert s_\theta(x)-\nabla_x \log p_{\text{data}}(x)\rVert^2\right]$$ 형태로 로그밀도 기울기를 근사합니다.[2][3]
- 노이즈 주입 버전(다중 $$\sigma$$)은 $$p_\sigma(x)=p_{\text{data}}\ast \mathcal{N}(0,\sigma^2 I)$$의 스코어를 학습하여 다양한 스케일에서 안정적으로 훈련합니다.[3][8][2]
- Score-SDE의 전방 확산: $$\mathrm{d}x=f(x,t)\mathrm{d}t+g(t)\mathrm{d}w$$. 시간역행 SDE: $$\mathrm{d}x=\big[f(x,t)-g(t)^2\nabla_x \log p_t(x)\big]\mathrm{d}t+g(t)\mathrm{d}\bar{w}$$로, 여기서 $$\nabla_x \log p_t(x)$$를 신경망으로 근사해 샘플링합니다.(Fokker-Planck 방정식, Anderson(1982) 에서 증명됨.) [1][8]

## Image = Distribution 주장에 대한 평가
- “이미지는 확률분포다”: 표현은 약간 은유적이지만, “이미지 데이터셋은 고차원 분포에서의 샘플”이라는 표준 관점으로 해석하면 타당합니다. 명확성을 위해 “이미지의 집합이 이루는 데이터 분포”라고 기술하는 것을 권장합니다.[6][4]
- “픽셀 히스토그램→조건부→결합분포” 전개: 개념적 도입으로 적절합니다. 다만 실제 생성모델은 픽셀 독립 가정을 두지 않으며, 고수준 표현으로 복잡한 상관을 학습함을 분명히 하는 보완이 필요합니다.[4]
- “스코어로 제약을 우회”: 정확합니다. 정규화상수(적분 1)의 부담 없이 로그밀도 기울기를 학습하고, Langevin/SDE 역과정으로 샘플링한다는 설명은 문헌과 합치합니다.[1][2][3]
- “Gaussian만으로는 부족”: 고차원 멀티모달 분포에 대한 GMM/단순 가우시안의 한계를 지적한 방향은 적절합니다. 현대 기법(Flow/Autoregressive/Diffusion/Score)의 동기와도 일치합니다.[6][4]
- 보강 제안: CIFAR-10 통계 플롯의 출처와 정규화 정의를 명확히 하고, score-based의 이론·실무 레퍼런스(2019 SMLD, 2021 Score-SDE, 코드 리포)를 함께 제시하면 신뢰성과 재현성이 높아집니다.[9][7][3][1]

## 연구 수준 요약 포인트
- 목적: $$p_\theta(x)\approx p_{\text{data}}(x)$$ 또는 스코어 $$s_\theta(x)\approx \nabla_x \log p_{\text{data}}(x)$$를 학습하여 새로운 샘플을 생성합니다.[3][4]
- 스코어-확산 통합: 전방 확산 SDE로 데이터를 노이즈로 보낸 뒤, 추정된 스코어로 역 SDE/ODE를 풀어 노이즈에서 샘플을 생성합니다. Predictor–Corrector, ODE 샘플러로 품질·속도를 개선합니다.[8][1]
- 성능: Score-SDE는 CIFAR-10에서 FID/IS 등 강력한 수치를 달성하였고, 공식 구현이 공개되어 재현 가능합니다.[9][1]

## 최소 예시 코드(PyTorch, 스코어 학습 + Langevin 샘플링)

```
Score Matching with Langevin Dynamics(SMLD)

지금까지의 내용을 바탕으로 요약하면,

Score Matching with Langevin Dynamics는 데이터 분포의 score(로그밀도 함수의 기울기)를 학습하여, Langevin dynamics(확률적 경사 상승법)를 통해 샘플을 생성합니다.
이 방법은 score function이 정확할 때 효과적이며, Langevin dynamics를 여러 단계 수행하여 점차 데이터 분포로 샘플을 이동시킵니다.
```

- 목적: 2D 장난감 데이터(멀티모달)에서 스코어 네트워크를 훈련하고, Annealed Langevin Dynamics로 샘플링하는 실습입니다. 아이디어는 Song–Ermon(2019)의 SMLD를 축약·교육용으로 재구성한 것입니다.[2][8][3]
- 주의: 학습 안정화를 위해 다중 $$\sigma$$ 스케줄, 정규화, 학습률 스케줄이 중요합니다. 고차원 이미지로 확장하려면 U-Net 아키텍처와 SDE 프레임워크(Score-SDE) 사용을 권장합니다.[8][1][9]

```python
# PyTorch 2D toy example of score matching + Annealed Langevin Dynamics
# Ref: Score matching + NCSN (Song & Ermon, 2019) concepts [5][3][18]

import math, torch, torch.nn as nn, torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Toy dataset: Mixture of Gaussians in 2D
def sample_data(n):
    centers = torch.tensor([[0,0],[3,0],[0,3],[-3,0],[0,-3]], dtype=torch.float32)
    idx = torch.randint(len(centers), (n,))
    x = centers[idx] + 0.5*torch.randn(n,2)
    return x.to(device)

# 2) Score network: small MLP s_theta(x, sigma) -> R^2
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2+1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x, sigma):
        # concatenate log-sigma as conditioning
        if sigma.dim()==1: sigma = sigma[:,None]
        h = torch.cat([x, torch.log(sigma)], dim=1)
        return self.net(h)

# 3) Multi-sigma training objective (denoising score matching)
#    Minimize E || s_theta(x_t, sigma) + (x_t - x) / sigma^2 ||^2,
#    where x_t = x + sigma * z, z ~ N(0,I)
sigmas = torch.exp(torch.linspace(math.log(1.0), math.log(0.01), 10)).to(device)
model = ScoreNet().to(device)
opt = optim.Adam(model.parameters(), lr=2e-4)

def loss_step(batch):
    n = batch.size(0)
    # sample a sigma per sample
    idx = torch.randint(len(sigmas), (n,), device=device)
    sigma = sigmas[idx].view(n,1)
    noise = torch.randn_like(batch)
    xt = batch + sigma*noise
    # target score of xt under p_sigma: -(xt - x)/sigma^2
    target = -(xt - batch)/(sigma**2)
    pred = model(xt, sigma.squeeze(1))
    return ((pred - target).pow(2).sum(dim=1).mean())

# 4) Train
for it in range(5000):
    x = sample_data(512)
    loss = loss_step(x)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (it+1)%500==0:
        print(it+1, float(loss))

# 5) Annealed Langevin Dynamics sampling
@torch.no_grad()
def ald_sample(n_steps=100, step_factor=0.1, n=2048):
    x = torch.randn(n,2, device=device)
    for sigma in sigmas:  # coarse-to-fine
        eps2 = (step_factor * (sigma**2)).item()
        for _ in range(n_steps):
            grad = model(x, torch.full((n,), sigma.item(), device=device))
            x = x + 0.5*eps2*grad + math.sqrt(eps2)*torch.randn_like(x)
    return x

samples = ald_sample()
# Save to CSV for downstream plotting
import pandas as pd
pd.DataFrame(samples.cpu().numpy(), columns=['x1','x2']).to_csv('ald_samples.csv', index=False)
print('Wrote ald_samples.csv')
```


설명 포인트:
- 손실식 유도: 가우시안 노이즈가 섞인 분포 $$p_\sigma(x)$$에 대해 $$\nabla_x \log p_\sigma(x_t)=-(x_t-x)/\sigma^2$$를 이용하여 스코어를 회귀합니다. 이렇게 다양한 $$\sigma$$에서 학습하면 거친→미세 스케일의 구조를 안정적으로 포착합니다.[3][2]
- 샘플링: Annealed Langevin은 큰 $$\sigma$$에서 시작해 점차 줄이며, 단계별로 $$x \leftarrow x + \frac{\epsilon^2}{2}s_\theta(x,\sigma) + \epsilon z$$를 반복합니다. 이는 스코어 기반 샘플러의 기본형이며, SDE 기반 프레임워크로 일반화됩니다.[1][8][3]

## 이미지로 확장하려면
- 아키텍처: 2D MLP 대신 U-Net 백본과 사이클릭·시간 임베딩을 사용합니다. 고정/학습형 $$\sigma(t)$$ 스케줄과 SDE/ODE 샘플러(Predictor–Corrector, Heun/DPMSolver)를 도입합니다.[8][1]
- 구현 참고: Score-SDE 공식 코드는 CIFAR-10, CelebA-HQ 등을 포함하고, 훈련/샘플링 스크립트와 설정파일이 제공됩니다. 재현 가능성과 실험 설계 참고에 유용합니다.[9][1]

## 자주 나오는 이론·실무 질문
- 정규화 상수는 왜 필요 없나? 로그밀도의 기울기 $$\nabla_x \log p(x)=\nabla_x \log \tilde{p}(x)$$에서 $$\log Z$$의 기울기는 0이기 때문입니다. 따라서 스코어 학습은 정규화 제약을 우회합니다.[2][3]
- 왜 노이즈를 섞어 훈련하나? 고차원에서 데이터 지지집합이 얇아 스코어 추정이 불안정합니다. 다양한 $$\sigma$$에서의 스코어를 학습하면 학습이 원활하고, 샘플러도 거친 구조부터 복원할 수 있습니다.[3][8]
- 성능과 이론 보장? Score-SDE는 강력한 경험적 성능을 보이며, 수렴·근사 오차에 대한 이론 결과도 축적되고 있습니다(예: DDPM/SGM 수렴 분석).[10][1][9]

## 블로그에 넣을 권장 보강 자료
- 개념 글: Yang Song의 스코어 기반 모델 블로그 포스트(직관·응용 요약).[2]
- 핵심 논문: SMLD(2019), Score-SDE(ICLR 2021 Oral).[1][3]
- 실무 가이드: Lil’Log의 Diffusion 개요와 NCSN 연결.[8]
- 공식 코드: score_sde GitHub(설정, 스크립트, 재현성).[9]

결론: 원문은 “이미지를 확률분포로 본다”는 핵심 관점을 잘 짚었고, 스코어 기반 확산이 정규화 제약을 우회해 고차원 분포를 모델링한다는 주장도 연구 결과와 합치합니다. 픽셀 히스토그램 직관은 입문에 유익하나, 실제 모델링은 고차원 표현과 스코어 추정을 통해 전역적 구조를 학습한다는 점을 보완하면 더 탄탄한 가이드가 됩니다.[10][7][6][1][3][9][4][8][2]

[1](https://arxiv.org/abs/2011.13456)
[2](https://yang-song.net/blog/2021/score/)
[3](https://arxiv.org/abs/1907.05600)
[4](https://cs231n.github.io/generative-models/)
[5](https://lamarr-institute.org/blog/generative-neural-models/)
[6](https://www.sciencedirect.com/science/article/abs/pii/S1574013720303853)
[7](https://www.biorxiv.org/content/10.1101/2021.02.18.431827v1.full.pdf)
[8](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[9](https://github.com/yang-song/score_sde)
[10](https://proceedings.mlr.press/v201/lee23a/lee23a.pdf)
[11](https://www.eungbean.com/59220575-32d4-44e5-9f5d-f403cb8034fa)
[12](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)
[13](https://arxiv.org/html/2403.12636v1)
[14](https://openreview.net/pdf/1f5a22ba1210509f92301368c19958be1dffd97a.pdf)
[15](https://neurips.cc/virtual/2022/session/64303)
[16](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001268)
[17](https://arxiv.org/html/2501.00744v1)
[18](https://jmlr.org/papers/volume26/23-1472/23-1472.pdf)
[19](https://link.aps.org/doi/10.1103/PhysRevE.111.045304)
[20](http://alvarestech.com/temp/deep/Deep%20Learning%20by%20Ian%20Goodfellow,%20Yoshua%20Bengio,%20Aaron%20Courville%20(z-lib.org).pdf)
[21](https://arxiv.org/html/2411.17006v1)

https://www.eungbean.com/59220575-32d4-44e5-9f5d-f403cb8034fa

