## Score Function이란?

**Score function**은 확률 모델에서 **로그 우도 함수(log-likelihood function)**를 파라미터에 대해 미분한 값, 즉 **기울기(gradient)**를 의미합니다. 수식으로는 다음과 같이 표현합니다:

$$
s(\theta) = \nabla_\theta \log p(x | \theta)
$$

여기서 $$s(\theta)$$는 score function이고, $$\nabla_\theta$$는 파라미터 $$\theta$$에 대한 미분, $$p(x|\theta)$$는 파라미터 $$\theta$$ 하에서 데이터 $$x$$의 확률입니다. 이 함수는 주어진 데이터에 대해 모델의 파라미터가 얼마나 잘 맞는지, 즉 모델의 예측과 실제 데이터 간의 차이를 측정하는 데 사용됩니다[1][2].

### Score Function의 역할

- **파라미터 추정**: 우도(likelihood)를 최대화하는 방향을 알려줍니다.
- **모델 최적화**: 파라미터 업데이트에 사용되어, 모델 성능을 높이는 데 기초적인 정보를 제공합니다.
- **확률적 모델**: 다양한 확률 모델과 생성 모델에서 활용됩니다[1].

---

## Score Matching이란?

**Score matching**은 실제 확률분포(데이터의 진짜 분포)의 score function과 모델이 추정한 score function 사이의 차이를 최소화하여, 복잡한 확률분포를 직접 추정하지 않고도 모델을 학습하는 방법입니다[2].

### 왜 Score Matching이 필요한가?

전통적인 확률모델(예: energy-based model)에서는 확률분포의 정규화 상수(normalizing constant, $$Z$$)를 계산해야 하는데, 이 값은 보통 계산이 매우 어렵습니다. 하지만 score function을 사용하면, 로그 우도 함수의 미분 과정에서 이 상수가 사라지므로 계산이 훨씬 간단해집니다[2].

### Score Matching의 원리

- **기본 아이디어**: 데이터의 진짜 분포 $$p_{data}(x)$$의 score function과 모델이 추정한 score function $$s_\theta(x)$$ 사이의 차이(Fisher divergence)를 최소화합니다.
- **수식**: Fisher divergence는 다음과 같이 정의됩니다.

$$
D_F(p_{data} \| p_\theta) = \int p_{data}(x) \left\| \nabla_x \log p_{data}(x) - \nabla_x \log p_\theta(x) \right\|^2 dx
$$

- **실제 구현**: 진짜 분포의 score function을 직접 알 수 없으므로, 부분적분을 이용해 계산이 가능한 형태로 loss function을 변형합니다. 이 과정에서 모델이 추정한 score function에만 의존하는 손실함수를 만들 수 있습니다[2].

---

## 쉽게 이해하는 예시

1. **Score Function**: "이 데이터가 이 모델 파라미터에서 얼마나 잘 설명되는지"를 나타내는 기울기(방향 벡터)입니다.
2. **Score Matching**: "데이터가 실제로 분포하는 방향(진짜 score function)과, 모델이 생각하는 방향(추정 score function)이 최대한 비슷해지도록" 모델을 학습하는 방법입니다.

---

## 실제 활용

- **Diffusion Model**: 최근 이미지 생성 등에서 각광받는 diffusion model의 핵심 objective로 score matching이 널리 사용됩니다. 복잡한 분포를 정확히 모델링하지 않고도, 데이터의 구조를 효과적으로 학습할 수 있기 때문입니다[2].
- **확률적 생성 모델**: 복잡한 데이터의 분포를 추정할 때, score matching은 계산 효율성과 성능 모두에서 강점을 보입니다[1][2].

---

## 요약

- **Score function**: 로그 우도 함수의 기울기(gradient)로, 파라미터 최적화와 모델 학습에 필수적인 도구입니다[1][2].
- **Score matching**: 진짜 데이터 분포의 score function과 모델의 score function이 최대한 비슷해지도록 학습하는 방법으로, 복잡한 확률분포를 직접 계산하지 않고도 효과적인 모델링이 가능합니다[2].

이해를 돕기 위해, score function은 "방향", score matching은 "그 방향을 최대한 맞추는 과정"이라고 생각하면 됩니다.

[1] https://askai.glarity.app/ko/search/Score-function%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94
[2] https://process-mining.tistory.com/211
[3] https://nswa.sciencesforce.com/journal/vol22/iss1/2
[4] https://www.ijsr.net/getabstract.php?paperid=SR241004074146
[5] https://dl.acm.org/doi/10.1145/3459637.3482450
[6] https://www.nature.com/articles/s41440-019-0305-8
[7] https://www.emerald.com/insight/content/doi/10.1108/CG-01-2023-0033/full/html
[8] https://www.biomedicine.video/animated-videos/propensity-score-matching-methodology-why-and-how-it-is-used
[9] https://www.reddit.com/r/datascience/comments/jk1dyj/propensity_score_matching_basic_statistical/
[10] https://www.semanticscholar.org/paper/4cd11be7eabde93c4b0b8986a66dd59b767b5e35
[11] https://ieeexplore.ieee.org/document/10826052/
[12] https://www.aclweb.org/anthology/2020.acl-main.435
[13] https://aircconline.com/ijaia/V12N1/12121ijaia02.pdf
[14] http://link.springer.com/10.1007/JHEP01(2016)023
[15] https://builtin.com/data-science/propensity-score-matching
[16] https://andrewcharlesjones.github.io/journal/21-score-matching.html
[17] https://jaketae.github.io/study/sliced-score-matching/
[18] https://statsnotebook.io/blog/analysis/matching/
[19] https://help.sap.com/doc/saphelp_gds20/2.0/en-US/e3/0ca76c664d49edb7fb3193864b7592/content.htm?no_cache=true
[20] https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.make_scorer.html
