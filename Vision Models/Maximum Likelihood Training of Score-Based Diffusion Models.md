# Maximum Likelihood Training of Score-Based Diffusion Models | 2021 · 839회 인용, Image generation

**핵심 주장 및 주요 기여**  
- 이 논문은 **확산(denoising diffusion) 모델**을 점수 매칭(score matching) 방식이 아닌 **최대 우도(maximum likelihood)** 관점에서 직접 학습할 수 있는 이론적 틀과 실제 학습 알고리즘을 제안한다.  
- 기존의 점수 기반 확산 모델이 근사적 denoising score matching에 의존했던 반면, 이 연구에서는 **연속 시간 확산 SDE(Stochastic Differential Equation)** 를 이용해 우도 하한치(ELBO)를 유도하고, 이를 통해 정확한 확률 밀도 학습이 가능함을 보인다.  
- 수치 실험을 통해 CIFAR-10, ImageNet 등에서 로그우도와 샘플 품질 모두에서 최첨단 성능을 달성하며, 우도 관점의 학습이 표본 생성 모델 품질 개선에 효과적임을 입증했다.

## 1. 해결하고자 하는 문제  
확산 모델은 데이터 $$x$$에 점진적 노이즈를 추가하는 정방향 SDE(forward SDE)와, 이를 역전해 샘플을 생성하는 역방향 SDE(reverse SDE)를 기반으로 한다.  
기존에는 주로 **denoising score matching**으로 점수 함수 $$\nabla_x \log p_t(x)$$를 직접 학습해 왔으나,  
- 이 방식은 **우도 최적화(objective-likelihood mismatch)** 문제가 있고  
- 모델 평가 시에도 정확한 데이터 우도를 계산할 수 없다는 단점이 있다.  

본 연구는 이 격차를 해소하기 위해 **연속 시간 SDE**를 통해 **우도 하한치(ELBO)** 를 엄밀히 유도하고, 이를 직접 최대화하는 학습 방식을 제안한다.  

## 2. 제안 방법  
### 2.1 연속 확산 SDE와 역 SDE  
정방향 확산 프로세스는 다음 연속 SDE로 기술된다:  

$$
dx = f(x,t)\,dt + g(t)\,dw,
$$  

여기서 $$f, g$$는 설계 가능한 drift와 diffusion 계수, $$w$$는 표준 위너 프로세스이다. 역방향 프로세스 역시 Girsanov 정리를 통해 아래 형태로 표현된다:  

$$
dx = \bigl[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\bigr]dt + g(t)\,d\bar w.
$$  

### 2.2 ELBO 유도  
데이터 분포 $$p_0(x)$$의 로그우도 $$\log p_0(x)$$는 연속 시간 ELBO로 나타낼 수 있으며, 아래 항들의 합으로 분해된다:  
1. 초기 노이즈 레벨에서의 데이터 재구성 오차  
2. 시간에 따른 점수 네트워크의 예측 오차 (weighted score matching)  
3. 경계 조건 관련 항 (KL divergence term)  

이를 수식으로 정리하면,  

$$
\mathcal{L} = \mathbb{E}_{q}\Bigl[\underbrace{\|\nabla_x \log q_t(x)-s_\theta(x,t)\|^2}_{\text{score matching}} + \dots \Bigr] + \text{const.}
$$  

여기서 $$s_\theta(x,t)$$는 시·공간 입력을 받는 **점수 네트워크**이다.

### 2.3 네트워크 구조  
- **UNet 기반** 아키텍처에 시간 임베딩(time embedding) 모듈을 결합하여, 각 노이즈 레벨 $$t$$에 대응하는 점수 함수를 학습한다.  
- Attention 모듈과 residual block을 활용해 깊은 모델에서도 안정적 학습을 도모한다.

## 3. 성능 향상 및 한계  
### 3.1 성능 검증  
- **로그우도(log-likelihood)**: CIFAR-10에서 기존 최첨단 모델 대비 우도 상한치가 대폭 상승.  
- **샘플 품질**: FID/FID-scores 지표에서도 경쟁 모델을 능가.  
- **속도-정밀도 트레이드오프**: 학습 및 샘플링 단계에서 균형 잡힌 성능을 보여준다.

### 3.2 일반화 성능 향상  
- 우도 기반 목표함수는 다양한 노이즈 레벨에 대한 모델의 **일반화 능력**을 수학적으로 보장하는 성질을 지닌다.  
- 특히, **importance weighting**을 이용해 확산 경로 전체에서 균일한 학습 집중을 유도함으로써, 노이즈 심도가 다른 입력에서도 점수 예측 성능이 고루 개선된다.

### 3.3 한계  
- **계산비용**: 정밀한 ELBO 추정을 위해 많은 시간 분할 단계(time discretization)가 필요해 학습·샘플링 비용이 증가한다.  
- **분산 불안정성**: Noise schedule에 따라 gradient 분산이 커져 학습이 불안정해질 수 있다.  
- **응용 범위 제약**: 고차원 이미지나 복잡한 조건부 생성(task-conditioned generation)으로 확장할 때, 추가적인 구조 설계가 필요하다.

## 4. 향후 연구 영향 및 고려 사항  
- **이론적 확장**: SDE 기반 ELBO 유도 기법은 다른 연속 생성 모델(예: Score-Flow, Normalizing Flow)로의 확장이 기대된다.  
- **노이즈 스케줄 최적화**: 일반화 성능과 학습 안정성을 위한 **adaptive noise schedule** 연구가 중요해질 것이다.  
- **모델 경량화**: 계산비용 문제를 해결하기 위한 **효율적 차원 축소** 기법 및 **교사-학생(distillation)** 전략이 필요하다.  
- **응용 분야 확장**: 의료영상, 음성합성, 시계열 예측 등 다양한 도메인에서 우도 기반 확산 모델의 **실용성**을 탐구해야 한다.
