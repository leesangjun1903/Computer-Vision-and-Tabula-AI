# LSGM : Score-based Generative Modeling in Latent Space | Image generation

**Score-based Generative Modeling in Latent Space (LSGM)**[1]는 데이터 공간이 아닌 VAE의 잠재 공간에서 연속시간 확산 기반(score-based) 생성 모델을 학습함으로써  
1) 샘플링 속도를 수백 배 단축,  
2) 모델 표현력을 향상,  
3) 이산 또는 비연속형 데이터에도 자연스럽게 적용  
가능함을 보였다.

# 1. 해결 과제  
기존 연속-시간 확산 기반 생성 모델(SGM)은  
- 고품질 샘플과 높은 분포 커버리지 달성  
- 그러나 샘플링 시 수천 회의 신경망 평가 필요→실시간성 부재  
- 데이터 공간에서 직접 확산→이산 데이터(이진·범주형)에 적용 어려움  

이를 타개하고자, 저자들은 **잠재 공간**에서 확산 과정을 수행하는 LSGM을 제안한다.

# 2. 제안 방법

## 2.1 모델 구조  
- 기존 VAE(qφ(z₀|x), pψ(x|z₀))에  
- 잠재 변수 z₀에 대한 **SGM prior** pθ(z₀)를 결합  
- 생성: z₁∼𝒩(0,I) → 역확산(reverse SDE/ODE)으로 z₀ 생성 → 디코더로 x 생성  

[image:1]

## 2.2 학습목표  
전체 목적은 변분하한식(VAE ELBO)에 SGM prior를 결합:

$$
\mathcal{L}(x)=\underbrace{\mathbb{E}\_{qφ(z₀|x)}[-\log pψ(x|z₀)]}_{\text{재구성}}
+KL\bigl(qφ(z₀|x)\,\|\,pθ(z₀)\bigr).
$$

문제는 cross-entropy $$−\log pθ(z₀)$$가 SGM의 score function ∇ₙ log pθ(zᵗ)를 필요로 하며,  
데이터 분포의 marginal score ∇ₙ log q(zᵗ) 불가분 표현이 등장한다는 점.  

### 2.2.1 교차엔트로피의 denoising score matching 전환  
정리(Thm.1)[1]에 따라

```math
CE\bigl(q(z₀|x)\|pθ(z₀)\bigr)
=
\mathbb{E}_{t\sim U[1]}\Bigl[\tfrac{g(t)^2}{2}\,
\mathbb{E}_{qφ(z₀|x)q(zᵗ|z₀)}\|\nabla_{zᵗ}\log q(zᵗ|z₀)-\nabla_{zᵗ}\log pθ(zᵗ)\|^2\Bigr]
+ \mathrm{const.}
```

로 변환하여 $$\nabla_{zᵗ}\log q(zᵗ)$$ 의존 문제를 제거했다.

### 2.2.2 혼합 스코어 파라미터화  
잠재 prior를 표준정규와 학습가능 SGM의 기하학적 섞임으로 정의하여  

$$
\nabla_{zᵗ}\log p(zᵗ)
=-(1-\alpha)\,zᵗ + \alpha\,\nabla_{zᵗ}\log p'θ(zᵗ).
$$

이로써 모델은 정규분포와의 “불일치”만 학습 → 역확산이 선형항 주도로 빠르게 수렴.

### 2.2.3 분산저감 기법  
- **Geometric VPSDE**: $$\sigma_t^2=\sigma_{min}^2(\sigma_{max}^2/\sigma_{min}^2)^t$$ 로 일정 $$\tfrac{d\log\sigma_t^2}{dt}$$  
- **Importance Sampling**: $$r(t)\propto\tfrac{d\log\sigma_t^2}{dt}$$ 등으로 $$t$$ 샘플링 분산 최소화  

# 3. 성능 및 일반화

| 데이터셋      | FID   | NELBO (nats) | 샘플링 속도      |
|:-------------:|:-----:|:------------:|:----------------:|
| CIFAR-10      | **2.10**[1]  | 2.87     |  0.11 s (16개)  |
| CelebA-HQ-256 | 7.22[1] | ≤0.70    |  4.15 s (16개)  |
| OMNIGLOT      | –     | **87.79**[1]  | –                |
| MNIST         | –     | **78.47**[1]  | –                |

- **샘플링 속도**: 픽셀 공간 SGM 대비 56×–637× 가속  
- **이산 데이터**: Bernoulli 디코더로 이진 이미지에 자연 확장  

**일반화 성능**  
- 잠재 공간 확산 덕분에 **네트워크 규모 축소** → 오버피팅 위험 감소  
- 분산저감으로 **훈련 안정성** 확보  
- 혼합 스코어가 **정규분포 바이어스** 부여 → 소수 표본에서도 일관된 역확산  

# 4. 한계 및 고려사항

- **실시간성**: 여전히 인터랙티브한 수준(d≅0.01 s)에는 미치지 못함  
- **구조 제약**: VAE 백본 의존 → 대형 VAE 학습 불안정성 상존  
- **하이퍼파라미터**: σₘᵢₙ, σₘₐₓ, β(t) 등 스케줄 설정 필요  

# 5. 미래 영향 및 연구 시 고려점

- **다양한 데이터 타입**: 범주형·그래프·음악·분자 생성에 적용 가능  
- **Semi-supervised**: 잠재 표현 학습 강화 → 다운스트림 태스크 일반화  
- **실시간 샘플링**: ODE solver 최적화·distillation 기법 결합 연구  
- **하이퍼파라미터 자율설정**: 학습 중 스케줄 적응·메타러닝 도입  

LSGM은 확산 기반 생성 모델의 **속도-표현력 트레이드오프**를 극복할 혁신적 프레임워크로,  
차세대 생성 및 표현 학습 연구의 토대가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6c182dd7-c329-492f-b328-e63f2e06239a/2106.05931v3.pdf
