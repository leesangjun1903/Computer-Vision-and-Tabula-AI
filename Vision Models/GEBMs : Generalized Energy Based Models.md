# Generalized Energy Based Models | 2020 · 107회 인용, Image generation

**핵심 주장:** Generalized Energy Based Models (GEBMs)는 저차원 잠재 지지(support)를 학습하는 암묵적 모델(implicit model)과 그 지지 위에서 확률 질량을 정교하게 조정하는 에너지 함수(energy function)를 결합함으로써, 전통적 EBM과 GAN의 장점을 동시에 취한다.[1]

**주요 기여:**  
- 잠재 공간(latent space)을 학습하는 베이스 분포(base distribution)와 에너지 함수를 번갈아 학습하는 새로운 학습 절차 제안.[1]
- Donsker–Varadhan 하한 및 Fenchel 이중성을 이용한 일반화된 우도(generalized likelihood) 정의 및 KALE(KL Approximate Lower-bound Estimate) 손실함수 제안.[1]
- 낮은 차원의 잠재 공간에서 MCMC를 통한 샘플링 알고리즘(Unadjusted Langevin Algorithm, Kinetic Langevin Algorithm) 제시하여 고품질 샘플 획득.[1]
- 잠재 공간에서의 에너지 기반 재가중치(re-weighting)로 다중 모달리티(multimodality) 및 일반화 성능 강화.[1]

***

## 1. 해결하고자 하는 문제

고차원 데이터 분포 $$P$$는 종종 저차원 매니폴드(manifold)에 지지되어 있으며,  
- 전통 EBM은 전체 공간에 질량을 할당해 매니폴드 위에서 모호한(blurry) 표현 발생  
- GAN은 잠재 공간에서 매니폴드를 학습하나, 고정된 잠재 분포로 인해 지지 위 질량 재분배에는 한계  

GEBM은 이 두 접근의 한계를 **잠재 지지 학습** + **에너지 재분배**로 동시에 해결한다.[1]

***

## 2. 제안 방법

### 2.1 모델 정의  
베이스 분포 $$G$$와 에너지 $$E$$에 의해 생성된 GEBM $$Q$$의 밀도는  

$$
Q(dx)=\exp\bigl(-E(x)-A_{G,E}\bigr)\,G(dx),
\quad
A_{G,E}=\log\int\exp\bigl(-E(x)\bigr)\,G(dx).
$$

[1]

잠재 공간 $$Z$$에서 베이스 $$G$$는 $$x=G(z)$$, $$z\sim\eta(z)$$로 샘플링되며,  
에너지값을 반영한 잠재 사후분포(posterior)  

$$
\nu(z)\propto\eta(z)\exp\bigl(-E(G(z))\bigr)
$$

를 통해 **MCMC**로 샘플링.[1]

### 2.2 일반화된 우도와 KALE  
표준 우도는 잠재 지지 불일치 시 정의 불가능하므로,  

```math
\mathcal{L}_{P,G}(E):=-\mathbb{E}_{P}[E(x)]-A_{G,E}
```

를 *Generalized Likelihood*로 정의.[1]
에너지 학습은 이 우도를 최대화하며, 베이스 학습은 KALE 손실  

```math
\mathrm{KALE}(P\|G)
=\sup_{E,A}\Bigl\{-\mathbb{E}_{P}[E(x)+A]-\mathbb{E}_{G}\bigl[e^{-(E(x)+A)}\bigr]+1\Bigr\}
```

를 최소화.[1]

### 2.3 학습 절차 (Algorithm 1)  
1. 에너지 파라미터 $$\psi$$ 및 로그정규화 상수 $$A$$ 업데이트 by maximizing $$F_{P,G}(E_\psi+A)$$.[1]
2. 고정된 에너지로 베이스 파라미터 $$\theta$$ 업데이트 by minimizing KALE gradient.[1]
3. 위 과정을 번갈아 수행.

***

## 3. 모델 구조 및 샘플링

- **베이스**: GAN generator(implicit model) 또는 normalizing flow  
- **에너지**: GAN discriminator 구조 활용 가능  
- **샘플링**:  
  - Overdamped Langevin (ULA):  

$$\displaystyle Z_{k+1}=Z_k+\lambda\nabla_z[\log\eta(Z_k)-E(G(Z_k))]+\sqrt{2\lambda}\,\xi$$  
  
  - Kinetic Langevin (KLA): momentum 도입해 모드 간 이동성 향상.[1]

***

## 4. 성능 향상 및 일반화 성능

- **이미지 생성**: CIFAR-10, ImageNet 등에서 동일 네트워크 기준 GAN 대비 FID 대폭 개선.[1]
- **밀도 추정**: Real NVP 기반 베이스와 에너지로 UCI 데이터셋 NLL이 CD 및 ML과 동등 성능.[1]
- **일반화 이론**:  
  - KALE은 KL 하한이며, Lipschitz 조건 하에 약한 수렴(weak convergence)을 측정 가능.[1]
  - 에너지 파라미터 및 베이스 파라미터에 대한 학습 안정성과 수렴 보장(gradient well-defined, smoothness).[1]

***

## 5. 한계

- **MCMC 비용**: 잠재 공간 MCMC는 베이스 차원에 의존하므로 잠재 차원이 클 경우 비용 증가  
- **에너지 공간 설계**: Lipschitz, smoothness 보장을 위한 네트워크 설계 및 정규화 필요  
- **이론적 확장**: 비compact 도메인, 비유한 가정 하 일반화 이론 추가 연구 요구  

***

## 6. 향후 연구 방향 및 고려 사항

- **효율적 잠재 샘플링**: Riemannian Langevin, adaptive MCMC로 샘플링 가속화  
- **에너지 함수 설계**: 스펙트럴 정규화, gradient penalty 최적화해 안정성 강화  
- **다양한 베이스**: 변분 흐름(normalizing flows), VAEs, autoregressive 모델 등과의 결합  
- **응용 확장**: 고해상도 이미지, 시계열, 그래프 데이터 등 다양한 도메인 적용 검토  
- **이론 심화**: 비compact 지원에서의 KALE 수렴 조건 완화 및 일반화 보장 연구  

이상의 방향을 통해 GEBM은 **저차원 지지 학습**과 **에너지 기반 재분배**를 결합한 강력한 생성 모델로서, 다양한 분야에서 일반화 성능과 샘플 품질을 획기적으로 향상시킬 잠재력을 지닌다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7cee3f83-3068-43c0-9684-d529c9005298/2003.05033v5.pdf)
