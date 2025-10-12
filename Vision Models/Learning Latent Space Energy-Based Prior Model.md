# Learning Latent Space Energy-Based Prior Model | 2020 · 170회 인용, Image generation

**주요 주장 및 기여**  
“Learning Latent Space Energy-Based Prior Model”(NeurIPS 2020)는 생성 모델(generator model)의 잠재 공간(latent space)에 에너지 기반 모델(EBM)을 **사전(prior) 분포**로 도입하여, 기존의 단순 Gaussian 사전 대신 데이터의 구조적 규칙을 학습하도록 한다. 저차원 잠재 공간이기에 MCMC 샘플링이 효율적이며, EBM과 생성 모델을 **공동 최대우도학습(MLE)**으로 학습하여 이미지·문장 생성과 이상치 탐지 성능을 크게 향상시킨다.  
주요 기여는 다음과 같다.  
- 잠재 공간 상단에 EBM 사전 분포를 결합한 새로운 생성 모델 제안.[1]
- 단기(short-run) MCMC 샘플링 기반의 효율적 MLE 학습 알고리즘 개발.[1]
- 이론적 기반 제공: MLE와의 관계, perturbation 관점에서 단기 MCMC 학습 정당화.[1]
- 짧은 MCMC만으로도 잠재 공간 샘플링의 수렴성·mixing 성능 향상 입증.[1]
- 이미지·텍스트 생성 및 이상치 탐지에서 SOTA급 성능 달성.[1]

***

## 1. 해결하려는 문제  
기존 VAE·GAN 계열 생성 모델은 잠재 벡터 z에 대개 **균일 또는 등방 가우시안**을 사전 분포로 가정하지만, 이는 실제 데이터의 복잡한 다봉성(multi-modality)을 반영하지 못한다.  
따라서  
- 단순 사전으로는 생성 모델의 표현력이 제한되고,  
- 데이터 공간에서 직접 EBM을 학습하면 MCMC 샘플링이 고차원·다봉성 공간에서 비효율적임  

두 문제를 동시에 해결하기 위해 **잠재 공간**에 EBM을 두고, 생성 네트워크(top-down network)의 표현력을 활용하여 효율적이고 풍부한 사전 분포를 학습하고자 한다.[1]

***

## 2. 제안 방법

### 2.1 모델 구조  
관찰 데이터 $$x\in\mathbb{R}^D$$, 잠재 벡터 $$z\in\mathbb{R}^d$$에 대하여  

$$
p_{\theta,\alpha}(x,z)
  = p_\alpha(z)\,p_\theta(x\mid z),
$$  

사전 분포 $$p_\alpha(z)$$는 **EBM**으로 정의:

$$
p_\alpha(z)
  = \frac{1}{Z(\alpha)}\exp\bigl(f_\alpha(z)\bigr)\,p_0(z),
$$

여기서 $$p_0(z)$$는 기준 분포(등방 가우시안), $$f_\alpha(z)$$는 MLP 형태의 **negative energy** 함수이다.  
생성 분포 $$p_\theta(x\mid z)$$는 이미지용 컨볼루션 디코더 또는 텍스트용 RNN 디코더로 구성된다.[1]

### 2.2 학습: 단기 MCMC 기반 MLE  
MLE의 그래디언트는  

$$
\nabla_\alpha\log p_\alpha(z)
  = \mathbb{E}_{p_\theta(z\mid x)}[\nabla f_\alpha(z)]
    - \mathbb{E}_{p_\alpha(z)}[\nabla f_\alpha(z)]
$$  

$$
\nabla_\theta\log p_\theta(x\mid z)
  = \mathbb{E}_{p_\theta(z\mid x)}[\nabla_\theta \log p_\theta(x\mid z)].
$$

하지만 정확한 샘플링은 고비용이므로, **단기(short-run) MCMC**를 도입한다.  
- 사전 샘플링: $$z^{(0)}\sim p_0$$, K 단계 Langevin dynamics 적용  
- 사후 샘플링: $$z^{(0)}\sim p_0$$, K 단계 Langevin dynamics를 $$p_\theta(z\mid x)$$ 목표로 수행  

$$
z^{(k+1)}=z^{(k)}+\tfrac{s}{2}\nabla \log p(z^{(k)})+\sqrt{s}\,\epsilon^{(k)}.
$$

이후 이 샘플들로 그래디언트를 근사하여 $$\alpha,\theta$$를 갱신한다.[1]

***

## 3. 성능 향상 및 한계

### 3.1 이미지·텍스트 생성 성능  
SVHN·CIFAR-10·CelebA에서 **FID** 및 **MSE** 지표에서 VAE·SRI·2sVAE·RAE 대비 우월한 성능을 보인다.  
예: SVHN FID 29.44 (기존 최저 35.23) 및 MSE 0.008 (기존 0.011).[1]
텍스트 생성에서도 FPPL·RPPL·NLL 지표에서 SOTA 성능 달성.[1]

### 3.2 이상치 탐지  
MNIST 각 클래스 이상치 설정 시 **AUPRC** 지표에서 VAE·MEG·BiGAN-based 방법 대비 평균 0.45 이상으로 우수.[1]

### 3.3 계산 비용  
단기 MCMC로 인해 VAE 대비 **4배 느린 학습 속도** 발생. 그러나 텍스트 모델은 우수한 후행 샘플 품질로 총 학습 시간은 큰 차이 없음.[1]

### 3.4 한계  
- MCMC 단계 수 $$K$$가 적으면 편향 발생, 많으면 계산 비용 증가.  
- 잠재 공간 차원 $$d$$가 증가하면 EBM 복잡도·파라미터 증가로 학습 불안정 가능.  
- 데이터 공간 EBM 확장 시 MCMC 난제 여전.

***

## 4. 일반화 성능 향상 관점

잠재 공간 EBM은 낮은 차원에서 MCMC가 잘 수렴하고, 생성 네트워크가 다봉성 데이터 분포를 보완하므로 **샘플 다양성·표현력**이 크게 개선된다.  
단기 MCMC는 빠르게 모드를 탐색하고, empirically posterior와 prior 간 차이를 줄여 **잠재 표현의 견고성**과 **일반화 성능**을 높인다.[1]
특히, **장기 MCMC**를 실시해도 오버시추레이션(oversaturating) 없이 안정적 합성 결과를 보였고, 이는 잠재 EBM의 **mixing 성능**과 **mode covering** 능력을 나타낸다.[1]

***

## 5. 향후 연구의 영향 및 고려 사항

- **Amortized Inference 통합**: 잠재 및 합성 MCMC를 별도 네트워크로 근사하여 학습 효율 제고 가능.[1]
- **다층 층위 EBM**: 잠재 공간뿐 아니라 중간 층에도 EBM correction 적용해 데이터 구조 반영력 강화 연구.  
- **하이퍼파라미터 튜닝**: MCMC 스텝 수(K), 스텝사이즈(s) 민감도 분석 및 adaptive scheme 도입 필요.  
- **잠재 공간 확장**: 고차원 잠재 공간에서 MCMC 효율성·안정성을 유지하는 새로운 샘플링 기법 모색.  
- **응용 분야**: 이상치 탐지·표현 학습·반지도 학습·분자 생성 등 다양한 도메인 확장 연구 기대.

이 논문은 생성 모델에 **잠재 공간 EBM**을 더함으로써 표현력과 일반화 성능을 동시에 끌어올릴 수 있음을 보였으며, **에너지 기반 모델의 실용적 학습** 방향성을 제시했다. 향후 연구에서는 MCMC 효율화, 층위별 EBM 적용, Amortized 기법 통합을 통해 더 넓은 응용 및 성능 향상이 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7d383664-d629-4000-a009-d06d93d3bb59/2006.08205v2.pdf)
