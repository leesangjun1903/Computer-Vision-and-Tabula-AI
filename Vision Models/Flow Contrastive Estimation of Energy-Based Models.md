# Flow Contrastive Estimation of Energy-Based Models | 2019 · 131회 인용, Image generation

**핵심 주장 및 주요 기여**  
Flow Contrastive Estimation(FCE)은 에너지 기반 모델(EBM)과 흐름 기반 모델(flow model)을 **공동 학습**시켜, EBM의 표현 유연성과 flow model의 계산 효율성을 동시에 확보하는 새로운 학습 프레임워크를 제안한다.  
-  EBM 업데이트에 **노이즈 대비 추정(noise contrastive estimation, NCE)** 방식을 적용하되, flow model을 강력한 노이즈 분포로 활용  
-  Flow model 업데이트는 데이터 분포와의 **Jensen–Shannon divergence**를 근사 최소화  
-  GAN과 달리 두 모델 모두 **명시적 확률 분포**를 학습하며, semi-supervised learning에도 손쉽게 확장 가능[1]

***

## 1. 해결하고자 하는 문제  
기존 flow 모델은 밀도 평가와 샘플링이 효율적이지만 분포 가정(invertibility, 효율 정규화)이 실제 데이터에 부합하지 않을 수 있고, EBM은 유연하지만 MCMC 기반 샘플링 비용이 크다.  
따라서 이 두 접근의 **단점을 보완**하며 EBM의 정확도와 flow 모델의 합성 품질을 동시에 향상시킬 방법이 필요하다.[1]

***

## 2. 제안 방법  
### 2.1 에너지 기반 모델(NCE 기반)  
EBM은 에너지 함수 $$f_\theta(x)$$로 정의된 비정규화 밀도  

$$
p_\theta(x)=\frac{1}{Z(\theta)}\exp\bigl(f_\theta(x)\bigr)
$$  

NCE 목표: real vs. noise 샘플을 구분하는 로지스틱 회귀  

$$
\max_\theta\; \mathbb{E}_{p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)+q(x)} + \mathbb{E}_{q}\log\frac{q(x)}{p_\theta(x)+q(x)}
$$  

여기서 $$q(x)$$는 flow model이 생성하는 대조 분포.[1]

### 2.2 흐름 기반 모델(JSD 최소화)  
flow 모델은 가역 변환 $$x=g_\phi(z)$$, $$z\sim q_0(z)$$로 정의되며  

$$
q_\phi(x)=q_0\bigl(g_\phi^{-1}(x)\bigr)\bigl|\det \tfrac{\partial g_\phi^{-1}(x)}{\partial x}\bigr|
$$  

FCE 업데이트는 EBM과 동일한 **가치 함수(V)**의 minimax 게임  

$$
\min_\phi\max_\theta\;V(\theta,\phi)
=\mathbb{E}_{p_{\text{data}}}\log\frac{p_\theta(x)}{p_\theta(x)+q_\phi(x)}
+\mathbb{E}_{q_0}\log\frac{q_\phi(g_\phi(z))}{p_\theta(g_\phi(z))+q_\phi(g_\phi(z))}
$$  

Flow 모델의 $$\phi$$ 업데이트는 EBM이 데이터와 동일해질 때 **Jensen–Shannon divergence**  

$$\mathrm{JSD}(p_{\text{data}}\|q_\phi)$$ 근사 최소화 효과 [1].

***

## 3. 모델 구조  
- EBM: 간단한 4-layer CNN 또는 fully-connected 네트워크  
- Flow 모델: Glow 아키텍처(다수의 affine coupling 레이어)  
두 모델 모두 **매 iteration**마다 EBM 몇 회, flow model 몇 회를 교대로 업데이트.[1]

***

## 4. 성능 향상 및 한계  
### 4.1 합성 품질 개선  
- CIFAR-10, SVHN, CelebA 데이터에서 Glow-MLE 대비 FID 대폭 감소  
  - SVHN: 41.70 → 20.19  
  - CIFAR-10: 45.99 → 37.30  
  - CelebA: 23.32 → 12.21[1]

### 4.2 밀도 추정 정확도  
- 2D synthetic data에서 NCE 대비 log-density MSE 감소  
- 랜덤 초기화한 Glow에서도 빠른 수렴[1]

### 4.3 Unsupervised feature learning  
- SVHN에서 top-layer feature + linear classifier: labeled 예시 수가 적을 때(supervised 대비) 더 우수한 성능  
- multi-layer feature + SVM: GAN, CD 기반 EBM 등 대비 낮은 분류 오류[1]

### 4.4 한계  
- Flow model 초기화에 따라 학습 안정성 편차  
- 높은 차원 데이터에서 MCMC-free EBM 학습의 이론적 수렴 보장 미흡  
- 계산량: 매 iteration마다 두 모델 업데이트로 리소스 요구 증가

***

## 5. 일반화 성능 향상 관점  
FCE는 EBM이 flow 모델과 적응적으로 경쟁하며 주요 모드를 정확히 포착하도록 유도하므로, **일반화 성능**이 크게 향상될 수 있다.  
- JSD 최소화 관점에서 flow 모델이 과도한 퍼짐(over-dispersion)을 수정  
- NCE의 대조 분포가 학습 과정에 따라 점진 개선되어, EBM이 **실제 데이터 분포**에 더욱 근접  
- semi-supervised 설정 시, 소수 라벨 데이터만으로도 EBM이 **매끄러운 클러스터링** 학습[1]

***

## 6. 향후 연구 방향 및 고려 사항  
- **다른 정규화 모델**(auto-regressive 등)과의 공동 학습 일반화  
- adversarial contrastive divergence, divergence triangle 등 **다양한 결합 전략** 탐색  
- 대규모·고차원 이미지에 대한 **안정성** 및 **수렴 속도** 개선  
- EBM의 **정규화 상수 추정** 이론 강화 및 MCMC-free 학습에 대한 **수렴 보장** 연구

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1c77702e-3cd2-42c1-91c4-35b7c8dca328/1912.00589v2.pdf)
