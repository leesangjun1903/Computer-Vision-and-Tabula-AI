# Score-Based Generative Modeling through Stochastic Differential Equations | Image generation, Density estimation, Image inpainting

## 1. 핵심 주장 및 주요 기여  
Score-based generative modeling을 SDE(확률 미분방정식) 프레임워크로 통합하여,  
-  데이터 → noise로의 연속적 전환(순방향 SDE)과 noise → 데이터로의 역방향 SDE를 정의  
-  시간 의존 score 함수 ∇ₓ log pₜ(x)만으로 역방향 SDE를 풀어 새로운 샘플을 생성  
-  기존 SMLD, DDPM 방법을 각각 Variance Exploding(VE) SDE, Variance Preserving(VP) SDE의 이산화로 재해석  
-  Predictor–Corrector(PC) 샘플러, probability flow ODE(신경 ODE) 기반 샘플러, exact likelihood 계산, inverse problem 해결 등 다기능성 확보  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 문제  
- 기존 score matching 기반 모델(SMLD, DDPM)은 discrete noise scale step들로만 노이즈를 주입/제거  
- 복수 noise scale 간 불연속성, 샘플링 효율·질, likelihood 계산, inverse problem 제약  

### 제안 방법  
1) **연속 SDE 프레임워크**  
   순방향 Itô SDE:
   
   $$dx = f(x,t)\,dt + g(t)\,dw$$
   
   역방향 SDE(Anderson, 1982):
   
   $$dx = [\,f(x,t) - g(t)^2 ∇_x \log p_t(x)\,]\,dt + g(t)\,d\bar w$$

3) **Score 추정**  
   시간 의존 denoising score matching:

$$
   θ^* = \arg\min_θ \mathbb{E}\_{t∼U[0,T]}\mathbb{E}\_{x(0),x(t)|x(0)} \big\|s_θ(x(t),t) - ∇_{x(t)}\log p_t(x(t)|x(0))\big\|^2
$$

5) **VE·VP·sub-VP SDE 예시**  
   - VE: $$dx = \sqrt{\tfrac{d}{dt}σ^2(t)}\,dw$$  (분산 폭발)  
   - VP: $$dx = -\tfrac12β(t)\,x\,dt + \sqrt{β(t)}\,dw$$  (분산 보존)  
   - sub-VP: VP의 분산 상한 개선 버전  

6) **샘플링 알고리즘**  
   - Reverse diffusion sampler (일반적 Euler–Maruyama)  
   - Predictor–Corrector (PC) sampler: 역방향 SDE 예측 후 Langevin corrector 반복  
   - Probability flow ODE(신경 ODE) 샘플러: 적응적 스텝으로 빠른 샘플링, exact likelihood  

7) **모델 구조 개선**  
   - Anti-aliasing FIR 업/다운샘플, skip-rescale, BigGAN-type ResBlock, deeper residual block, progressive architecture  

### 성능 향상  
- CIFAR-10 unconditional 샘플: FID 2.20, Inception 9.89 (NCSN++ cont. deep, VE)  
- CIFAR-10 likelihood: 2.99 bits/dim (DDPM++ cont. deep, sub-VP)  
- CelebA-HQ 1024×1024 high-fidelity 샘플 최초 달성  
- class-conditional generation, inpainting, colorization 등 inverse problem 해결  

### 한계  
- SDE 샘플링은 GAN 대비 여전히 느림  
- 수많은 sampler hyper-parameter 재설정 필요  
- VE vs VP/sub-VP 중 선택, time discretization 민감  

## 3. 일반화 성능 향상 가능성  
- **연속 시간 프레임워크**: discrete noise scale 의존성 제거 → 새로운 SDE 설계로 다양한 data domain 적응  
- **Probability flow ODE**: exact likelihood, latent encoding 제공 → representation learning, transfer learning 활용  
- **PC 샘플러**: predictor-corrector 구조로 샘플 bias 교정 → out-of-distribution 샘플링 안정성 기대  
- **Architecture 개선**: anti-aliasing, deeper ResNet 블록 등으로 더욱 높은 표현력 확보 가능  

## 4. 향후 연구 영향 및 고려 사항  
- SDE 기반 생성 모델의 **다양한 SDE 설계**(nonlinear drift/diffusion, manifold SDE) 연구  
- **샘플링 속도 개선**: implicit/초해상 GAN 결합, adaptive sampler 자동 튜닝  
- **조건부·제약 생성**: inverse problem generalization, 다른 modality(텍스트·오디오)로 확장  
- representation 분석: probability flow latent 특성 연구, feature interpolation 및 활용  
- 실제 대규모 고해상도 데이터셋에 대한 **스케일 업**과 **효율적 학습** 전략 마련

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b1572b0c-2f02-4bda-bc35-8edb92bdbf7b/2011.13456v2.pdf

확률적 생성 모델에는 성공적인 분야가 두 가지 있는데, 하나는 Score matching with Langevin dynamics (SMLD)이고, 다른 하나는 DDPM (논문리뷰)이다.  
두 클래스 모두 천천히 증가하는 noise로 학습 데이터를 순차적으로 손상시키고 이를 되돌리는 방법을 생성 모델이 학습한다.  
SMLD는 score(ex. 로그 확률 밀도의 기울기)를 각 noise scale에서 학습하고 Langevin dynamics로 샘플링한다.  
DDPM은 학습이 tractable하도록 reverse distribution을 정의하여 각 step을 reverse한다. 연속적인 state space의 경우 DDPM의 목표 함수는 암시적으로 각 noise sclae에서 점수를 계산한다.  
따라서 이 두 모델 클래스를 함께 score 기반 생성 모델이라고 한다.

Score 기반 생성 모델은 이미지나 오디오 등의 다양한 생성 task에 효과적인 것으로 입증되었다.  
저자들은 확률적 미분 방정식(Stochastic Differential Equations, SDE)을 통해 이전 접근 방식을 일반화하는 통합 프레임워크를 제안한다.  
이를 통해 새로운 샘플링 방법이 가능하고 score 기반 생성 모델의 기능이 더욱 확장된다.

특히, 저자들은 유한 개의 noise 분포 대신 diffusion process에 의해 시간이 지남에 따라 진화하는 분포의 연속체(continuum)를 고려하였다.  
이 프로세스는 점진적으로 데이터 포인트를 random noise로 확산시키고 데이터에 의존하지 않고 학습 가능한 parameter가 없는 SDE에 의해 가능하다.  
이 프로세스를 reverse하여 random noise를 샘플 생성을 위한 데이터로 만들 수 있다.  
결정적으로, 이 reverse process는 reverse-time SDE를 충족하며, 이는 시간의 함수로서 주변 확률 밀도의 score가 주어진 forward SDE에서 유도할 수 있다.  
따라서 시간에 의존하는 신경망을 학습시켜 score를 추정한 다음 numerical SDE solver로 샘플을 생성하여 reverse-time SDE를 근사할 수 있다.



# Reference
- https://dlaiml.tistory.com/entry/Score-Based-Generative-Modeling-through-Stochastic-Differential-Equations
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/sbgm/
