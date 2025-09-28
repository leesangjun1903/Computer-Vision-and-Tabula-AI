# Denoising MCMC for Accelerating Diffusion-Based Generative Models | Image generation, Accelerating Technique

# 핵심 요약  
**Denoising MCMC(DMCMC)**는 데이터-확산시간 결합 공간에서 MCMC를 수행한 후, reverse-SODE 적분기로 후처리하여 확산 모델의 샘플링 속도와 품질을 동시에 향상시키는 **새로운 범용 프레임워크**입니다.[1]

***

## 1. 문제 정의  
기존 **확산 모델**은 노이즈에서 깨끗한 데이터를 생성하기 위해 수천 단계의 reverse-SODE/ODE 적분을 필요로 하며 연산 비용이 크고, 전통적 **MCMC**는 고차원 다봉(multimodal) 분포에서 모드 간 이동성이 부족합니다.[1]

***

## 2. 제안 방법  
### 2.1 데이터-확산시간(product space)에서의 MCMC  
- 확산시간 $$t\in[\tau_{\min},\tau_{\max}]$$를 데이터 공간 $$X$$에 결합하여 $$(x,t)$$ 공간에서  

$$
    p(x,t)\propto p_t(x)\,p(t)
  $$  
  
형태의 조인트 분포로 MCMC를 수행.[1]

- 여기서 $$p_t(x)=\int p(x|\tilde x)p_{\tilde x}(\tilde x)\,d\tilde x$$는 노이즈 레벨 $$t$$ 하의 데이터 분포이고, $$p(t)$$는 원하는 시간 분포(prior)이다.  

### 2.2 Denoising 단계: reverse-SODE 적분  
- MCMC로 얻은 샘플 $$(x_n, t_n)$$에 대해, reverse-SODE를 $$t_n\to 0$$ 구간만 적분하여 최종 깨끗한 샘플을 생성  

$$
    x_{t=0} = \mathrm{Integrate}\bigl[\mathrm{reverse\text{-}SODE}\bigr]\bigl(x_n, t_n\to0\bigr)
  $$

- MCMC가 데이터 매니폴드 부근에서 탐색하므로 $$t_n$$이 작게 유지되어 적분 구간이 짧아지고, 동일 연산 예산에서 오차가 감소하여 효율이 대폭 개선된다.[1]

***

## 3. Denoising Langevin Gibbs(DLG)  
DMCMC의 구체적 구현으로, **Langevin dynamics**와 **Gibbs 샘플링**을 결합한 DLG를 제안:  
1. **$$x$$ 업데이트** (Eq. 15):  

$$
     x_{n+1} = x_n + \tfrac{\epsilon}{2}\,\nabla_x\log p(x_n|t_n) + \sqrt{\epsilon}\,\eta
     \quad(\eta\sim\mathcal{N}(0,I))
   $$

2. **$$t$$ 업데이트** (Eq. 16):  
   - 사전 학습된 노이즈-레벨 분류기 $$q(x)$$로 $$x_{n+1}$$의 레벨을 추정  
   - 가장 높은 확률의 레벨 인덱스를 $$t_{n+1}$$로 선택  
3. **Denoising**: 얻어진 $$(x_n,t_n)$$에 reverse-SODE 적분  
4. **하이퍼파라미터**: step size $$\epsilon$$, MCMC 스킵 간격 $$n_{\mathrm{skip}}$$, denoise NFE 비율 $$n_{\mathrm{den}}/n$$ 등을 통해 균형 조정 가능  

***

## 4. 성능 향상  
- **CIFAR-10**:  
  - 10 NFE에서 **3.86 FID**, 20 NFE에서 **2.63 FID** 달성, 기존 최고치(4.17/2.86 FID) 대비 유의미한 개선.[1]
- **CelebA-HQ-256**:  
  - 160 NFE에서 **6.99 FID**, 기존 최고의 4000 NFE 기반 7.16 FID 기록을 크게 경신.[1]
- 모든 테스트한 6개 reverse-SODE 적분기에 대해 유의미한 가속 및 품질 향상 확인  

***

## 5. 한계 및 일반화 성능  
- **하이퍼파라미터 민감도**: $$\epsilon$$과 $$n_{\mathrm{den}}/n_{\mathrm{skip}}$$ 비율에 따라 최적 성능 구간이 존재하므로, 데이터 차원별 튜닝 필요.[1]
- **분류기 학습**: 노이즈-레벨 분류기 $$q$$는 데이터셋마다 재학습이 필요하며, 고해상도에서는 비용 증가 가능성 존재.  
- **일반화**: MCMC가 모델 학습 분포 외 영역으로 과도하게 이동하는 경우, 품질 보장이 어려울 수 있음.  

***

## 6. 향후 연구 영향 및 고려 사항  
- **확산 모델 가속화의 새로운 방향**: reverse-SODE 적분 기법과 독립적으로 적용 가능하며, 향후 다양한 MCMC 기법(Hamiltonian, Riemannian 등)과 결합 연구 기대  
- **범용성 확대**: 비주얼 데이터 외 텍스트·오디오·과학 시뮬레이션 등 고차원 분포 샘플링에도 적용 가능성  
- **하이퍼파라미터 자동화**: 메타러닝·AutoML을 통한 $$\epsilon$$, $$n_{\mathrm{den}}$$ 셋업 자동 최적화 연구  
- **안정성·일반화**: 분포 이동·도메인 적응 상황에서 MCMC 탐색 안정성 확보 방안 모색  

DMCMC는 diffusion 기반 생성 모델의 샘플링 효율을 근본적으로 개선할 수 있는 **orthogonal한 접근법**으로, 이후 다양한 모델·분포에서의 확장과 자동화 연구가 중요한 과제로 남는다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/10c15702-4ffb-4069-8609-9626a12a8b11/2209.14593v1.pdf)
