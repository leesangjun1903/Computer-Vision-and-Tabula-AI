# iDDPM : Improved Denoising Diffusion Probabilistic Models | Image generation, Image denoising

## 1. 핵심 주장 및 주요 기여  
**Improved DDPM**은 기존 DDPM의 높은 샘플 품질을 유지하면서  
1) **로그-우도(log-likelihood) 성능을 크게 향상**시키고,  
2) **샘플링 속도를 수십 배** 단축할 수 있음을 보인다.  

이를 위해  
- 역확산 과정의 분산(Σθ)을 학습 가능한 파라미터로 도입  
- 선형 대신 **코사인(noise) 스케줄** 채택  
- 변분 하한(VLB) 최적화 시 **중요도 샘플링** 적용  
- μθ와 Σθ에 상이한 손실을 결합한 **하이브리드 목적 함수** 제안  

## 2. 해결 과제, 제안 방법, 모델 구조  

### 2.1. 해결 과제  
- 기존 DDPM은 FID 기준 샘플 품질이 우수하나, 로그-우도가 Autoregressive 모델 대비 열위  
- 샘플링 단계(T≈1,000) 수가 매우 많아 실제 활용 시 속도 문제  

### 2.2. 제안 방법  
1) **학습 가능한 분산 Σθ(xt,t)**  
   - 기존: $Σθ$ 고정($β_t$ 또는 $\tildeβ_t$)  
   - 제안: $v(x_t,t)$ 출력 후  

  $$
       Σθ = \exp\bigl(v\ln β_t + (1-v)\ln\tildeβ_t\bigr)
  $$  

2) **하이브리드 목적 함수**  

$$
     L_{\mathrm{hybrid}} = L_{\mathrm{simple}} + λ L_{\mathrm{vlb}},\quad λ=10^{-3}
$$  
   - $$L_{\mathrm{simple}} = \mathbb{E}_{t,x_0,ϵ}\|\;ϵ - ϵ_θ( x_t,t)\|^2$$  
   - $$L_{\mathrm{vlb}}$$는 변분 하한(식 4–7)  
3) **코사인(noise) 스케줄**  
   
$$
     \barα_t = \frac{\cos^2\bigl((t/T + s)/(1+s)\cdot \tfrac\pi2\bigr)}{\cos^2\bigl(s/(1+s)\cdot \tfrac\pi2\bigr)},\quad s=0.008
$$  

4) **중요도 샘플링**으로 VLB 용어 분산 감소  

$$
     p_t \propto \sqrt{\mathbb{E}[L_t^2]},\quad L_{\mathrm{vlb}} 
     = \mathbb{E}_{t\sim p_t}[L_t/p_t]
$$  

### 2.3. 모델 구조  
- UNet 기반 아키텍처  
- 멀티헤드 어텐션(4 heads) 사용, 해상도 16×16·8×8에서도 적용  
- 시간 t 및 클래스 조건(class-conditional)은 GroupNorm 가중치·바이어스로 삽입  

## 3. 성능 향상 및 한계  

| 데이터셋      | 기본 DDPM (Ho et al.) | Improved DDPM            |
|---------------|-----------------------|--------------------------|
| CIFAR-10 NLL  | 3.70 bits/dim         | 2.94 bits/dim            |
| CIFAR-10 FID  | 25.0                  | 2.90                     |
| ImageNet-64 NLL | 3.77 bits/dim       | 3.57 bits/dim            |
| ImageNet-64 FID | 32.5                | 19.2                     |
| 샘플링 단계   | 1,000                | **50–100**로 고품질 유지 |

- **로그-우도**: 주요 기존 모델과 비슷하거나 우월  
- **샘플 품질(FID)**: GAN 급(클래스조건 BigGAN-deep 대비 FID 2.92 vs. 4.06)  
- **모드 커버리지**: Precision은 유사, Recall 대폭 향상 → 더 넓은 분포 커버  
- **샘플링 속도**: 단계 수 4K→50로, 수십 배 단축  

**한계**  
- VLB 최적화 시 샘플 품질(FID) 악화  
- 대용량 모델·장시간 학습 시 과적합 관찰  
- 로그-우도와 샘플 품질 간 트레이드오프  

## 4. 일반화 성능 향상 가능성  
- 학습 가능한 Σθ와 중요도 샘플링은 **최초 몇 단계(t≈0, T)** 에 집중된 VLB 손실을 줄여 전 범위에 걸친 **정보 보존**을 강화  
- **코사인 스케줄**은 초기·후기 노이즈 변화를 완만하게 유지 → 과도한 노이즈 파괴 방지  
- 이들 기법은 **다양한 해상도·도메인**(e.g. 오디오, 고해상도) 일반화 가능성 보유  

## 5. 향후 연구에 미치는 영향 및 고려 사항  
- **속도-품질 트레이드오프** 최적화: 단계 수 절감과 품질 유지 기법 추가 탐색  
- **VLB 직접 최적화** 안정화: 중요도 샘플링 외 분산 축소 기법 연구  
- **과적합 방지**: 대규모·고해상도 데이터에서 regularization·early stopping 전략  
- **다양한 조건부 확산모델**: 텍스트·음성 등 복합 조건(task)으로 확장  

> **주요 시사점**: Improved DDPM은 likelihood-based 모델과 GAN 장점을 결합, 실제 응용에 적합한 고품질·효율적 생성모델의 새 지평을 열었다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a7c22396-d491-4070-8e0d-4b435dac7116/2102.09672v1.pdf
