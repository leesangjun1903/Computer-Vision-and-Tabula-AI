# DiffPIR : Denoising Diffusion Models for Plug-and-Play Image Restoration | 2023 · 323회 인용, Image restoration

# 주요 주장 및 기여 요약  
**“Denoising Diffusion Models for Plug-and-Play Image Restoration”** 논문은 기존 Plug-and-Play 방식이 주로 판별적(Discriminative) Gaussian 디노이저에 의존하는 한계를 지적하고, **생성적(Generative) 디노이저로서의 확산 모델(Diffusion Models)을 Plug-and-Play 프레임워크에 통합**한 **DiffPIR**을 제안한다. 이를 통해 낮은 신호 대 잡음 환경에서도 **100회 이하의 Neural Function Evaluations(NFEs)**로 **최고 수준의 복원 충실도(PSNR)와 지각 품질(FID, LPIPS)**를 동시에 달성한다.[1]

***

## 1. 해결하고자 하는 문제  
이미지 복원(Inverse Problems)은 저해상도·블러·결측 등 다양한 열화(Degradation) 모델 $$y=H x + n$$에서 원본 $$x$$를 복원하는 문제다. Plug-and-Play 방법은 변수 분리(ADMM/HQS)를 통해  

$$
\min_x \frac{1}{2}\|y - Hx\|_2^2 + \lambda P(x)
$$  

를 데이터 항과 사전 항으로 분할한 뒤, 사전 항을 Gaussian 디노이저로 풀어 안정적 복원을 이룬다. 하지만 기존 방식은 복잡한 분포를 모델링하는 데 한계가 있다.[1]

***

## 2. 제안 방법  
### 2.1. HQS 기반 분할  
Auxiliary 변수 $$z$$를 도입하여  

$$
\begin{aligned}
z_k &= \arg\min_z \tfrac{1}{2\beta}\|z - x_k\|_2^2 + P(z),\\
x_{k+1} &= \arg\min_x \tfrac{1}{2\sigma^2}\|y - Hx\|_2^2 + \tfrac{1}{2\beta}\|x - z_k\|_2^2
\end{aligned}
$$  

로 나누어 반복 해결한다.[1]

### 2.2. 확산 모델을 Generative 디노이저로 활용  
- **Score-based Diffusion**: 노이즈 단계 $$t$$에서의 score 함수 $$s(x_t,t)$$는 이상적인 Gaussian 디노이저이며, 역과정에서 본래 이미지를 점진 복원한다.  
- **DDPM/DDIM**: U-Net 기반 노이즈 예측기 $$\epsilon_\theta(x_t,t)$$를 학습하고, 역확산 스텝  

$$
  x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr) + \sigma_t z
  $$  
 
  으로 샘플링한다.[1]

### 2.3. DiffPIR 알고리즘  
매 스텝마다  
1. **Prior 서브문제**: $$z_t\leftarrow$$ 확산 모델로부터 예측된 노이즈 제거(Generative 디노이저).  
2. **Data 서브문제**: $$x_{t-1}\leftarrow$$ 측정 $$y$$와 결합한 proximal 연산(Closed-form 또는 수치적 추정)  
3. **Noise Injection**: 역확산 샘플링으로 노이즈 재주입  
으로 구성된다.[1]

***

## 3. 모델 구조  
- **U-Net 기반 노이즈 예측기** $$\epsilon_\theta$$를 사전 훈련된 DDPM/DDIM 체크포인트로 사용.  
- **HQ S–DDIM**: HQS 분할과 DDIM 가속 샘플링을 결합하여 **최대 100 NFEs** 내에서 고품질 복원을 실현.  
- **Data 서브문제 해법**  
  - Inpainting: $$y = M\odot x$$ → $$x = M\odot y + (1-M)\odot z$$  
  - Deblurring/SR: FFT나 IBP, 근사 커널을 이용한 Closed-form 해.  

***

## 4. 성능 향상  
- **정량적**: FFHQ·ImageNet에서 Gaussian Deblur·Motion Deblur·×4 SR 모두에서 **FID, LPIPS 최상위**, PSNR 또한 경쟁력 확보(≤100 NFEs).[1]
- **정성적**: 기존 DDRM·DPS 대비 디테일 보존, 낮은 블러 현상, 다양성 있는 샘플 생성 가능.  
- **가속화**: DDIM 기반 비마르코프 샘플링으로 1000→100단계 축소 시에도 품질 유지.

***

## 5. 한계  
- **연산 비용**: 100 NFEs라도 대규모 배치나 고해상도에선 여전히 부담.  
- **사전 학습 모델 의존**: 확산 모델 품질에 복원 성능이 크게 좌우됨.  
- **하이퍼파라미터 민감도**: $$\lambda,\,\beta$$ 및 샘플링 스케줄에 따라 결과 편차 존재.[1]

***

## 6. 일반화 성능 향상 관점  
- **Generative Prior**: 확산 모델의 학습된 데이터 분포 표현력이 Plug-and-Play prior로 활용되어 **분포Shift 저항성 강화**.  
- **Task-agnostic**: Blurring·SR·Inpainting 등 다양한 열화 모델에 일관된 알고리즘 적용 가능.  
- **하이퍼파라 조정**: $$\beta$$ 및 시작 단계 $$t_{\text{start}}$$ 조정으로 NFEs 감소 및 특정 작업에 맞춘 샘플링 최적화.

***

## 7. 연구적 영향 및 향후 고려점  
- **플러그인 구조 확장**: 타 Generative 모델(텍스트·음성 등)에도 유사한 HQS–Diffusion 통합 적용 가능.  
- **샘플링 가속화**: 더욱 효율적 비마르코프·스케줄링 연구로 실시간 복원 목표.  
- **하이퍼파라미터 자동화**: 메타러닝 기반 자동 최적화로 민감도 완화.  
- **실제 의료·위성 영상 적용**: 고해상도·특수 노이즈 환경에서 성능 검증 및 모델 경량화 연구 필요.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/11efb429-b9ef-4f2c-8ece-f5b7bc4be96b/2305.08995v1.pdf)
