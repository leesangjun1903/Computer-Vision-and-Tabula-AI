# SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models | Super resolution

## 핵심 주장 및 주요 기여 
SRDiff는 **단일 이미지 초해상도(SISR)** 문제를 위해 최초로 **확산 확률 모델(diffusion probabilistic model)** 을 적용한 방법으로,  
1. **다양하고 현실감 있는 SR 결과**를 생성하며 mode collapse를 방지  
2. **잔차 예측(residual prediction)** 으로 수렴 속도를 높이고 안정적 학습  
3. **GAN 또는 흐름 기반(flow-based) 모델의 단점**인 불안정 학습·큰 모델 용량을 해소  
4. **잠재 공간 보간(latent interpolation)** 및 **콘텐츠 융합(content fusion)** 과 같은 유연한 이미지 조작 기능 제공  

## 1. 해결하고자 하는 문제
- **본질적 불완정(ill-posed) 문제**: 하나의 저해상도(LR) 영상에 다수의 고해상도(HR) 영상이 대응  
- 기존 PSNR 지향 방식은 **과도한 평활화(over-smoothing)**, GAN 기반은 **mode collapse** 및 **학습 불안정**, 흐름 기반은 **매우 큰 모델 용량** 문제

## 2. 제안 방법
### 2.1 확산 모델 기반 SISR
- **확산 과정(diffusion process)** (q): HR 잔차 $$x_0 = x_H - \mathrm{up}(x_L)$$에 점진적 화이트 노이즈 $$\epsilon$$ 추가  

$$
    q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I),\quad
    x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
  $$
  
- **역확산 과정(reverse process)** ($$p_\theta$$): 노이즈 예측기 $$\epsilon_\theta$$로 잔차 복원  

$$
    p_\theta(x_{t-1}|x_t) = \mathcal{N}\bigl(x_{t-1};\mu_\theta(x_t,t),\sigma_t^2I\bigr),\quad
    \mu_\theta = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Bigr)
  $$
  
- **학습 목표**: 변분 하한(ELBO) 기반 잔차 노이즈 예측  

$$
    \min_\theta \mathbb{E}\_{x_0,\epsilon,t}\bigl\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\,t)\bigr\|_1
  $$

### 2.2 모델 구조
- **잔차 예측(residual prediction)**: HR 영상 $$x_H$$가 아닌 LR 업샘플 $$\mathrm{up}(x_L)$$과의 차이 $$x_0$$를 예측  
- **LR 인코더**: RRDB 기반, LR 정보 $$x_L$$를 고차원 특징 $$\mathbf{e}$$로 인코딩  
- **노이즈 예측기**: U-Net 구조
  - 입력: 노이즈 있는 $$x_t$$ (3채널), 타임 임베딩 $$t$$, LR 인코딩 $$\mathbf{e}$$  
  - contracting–expanding 경로 및 multi-scale skip connection  

## 3. 성능 향상 및 한계
| 데이터셋 | 비교 방법     | PSNR↑  | SSIM↑  | LPIPS↓ | LR-PSNR↑ | σ(다양성)↑ |
|----------|---------------|-------|-------|-------|----------|-----------|
| CelebA 8×| SRFlow        | 25.32 | 0.72  | 0.109 | 51.15    | 5.32      |
|          | **SRDiff**    | **25.38** | **0.74** | **0.106** | **52.34** | **6.13**   |
| DIV2K 4× | SRFlow        | 27.09 | 0.76  | **0.120** | 49.96    | 5.14      |
|          | **SRDiff**    | **27.41** | **0.79** | 0.136 | **55.21** | **6.09**   |

- **잔차 예측**: PSNR 0.5dB↑, 학습 속도 2×  
- **T=100, 채널수=64** 트레이드오프로 빠른 추론 및 높은 품질 유지  
- **제한점**: 역확산 단계 수(T)가 늘어날수록 추론 속도 감소; 매우 고해상도 이미지에 대한 적용성 검증 필요

## 4. 일반화 성능 향상 가능성
- **잔차 대상 확대**: 블러 제거·노이즈 제거에도 적용 가능  
- **조건부 확산**: 다양한 입력 조건(세그멘테이션, 엣지 맵)으로 일반화  
- **잠재 보간**: 잠재 공간에서 보간·융합 실험을 통해 **도메인 간 전이** 가능성 확인

## 5. 향후 연구에 미치는 영향 및 고려사항
- **확산 모델의 SISR 활용**: 그간 GAN/Flow 기반에 집중된 SISR 분야에 새로운 패러다임 제시  
- **학습 안정성**: 단일 L1 손실로 안정적 학습, 다른 복원 과제(탈흐림, 탈노이즈)에 적용 기대  
- **고속 추론**: 단계 수 축소·지능형 스케줄링 연구 필요  
- **모델 경량화**: 모바일·엣지 환경에서 실시간 적용을 위한 경량 확산 모델 설계 검토  

---  

**주요 참고**  
SRDiff는 **확산 기반 생성 방식**이 SISR 분야에서 **과도한 매끄러짐, 모드 붕괴, 대형 모델** 문제를 동시에 해결하며, **다양성 및 응용 유연성**을 크게 향상시킨 최초의 방법이다. 앞으로 확산 모델을 활용한 다양한 영상 복원 및 합성 연구가 촉진될 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b6f7b483-977f-4246-98ca-9fc965cb5363/2104.14951v2.pdf
