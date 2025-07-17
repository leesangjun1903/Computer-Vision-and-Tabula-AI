# Diffusion Models, Image Super-Resolution And Everything: A Survey

**주장**  
“Diffusion Models, Image Super-Resolution And Everything: A Survey” 논문은 확산모델(Diffusion Models; DMs)이 이미지 초해상도(Super-Resolution; SR) 분야에서 기존 기법(GAN, VAE, Flow-based) 대비  
- 학습이 안정적이고  
- 샘플 품질(인간 지각 품질) 우수  
- 인간 평가자에게 가장 현실적으로 인식됨  

을 보여주며, SR 연구 전반에 걸쳐 DMs의 원리·응용·개선점을 통합적으로 정리한다.  

**주요 기여**  
1. **통합적 이론 정리**:  
   - DDPM, Score-based SGM, SDE 기반 확산의 수학적 토대 비교  
   - GAN/VAE/Flow와의 관계 및 SR 분야에서의 장단점 대비  
2. **SR 특화 기법 분류**:  
   - **상태 도메인(State Domain)**: 픽셀·잠재공간·주파수·잔차 기반 SR  
   - **조건화(Conditioning)**: LR 참조·예측 SR·특징 참조·텍스트 임베딩  
   - **가이드(Training Guidance)**: Classifier guidance vs. classifier-free guidance  
   - **오염 공간(Corruption Space)**: Gaussian 노이즈, Cold Diffusion, I2SB 등 대체 오염 기법  
3. **성능 분석**:  
   - 대표 모델(SR3, SRDiff, Latent Diffusion 등)의 PSNR·SSIM·LPIPS 성능 비교  
   - 4×·16× SR에서 정량·정성 우수성 및 일관성(consistency) 검증  
4. **응용 분야 정리**:  
   - 의료영상(MRI)·얼굴 복원·원격 탐사 등 도메인별 맞춤 SR 기법  
5. **현안·미래 과제**:  
   - 고연산 비용·효율적 샘플링·배치 크기 제약에 따른 색 편향(color shift)·비지도(Zero-shot) SR 강화 등  

# 상세 설명  

## 1. 해결 과제  
- SR의 본질적 난제: 여러 HR 해상도가 가능해 *다해성(ill-posedness)* 존재  
- 기존 딥러닝 SR(Regression, GAN, VAE, Flow)의 한계  
  -  부드러운 질감(Regression)  
  -  불안정한 학습·모드 붕괴(GAN)  
  -  낮은 해상도 복원(VAE)  
  -  고컴퓨팅 비용(Flow)  

## 2. 제안 방법론  

### 2.1 확산모델 수식  
- **Forward (q)**: $$q(z_t|z_{t-1})=\mathcal{N}(z_t;\sqrt{1-\alpha_t}z_{t-1},\,\alpha_t I)$$  
- **Reverse (p\_θ)**: $$p_\theta(z_{t-1}|z_t)=\mathcal{N}(z_{t-1};\mu_\theta(z_t,t),\,\Sigma_\theta(z_t,t))$$  
- 손실: 변분 하한(VLB) 또는 노이즈 재구성 손실 $$\mathbb{E}\_{t,\,z_0,\,\epsilon}\|\epsilon-\epsilon_\theta(z_t,t)\|^2$$  

### 2.2 모델 구조  
- **기본 네트워크**: U-Net 기반 잔차 블록  
- **SR3**: SR3 φθ(x,z_t,γ_t)로 노이즈 예측, 식 (29)–(31) 적용  
- **SRDiff**: HR–LR 차이(Residual) 예측으로 수렴 속도·안정성 개선  
- **Latent Diffusion (LDM)**: VQ-GAN 잠재공간에서 확산, 계산량 감소  
- **Wavelet Domain**: DiWa, WaveDM 등 주파수 분리 후 고주파 처리  

# V: 이미지 초해상도(Image SR)를 위한 확산 모델(Diffusion Models)

이 절에서는 확산 모델이 이미지 초해상도(Super-Resolution, SR) 문제에 어떻게 적용되는지, 주요 기법과 구성 요소를 중심으로 다룹니다. 크게 네 가지 토픽으로 구성됩니다.

## 1. 확산 모델의 구체적 구현 (Concrete Realization)

- **DDPM 기반 SR3**  
  - 입력: 저해상도 이미지 $$x$$  
  - 노이즈 생성: 순방향 프로세스에서 $$x$$에 점차 가우시안 노이즈를 더해 $$z_T$$를 얻음  
  - 역방향 복원: U-Net 기반 모델 $$\phi_\theta(x, z_t, t)$$이 시점 $$t$$의 노이즈 $$z_t$$에서 원본 HR 이미지 $$z_0$$를 예측  
  - 한 스텝 복원 식:  

$$
      z_{t-1}
      = \frac{1}{\sqrt{\alpha_t}}\Bigl(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\;\phi_\theta(x,z_t,t)\Bigr)
      + \sqrt{1-\alpha_t}\,\epsilon_t
      \quad(\epsilon_t\sim\mathcal{N}(0,I))
$$

  - 특징: 안정적 학습, 높은 지각 품질

- **잔차 예측 SRDiff**  
  - SR3와 유사하지만 HR 이미지 자체가 아니라 “업샘플된 LR”과 HR의 차이(residual)를 예측  
  - 잔차만 학습하므로 수렴 속도 및 안정성 향상  

## 2. 가이드(Guide) 기법

모델이 LR 조건을 충분히 반영하도록 **학습 중** 또는 **샘플링 시** 조정하는 방법

- **Classifier Guidance**  
  - 별도 분류기(classifier)로부터 $$\nabla_z\log p(x|z)$$를 샘플링 과정에 추가  
  - 고품질 샘플 생성 가능하나, 노이즈 입력을 잘 처리하는 분류기 필요  

- **Classifier-Free Guidance**  
  - 동일 모델에서 조건(condition)과 무조건(unconditional) 예측을 모두 수행  
  - 식:  

$$
      s_\text{guided} = (1-\lambda)\,s_\text{uncond} + \lambda\,s_\text{cond}
      \quad(\lambda>1\text{일수록 조건 반영↑})
  $$
  
  - 추가 분류기 없이도 성능 개선  

## 3. 상태 도메인(State Domain)

확산 과정이 동작하는 표현 공간을 달리함으로써 계산량·성능을 최적화

1. **픽셀 공간 (Pixel Space)**  
   - SR3, SRDiff처럼 RGB 화소 그대로 확산  
   - 직관적이지만 계산량 큼  

2. **잠재 공간 (Latent Space)**  
   - VQ-GAN, VAE 등으로 입력 이미지를 저차원 잠재코드로 인코딩 후 그 공간에서 확산  
   - 예: Latent Diffusion Model (LDM)  
   - 장점: 메모리·연산량 감소, 속도 향상  

3. **주파수 공간 (Frequency/Wavelet Space)**  
   - Wavelet 변환을 통해 고·저주파 성분 분리  
   - 확산은 특정 밴드(예: 저주파)만 또는 전체 밴드에서 수행  
   - 예: DiWa, WaveDM  

4. **잔차 공간 (Residual Space)**  
   - LR→HR 차이 정보만 확산  
   - 학습 목표 집중, 수렴 안정성  

## 4. 오염 공간(Corruption Space)

순방향(noising) 과정에서 어떤 형태의 왜곡을 사용하는지에 따라 SR 특성 변화

- **가우시안 노이즈** (표준)  
- **Cold Diffusion**  
  - 임의 변형·왜곡(예: 블러, 색상 변환 등) 반복 적용 후 복원  
- **Image-to-Image Schrödinger Bridge (I2SB)**  
  - 순방향 종단을 “실제 저해상도 이미지”로 설정  
  - 복원 과정(step 수 2~10)에서 높은 효율성  
- **InDI**  
  - 직접 매핑 방식으로 SR  

## 요약 및 활용 시 고려사항

- **연산 비용 vs. 품질**: 픽셀 공간 확산은 품질 우수하나 비용이 크고, 잠재·주파수 공간 확산은 경량화 가능  
- **가이드 조절**: 조건 반영 강도를 $$\lambda$$로 제어하여 품질·다양성 균형  
- **오염 스페이스 실험**: 다양한 왜곡 기법을 활용하면 모델의 복원 성능 및 효율 개선 가능  

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e16bfc65-f12b-4bb8-9528-a974f5a6cdf4/2401.00736v3.pdf

### 2.3 성능 향상 기법  
- **Guidance**:  
  -  Classifier guidance: $$\nabla_z\log p(z|x)=\nabla\log p(z)+\lambda\nabla\log p(x|z)$$  
  -  Classifier-free: $$(1-\lambda)\nabla\log p(z)+\lambda\nabla\log p(z|x)$$  
- **Efficient Sampling**: DDIM, DPM-solver, 지식 증류  
- **Likelihood 개선**: iDDPM(cosine noise schedule), likelihood weighting  

## 3. 일반화 성능 향상  

- **Zero-shot SR**: ILVR, DDNM, DPS, DDRM 등 비학습 SR 기법  
- **Correlation**:  
  -  *Consistency* MSE 측정 → DMs의 낮은 랜덤 편차  
  -  **데이터 다양성**: 다양한 오염 스케줄·도메인 적용으로 실세계 적용성 제고  
- **Multi-domain Conditioning**: 텍스트·특징·사전 학습 SR 참조로 범용성 강화  

# VI. Diffusion-기반 제로샷(Zero-Shot) 초해상도(SR)

확산 모델(Diffusion Models)을 이용한 **제로샷 제너레이션(Zero-Shot Generation)** 방식은, 사전 학습된 확산 모델을 **추가 학습 없이** 그대로 활용해 단일 이미지 내 정보만으로 초해상도를 수행하는 기법입니다. 제로샷 SR은 훈련 데이터의 LR-HR 쌍이 없어도 잘 동작하며, 특히 실제 열악한 저해상도 이미지 복원에 유용합니다. 주요 접근법을 크게 세 가지 범주로 나누어 살펴보겠습니다.

## 1. 투영 기반(Projection-Based) 방법  
저해상도(LR) 이미지의 **내재(low-frequency) 정보**를 고해상도 추정 과정에 반복 투영하여 “데이터 일관성(data consistency)”을 유지하는 기법입니다.  
- **ILVR** (Iterative Latent Variable Refinement)  
  - 사전 학습된 무조건 확산 모델의 생성 단계마다, 생성된 중간 결과의 저주파 성분만 LR 이미지의 저주파 성분으로 교체(투영)  
  - 각 반복에서 LR과 일치하도록 보정하면서 점진적 고해상도 복원  
- **RePaint 응용**  
  - 원본 RePaint는 부분 인페인팅에 투영 아이디어 사용  
  - SR에 그대로 적용해 관심 영역(주로 저주파)에만 복원을 집중  

## 2. 분해 기반(Decomposition-Based) 방법  
열화 연산자 $$A$$의 **스펙트럼 분해**(예: 특이값 분해)를 활용해  

$$
x = Ay \quad\longrightarrow\quad y = A^{\dagger}x + (I - A^{\dagger}A)\bar y
$$

형태로 HR 이미지를 범위 공간(range space)과 영(Null) 공간으로 분리해 생성합니다.  
- **SNIPS / DDRM**  
  - $$A$$를 내장된 선형 연산(예: 평균 풀링)으로 매트릭스로 표현  
  - 투영 단계 없이 스펙트럼 도메인에서 직접 SR 구현  
- **DDNM** (Denoising Diffusion Null-Space Model)  
  1. 일반 DDPM 수식을 통해 중간 역확산 상태 $$z_t$$로부터 “예상 깨끗 이미지” $$\hat z_{0|t}$$ 계산  
  2. $$\hat z_{0|t}$$의 범위 공간은 $$A^{\dagger}x$$로, 영공간은 확산 출력 그대로 사용해  

     $$\displaystyle \hat z_{0|t}^{\rm proj}=A^{\dagger}x + (I - A^{\dagger}A)\hat z_{0|t}$$
     
  3. 이 수정된 $$\hat z_{0|t}^{\rm proj}$$로부터 다음 시점 $$z_{t-1}$$ 샘플링  
  - **장점**: 다양한 선형 복원 문제(컬러화·인페인팅·SR 등)에 일관된 프레임워크 적용  
  - **제약**: 선형 열화 모델 가정, 연산 비용 증가  

## 3. 사후 확률 추정(Posterior-Estimation) 방법  
베이즈 정리를 활용해  

$$
p(z_t\mid x) \propto p(x\mid z_t)\,p(z_t)
$$ 

를 직접 추정하여 확산 과정의 스코어 추정치에 수정항을 더하는 방식입니다.  
- **DPS** (Diffusion Posterior Sampling)  
  - 중간 상태의 “깨끗 이미지 기대값” $$\mathbb{E}[z_0\mid z_t]$$ 를 트위디 공식으로 예측  
  - $$x$$와 $$\mathbb{E}[z_0\mid z_t]$$ 간 MSE 거리 그라디언트를 가중치로 활용해 샘플링 경로 조정  
- **GDP** (Generative Diffusion Prior)  
  - 사후 확률을 $$\exp\bigl(-\|D(\hat z_0)-x\|\bigr)$$와 화질 메트릭 결합 형태로 근사  
  - 노이즈 레벨 차이를 해소하기 위해 $$\hat z_0$$에 열화 연산 적용  
- **특징**  
  - 투영 기반보다 데이터 일관성 훼손이 적고, 분해 기반보다 비선형 복원에도 유연  
  - 다만 사후 분포 근사가 정확해야 효과적  

## 4. 주요 성능 비교  
Li 등 정리 기준 DIV2K·CelebA 벤치마크에서, ILVR·DDRM·DDNM·DPS·GDP 등은 PSNR 27–31 dB, SSIM 0.83–0.95, LPIPS 0.14–0.22 범위에서 경쟁력 있는 성능을 보이며, 처리 시간과 FLOPs(연산량) 측면에서도 차이가 있습니다.  

## 5. 사용 시 고려 사항  
- **데이터 일관성 vs. 다양성**: 투영·분해 기법은 일관성 유지 우수하나 생성 다양성 감소  
- **복원 대상 열화 모델**: 선형 열화인지, 노이즈 포함인지에 따라 기법 선택  
- **연산 효율성**: 추가 투영·사후 계산 단계로 인한 연산 비용 상승  

제로샷 SR은 **사전 학습된 확산 모델을 그대로 이용**해, 별도 LR-HR 데이터 없이도 단일 이미지 내 정보만으로 고품질 SR을 달성하는 혁신적 접근법입니다. 투영, 분해, 베이지안 사후 추정이라는 세 가지 주요 패러다임을 중심으로, **ILVR**, **DDRM**, **DDNM**, **DPS**, **GDP** 등 다양한 기법들이 제안되어 왔으며, 실제 실험 결과 이들은 전통적 비(非)학습 기반 기법을 뛰어넘는 성능과 유연성을 제공하고 있습니다. 앞으로는 **비선형 열화**, **실시간 제로샷 SR**, **다중 이미지 조건** 등을 다루는 연구가 활발히 진행될 전망입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e16bfc65-f12b-4bb8-9528-a974f5a6cdf4/2401.00736v3.pdf

## 4. 한계 및 고려사항  
- **계산 자원**: 큰 배치 필요 → 색 편향 발생, 고비용 학습·추론  
- **비교 어려움**: 데이터셋·메트릭 통일성 부재  
- **샘플링 효율**: 여전히 수백 스텝, 스케줄 자동화 미흡  
- **Corruption Assumption**: Gaussian 외 실세계 오염 모델 연구 필요  

# 향후 연구 영향 및 주의점  

- **Benchmark 표준화**: 통합 데이터셋·IQA 메트릭(PSNR·SSIM·LPIPS·DeepQA) 제정  
- **경량화·가속화**: Latent/Wavelet·지식증류·효율적 샘플러 심화  
- **다양한 오염 스페이스**: I2SB·InDI 비디오·의료·위성 영상 특화 모델 확장  
- **일반화 강화**: Zero-shot·도메인 적응·다중 조건화로 범용 SR 모델 구축  
- **색 편향 이론화**: 소규모 배치 시 색 왜곡 원인 분석 및 설계 가이드 제안  

이 논문은 확산모델을 이미지 SR에 체계적으로 정리·비교하여 **차세대 SR 연구의 로드맵**을 제시했으며, 위 과제를 중심으로 모델 효율화·일반화·실세계 적용성 연구에 지침을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e16bfc65-f12b-4bb8-9528-a974f5a6cdf4/2401.00736v3.pdf
