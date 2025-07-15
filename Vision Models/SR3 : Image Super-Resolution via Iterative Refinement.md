# SR3 : Image Super-Resolution via Iterative Refinement | Super resolution

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
SR3는 노이즈 제거 확산 확률 모델(Denoising Diffusion Probabilistic Model, DDPM)을 조건부 이미지 생성에 확장하여, 순차적 반복 정제를 통해 고품질 초해상도 이미지를 생성한다.  

**주요 기여**  
- DDPM을 단일 이미지 초해상도(Conditional SR)에 적용한 SR3 기법 제안.  
- U-Net 기반 네트워크에 저해상도 입력을 채널 축으로 단순 결합(concatenation)하여 조건부 생성 구현.  
- 8× 얼굴 SR(16×16→128×128)에서 GAN 계열 최고 성능 대비 휴먼 포얼(fool)율 50%에 근접, 기존 GAN 최대 34% 달성[1].  
- 다단계(cascaded) 모델 체이닝으로 최대 1024×1024 해상도 생성, ImageNet 256×256에서 FID 11.3 확보.  

## 2. 문제 정의, 제안 기법, 모델 구조, 성능, 한계  

### 2.1 해결하고자 하는 문제  
- 저해상도 $$x$$로부터 원본 고해상도 $$y$$의 조건부 분포 $$p(y|x)$$를 표현  
- 다중 해답이 가능한 역문제(inverse problem)로, MSE 기반 회귀식으로는 고주파 세부묘사가 부족  

### 2.2 제안 방법  
**확산 과정(forward diffusion)**  

$$
q(y_t \mid y_{t-1}) = \mathcal{N}\bigl(y_t; \sqrt{\alpha_t}\,y_{t-1},\,(1-\alpha_t)\mathbf{I}\bigr)
$$

$$
q(y_t \mid y_0) = \mathcal{N}\bigl(y_t; \sqrt{\gamma_t}\,y_0,\,(1-\gamma_t)\mathbf{I}\bigr),\quad \gamma_t=\prod_{i=1}^t\alpha_i
$$

**역확산 과정(reverse diffusion)**  

$$
p_\theta(y_{t-1}\mid y_t,x)
=\mathcal{N}\bigl(y_{t-1};\mu_\theta(x,y_t,\gamma_t),\,(1-\alpha_t)\mathbf{I}\bigr)
$$

$$
\mu_\theta(x,y_t,\gamma_t)
=\frac{1}{\sqrt{\alpha_t}}\Bigl(y_t-\frac{1-\alpha_t}{\sqrt{1-\gamma_t}}\,f_\theta(x,y_t,\gamma_t)\Bigr)
$$

여기서 $$f_\theta$$는 노이즈 $$\epsilon$$을 예측하도록 학습된 U-Net 모델.  

**학습 손실**  

$$
\mathcal{L}
= \mathbb{E}\_{(x,y_0),\gamma,\epsilon}\bigl\|
f_\theta\bigl(x,\sqrt{\gamma}\,y_0+\sqrt{1-\gamma}\,\epsilon,\gamma\bigr)
-\epsilon
\bigr\|_p^p
$$

주로 $$p=2$$ (MSE) 사용[1].  

### 2.3 모델 구조  
- **백본**: BigGAN형 residual block이 삽입된 U-Net  
- **조건부 입력**: 저해상도 이미지를 목표 해상도로 업샘플링 후 고해상도 노이즈 이미지 $$y_t$$와 채널 결합  
- **스케줄**: 학습 시 $$T=2000$$ 스텝, 추론 시 최대 100 스텝의 효율적 스케줄 선택  

### 2.4 성능 향상  
- **휴먼 포얼율**  
  - 16×16→128×128 얼굴 SR에서 SR3: 54.1% vs. FSRGAN 8.9%, PULSE 24.6%[1].  
  - 64×64→256×256 자연 이미지 SR에서 SR3: 38.8% vs. 회귀 16.8%[1].  
- **자동 지표**  
  - 얼굴 SR(8×): PSNR/SSIM은 회귀식에 미치지 못하나, 일관성(consistency) MSE 최고(2.68) 기록[1].  
  - 자연 SR(4×): FID 5.2, IS 180.1로 회귀(15.2/121.1) 대비 질적 우위[1].  
- **고해상도 합성**  
  - 256×256 ImageNet 클래스SR FID 11.3, VQ-VAE-2 38.1, BigGAN(Trunc=1.5) 11.8[1].  

### 2.5 한계  
- **속도**: DDPM 계열 반복 정제는 여전히 느림 (최대 100 스텝).  
- **모드 드롭핑**: 동일 입력에 유사 출력 반복, 피부 잡티·문신 등 세부 묘사 부족 관찰.  
- **편향**: 얼굴 SR에서 특정 소수자 특성 누락 가능성.  
- **텍스트 복원 실패**: 알파벳 형태 학습 미흡해 자연스러운 문장 재현 어려움[1].  

## 3. 일반화 성능 향상 가능성  
- **노이즈 스케줄 최적화**: 추론 스텝 감소에도 품질 유지, 저해상도 단계 더 많은 스텝 활용 제안.  
- **데이터 증강**: Gaussian blur 등 입력 변이 추가 시 FID 13.1→11.3 개선[1].  
- **목표 함수**: $$L_1$$ vs. $$L_2$$ 비교에서 $$L_1$$ 소폭 우세, 다양한 노이즈 분포로 일반화 강화 가능.  
- **조건부 방식**: 단순 결합 대안으로 FiLM·어텐션 기반 conditioning 연구해 입력-출력 일관성 강화 여지.  
- **모드 커버리지**: 스코어 매칭 관점에서 다중 가우시안 혼합 모델로 확장해 다양한 해상도 세부묘사 학습 촉진 가능.  

## 4. 향후 연구 영향 및 고려사항  
- **확산 모델 응용 확장**: SR 외에도 컬러라이제이션·인페인팅 등 다른 조건부 생성 문제에 확산 기법 적용 가능성.  
- **추론 효율화**: 스텝 수 감소 알고리즘(예: DDIM, stochastic sampler) 도입 연구 필요.  
- **편향 완화**: SR3의 세부 누락·편향 문제 분석 후 페어링된 디버깅·공정성 제어 기법 개발 필수.  
- **일관성 검증**: 자동화된 perceptual 일관성 메트릭 개발로 human evaluation 의존도 감소 방안 모색.  
- **범용성 검증**: 자연 이미지·의료 영상·위성 영상 등 다양한 도메인에서도 SR3 재학습·미세조정(fine-tuning) 성능 검증 필요.  

---  
References  
[1] Saharia et al., “Image Super-Resolution via Iterative Refinement” (arXiv:2104.07636v2)

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4b557eee-191f-4064-bbf8-2353017ff4a1/2104.07636v2.pdf

# Abs

# Introduction
Super-resolution은 입력된 저해상도 이미지와 일치하는 고해상도 이미지를 생성하는 프로세스이다.  
Colorization, inpainting, de-blurring과 함께 광범위한 image-to-image translation task에 속한다.  
이러한 많은 inverse problem과 마찬가지로 super-resolution은 여러 출력 이미지가 하나의 입력 이미지와 일치할 수 있고, 주어진 입력에 대한 출력 이미지의 조건부 분포가 일반적으로 간단한 parametric 분포이다.  
따라서 feedforward convolutional network를 사용하는 단순 회귀 기반 방법은 낮은 배율에서 super-resolution에 사용할 수 있지만 높은 배율에는 디테일이 부족한 경우가 많다.  

심층 생성 모델은 이미지의 복잡한 경험적 분포를 학습하는 데 성공했다.  
Autoregressive model, VAE, Normalizing Flow (NF), GAN은 설득력 있는 이미지 생성 결과를 보여주었고 super-resolution과 같은 조건부 task에 적용되었다.  
그러나 이러한 접근 방식에는 종종 다양한 제한 사항이 있다.  
예를 들어, autoregressive model은 고해상도 이미지 생성에 엄청나게 비싸고, NF와 VAE는 종종 좋지 못한 샘플 품질을 보이며, GAN은 최적화 불안정성과 mode collapse를 피하기 위해 신중하게 설계된 정규화 및 최적화 트릭이 필요하다.  

본 논문은 조건부 이미지 생성에 대한 새로운 접근 방식인 Super-Resolution via Repeated Refinement (SR3)를 제안한다.  
이는 DDPM과 denoising socre matching에서 영감을 얻었다.  
SR3는 Langevin 역학과 유사한 일련의 refinement step을 통해 표준 정규 분포를 경험적 데이터 분포로 변환하는 방법을 학습하여 작동한다.  
핵심은 출력에서 다양한 레벨의 noise를 반복적으로 제거하기 위해 denoising 목적 함수로 학습된 U-Net 아키텍처이다.  
U-Net 아키텍처에 대한 간단하고 효과적인 수정을 제안하여 DDPM을 조건부 이미지 생성에 적용한다.  
GAN과 달리 잘 정의된 loss function을 최소화한다.  
Autoregressive model과 달리 SR3는 출력 해상도에 관계없이 일정한 수의 inference step을 사용한다.  

SR3는 다양한 배율과 입력 해상도에서 잘 작동한다.  
SR3 모델은 예를 들어 64×64에서 256×256으로, 그 다음에는 1024×1024로 cascade할 수도 있다.  
Cascading model을 사용하면 배율이 높은 단일 대형 모델이 아닌 몇 개의 작은 모델을 독립적으로 학습시킬 수 있다.  
고해상도 이미지를 직접 생성하려면 동일한 품질에 대해 더 많은 refinement step이 필요하기 때문에 cascading model이 보다 효율적인 inference를 가능하게 한다.  
또한 저자들은 조건부 생성 모델을 SR3 모델과 연결하여 unconditional한 고충실도 이미지를 생성할 수 있음을 발견했다.  
특정 도메인에 초점을 맞춘 기존 연구들과 달리 SR3는 얼굴과 자연스러운 이미지 모두에 효과적임을 보여준다.

PSNR과 SSIM과 같은 자동화된 이미지 품질 점수는 입력 해상도가 낮고 배율이 큰 경우 인간의 선호도를 잘 반영하지 못한다.  
합성 디테일이 레퍼런스 디테일과 완벽하게 일치하지 않기 때문에 이러한 품질 점수는 종종 머리카락 질감과 같은 합성 고주파 디테일에 불이익을 준다.  
저자들은 SR3의 품질을 비교하기 위해 사람의 평가에 의지한다.  
인간 피험자에게 저해상도 입력을 보여주고 모델 출력과 ground-truth 이미지 중에서 선택해야 하는 2-alternative forced-choice(2AFC)를 채택한다.  
이를 기반으로 이미지 품질과 저해상도 입력으로 모델 출력의 일관성을 모두 캡처하는 fool rate 점수를 계산한다.  
SR3는 SOTA GAN 방법과 강력한 회귀 baseline보다 훨씬 더 높은 fool rate를 달성하였다.

# Conditional Denoising Diffusion Model
모르는 조건부 분포 p(y|x) 에서 추출한 샘플을 나타내는 D로 표시된 입력-출력 이미지 쌍의 데이터셋이 제공된다.  
이는 많은 타겟 이미지가 하나의 소스 이미지와 일치할 수 있는 일대다 매핑이다.  
소스 이미지 x를 target 이미지 y 에 매핑하는 확률적 반복 정제 프로세스를 통해 p(y|x) 에 대한 parametric 근사를 학습하는 데 관심이 있다.  
저자들은 조건부 이미지 생성에 대한 DDPM 모델을 적용하여 이 문제에 접근한다.  

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig2.PNG)

조건부 DDPM 모델은 T개의 refinement step에서 타겟 이미지 y_0을 생성한다.  
모델은 순수한 noise 이미지 y_T 에서 시작하여 y_0를 만족하는 transition 분포에 따라 이미지를 반복적으로 정제한다.  
Inference chain에서 중간 이미지의 분포는 q(yt|yt-1) 로 표시되는 고정 Markov chain을 통해 신호에 Gaussian noise를 점진적으로 추가하는 forward diffusion process로 정의된다.  
모델의 목적 함수는 x로 컨디셔닝된 reverse Markov chain을 통해 noise에서 신호를 반복적으로 복구하여 Gaussian diffusion process를 reverse시키는 것이다.  
원본 이미지와 noisy한 타겟 이미지를 입력으로 사용하고 noise를 추정하는 denoising model fθ를 사용하여 reverse chain을 학습한다.

# Gaussian Diffusion Process

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.21.33.png)

# Optimizing the Denoising Model

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.21.49.png)

# Inference via Iterative Refinement

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.22.13.png)

# SR3 Model Architecture and Noise Schedule

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.22.24.png)

# Related Work

# Experiments

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-29%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.23.45.png)

## Qualitative Results
### Natural Images
다음은 ImageNet에서 학습한 SR3 model을 ImageNet 테스트 이미지에서 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig3.PNG)

### Face Images
다음은 FFHQ에서 학습한 SR3 model을 학습 셋에 포함되지 않는 이미지에서 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig4.PNG)

## Benchmark Comparison
### Automated metrics
얼굴 super-resolution에 대한 PSNR과 SSIM을 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-table1.PNG)

다음은 SR3와 Regression baseline을 ImageNet validation set으로 성능을 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-table2.PNG)

다음은 ImageNet Validation set의 이미지 1,000개에서 4× 자연 이미지 super-resolution에 대한 classification 정확도를 비교한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-table3.PNG)

### Human Evaluation (2AFC)
다음은 얼굴 super-resolution의 fool rates를 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig6.PNG)

다음은 ImageNet super-resolution의 fool rates를 평가한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig7.PNG)

## Quantitative Results
얼굴 super-resolution task에 대한 다양한 방법을 비교한 것이다.
![](https://kimjy99.github.io/assets/img/sr3/sr3-fig5.PNG)

## Cascaded High-Resolution Image Synthesis
unconditional diffusion model로 샘플링한 뒤 SR3 model 2개를 통과시켜 얼굴 이미지를 생성한 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig8.PNG)

클래스 조건부 diffusion model로 샘플링한 뒤 SR3 4x model을 통과시킨 것이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-fig9.PNG)

FID 점수이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-table4.PNG)

## Ablation Studies
SR model에 대한 ablation study 결과이다.

![](https://kimjy99.github.io/assets/img/sr3/sr3-table5.PNG)
