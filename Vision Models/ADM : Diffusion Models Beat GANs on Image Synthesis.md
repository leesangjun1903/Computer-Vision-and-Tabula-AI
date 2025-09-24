# ADM : Diffusion Models Beat GANs on Image Synthesis | Image generation
## 2021 · 10611회 인용

# Classifier Guidance in Diffusion Models

**핵심 요약**  
Classifier guidance는 사전 학습된 분류기(classifier)의 그래디언트를 활용해 확산(diffusion) 샘플링 과정을 특정 클래스 방향으로 “안내”함으로써, **생성 이미지의 충실도(fidelity)를 높이고 다양성(diversity)을 제어**하는 기법입니다[1].  

## 1. 왜 필요한가?  
- **GAN의 Truncation/Temperature 기법 부재**  
  GAN은 latent 벡터 샘플링 분포를 조절해 다양성과 충실도의 균형을 맞추는 *truncation trick*이나 *low‐temperature sampling*을 사용합니다.  
- **확산 모델의 한계**  
  표준 diffusion 모델은 점진적인 노이즈 제거 절차만으로 학습되어, 이러한 직접적인 샘플링 분포 조절이 어렵습니다. 이로 인해 원하는 클래스나 속성의 샘플 품질을 제어하기 힘듭니다.  

Classifier guidance는 이 문제를 해결하기 위해, **조건(클래스) 정보**를 분류기 그래디언트로 제공하여 생성 과정을 지시합니다[2].  

## 2. 기법의 핵심 아이디어  
1. **기본 확산 샘플링**  
   - 확산 모델은 시간 단계 $$t$$에서 잡음을 제거하며 $$x_t \to x_{t-1}$$으로 점진적으로 복원합니다.  
   - 각 단계에서 모델은 $$\mu_\theta(x_t,t)$$와 $$\Sigma_\theta(x_t,t)$$로 기술되는 가우시안 분포로부터 샘플링합니다.

2. **분류기 그래디언트 적용**  
   - 사전 학습된 분류기 $$p_\phi(y\mid x_t)$$를 이용해 목표 클래스 $$y$$에 대한 로그 확률 $$\log p_\phi(y\mid x_t)$$의 그래디언트 $$\nabla_{x_t}\log p_\phi(y\mid x_t)$$를 계산합니다.  
   - 이 그래디언트를 확산 모델의 예측 평균 $$\mu_\theta$$에 추가하여 **mean shifting**을 수행합니다[3]:
     
  $$
       \tilde\mu = \mu_\theta(x_t,t) + s\Sigma_\theta(x_t,t)\nabla_{x_t}\log p_\phi(y\mid x_t),
  $$  
     
여기서 $$s$$는 guidance 강도(스케일)입니다.  

3. **가이던스 스케일**  
   - $$s$$가 클수록 충실도(fidelity)는 증가하지만 다양성(diversity)은 감소합니다.  
   - 적절한 $$s$$ 값을 선택하면 FID/IS(또는 Precision/Recall) 간의 완만한 절충(trade-off)을 얻을 수 있습니다[1][2].  

## 3. 수학적 정리  
확산 역방향 전이 확률을 $$p_\theta(x_{t-1}\mid x_t)\approx\mathcal{N}(\mu,\Sigma)$$라 할 때, **조건부** 전이는 베이즈 정리에 따라  

$$
  p(x_{t-1}\mid x_t,y)\;\propto\;p_\theta(x_{t-1}\mid x_t)\;p_\phi(y\mid x_{t-1}).
$$

가우시안 가정 하에서 평균이  
$$\mu+\Sigma\nabla_{x_{t-1}}\log p_\phi(y\mid x_{t-1})$$로 이동함을 보일 수 있습니다[1].  
DDIM 샘플러의 경우에도, 노이즈 예측 $$\epsilon_\theta(x_t)$$에 분류기 그래디언트를 추가해  

$$
  \hat\epsilon = \epsilon_\theta(x_t) \;-\;\sqrt{1-\bar\alpha_t}\;\nabla_{x_t}\log p_\phi(y\mid x_t)
$$

으로 대체함으로써 동일한 효과를 얻습니다[1].  

## 4. 직관적 이해  
- **“확실한” 사례 중심으로 샘플링**  
  분류기가 높은 확률로 목표 클래스에 속한다고 평가하는 영역으로 샘플링 경로를 유도합니다.  
- **GAN의 Truncation과 유사**  
  GAN이 latent를 잘 분류된 영역(truncated normal)에서만 샘플링해 고충실 이미지를 얻는 것과 원리가 비슷합니다.  
- **스케일 조절**  
  낮추면 다양성을, 높이면 충실도를 강조하는 *vice versa* 효과를 냅니다.  

## 5. 구현상 유의점  
1. **분류기 학습**  
   - 생성 모델과 동일한 노이즈 수준의 데이터를 사용해 분류기를 훈련해야 합니다.  
2. **가이던스 강도 $$s$$**  
   - 경험적으로 1∼10 범위를 탐색하며 FID/IS의 최적점을 찾습니다.  
3. **추론 속도**  
   - 각 샘플링 스텝마다 분류기 역전파가 필요하므로, 비용이 늘어납니다.  

## 6. 활용 사례와 한계  
- **활용**: ImageNet 클래스별 고충실 샘플링, 텍스트-이미지 모델에 CLIP 분류기 가이던스 적용 등[1].  
- **한계**: 느린 샘플링 속도, 레이블 의존성(labelled data 필요), 무라벨(unlabeled) 데이터로 확장 어려움.  

**요약**  
Classifier guidance는 단순히 확산 모델만으로 어렵던 **목표 클래스 지향성 샘플링**을 구현하며, **FID와 IS(또는 Precision/Recall) 간의 균형**을 스케일 $$s$$로 조절할 수 있는 강력한 기법입니다[1][2]. 그러나 속도 및 레이블 의존성 문제가 있어, 이를 보완하는 classifier-free guidance 등 후속 연구도 활발합니다.  

33 / 33

[1] https://www.semanticscholar.org/paper/64ea8f180d0682e6c18d1eb688afdb2027c02794
[2] https://milvus.io/ai-quick-reference/what-is-classifier-guidance-in-diffusion-models
[3] https://ostin.tistory.com/133
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f29ae6da-bbc4-4951-bc47-469c1bae51c1/2105.05233v4.pdf
[5] https://www.sec.gov/Archives/edgar/data/1053691/000143774923013739/dffn20230501_s4.htm
[6] https://www.sec.gov/Archives/edgar/data/1053691/000143774923022535/dffn20230630_10q.htm
[7] https://www.sec.gov/Archives/edgar/data/1053691/000143774923021941/dffn20230803_8k.htm
[8] https://www.sec.gov/Archives/edgar/data/1053691/000143774923022773/dffn20230808_8k.htm
[9] https://www.sec.gov/Archives/edgar/data/1053691/000143774923014359/dffn20230331_10q.htm
[10] https://www.sec.gov/Archives/edgar/data/1053691/000143774923008519/dffn20230329_8k.htm
[11] https://www.sec.gov/Archives/edgar/data/1053691/000143774923007795/dffn20221231_10k.htm
[12] https://arxiv.org/abs/2207.12598
[13] https://ieeexplore.ieee.org/document/10822129/
[14] https://arxiv.org/abs/2406.08070
[15] https://arxiv.org/abs/2403.11968
[16] https://arxiv.org/abs/2407.02687
[17] https://ieeexplore.ieee.org/document/10377202/
[18] https://arxiv.org/abs/2408.05900
[19] https://juniboy97.tistory.com/55
[20] https://velog.io/@yhyj1001/Classifier-Free-Diffusion-Guidance
[21] https://velog.io/@hero981001/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Classifier-Free-Diffusion-Guidance
[22] https://www.kaggle.com/code/vikramsandu/guided-diffusion-by-openai-from-scratch
[23] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/cfdg/
[24] https://github.com/tsmatz/diffusion-tutorials/blob/master/06-classifier-guidance.ipynb
[25] https://theaisummer.com/classifier-free-guidance/
[26] https://kyujinpy.tistory.com/131
[27] https://paperswithcode.com/method/classifier-guidance
[28] https://ffighting.net/deep-learning-paper-review/diffusion-model/classifier-guidance/
[29] https://arxiv.org/abs/2411.15393
[30] https://arxiv.org/abs/2408.08252
[31] https://arxiv.org/pdf/2403.03938.pdf
[32] https://arxiv.org/pdf/2208.08664.pdf
[33] https://arxiv.org/html/2412.18604
[34] http://arxiv.org/pdf/2406.08070.pdf
[35] https://arxiv.org/pdf/2304.12536.pdf
[36] https://arxiv.org/html/2406.17399v1
[37] http://arxiv.org/pdf/2405.14677.pdf
[38] https://arxiv.org/html/2412.10193v2
[39] https://arxiv.org/html/2502.07849
[40] http://arxiv.org/pdf/2408.04220.pdf

## 1. 핵심 주장과 주요 기여

**핵심 주장**: 이 논문은 diffusion 모델이 기존 최고 성능의 생성 모델인 GAN을 능가하는 이미지 합성 품질을 달성할 수 있음을 보여줍니다[1].

**주요 기여**:
- **아키텍처 개선**: 체계적인 ablation study를 통해 diffusion 모델의 UNet 아키텍처를 개선
- **분류기 가이던스(Classifier Guidance)**: 분류기의 그래디언트를 활용하여 다양성과 충실도를 조절할 수 있는 새로운 조건부 생성 방법 제안
- **SOTA 성능 달성**: ImageNet에서 FID 2.97 (128×128), 4.59 (256×256), 7.72 (512×512) 달성
- **효율성 개선**: 25번의 forward pass만으로도 BigGAN-deep과 비슷한 성능 달성

## 2. 문제, 제안 방법, 모델 구조 및 성능

### 해결하고자 하는 문제

**GAN의 한계점**[1]:
- 우도 기반 모델보다 낮은 다양성 포착
- 훈련과 스케일링의 어려움
- 세심한 하이퍼파라미터 튜닝 필요
- 모드 붕괴(mode collapse) 문제

**Diffusion 모델의 격차**[1]:
- LSUN, ImageNet과 같은 어려운 데이터셋에서 GAN보다 성능 저하
- GAN만큼 정교하지 않은 아키텍처
- 다양성과 충실도 간의 트레이드오프 조절 능력 부족

### 제안하는 방법

#### 1. 아키텍처 개선

**핵심 개선사항**[1]:
- **모델 크기 최적화**: 깊이보다 너비 증가 (160 채널, 2 residual blocks)
- **멀티헤드 어텐션**: 64 채널/헤드로 어텐션 헤드 증가
- **멀티해상도 어텐션**: 32×32, 16×16, 8×8 해상도에서 어텐션 적용
- **BigGAN 스타일 residual blocks**: 업샘플링/다운샘플링용
- **Adaptive Group Normalization (AdaGN)**:

$$ \text{AdaGN}(h, y) = y_s \cdot \text{GroupNorm}(h) + y_b $$

#### 2. 분류기 가이던스

**수학적 원리**[1]:

조건부 역방향 과정:

$$ p_{\theta,\phi}(x_t|x_{t+1}, y) = Z p_\theta(x_t|x_{t+1}) p_\phi(y|x_t) $$

가우시안 근사 하에서:

$$ \mu_{\text{guided}} = \mu + s \cdot \Sigma \cdot \nabla_{x_t} \log p_\phi(y|x_t) $$

DDIM 확장:

$$\hat{\epsilon}(x_t) = \epsilon_\theta(x_t) - \sqrt{1-\bar{\alpha}\_t} \nabla_{x_t} \log p_\phi(y|x_t) $$

여기서 스케일 팩터 $$s$$는 다양성과 충실도 간의 트레이드오프를 조절합니다[1].

### 모델 구조

**기본 아키텍처**: UNet 인코더-디코더 구조[1]

**핵심 구성요소**:
- Skip connection을 가진 Residual blocks
- 다운샘플링/업샘플링 convolution
- 멀티해상도 어텐션 메커니즘
- 타임스텝 임베딩 주입
- 클래스 임베딩 통합 (조건부 모델)

### 성능 향상

**무조건부 생성**[1]:
- LSUN Bedroom: FID 1.90 (StyleGAN 2.35 대비)
- LSUN Horse: FID 2.57 (StyleGAN2 3.84 대비)
- LSUN Cat: FID 5.57 (StyleGAN2 7.25 대비)

**조건부 생성**[1]:
- ImageNet 128×128: FID 2.97 (BigGAN-deep 6.02 대비)
- ImageNet 256×256: FID 4.59 (BigGAN-deep 6.95 대비)
- ImageNet 512×512: FID 7.72 (BigGAN-deep 8.43 대비)

**효율성**: 25 DDIM 스텝으로 250 전체 스텝과 비슷한 성능 달성[1]

### 한계점

- **샘플링 속도**: 여러 디노이징 스텝으로 인해 GAN보다 느림
- **레이블 데이터 의존성**: 분류기 가이던스가 레이블된 데이터셋에 제한됨
- **무레이블 데이터**: 무레이블 데이터를 위한 대안 전략 필요

## 3. 일반화 성능 향상

### 분포 커버리지

**핵심 통찰**: Diffusion 모델은 GAN보다 더 나은 분포 커버리지를 유지합니다[1].

**증거**:
- GAN 대비 높은 recall 값 (다양성 측정)
- 데이터의 소수 모드에 대한 더 나은 커버리지
- Figure 6에서 보여지는 더 다양한 샘플

### 스케일링 특성

**계산 스케일링**: 모델이 증가된 계산량과 함께 안정적으로 향상[1]
**아키텍처 스케일링**: 여러 개선사항을 결합할 때 이점이 누적됨
**데이터셋 스케일링**: 다양한 데이터셋(LSUN, ImageNet)에서 일관된 개선

### 견고성

**훈련 안정성**: GAN과 달리 정상 훈련 목표 함수[1]
**하이퍼파라미터 민감성**: GAN보다 하이퍼파라미터에 덜 민감
**모드 붕괴**: GAN과 달리 모드 붕괴 문제 없음

## 4. 미래 연구에 미치는 영향과 고려사항

### 연구 영향

**즉각적 영향**[1]:
- Diffusion 모델을 GAN의 진지한 경쟁자로 확립
- Diffusion 모델에서 아키텍처 설계의 중요성 입증
- 분류기 가이던스를 일반적인 기법으로 도입
- Diffusion 모델의 스케일링 특성 실증

### 후속 연구 방향

**기술적 개선**[1]:
- **더 빠른 샘플링**: 증류(distillation), 더 적은 스텝 방법
- **무레이블 데이터 가이던스**: 클러스터링 기반 synthetic label 또는 판별 모델
- **텍스트-이미지 생성**: CLIP 가이던스를 통한 확장

**연구 시 고려사항**:

1. **효율성 vs 품질**: 샘플링 속도와 생성 품질 간의 균형
2. **조건부 생성**: 다양한 조건 형태에 대한 가이던스 방법 개발
3. **스케일링 법칙**: 모델 크기, 데이터, 계산량 간의 관계 이해
4. **평가 지표**: FID, Precision, Recall 외의 더 포괄적인 평가 방법
5. **실용적 응용**: 실제 응용에서의 계산 효율성과 메모리 사용량 고려

이 논문은 생성 모델링 분야에서 패러다임 전환을 가져왔으며, 현재 DALL-E 2, Imagen, Stable Diffusion 등의 대규모 텍스트-이미지 생성 모델의 기반이 되었습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f29ae6da-bbc4-4951-bc47-469c1bae51c1/2105.05233v4.pdf
