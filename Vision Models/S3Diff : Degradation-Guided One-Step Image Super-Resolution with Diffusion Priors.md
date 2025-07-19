# Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors | Super resolution

**Main Takeaway:** Introducing a one-step, degradation-aware fine-tuning strategy for text-to-image diffusion models achieves high-quality real-world super-resolution in a single forward pass, drastically reducing inference time while preserving generative priors.

## 1. 핵심 주장 및 주요 기여
이 논문은 **대규모 사전학습된 텍스트-투-이미지(T2I) 확산 모델**(Stable Diffusion Turbo)을 활성화하여 단일 단계(one-step)로 고해상도 복원을 수행하는 **S3Diff** 모델을 제안한다.  
– **Degradation-Guided LoRA:** 저해상도(LR) 이미지의 잡음 및 블러 정도를 예측한 2차원 벡터를 입력으로, 각 블록별 LoRA(A · B) 가중치를 동적으로 보정하는 모듈을 도입.  
– **Online Negative Prompting:** 학습 시 일부 HR 타깃을 다시 열화시킨 LR로 대체하여 “저품질” 개념을 negative prompt로 학습, 추론 단계에서 Classifier-Free Guidance로 시각적 품질 크게 향상.  
– **효율성:** 한 번의 U-Net 및 VAE encoder forward만으로 복원, 기존 수십 단계 확산 모델 대비 30× 빠른 추론.  
– **성과:** DIV2K-Val에서 LPIPS=0.2571, DISTS=0.1730, FID=19.35 달성하며, REAL-ESRGAN 등 최첨단 대비 **비교 불가 성능** 및 **추론 시간 절약** 입증.  

## 2. 해결 과제 및 제안 기법

### 2.1 해결하려는 문제  
현실 세계 이미지 SR은  
1) **여러 단계 확산** 모델의 높은 계산 비용  
2) **알 수 없는 열화(degradation) 모델** 부재  
문제가 있으며, 대규모 T2I 모델의 풍부한 prior를 단일 단계 SR에 활용하는 방법이 필요하다.

### 2.2 제안하는 방법

#### 2.2.1 One-Step Adaptation  
– 입력 LR 이미지를 거의 무잡음 상태로 U-Net에 투입하여 확산 역과정을 한 단계로 압축.  
– SD-Turbo의 4개 distilled noise level 중 하나에 해당하도록 Fine-tuning.  

#### 2.2.2 Degradation-Guided LoRA  
모델 파라미터 W 업데이트를  

$$
W_{\text{new}} = W + A (C B)
$$ 

로 정의하며,  
– $$d = [d_n, d_b]\in[1]^2$$ : Mou 등(2022)의 degradation estimator로부터 예측한 noise/blur 스코어.  
– Fourier embedding 및 MLP 통해 블록ID $$l_i$$와 결합하여 블록별 보정 행렬 $$C_i$$ 생성.  
– 각 U-Net 및 VAE encoder 레이어에 rank-r LoRA(A∈ℝ^{d×r}, B∈ℝ^{r×n}) 삽입, 동적으로 파라미터 조정.

#### 2.2.3 Online Negative Prompting  
– 확률 $$p_n$$로 minibatch의 HR 타깃을 해당 LR로 대체.  
– Positive prompt: “a high-resolution image full of vivid details, …”  
– Negative prompt: “oil painting, cartoon, blur, dirty, … low quality”  
– 학습 중 UNet에 두 가지 condition을 주어 각각 $$z_{pos}, z_{neg}$$ 예측 후, 추론 시  

$$
z_{\text{out}} = z_{neg} + \lambda_{cfg}\,(z_{pos}-z_{neg})
$$  

로 fusion.  

### 2.3 모델 구조  
– **기본 모델:** SD-Turbo (VAE encoder + UNet + frozen VAE decoder)  
– **Adaptation:** VAE encoder 및 U-Net에 rank-{ $$r_{\text{enc}}=16$$, $$r_{\text{unet}}=32$$ } LoRA 삽입  
– **Degradation Estimation Network:** Mou 등(2022) 모델 (2.36 M 파라미터)  
– **Discriminator:** DINO 백본 + 다중 레벨 분류기

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
– **Synth. / Real-World Benchmarks:** DIV2K-Val, RealSR, DRealSR, RealSet65 전반에서 LPIPS, DISTS, FID, NIQE, MANIQA, MUSIQ 등 reference / non-reference 지표 최상위[Table 1][Table 2].  
– **추론 속도:** 1 forward만으로 0.62 s 처리, StableSR(200 steps, 17.75 s) 대비 30× 빠름[Table 3].  
– **Generalization:** 실험 전 범위(합성열화↔실제열화) 뛰어난 적응성 및 견고성.  

### 3.2 한계  
– **텍스트 없는 SR:** 텍스트 prompt 활용 시 reference 지표 소폭 저하 및 텍스트 추출 모델(1.4 B–7 B 파라미터) 의존 문제.  
– **복잡도:** rank 증가 시 파라미터 급증.  
– **소규모 텍스트·얼굴 영역:** 작은 scene text, 얼굴 복원 약점 보유.  

## 4. 일반화 성능 향상 관점  
– **Degradation Control:** 사용자가 noise/blur 스코어 수동 조정으로 SR 결과 조절 가능, generalized한 “interactive” SR 파이프라인 구현[Fig 11].  
– **Diffusion Prior 선택:** SD-Turbo, Stable Diffusion 1.5/2.1 모두 수렴 시 유사 성능, 사전학습된 대규모 prior 활용 시 학습 속도 및 메모리 절감[Fig 9].  
– **LoRA rank 조정:** encoder 16 + unet 32 조합이 SR↔생성력 균형 최적[Table 7].  
– **Noise Level 시작 단계:** 0 noise or 250-step noise 모두 무난하나 더 많은 noise 시 perceptual 증가, PSNR 소폭 감소[Table 8].  

## 5. 향후 연구 과제 및 고려 사항  
1. **대규모 Prior 확장:** SDXL/SD3 등 더 발전된 모델 fine-tuning으로 소규모 텍스트·얼굴 영역 성능 강화.  
2. **모델 경량화:** LoRA rank 최적화 및 block embedding 압축을 통한 파라미터·메모리 절감.  
3. **Prompt-Free vs. Prompted 균형:** 효율성을 유지하면서 선택적 텍스트 안내(태그·캡션) 통합 방안 연구.  
4. **Blind SR 강화:** 불확실 degradation 환경에서 더욱 robust한 스코어 추정 및 adaptive guidance 메커니즘 개발.  

**영향:** S3Diff는 T2I 확산 prior를 SR에 단일 단계로 적용하는 새로운 패러다임을 제시했으며, 효율·품질·상호작용 측면에서 SR 연구 방향을 크게 확장할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9a60b79d-e29c-4927-86f6-934170bb84f0/2409.17058v1.pdf
