# StableVSR : Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models | Super resolution

**주요 주장 및 기여**  
이 논문은 **Diffusion Models**(DMs)를 활용해 비디오 슈퍼해상도(VSR)의 **지각적 품질(perceptual quality)**을 대폭 향상시키면서, **프레임 간 세부 묘사의 시간적 일관성(temporal consistency)**을 보장하는 **StableVSR** 기법을 제안한다.  
1. **생성적 패러다임 도입**: 기존 VSR 기법들이 재구성 정확도(reconstruction quality)에 집중해 디테일이 부족한 반면, StableVSR는 DMs의 강력한 생성 능력을 이용해 사실적인 텍스처를 합성.  
2. **Temporal Conditioning Module (TCM)**: SISR용 사전 학습된 LDM에 추가되어, 인접 프레임에서 합성된 디테일을 현재 프레임의 생성 과정에 주입.  
3. **Temporal Texture Guidance**: 노이즈 제거 과정의 각 단계 $$t$$에서, 인접 프레임의 예측 $$\tilde x_0$$를 VAE 디코더로 RGB로 복원하고, 광류(optical flow) 기반 모션 보상을 거쳐 공간 정렬된 텍스처 정보를 제공(식 (7)).  
4. **Frame-wise Bidirectional Sampling**: 모든 프레임에 대해 동일한 노이즈 단계 $$t$$를 수행한 뒤, 순차적으로 과거·미래 정보를 번갈아 주입하며 역방향으로 진행(알고리즘 1).  

## 1. 해결 문제  
- **재구성-지각 품질 트레이드오프**: 한정된 모델 용량에서 PSNR·SSIM 같은 픽셀 기반 지표를 높이면 실제 보이는 품질은 떨어짐.  
- **단일 이미지 SR 적용의 한계**: SISR 모델을 프레임별로 독립 적용하면 디테일은 생성되나 시간적 불연속 문제 발생.  
- **기존 VSR 기법의 디테일 부족**: BasicVSR, RVRT 등은 흐릿한 결과, GAN 기반 RealBasicVSR는 어느 정도 개선하나 세밀한 일관성 부족.  

## 2. 제안 방법  
### 2.1 모델 구조  
- **사전 학습 LDM (SD×4 Upscaler)**  
  -  Latent Diffusion 기반 UNet + VAE 디코더  
- **Temporal Conditioning Module (TCM)**  
  -  ControlNet 스타일로 디노이징 UNet 디코더 내 삽입  
  -  입력:  
    – Noisy latent $$x_t$$ (4채널) + LR 프레임(3채널)  
    – 시간적 텍스처 가이드 $$\widetilde{\mathrm{HR}}^{i-1\to i}$$ (3채널)  

### 2.2 핵심 수식  
1. **노이즈 제거 예측**

$$
  x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \tfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\tilde\epsilon\Bigr) + \sigma_t z
     \quad(\tilde\epsilon = \epsilon_\theta(x_t,t,\dots))
$$  
  
2. **노이즈 없는 가이드 $$\tilde x_0$$**

$$
     \tilde x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\Bigl(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)\Bigr)
$$ 

3. **Temporal Texture Guidance**  

$$
     \widetilde{\mathrm{HR}}^{i-1\to i}
     = \mathrm{MC}\bigl(\mathrm{ME}(LR^{i-1},LR^i),\,
       D(\tilde x_0^{\,i-1})\bigr)
$$  

### 2.3 학습 및 추론  
- **학습**: TCM만 2만 스텝 학습(Adam, lr=1e−5, 배치 32)  
- **추론**: $$T=50$$ 단계 DDPM 샘플링, 프레임당 약 100 s  

## 3. 성능 향상 및 한계  
| 메트릭             | StableVSR (ours)    | 최적 비교 기법            |
|-------------------|---------------------|--------------------------|
| LPIPS⋆↓ (REDS4)   | **0.045**           | RVRT: 0.067              |
| tLP- ↓ (Vimeo-90K-T)| **3.89**           | RVRT: 4.28               |
| PSNR⋄↑ (REDS4)    | 27.97               | RVRT: 32.74 (높으나 지각 품질↓) |
| SSIM⋄↑ (Vimeo-90K-T)| 0.877              | 낮음 (지각 품질 극대화)     |

- **장점**: LPIPS·DISTS 등 지각 품질 및 tLP·tOF 일관성 모두 최고치 달성[Table 1].  
- **단점**: 모델 규모·추론 시간 대폭 증가(파라미터 약 712 M, 프레임당 100 s).  

## 4. 일반화 성능 향상 가능성  
- **Temporal Texture Guidance**는 인접 프레임 디테일을 동적으로 활용하므로, 다양한 영상 콘텐츠·프레임 레이트에 대해 **적응적**으로 작동  
- **Bidirectional Sampling** 구조는 긴 시퀀스에도 오류 누적을 억제하여 **긴 클립** 일반화에 유리  
- 향후 **샘플링 경량화 연구** 적용 시, 실시간성 강화 및 **다양한 도메인(의료·위성 영상)** 전이 학습에 유리  

## 5. 향후 연구 방향 및 고려사항  
1. **추론 가속**: DDIM·ODE 기반 빠른 샘플링 도입  
2. **경량화 모델**: 프루닝·양자화로 메모리·연산량 저감  
3. **다중 시점·멀티스케일**: 멀티 해상도·카메라 뷰에 대한 일관성 보장  
4. **무감독 설정**: 테스트 시 실제 LR만 활용하는 **제로샷** VSR 강화  

이상으로, StableVSR는 생성적 DMs를 VSR에 성공적으로 적용해 **지각 품질**과 **시간적 일관성**을 동시에 대폭 개선했으며, 향후 **실시간화** 및 **경량화** 연구에서 큰 파급 효과가 예상된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c4c823a1-634b-439c-8963-37f13972ba2e/2311.15908v2.pdf


# Reference
https://huggingface.co/claudiom4sir/StableVSR  
https://github.com/claudiom4sir/StableVSR
