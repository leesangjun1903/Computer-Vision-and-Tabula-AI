# CDM : Cascaded Diffusion Models for High Fidelity Image Generation | Image generation, Super resolution
# 핵심 요약

**핵심 주장**  
Cascaded Diffusion Models(CDM)는 여러 해상도의 순차적 확산 모델을 파이프라인으로 연결해 고해상도 이미지를 생성하며, 보조 분류기(classifier) 없이도 ImageNet class-conditional 생성에서 BigGAN-deep 및 VQ-VAE-2를 능가하는 샘플 품질(FID, CAS)을 달성한다[1].

**주요 기여**  
1. **Cascading**: 32×32 기본 모델 → 64×64, 128×128, 256×256 super-resolution 모델 순차적 구성[1].  
2. **Conditioning Augmentation**: 각 super-resolution 단계에서 저해상도 조건(z) 입력에 가우시안 노이즈 또는 블러를 적용해 train–test 불일치로 인한 오차 누적을 억제[1].  
3. **최고 성능**:  
   - FID: 64→1.48, 128→3.52, 256→4.88  
   - CAS (Top-1/Top-5): 256→63.02%/84.06% (Real: 73.09%/91.47%)[1].

# 문제 정의 및 제안 기법

## 해결하고자 하는 문제  
- 대규모 고해상도(high-fidelity) 데이터셋(예: ImageNet)에서 보조 분류기 없이 pure generative diffusion 모델의 샘플 품질 한계.

## 제안 방법  
1) **Cascaded Pipeline**  
   - 저해상도 생성 pₜi(z₀|c) → 고해상도 pₓ(x₀|zₖ, c) 순차적 ancestral sampling.  
2) **Conditioning Augmentation**  
   - **Truncated**: z₀ → zₛ (timestep s에서 중단)  
  $zₛ = \sqrt{\barαₛ} z₀ + \sqrt{1-\barαₛ}ϵ,\quad ϵ∼\mathcal{N}(0,I)$  
   - **Non-truncated**: 샘플된 z₀에 추가 노이즈 q(zₛ|z₀) 적용[1].  
3) **수식 (ELBO 변형)**  
   - Cascaded ELBO:  

  $$
       -\log p^s_θ(x₀) \leq E\bigl[L_T(z₀)+\sum_{t>s}D_{KL}(q(z_{t-1}|z_t,z₀)\|p_θ(z_{t-1}|z_t)) + L_θ(x₀|z_s)\bigr].
  $$

## 모델 구조  
- **U-Net 기반** 각 단계  
  - Class embedding + timestep embedding  
  - Super-res 단계: 저해상도 업샘플(bilinear)된 z 입력 채널 연결[1].

# 성능 향상 및 한계

## 성능 향상  
- **Compounding Error 감소**: Conditioning augmentation으로 train/test mismatch 완화 → 샘플 품질(FID) 비선형 최적화 (s 중간값에서 최고 성능)[1].  
- **효율적 Sampling**: 256×256 단계에서 100 스텝만으로도 FID ≈ 4.88 달성[1].

## 한계  
- **샘플링 속도**: 다수 단계와 수천 timestep 필요 → 생성 지연.  
- **하이퍼파라미터**: s, σ 범위, augmentation 비율 등 민감 → 별도 탐색 필요.  
- **자원 소모**: 다단계 모델 독립 학습·추론 시 연산·메모리 비용 증가.

# 일반화 성능 향상 관점

- **다양한 데이터셋 적용성**: LSUN 등의 비클래스 조건(무조건) 데이터셋에서도 conditioning augmentation 효과 확인[1].  
- **고해상도 확장**: 추가 super-res 단계로 512×512 이상 확장 가능성, 다단계 조합 시 안정적 품질 확보 기대.  
- **Train/Test Mismatch 이론적 이해**: 비감독 학습·순차 모델의 exposure bias 해결책으로 확산 가능.

# 향후 연구 영향 및 고려사항

1. **교차 기법 통합**: Classifier guidance, 시맨틱 지도 정보, 텍스트-이미지 조건부 확산과 결합 시 더욱 향상된 품질 기대.  
2. **샘플링 효율화**: DDIM, ODE 기반 deterministic 샘플러 통합으로 speed–quality trade-off 개선 연구.  
3. **하이퍼파라미터 자동화**: s 및 σ 최적화 자동화(AutoML)로 대규모 튜닝 비용 절감.  
4. **안정성·윤리성**: 고품질 합성 이미지의 악용 방지 및 데이터 편향 최소화 위한 거버넌스 필요.

[1] Cascaded Diffusion Models for High Fidelity Image Generation, Ho et al. (2021).

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cf60b812-cf49-4761-8222-3fea19669610/2106.15282v3.pdf
