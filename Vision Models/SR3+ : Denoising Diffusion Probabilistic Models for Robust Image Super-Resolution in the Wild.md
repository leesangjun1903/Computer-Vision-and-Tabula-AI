# SR3+ : Denoising Diffusion Probabilistic Models for Robust Image Super-Resolution in the Wild | Super resolution

**핵심 주장 및 주요 기여**  
- **주장**: 기존 GAN 기반 방법이 “야생” 이미지(unknown degradations)에 취약한 반면, 제안된 SR3+는 고차원(deeper) 파라메트릭 열화(degradation)와 noise-conditioning augmentation을 결합한 확산(diffusion) 모델로서, zero-shot blind super-resolution에서 SOTA 성능을 달성한다.  
- **주요 기여**:  
  1. 복합 파라메트릭 열화(blur, JPEG, resizing 등)와 noise-conditioning augmentation을 조합한 self-supervised 학습 파이프라인 설계.  
  2. 40M vs. 400M 파라미터 규모와 800K→61M 이미지 규모 확장을 통한 성능 스케일링 입증.  
  3. 실험을 통해 RealSR/DRealSR 벤치마크에서 FID 36.82→32.37까지 개선하며 기존 SR3 및 Real-ESRGAN을 능가.  
  4. noise conditioning의 test-time 활용(teval)을 통한 텍스처 hallucination 제어 및 일반화 강화.  

## 1. 해결 과제  
야생 이미지에는 unknown blur, noise, JPEG 압축, downsampling 등 복합적이며 예측 불가한 열화가 섞여 있어, 학습 시 정해진 열화 모델만을 사용한 SR3나 GAN 모델은 zero-shot 상황에서 품질 저하(oversmoothing, excessive contrast 등)가 발생함.  

## 2. 제안 방법  
### 2.1. 데이터 파이프라인  
1) **Higher-order degradations**:  
   - 두 단계(repeat=2)의 랜덤 열화 수행  
   - Blur (Gaussian, generalized Gaussian, plateau, sinc), resizing (area/bicubic/bilinear), JPEG 압축 + 추가 sinc  
2) **Noise-conditioning augmentation**:  
   - 학습 시 upsampled LR 입력에 τ ∼ Uniform(0, τmax=0.5) 샘플링 후 forward diffusion noise 추가, τ도 조건으로 인가  
   - 손실: L(θ)=E‖ϵθ(zt,t,cτ)−ϵ‖² (Eq.2)  
   - 테스트 시 teval 조절을 통해 conditioning 신호 대비 hallucination 강도 trade-off 제공  

### 2.2. 모델 구조  
- 기본 UNet 기반(“Efficient U-Net v3”)  
- Self-attention 제거로 arbitrary resolution/general aspect ratio 지원  
- 두 크기: 40M & 400M 파라미터  

### 2.3. 학습 및 평가  
- 학습 데이터: DF2K+OST(800+2650+300) → 61M 대규모 이미지  
- 1.5M 스텝, batch 256→512, 64×64→256×256 zero-shot 테스트  
- 평가: PSNR, SSIM(참조 기반) 및 FID-10K(통계 기반)  

## 3. 성능 향상 및 한계  
| 모델                           | RealSR FID↓ | DRealSR FID↓ | PSNR↑    | SSIM↑     |  
|--------------------------------|------------|-------------|----------|-----------|  
| Real-ESRGAN                     | 34.21      | 37.22       | 25.14    | 0.728     |  
| SR3+ (40M, DF2K+OST)            | 31.97      | 40.26       | 24.84    | 0.683     |  
| SR3+ (400M, DF2K+OST)           | 27.34      | 36.28       | 23.84    | 0.662     |  
| SR3+ (400M, 61M 이미지)         | **24.32**  | **32.37**   | 24.89    | 0.692     |  

- **스케일링 효과**: 파라미터·데이터 확대 시 FID 34→24, 37→32 대폭 개선.  
- **Ablation**:  
  - 열화 제거 시 FID ↑≈10  
  - noise-conditioning 제거 시 FID ↑≫10  
  - 두 기법 모두 제거(SR3) 시 FID 85→93 (심각한 블러)  
- **한계**:  
  - 과도한 teval 적용 시 conditioning misalignment 및 텍스트 왜곡  
  - 더 복잡한 noise-conditioning으로 학습 난이도 상승 → 추가 학습 스텝 필요  

## 4. 일반화 성능 향상 관점  
- **Higher-order degradations**로 실제 야생 이미지 분포 근사  
- **Noise-conditioning augmentation**으로 unseen noise·blur·compression 분포 자동 보정  
- **UNet without self-attention**: 임의 해상도/종횡비 적용 시 overfitting 억제  
- **테스트 시 teval 튜닝**: hallucination 대 conditioning alignment 간 균형 제어  

## 5. 향후 영향 및 연구 고려사항  
- **확산 모델 강화**: cascading text-to-image, deblurring 등 다양한 vision 과제에 SR3+ 기법 적용  
- **데이터·모델 스케일링**: 파라미터 수·훈련 이미지 규모 확장에 따른 품질 향상 지속 확인  
- **조건 신호 처리**: noise-conditioning 범위(teval) 자동 최적화, 텍스트·의료영상 등 특수 도메인 대응  
- **아키텍처 개선**: self-attention 대체 메커니즘 연구로 구조적 일반화 성능 추가 확보  

**결론**  
SR3+는 복합적 열화와 noise-conditioning을 결합한 diffusion 모델로서, blind SR에서 “야생” 이미지 일반화 성능을 획기적으로 향상시켰다. 이 접근은 대규모 학습 및 다양한 downstream vision 태스크로의 확장 가능성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c29a2715-f25e-4def-b257-66f356265758/2302.07864v1.pdf
