# LPTN : High-Resolution Photorealistic Image Translation in Real-Time: A Laplacian Pyramid Translation Network | Image generation

# 핵심 주장 및 주요 기여 요약

**핵심 주장**  
이 논문은 고해상도(최대 4K) 이미지를 실시간으로 포토리얼리스틱하게 도메인 간 변환할 수 있는 새로운 네트워크 구조, **Laplacian Pyramid Translation Network (LPTN)**을 제안한다. 저주파수 성분에서는 조명 및 색상 같은 속성(attribute) 변환을 수행하고, 고주파수 성분에서는 내용 디테일(detail)을 마스킹(mask) 기반으로 효율적으로 보정함으로써 고해상도 영상에서도 기존 방법 대비 대폭 빠른 속도와 유사한(또는 우수한) 품질을 달성한다.

**주요 기여**  
- **주파수 분해 기반 분리**: Laplacian 피라미드를 이용해 입력 이미지를 저주파수 성분 $$I_L$$과 고주파수 성분 $$\{h_0, \dots, h_{L-1}\}$$으로 분해하여, 속성 변화와 디테일 보정을 별도 모듈로 처리.  
- **경량 속성 변환**: 저주파수 성분 $$I_L\in\mathbb{R}^{\frac{h}{2^L}\times\frac{w}{2^L}\times3}$$에만 5개 잔차블록(residual block) 네트워크를 적용하여 조명·색상 변환 수행.  
- **프로그레시브 마스킹**: 가장 낮은 해상도 고주파수 $$h_{L-1}$$에 mask $$M_{L-1}\in\mathbb{R}^{\frac{h}{2^{L-1}}\times\frac{w}{2^{L-1}}\times1}$$를 학습하고, 이를 선형 보간과 소규모 컨브 블록으로 단계별 상위 해상도로 확대해 각 $$h_l$$를  

$$
    \hat h_l = h_l \odot M_l
  $$
  
  으로 보정.  
- **실시간 처리**: GPU 한 대로 4K 영상도 실시간(≤ 0.1 s) 변환 가능.  
- **성능 비교**: CycleGAN, UNIT, MUNIT, White-box, DPE 등과 비교해 유사 PSNR/SSIM, 우수한 사용자 선호도.

***

# 상세 설명

## 1. 해결 문제  
- **고해상도 I2IT 한계**: 기존 포토리얼리스틱 I2IT 기법은  
  1) 인코더–디코더 기반 모델이 고해상도에서 연산량 과다  
  2) 스타일 트랜스퍼형 모델이 HD 이상에서 수초 내외 소요  
  3) 디테일·속성 분리가 어려워 구조 왜곡 발생  
  
  을 해결하고자 함.

## 2. 제안 방법  

### 2.1 Laplacian 피라미드 분해·재구성  
입력 $$I_0\in\mathbb{R}^{h\times w\times3}$$를 Laplacian 피라미드로 분해해  

$$
  \{h_0, h_1, \dots, h_{L-1},\,I_L\}
$$

를 얻고, 역방향 합성으로 원본 복원이 가능.

### 2.2 저주파수 변환  

$$
  \hat I_L = \mathrm{TNet}(I_L)\,,
$$

여기서 TNet은 1×1 컨브로 채널 확장 후 5개의 residual block(각 3×3 컨브+LeakyReLU×2)으로 구성되며, 출력에 잔차 연결 및 Tanh를 적용.

### 2.3 고주파수 디테일 보정  
가장 낮은 해상도 고주파수 $$h_{L-1}$$에 대해  

$$
  M_{L-1} = \sigma\bigl(\mathrm{MaskNet}(\,[h_{L-1},\,\mathrm{up}(I_L),\,\mathrm{up}(\hat I_L)])\bigr)
$$

을 학습하고  

$$
  \hat h_{L-1} = h_{L-1}\odot M_{L-1}\,.
$$

이후  

```math
  M_l = \mathrm{Conv}\bigl(\text{bilinear\_up}(M_{l+1})\bigr),\quad \hat h_l = h_l\odot M_l
```

로 $$l=L-2,\dots,0$$까지 단계적 확장.

### 2.4 학습 목표  
재구성 손실  

$$
  \mathcal{L}_{\mathrm{recons}} = \|\,I_0 - \hat I_0\|_2^2
$$

와 LS-GAN 기반 적대적 손실  

$$
  \mathcal{L}_{\mathrm{adv}}
$$

를 가중합  

$$\mathcal{L}=\mathcal{L}\_{\mathrm{recons}}+\lambda\,\mathcal{L}_{\mathrm{adv}}$$ 로 최적화.

## 3. 모델 구조  
- **분해 모듈**: 고정 Laplacian 필터(1D 커널 )[1]
- **속성 변환기**: 저해상도 $$I_L$$ 전담, 채널 확장→5× residual blocks→채널 축소  
- **마스크 생성기**: $$h_{L-1}$$ 및 up($$I_L$$ , $$\hat I_L$$) 입력, 소형 CNN  
- **점진적 보정**: 블록 수 적은 컨브로 마스크를 단계별 해상도↑

## 4. 성능 향상  
- **속도**: 4K에서 0.03 s (L=4), 1080p에서 0.007 s로 CycleGAN 대비 ×80 가속.  
- **화질**: MIT-Adobe FiveK에서 1080p 기준 PSNR 22.09 dB, SSIM 0.883로 DPE와 비등.[1]
- **사용자 선호도**: Day→Night 변환에서 photorealism 78.3%, aesthetic 57.5% 우위.

## 5. 한계  
- **고주파수 신규 디테일 미합성**: 피라미드 디테일 의존 → 완전한 콘텐츠 생성 불가  
- **마스킹 아티팩트**: 경계부 halo 현상 발생 가능  
- **도메인 특화**: 조명·색상 변환에는 강하나 구조 변경 과제

***

# 일반화 성능 향상 가능성

1. **다양한 도메인**: 도메인 간 조명·색상이 아닌 구조적 변형에도 Laplacian 분해를 응용하면, 다중 채널 피라미드나 wavelet 변환 도입으로 일반화 가능.  
2. **마스크 확장**: 마스킹 대신 **sparse convolution** 또는 **attention** 기반 보정 모듈로 고주파 디테일 적응성 향상.  
3. **다중 레퍼런스 학습**: 레퍼런스 이미지 간 스타일 간 차이를 latent로 학습해 변환 다양성 및 일반화 확보.  
4. **교차 도메인 적대학습**: 여러 도메인 샘플 포함한 멀티디스크리미네이터로 일반화된 adversarial regularization.

***

# 향후 연구 영향 및 고려 사항

- **실시간 고해상도 변환**: 가상·증강 현실, 게임 그래픽, 라이브 비디오 필터 등에 즉시 적용 가능.  
- **경량화 네트워크 설계 패러다임**: 주파수 분해 기반 경량 모델이 다양한 영상 처리(task)로 확대 전망.  
- **신규 콘텐츠 생성 연구**: Laplacian 프레임워크에 GAN 생성 능력 결합 연구 필요.  
- **아티팩트 저감**: 마스크 기반 보정의 경계 인공물 제거, **edge-aware** refinement 기법 채택할 것.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2d65059d-a54f-461c-8b02-4fad467f172c/2105.09188v1.pdf)
