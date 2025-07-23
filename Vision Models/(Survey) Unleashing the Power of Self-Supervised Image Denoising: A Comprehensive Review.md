# Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review

# 핵심 주장 및 주요 기여 요약  

“Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review”는 **노이즈-클린 쌍** 없이도 이미지 잡음을 효과적으로 제거할 수 있는 **셀프-슈퍼바이즈드(Self-Supervised) 딥러닝** 기법들을 정리한 최초의 종합 리뷰이다.  
1. 노이즈-클린 페어가 없는 실세계 조건에서의 **딥러닝 기반 이미지 복원** 수요를 제시  
2. 알고리즘을 세 가지 범주(General, BSN-기반, Transformer-기반)로 분류하고 각 기법의 **이론적 원리**, **구조**, **응용 시나리오**를 체계적으로 분석  
3. 다양한 공공 벤치마크(Grayscale, Color, rawRGB, sRGB 데이터셋)에 대한 **정량(PSNR/SSIM)·정성 비교 실험**을 통해 성능을 평가  
4. 현재 기법의 한계(공간적 독립성 가정, 복잡한 노이즈 모델 미흡 등)와 **미래 연구 방향**(BSN 개선, Transformer 융합, Diffusion 모델 도입) 제안  

# 1. 해결 과제  

- **노이즈-클린 이미지 쌍 부족**  
  - 실제 촬영 환경에서 완전한 클린 이미지 확보 어려움  
- **전통적·감독 학습 기반 딥러닝 한계**  
  - 합성 노이즈로 학습된 모델은 실세계 노이즈 일반화에 취약  
- **노이즈 모델링 다양성 결여**  
  - Gaussian, Poisson 외에 Multiplicative, Salt-&-Pepper, Spatially Correlated Noise 처리 미흡  

# 2. 제안 기법들  

## A. General Methods  
-  Noise2Noise (N2N)  
  – 손실: $$L=\|f(y_1)-y_2\|^2$$, 정렬된 노이즈-노이즈 쌍 필요  
-  Noise2Void (N2V), Noise2Self (N2S)  
  – Blind-spot 전략으로 입력 일부 마스킹, 잃어버린 픽셀 예측  
-  SURE-based  
  – MC-SURE를 이용해 **리스크 추정**만으로 네트워크 훈련  
-  Noisier2Noise, NAC, R2R, IDR, Noise2Score…  
  – “인위 노이즈 추가→자기 학습” 프레임워크, 통계적 리스크 추정, 반복 정제, 스코어 함수 학습+Tweedie 공식 등  

## B. Blind Spot Network (BSN) 기반  
-  **입력 마스킹**: N2V, PN2V, Noise2Same, S2S, B2UB  
  – 픽셀 단위 무작위/인접값/확률적 분포 마스킹  
  – B2UB Global-Aware Mask Mapper + Re-visible Loss: 모든 픽셀 활용  
-  **네트워크 마스킹**: Laine et al., DBSN, AP-BSN, MM-BSN, Li et al.  
  – 수평·수직 반 필터, PD/Patch-Mask, Dilated Conv/Transformer 융합, region-adaptive masking  

## C. Transformer 기반  
-  DT: **CNN+Transformer 하이브리드**  
  – Context-Aware Denoise Transformer (CADT) + Secondary Noise Extractor  
-  LG-BPN: DSPMC(Patch Mask) + Dilated Transformer Block  
-  SwinIA: Swin-Transformer Autoencoder, **마스킹 불필요**  

# 3. 성능 향상 및 한계  

| 범주               | 대표 기법     | 장점                                           | 한계                                                    |
|-------------------|--------------|----------------------------------------------|---------------------------------------------------------|
| General           | N2N, SURE    | 간단한 손실, 통계적 리스크 기반 복합 노이즈 적용 가능     | 노이즈 모델 가정 불완전, alignment 필요              |
| BSN-기반 (입력)      | B2UB         | 모든 픽셀 학습 활용, 공간적 독립성 가정 하 검사된 재현성    | 실세계 공간 상관 노이즈 파괴 시 텍스처 손실            |
| BSN-기반 (네트워크)  | AP-BSN, MM-BSN | 실제 sRGB에서 blind-spot 적용, 공간 상관 파괴 전략          | PD/Multi-mask로 텍스처 손상, 동적 노이즈 형상 대처 미비    |
| Transformer-기반   | LG-BPN, DT   | 글로벌 문맥 정보 포착으로 세밀한 디테일 보존              | 계산량 급증, CNN 대비 이점 불명확               |

- **일반화 성능**:  
  – BSN-기반은 **“잡음 독립성”** 가정↑, Gaussian→비정형 노이즈 일반화↑  
  – Transformer 융합은 **지역·전역 정보 균형** 가능성↑, 계산 효율화 필요  

# 4. 모델 구조 핵심  

- **Blind-Spot Network**: 중앙 픽셀 배제→주변 예측  
- **Patch Mask / PD**: 공간적 상관 파괴, stride 조절로 노이즈 패턴 해체  
- **Transformer Branch**: 채널 attention, dilated self-attention으로 global 문맥 습득  
- **Iterative Data Refinement**: 예비 복원 결과 재입력, 세밀도 vs. 과도한 스무딩 균형  

주요 수식 예) Noise2Score:  

$$
\text{score}(y) = \nabla_y \log p(y),\quad \hat{x} = y + \sigma^2 \cdot \text{score}(y)
$$ 

MC-SURE 기반 손실:  

$$
L(\theta) = \frac{1}{N}\sum_i \|f(y_i;\theta)-y_i\|^2 - 2\sigma^2\,\mathrm{div}\,f(y_i;\theta)
$$  

# 5. 일반화 성능 향상 관점  

1. **공간 상관성 파괴 기법 개선**  
   – 고정 PD/Mask → 노이즈 형상 적응적 마스킹 학습  
2. **잡음 모델 독립적 학습**  
   – Tweedie/Score Matching → 비지수 가족 노이즈 확장  
3. **Transformer 결합 최적화**  
   – 연산량 감소형 self-attention, Channel vs. Spatial attention 분리  
4. **엔드투엔드 Diffusion 모델 활용**  
   – 잡음 생성·제거 양방향 학습으로 다양한 노이즈 일반화  

# 6. 향후 연구 및 고려사항  

- **실세계 sRGB 노이즈**: 복합·공간 상관 잡음→Adaptive BSN+Transformer  
- **대규모 노이즈 구조**: Stripe·Speckle 등 특수 잡음 대응 메커니즘  
- **Diffusion 모델 통합**: AIGC 활용 self-supervision으로 노이즈 분포 학습  
- **실시간·경량화**: 모바일·임베디드 환경 적용을 위한 **연산 최적화**  

> *이 리뷰는 셀프-슈퍼바이즈드 이미지 복원 연구의 로드맵을 제시하며, 노이즈 일반화 및 전역 정보 통합 관점에서 향후 모델 설계에 실질적 영감을 제공할 것이다.*

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5928d84-d2de-4bce-ba1e-ee55d07d570d/2308.00247v4.pdf

# II. Self-Supervised Image Denoising Methods  
이 절에서는 노이즈–클린 쌍 없이 학습할 수 있는 세 가지 **셀프-슈퍼바이즈드(self-supervised)** 이미지 디노이징 기법 계열을 소개합니다.  
1. General Methods (일반 기법)  
2. Blind-Spot Network (BSN)-based Methods  
3. Transformer-based Methods  

각 기법은 **노이즈 모델링**, **마스킹 전략**, **네트워크 구조**, 그리고 **손실 함수** 측면에서 서로 다른 아이디어를 사용하며, 대표 모델을 중심으로 핵심 원리와 구현을 쉽게 풀어 설명합니다.

## 1. General Methods  
“General” 범주에는 입력에 별도의 마스킹 없이, **노이즈만 있는 데이터** 혹은 **노이즈–노이즈 쌍**만으로 학습하는 방법들이 포함됩니다.

### 1.1 Noise2Noise (N2N)  
- **핵심 아이디어**: 서로 다른 노이즈가 섞인 동일한 원본 이미지 쌍 $$(y_1 = x + n_1,\ y_2 = x + n_2)$$을 입력–타깃으로 사용  
- **손실 함수**  

$$
    L(\theta) = \|f(y_1;\theta) - y_2\|^2
  $$
  
- **장점**: 클린 이미지 없이도 “노이즈 평균”이 원본을 복원한다는 통계적 성질 이용  
- **제한**: 완벽히 정렬된 노이즈–노이즈 쌍 필요  

### 1.2 SURE-based Methods  
- **핵심 아이디어**: Stein’s Unbiased Risk Estimator (SURE)를 이용해 **클린 이미지 없이** 잔차 기반 리스크를 추정  
- **손실 함수 예시** (MC-SURE)  

$$
    L(\theta)=\frac1N\sum_i\|f(y_i;\theta)-y_i\|^2 - 2\sigma^2\,\mathrm{div}\,f(y_i;\theta)
  $$  
- **장점**: Gaussian뿐 아니라 리스크 추정이 가능한 다양한 노이즈 모델에 적용  
- **제한**: 노이즈 분산 $$\sigma^2$$ 사전 지식 필요, 하이퍼파라미터 민감  

### 1.3 Noisier2Noise, NAC, R2R, IDR, Noise2Score  
- **Noisier2Noise**: 단일 노이즈 이미지에 인위 노이즈를 두 번 추가해 노이즈–노이즈 쌍 생성  
- **Noise-As-Clean (NAC)**: “약한” 노이즈 이미지를 클린으로 간주하고, 동일 종류 합성 노이즈 추가로 쌍 생성  
- **Recorrupted-to-Recorrupted (R2R)**: 데이터 증강으로 인위 노이즈–노이즈 쌍 생성  
- **Iterative Data Refinement (IDR)**: 초기 디노이저로 노이즈를 단계별로 낮춰가며 반복 학습  
- **Noise2Score**: 스코어 매칭 기반으로 **로그 우도 기울기** 추정 후 Tweedie 공식으로 복원  

## 2. Blind-Spot Network (BSN)-based Methods  
BSN은 **공간적 노이즈 독립성(zero-mean, uncorrelated)** 가정을 이용해 “관심 픽셀”을 감추고 주변 정보만으로 예측하도록 네트워크를 설계합니다.

### 2.1 Mask in Input (입력 마스킹)  
입력 이미지의 일부 픽셀을 감추고, 원본(노이즈 포함) 전체를 타깃으로 학습합니다.

- **Noise2Void (N2V)**  
  – 입력에서 랜덤 픽셀을 이웃값으로 대체(Masking)  
  – $$\;L=\|f(y_{\text{masked}})-y_{\text{orig}}\|^2$$  
- **Noise2Self (N2S)**  
  – *J-invariance* 개념: 특정 위치 집합 $$J$$ 마스킹 → 예측  
- **PN2V** (Probabilistic N2V)  
  – 픽셀별 노이즈 분포를 학습해 확률적 예측  
- **Noise2Same**, **Self2Self (S2S)**, **Blind2Unblind (B2UB)** 등  
  – 마스킹 방식, 타깃 대체(랜덤·국지 평균), 재가시화(re-visible) 손실 다양화  

### 2.2 Mask in Network (네트워크 마스킹)  
입력은 완전하지만 네트워크 내부의 수용 영역(receptive field)에서 중앙 픽셀을 빼고 연산합니다.

- **Laine et al.** (Four-branch BSN)  
  – 상·하·좌·우 반평면만 보는 병렬 U-Net 네트워크  
- **Dilated BSN (DBSN)** + Knowledge Distillation  
- **Asymmetric PD-BSN (AP-BSN)**  
  – Pixel-Shuffle 다운샘플링 비대칭 적용  
- **Multi-Mask BSN (MM-BSN)**  
  – 다양한 모양(斜선, □ 등) 마스크로 대형 상관 노이즈 차단  
- **Spatially Adaptive BSN**  
  – 평탄 영역 vs. 텍스처 영역을 구분해 분리 학습  

## 3. Transformer-based Methods  
Transformer 구조를 도입해 **전역(global) 문맥**과 **국부(local) 정보**를 함께 활용합니다.  

### 3.1 DT (Denoise Transformer)  
- **입력 마스킹**: Blind2Unblind과 유사한 글로벌 마스크  
- **구조**  
  - Local branch: CNN + deformable conv  
  - Global branch: Self-attention 기반 transformer  
  - Secondary Noise Extractor: 경량 후처리 블록  
- **특징**: 잔차 학습으로 국부·전역 잡음을 추출해 결합  

### 3.2 LG-BPN (Local-Global Blind-Patch Network)  
- **Mask in Network**: DSPMC(Densely-Sampled Patch-Mask Conv)  
  – 일정 반경 내에서만 수용 영역 마스킹  
- **Dilated Transformer Block**: 공간 self-attention → 채널 attention 대체  

### 3.3 SwinIA  
- **완전 Transformer-Autoencoder**: Swin-Transformer 기반  
- **Blind-Spot 없이** 순수 **자기 지도 학습** 가능  

## 핵심 비교 및 시사점  
| 범주       | 대표 모델       | 학습 요구사항                   | 장점                                    | 한계                                   |
|----------|--------------|-----------------------------|---------------------------------------|--------------------------------------|
| General  | N2N, SURE    | 노이즈–노이즈 쌍 or 노이즈만        | 간단한 손실, 다양한 노이즈 모델 적용 가능       | 쌍 정렬 필요·모델 민감도 높음                |
| BSN-입력  | N2V, N2S, B2UB | 단일 노이즈 이미지                | 클린 불필요, 픽셀 은닉으로 오버핏 방지         | 정보 손실·텍스처 훼손 위험                 |
| BSN-네트워크 | Laine et al., AP-BSN, MM-BSN | 단일 노이즈 이미지                | 모든 픽셀 활용, 대형 상관 노이즈 차단           | 복잡도↑, 마스크 최적화 필요               |
| Transformer | DT, LG-BPN, SwinIA | 단일 노이즈 이미지                | 전역 문맥 포착 가능                       | 연산량 과다, CNN 대비 성능 향상 한계            |

이렇게 세 범주의 **셀프-슈퍼바이즈드 디노이징** 기법은  
- **General**: 데이터 쌍 없이도 통계적 손실로 학습  
- **BSN-based**: 마스킹을 통해 클린 없이 픽셀 예측  
- **Transformer**: 전역·국부 정보 융합  

각각의 **구조적·통계적 가정** 아래 **노이즈 독립성**, **공간 상관성 파괴**, **글로벌 문맥** 확보라는 서로 다른 전략을 구사하며, 실제 sRGB·rawRGB·의료·현실 노이즈 데이터 전반에 걸쳐 강력한 성능을 보여 줍니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5928d84-d2de-4bce-ba1e-ee55d07d570d/2308.00247v4.pdf
