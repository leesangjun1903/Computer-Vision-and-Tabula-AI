# ZeCon : Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer | 2023 · 100회 인용, Image-to-Image generation

**핵심 주장 및 주요 기여**  
사전 학습된 확산 모델 내에 이미 존재하는 공간적 정보를 패치 단위 대조 학습(contrastive learning)으로 활용하여, 추가 네트워크나 파인튜닝 없이 “제로샷”으로 텍스트 기반 이미지 스타일 전이(text-guided style transfer)를 수행한다. 이 방법은 스타일 변화와 콘텐츠 보존 간 균형을 효과적으로 달성하며, 별도의 학습 과정 없이 다양한 스타일 전이와 이미지 조작(image-to-image translation, manipulation)에 활용 가능하다.[1]

## 문제 정의  
기존 확산 모델은 무조건(unconditional) 샘플링 과정의 확률적(stochastic) 특성 때문에 스타일 전이 시 콘텐츠가 소실되는 문제를 가진다.  
- 페어 데이터셋이 필요한 conditional diffusion: 데이터 수집이 비현실적  
- DDIB, DiffusionCLIP, DiffuseIT: 추가 모델 학습 또는 파인튜닝이 필요해 계산비용이 크고, 도메인 변경 시 성능 저하 발생.[1]

## 제안 방법  
### 1) 제로샷 대조 손실(ZeCon Loss)  
사전 학습된 U-Net 노이즈 예측기(noise estimator)의 인코더(feature extractor)에서 추출한 패치 단위 특징맵 $$z_l$$을 활용한다. 원본 이미지 $$x_0$$와 역전 샘플링된 이미지 $$x_{0,t}$$의 동일 패치 위치는 positive, 다른 위치는 negative로 간주하여 대조 손실을 계산한다:  

$$
\mathcal{L}_{\text{ZeCon}}(x_{0,t},x_0)
= -\mathbb{E}_{s}\Big[ \log\frac{\exp\big(z_{s}^{\top} z'_{s}/\tau\big)}
{\sum_{i=1}^{S_l}\exp\big(z_{s}^{\top} z'_{i}/\tau\big)}\Big]
$$

[1]

### 2) 콘텐츠 손실 결합  
대조 손실 외에 VGG 기반 특징 손실과 픽셀 단위 MSE를 더해 콘텐츠 정밀도를 높인다:  

```math
\mathcal{L}_{\text{content}}
= \mathcal{L}_{\text{ZeCon}} + \lambda_{\text{VGG}}\|\phi(x_{0,t})-\phi(x_0)\|_2^2
+ \lambda_{\text{MSE}}\|x_{0,t}-x_0\|_2^2
```

[1]

### 3) 스타일 손실  
CLIP 기반 전역(global) 및 방향(direction) 손실을 결합한다.  
- 전역 CLIP 손실: 생성 이미지와 스타일 프롬프트 간 코사인 거리  
- 방향 CLIP 손실: CLIP 임베딩 방향 정렬  
패치 기반으로 확장해 모드 붕괴(mode collapse) 완화 및 고품질 스타일 반영을 달성한다.[1]

### 4) 샘플링 스킴  
- Forward: DDIM($$\eta\!=\!0$$)으로 안정적 콘텐츠 보존  
- Reverse: DDPM으로 풍부한 스타일 표현  
기본 타임스텝은 $$T\!=\!50$$, 스킵 시점 $$t_0\!=\!25$$로 설정하여 품질과 속도를 균형 조정한다.[1]

## 모델 구조  
사전 학습된 unconditional U-Net 확산 모델과 CLIP, VGG 네트워크를 그대로 활용하며, 별도 학습 파라미터는 없다. 각 역전 단계(reverse step)마다 다음 과정을 수행한다:  
1. 노이즈 예측기 인코더에서 $$x_0$$, $$x_{0,t}$$ 특징 추출  
2. ZeCon 대조 손실, VGG·MSE 콘텐츠 손실 계산  
3. CLIP 전역·방향 패치 손실 계산  
4. 손실 그라디언트를 $$x_{0,t}$$에 추가해 업데이트  
5. DDPM 역전 과정으로 다음 이미지 샘플링  

## 성능 향상 및 한계  
- **콘텐츠 보존**: 제안 기법은 GAN·기존 확산 모델 대비 구조적 세부 정보(창문, 얼굴 특징 등)를 뛰어나게 유지.[1]
- **스타일 전이**: 보이지 않는(unseen) 도메인(회화·초상화 등)에서도 높은 전이 품질과 속도(≈38초 vs DiffusionCLIP 293초) 보임.[1]
- **한계**:  
  1. 손실 가중치($$\lambda$$값) 및 패치 크기 등 하이퍼파라미터 조정이 필요  
  2. 일부 사례에서 텍스트 프롬프트가 이미지에 출력되는 현상 관찰됨.[1]

## 일반화 성능 관련 고찰  
- 사전 학습된 모델을 그대로 활용하므로, 새로운 도메인에도 별도 파인튜닝 없이 바로 적용 가능하며, 사용자 실험에서 사진(photo)·회화(painting)·초상화(portrait) 등 다양한 도메인 간 유사한 콘텐츠 보존·스타일 점수를 기록.[1]
- 패치 단위 대조 손실이 모델 내장 공간 정보를 효과적으로 활용하고, 확산 모델의 stochasticity를 제어해 일반화 성능을 안정적으로 개선한다.

## 연구 영향 및 향후 고려 사항  
- **영향**: 확산 기반 스타일 전이에 제로샷 학습 개념을 도입함으로써, 무학습(no-training) 이미지 편집 분야의 새로운 패러다임 제시. 다양한 응용(이미지 조작, 변환, 고해상도 합성)으로 확장 가능.  
- **향후 과제**:  
  - 자동 하이퍼파라미터 최적화 기법 도입  
  - 텍스트 프롬프트 오버레이 현상 제거  
  - 대규모·고해상도 스타일 전이 성능 연구  
  - 확산 모델 내 다양한 내장 임베딩 활용 사례 탐색  

---

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c9fa545b-3f49-417e-b7f3-749e4e0e34f6/2303.08622v2.pdf)
