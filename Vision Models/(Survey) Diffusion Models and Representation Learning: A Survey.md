# Diffusion models and representation learning a survey

**주요 주장 및 기여**  
이 논문은 **확산 모델(Diffusion Models)** 과 **표현 학습(Representation Learning)** 의 상호작용을 체계적으로 정리하고, 두 관점에서의 연구를 통합한 최초의 종합적 분류체계를 제안한다.  
1. 확산 모델을 **표현 학습**에 활용하는 방법  
2. 표현 학습 기법을 활용해 확산 모델을 **성능 개선**하는 방법  
3. 두 분야의 접근을 아우르는 **분류체계(taxonomy)** 및 **일반화된 프레임워크** 제시  
4. 향후 연구 방향 및 잠재적 도전 과제 제안  

## 1. 문제 설정  
- **배경**: 확산 모델은 최첨단 생성 성능을 보이나, 내부적으로 학습된 표현(representation)을 활용하여 인식·분류·분할 등 하위 작업에 전이하는 방법이 정리되지 않음.  
- **목표**: (1) 생성용으로 훈련된 확산 모델의 중간 활성화(feature)를 추출해 다운스트림 작업에 활용하는 기법, (2) 셀프-슈퍼바이즈드 표현 학습을 통해 확산 모델의 **안정성·다양성·속도**를 개선하는 기법을 통합 정리.

## 2. 제안된 분류체계 및 프레임워크  
### 2.1. 확산 모델 → 표현 학습  
1. **중간 활성화 활용**:  
   - U-Net 디코더 블록의 특정 레이어 출력을 추출하여, 분류·세그멘테이션·정합(correspondence) 등에 선형 프로빙  
2. **지식 전이(Knowledge Transfer)**:  
   - 강화학습 기반 시점 선택(RepFusion)  
   - 생성 모델의 feature를 교사-학생 구조로 증류(DreamTeacher)  
3. **모델 재구성(Reconstruction)**:  
   - 확산 모델을 DAE로 단순화(l-DAE)하여 노이즈 제거 과정 자체를 표현 학습 기회로 활용  
4. **통합 모델(Joint Models)**:  
   - 분류와 생성을 한 네트워크로 동시 최적화(HybViT, JDM)  

> **프레임워크**:  
> 1) 시점(time step) 및 레이어 선택  
> 2) 피처 추출 및 후처리(업샘플링·풀링)  
> 3) 분류 헤드(MLP/CNN/어텐션) 학습  

### 2.2. 표현 학습 → 확산 모델 개선  
1. **할당 기반(Guidance)**  
   - 사전 학습된 인코더(CLIP 등)로 라벨이나 클러스터를 생성→비지도 조건부 생성(Self-guidance, kNN-Diffusion, RDM)  
2. **표현 기반(Representation-based)**  
   - 확산 모델 자체를 특징 추출기로 활용(RCG)  
   - 내부 readout 헤드로 속성(깊이·포즈·에지) 예측→가이드로 사용(Readout Guidance)  
3. **목적함수 기반(Objective-based)**  
   - 확산망 중간 어텐션을 조작해 샘플 품질 향상(PAG, SAG, Depth-Aware Guidance)  

# Methods

Section 3에서는 “Diffusion Models and Representation Learning” 설문 논문이 제시하는 **두 가지 주요 연구 축**—(1) 확산 모델을 표현 학습에 활용하는 방법, (2) 표현 학습 기법을 확산 모델의 성능 개선에 활용하는 방법—에 대해 심도 있게 다룬다. 크게 **3.1 Diffusion Models for Representation Learning**와 **3.2 Representation Learning for Diffusion Model Guidance**로 구분된다.

## 3.1 Diffusion Models for Representation Learning  
이미 생성(pre-trained)된 확산 모델의 **내부 표현**을 활용하거나, 확산 모델의 **학습 과정을 변형**하여 다운스트림 인식 작업에 활용하는 여러 패러다임을 정리한다.

### 3.1.1 Leveraging intermediate activations  
- U-Net 또는 DiT 기반 확산 모델의 **디코더 블록 중간 활성화(feature map)**를 추출  
- 최적의 시점(timestep)과 레이어를 **그리드 서치**나 **학습 가능한 선택기**로 결정  
- 추출된 피처들을 upsample → concat → 경량 분류기(MLP/CNN/어텐션)에 입력  
- 분류·세그멘테이션·시맨틱 매칭 등 다양한 작업에서 **라벨 효율성** 및 **성능 향상** 입증  

### 3.1.2 General representation extraction framework  
모든 중간 활성화 활용 방법에 공통되는 **3단계** 프레임워크 제시  
1. Timestep t 및 디코더 블록 집합 B 선택:  

```math
(t^*, B^*) = \arg\min\_{t\in T, B\subseteq\mathcal B} \mathcal L_{\mathrm{discr}}(t,B)
```  

2. 입력 이미지와 t를 확산 모델에 통과시켜, B에 해당하는 블록들의 활성화 추출  
3. 후처리(upsample·pool·concat)된 피처로 분류 헤드(MLP/CNN/어텐션) 학습  

### 3.1.3 Knowledge transfer  
- **RepFusion**: 강화학습으로 최적 timestep 선택 → 확산 모델 피처를 학생 네트워크로 증류 → fine-tuning  
- **DreamTeacher**: U-Net 중간 피처를 teacher로, 이미지 백본(f)eatures regressor로 distill  
- **DiffusionClassifier**: 텍스트-이미지 확산 모델의 ELBO를 이용한 제로샷 분류자  

### 3.1.4 Diffusion model reconstruction  
- **l-DAE**: DDPM 구조를 점진적으로 변형해 denoising autoencoder로 환원 → diffusion 고유 이점 분석  
- **DiffAE / InfoDiffusion**: 유의미한 잠재(zsem)와 노이즈(zT) 분리 → 상호정보 정규화  
- **PDAE**: posterior mean gap을 보완하는 mean-shift 예측기로 DPM을 autoencoder로 개조  
- **MDM**: 마스킹 기반 디퓨전(SSIM 손실) → 마스크드 MAE 유사 자기지도 표현 학습  

### 3.1.5 Joint diffusion models  
- **HybViT**: ViT 백본으로 분류(lossCE)와 생성(lossdiff) 동시 최적화  
- **JDM**: U-Net+분류기 공유 파라미터 → cross-entropy 및 noise prediction 손실 통합  
- **ADDP**: VQ 토큰↔픽셀 교차 디노이징 → unified generation + recognition  

### 3.1.6 Generative augmentation  
- 확산 모델을 **데이터 증강**에 활용  
  - **Generative Augmentation**: LDM으로 새 뷰 생성 → contrastive/self-sup 학습에서 augmentation으로 사용  
  - **MA-ZSC**: Stable Diffusion으로 완전 합성 제로샷 분류 데이터 생성  
  - **ScribbleGen**: 제어넷 기반 스크리블→이미지 증강으로 부분 지도 분할 개선  

## 3.2 Representation Learning for Diffusion Model Guidance  
라벨 없는 데이터만으로도 **조건부 생성(생성 제어)**을 가능케 하는 “Assignment-based” 및 “Representation-based” guidance 기법들을 체계화한다.

### 3.2.1 Assignment-based guidance  
- **kNN-Diffusion**: CLIP 피처 → kNN 검색된 이웃 임베딩으로 조건부 확산  
- **RDM**: 외부 데이터베이스에서 CLIP 기반 retrieval → latent diffusion  
- **Self-guidance**: self-supervised 피처 추출(gϕ)→k-means self-annotation(f)→guidance  
- **Online guidance**: Sinkhorn-Knopp 클러스터링을 학습 중에 동적 생성  
- **DPT**: 분류기→pseudo-label→conditional diffusion→증강 데이터로 분류기 재학습  

#### General framework  
1. **Image encoder** E(x) → feature z  
2. **Self-annotation** f(z) → condition c  
3. **Denoising network** Dθ(xt, c, t) 학습  
4. Inference: 사용자 조건 k → E(k) → f(z) → Dθ로 controlled generation  

### 3.2.2 Representation-based guidance  
- **RCG**: Contrastive encoder(z)→RDM→pixel generator → classifier-free guidance  
- **Readout Guidance**: diffusion 모델 내부 readout head로 속성(depth/pose/edge) 예측→guidance  
- **SAG**: self-attention map 흐릿하게→잔여 정보로 adversarial guidance  
- **Perceptual loss**: diffusion 모델 자체를 퍼셉추얼 네트워크로 활용  

### 3.2.3 Objective-based guidance  
- **Depth-aware guidance**: 내부 피처로 depth predictor 학습 → depth consistency & prior guidance  
- **PAG**: self-attention perturbation → 암묵적 discriminator gradient로 샘플링 제어  

이상으로 Section 3에서 제시된 **방법론(Methodology)** 전반을 체계적으로 정리하였다. 각 서브섹션은 확산 모델과 표현 학습 간의 상호 보완적 활용을 탐구하며, 다운스트림 성능 향상과 완전 무라벨 조건부 생성의 실현 가능성을 보여준다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f96e297-65ec-44e8-8f22-2e0bc7ea35c0/2407.00783v1.pdf

> **프레임워크**:  
> 1) 이미지 인코더 E(x) → 임베딩 z  
> 2) 자가-주석 함수 f(z) → 조건 c  
> 3) Dθ(xt, c, t)로 노이즈 제거  

## 3. 수식 및 모델 구조  
- **DDPM 단순화 손실**  

$$
L_{\mathrm{simple}}
= \mathbb{E}\_{t,x_0,\epsilon_t}\big\|\epsilon_t - \epsilon_\theta(x_t,t)\big\|^2.
$$

- **Classifier-free Guidance**  

$$
\tilde{\epsilon}\_\theta(x_t,c)
= (1+w)\,\epsilon_\theta(x_t,c)\;-\;w\,\epsilon_\theta(x_t,\varnothing).
$$

- **Feature-distillation Loss (DreamTeacher)**  

$$
L_{\mathrm{MSE}}
=\frac{1}{L}\sum_{l=1}^L\big\|f^r_l - \mathrm{LayerNorm}(f^e_l)\big\|^2.
$$

대표적 모델 구조  
- **U-Net**: 대칭적 인코더-디코더 + 스킵커넥션  
- **DiT**: ViT 기반 패치 순차 처리 + adaLN-Zero 블록  
- **Hybrid**: ViT + DDPM 손실 + 분류 손실 동시 최적화  

## 4. 성능 향상 및 한계  
- **표현 활용**: 분류·세그멘테이션에서 라벨 효율성 최대 10% 미만 레이블로도 SOTA 수준 성능  
- **지식 전이**: DreamTeacher가 COCO·ADE20k에서 SSL 대비 +5∼10% 향상  
- **할당 기반 가이드**: Self-guidance가 완전 무라벨(Un-cond) 대비 FID·IS 지표 +20% 개선  
- **제약**:  
  - 중간 활성화 최적 시점·레이어 탐색 비용  
  - 사전 학습 인코더 의존성(CLIP) → 순수 무라벨 한계  
  - 생성과 분류 동시 학습 시 속도·모델 크기 증가  

## 5. 일반화 성능 향상 관점  
- **온라인 클러스터링**(Sinkhorn-Knopp): 훈련 중 실시간 특성 클러스터로 적응 → 데이터 도메인 변화에 **강건**  
- **Mutual-Information Regularization**(InfoDiffusion): 잠재 공간 구조 제어 → **도메인 일반화** 지원  
- **Flow Matching 모델 전이**: 확산 모델의 시계열 경로 대신 직선화된 궤도로 빠른 샘플링 및 **일반화** 가능성  

## 6. 향후 영향 및 고려사항  
- **향후 영향**:  
  - **표현↔생성 순환 발전**: 더 나은 표현 학습이 더 나은 조건부 생성으로, 더 강력한 생성 모델이 더 풍부한 표현으로 이어짐  
  - **비전-언어 멀티모달**: CLIP 대체 가능한 완전 무라벨 멀티모달 임베딩 개발 가속  
  - **실시간 제어·인터랙티브 생성**: 중간 활성화·어텐션 활용한 실시간 사용자 제어 강화  

- **연구 시 고려점**:  
  1. **모델 규모 vs. 효율성**: 성장하는 모델 크기 대비 계산·에너지 비용  
  2. **순수 무라벨 접근**: CLIP·대규모 라벨 의존도 탈피  
  3. **해석 가능성·제어성**: 잠재 공간의 의미론적 분리·해석 가능한 방향 탐색  
  4. **비전→로봇·의료 등 실환경 적용**: 일반화 성능과 안전성 검증  

**결론**: 이 논문은 확산 모델과 표현 학습의 경계를 허물며, 두 분야를 아우르는 분류체계와 프레임워크를 제시해 연구의 **통찰**과 **실용적 지침**을 제공한다. 앞으로는 **진정한 무라벨 멀티모달**, **효율적·해석 가능한 아키텍처**, **실시간 제어**를 고려한 연구가 핵심 과제로 떠오를 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f96e297-65ec-44e8-8f22-2e0bc7ea35c0/2407.00783v1.pdf

# Reference
https://arxiv.org/abs/2407.00783

