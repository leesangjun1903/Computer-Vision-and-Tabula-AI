# On Distillation of Guided Diffusion Models | Image generation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장:**  
이 논문은 기존의 **classifier-free guided diffusion 모델**이 우수한 생성 품질을 제공하지만, 추론 시 수십에서 수백 번의 모델 평가가 필요해 비효율적이라는 문제를 해결하기 위해, 이를 **2단계 프로그레시브 증류(two-stage progressive distillation)** 기법을 통해 **샘플링 단계를 10×–256×**까지 획기적으로 줄이면서도 원본 모델과 동등한 시각 품질을 유지할 수 있음을 보인다.

**주요 기여:**  
1. **Stage-1 증류:** 조건부(conditional) 및 무조건부(unconditional) 모델의 출력을 하나의 학생 모델이 모방하도록 학습하여, 가이드 지도를 위한 두 모델 평가를 한 번으로 대체.  
2. **Stage-2 증류:** DDIM 기반 프로그레시브 증류 기법을 이용해 학생 모델을 반복적으로 샘플링 단계 수를 반으로 줄여가며 재증류함으로써, **픽셀 공간 모델은 4–16단계**, **라텐트 공간 모델(Stable Diffusion)은 1–4단계** 만으로도 원본 성능을 달성.  
3. **일반화:** w(가이드 강도) 값을 입력으로 받아 다양한 가이드 강도에 걸쳐 **품질–다양성 트레이드오프**를 하나의 모델로 지원.  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- **비용 문제:** classifier-free guidance는 각 단계마다 조건부 및 무조건부 U-Net을 둘 다 평가해야 하므로, 수백 단계의 반복이 비용적·시간적 병목.  
- **기존 증류 한계:** 기존의 progressive distillation 기법은 비지도(diffusion) 모델에만 적용되어 왔으며, guided diffusion 모델 및 라텐트 공간에는 적용되지 않음.

### 2.2 제안 방법  

#### Stage-1: 단일 모델 증류  
- 목표: $$\hat x^w_\theta(z_t) = (1+w)\,\hat x_{c,\theta}(z_t) - w\,\hat x_\theta(z_t)$$ 출력을  

$$\hat x_{\eta_1}(z_t, w)$$ 학생 모델이 직접 예측하도록 학습  

- 손실:  

$$
    \mathbb{E}\_{t\sim U[1],\,w\sim U[w_{\min},w_{\max}],\,x\sim p_{\rm data}}
      \bigl\|\hat x_{\eta_1}(z_t,w) - \hat x^w_\theta(z_t)\bigr\|^2
  $$
  
  여기서 $$z_t\sim q(z_t|x)$$, 모델 입력에 $$w$$ 포리에 임베딩 적용.  

#### Stage-2: 프로그레시브 증류  
- $$N$$단계 DDIM 샘플링을 $$N/2$$단계로 줄이는 증류를 반복  
- 각 반복에서 “2단계 DDIM”→“1단계 모델” 타깃을 만들고, 학생 모델이 이를 예측하도록 학습  
- 단계마다 학생→교사로 교체, 초기 파라미터는 이전 교사 모델 복사

### 2.3 모델 구조  
- **U-Net 백본:** Stable Diffusion 및 픽셀 공간 DDPM과 동일 아키텍처  
- **w-conditioning:** 가이드 강도 $$w$$를 푸리에 임베딩 후 타임스텝 임베딩과 결합  
- **예측 대상:** $$v$$ 예측(parameterization)  

### 2.4 성능 향상  
- **픽셀 공간 (ImageNet 64×64, CIFAR-10):**  
  - 원본 1024×2단계 → **4–16단계**로 **최대 256× 속도 향상**, FID/IS 동등  
- **라텐트 공간 (ImageNet 256×256, LAION 512×512):**  
  - 원본 1000+단계 → **1–4단계**로 **≥10× 속도 향상**, FID/CLIP 동등 이상  
- **편집·인페인팅·스타일 변환:** 2–4단계에서도 실용적 고품질 결과

### 2.5 한계  
- **극저단계(1–2 단계) 품질:** 일부 조건(높은 $$w$$)에서 여전히 화질 저하  
- **훈련 비용:** Stage-2에서 각 단계별 긴 재훈련 필요  
- **이론적 이해 부족:** 왜 프로그레시브 증류가 guided 모델에 잘 작동하는지 이론적 뒷받침 미흡  

## 3. 모델의 일반화 성능 향상 가능성  
- **w-conditioned 모델:** 단일 모델이 다양한 guidance 세팅에 적응 가능하므로, 응용 분야별 튜닝 없이도 **일관된 성능** 보장  
- **라텐트 및 픽셀 모두에 적용:** 프레임워크 독립적으로 증류 가능 → 다른 도메인(의료 영상, 3D 등)으로 확장 가능  
- **결정론적·확률적 샘플러 지원:** 폭넓은 샘플링 전략에도 견고  
- **향후 연구 방향:**  
  - 작은 표본(step 수 1–2)에서 재현성·안정성 제고  
  - 효율적 재증류 알고리즘 개발로 훈련 비용 절감  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **실시간 생성 애플리케이션:** 대화형 UI, 웹 서비스, 모바일 디바이스에서 diffusion 모델 실용화  
- **다중 모달 증류:** 텍스트·음성·3D 등 멀티모달 diffusion에도 증류 기법 확장  
- **절충 최적화:** 품질·속도·모델 크기 간 균형을 위한 **공동 최적화(co-optimization)** 연구  
- **이론적 연구:** 증류가 guided diffusion의 내재적 구조를 어떻게 보존하는지 수학적 분석  
- **자동화된 증류 파이프라인:** 증류 단계·학습률·샘플링 스케줄 자동 튜닝으로 개발 편의성 향상  

이 논문은 **효율적 diffusion 모델 추론**을 위한 새로운 패러다임을 제시하며, 실시간 생성과 자원 제약 환경에서의 대규모 적용을 가능케 하는 기반을 마련했다. 앞으로 **낮은 단계 수** regime에서의 안정성 확보와 **자동화된 증류 워크플로우** 구축이 중요한 연구 과제가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9b088a27-8082-4619-b888-f7ac8a6da97c/2210.03142v3.pdf
