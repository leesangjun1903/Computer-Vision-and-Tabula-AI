# DMD : One-step Diffusion with Distribution Matching Distillation | 2023 · 355회 인용, Image generation

**주요 주장 및 기여**  
“One-step Diffusion with Distribution Matching Distillation” (Yin et al., 2024)은 기존의 다단계(diffusion) 샘플링 과정을 **단일 단계(one-step)**로 대체하면서도 고품질 이미지를 생성하는 방법을 제시한다. 핵심 기여는 다음과 같다:

- **분포 매칭(dist. matching) 증류**: 원본 다단계 diffusion 모델과 학생 모델의 **출력 분포**를 **KL 발산** 수준에서 정렬하도록 학습함으로써, 복잡한 노이즈-이미지 매핑을 직접 학습하는 대신 분포 전체를 일치시킨다.[1]
- **추가 회귀(regression) 손실**: LPIPS 손실을 이용해 사전에 계산된 다단계 샘플링 결과와 학생 모델 출력을 정합시켜, 모드 붕괴(mode collapse)와 과도한 발산을 방지한다.[1]
- **스코어 함수 활용**: 원본 및 학생(“가짜”) 분포에 대해 각각 diffusion denoiser를 스코어 함수로 활용하여, 분포 매칭 손실의 기울기를 “실제(real) 스코어 – 가짜(fake) 스코어” 차이로 근사한다.[1]
- **실시간 생성 속도**: FP16 추론에서 20 FPS로 512×512 이미지를 생성하며, ImageNet 64×64 클래스조건부에서 FID 2.62, COCO-30k 제로샷에서 FID 11.49를 달성해 Stable Diffusion과 유사 품질을 유지하면서 30× 이상 가속화한다.[1]

***

## 1. 해결하는 문제
- **다단계 샘플링의 느린 속도**: diffusion 모델은 수십~수백 단계의 반복 평가가 필요해 실시간 응용에 부적합하다.
- **고차원 맵핑의 학습 난이도**: 학생 네트워크가 원본 diffusion의 노이즈→이미지 매핑을 직접 회귀하는 것은 매우 어려운 과제이다.

***

## 2. 제안 방법

### 2.1 모델 구조
- **학생 네트워크 G**: 원본 diffusion denoiser 구조에서 **시간(t) 조건을 제거**한 형태로, 초기 가중치는 원본 모델과 동일하게 설정된다.
- **스코어 네트워크 real, fake**:  
  - real: 고정된 원본 diffusion 모델  
  - fake: fine-tuning 가능한 복제 모델(학생 생성물 분포 추정)

### 2.2 분포 매칭 손실 (L<sub>KL</sub>)
- 목표: 학생 생성 분포 $$p_{\mathrm{fake}}$$가 원본 분포 $$p_{\mathrm{real}}$$와 KL 발산 수준에서 일치하도록 학습  

$$
    \mathrm{DKL}(p_{\mathrm{fake}}\|\,p_{\mathrm{real}})
    = \mathbb{E}_{z\sim\mathcal{N}}\bigl[s_{\mathrm{real}}(x) - s_{\mathrm{fake}}(x)\bigr]\,\frac{\partial G(z)}{\partial\theta}
  $$  
  
여기서 $$s(x)=\nabla_x\log p(x)$$는 스코어 함수이다.

- **확률밀도 퍼짐(Perturbation)**: 가우시안 노이즈 레벨 $$t$$를 도입하여 두 분포가 전 영역에서 겹치도록 조정하고, diffusion denoiser로 스코어를 근사한다.

- **가중치 설계**: 노이즈 레벨별 기울기 안정화를 위해  

$$
    w_t = \frac{2\,t}{C\,S\,\|\,\mathrm{denoiser}(x_t,t)-x_t\|_1}
  $$  
  
(Eq.8) 형태로 정규화한다.

### 2.3 회귀 손실 (L<sub>reg</sub>)
- **노이즈-이미지 쌍** $$(z,y)$$를 사전 계산하여,  

$$
    L_{\mathrm{reg}} = \mathbb{E}_{(z,y)}\bigl[\mathrm{LPIPS}(G(z),\,y)\bigr]
  $$  
  
로 학습해 모드 붕괴를 방지하고 대규모 구조 정합을 보장한다.[1]

### 2.4 최종 목적 함수
$$
  L = L_{\mathrm{KL}} + \lambda\,L_{\mathrm{reg}}\quad(\lambda=0.25\;\text{기본값})
$$

***

## 3. 성능 평가 및 한계

- **ImageNet 64×64 (클래스조건부)**: FID 2.62, 원본 multi-step 모델 대비 차이 ≈0.3, 512배 빠름.
- **MS COCO 제로샷**: FID 11.49, CLIP Score ≈0.32, 30× 가속.
- **CIFAR-10**: 클래스조건부 FID 2.66, unconditional FID 3.77[E].

### 주요 한계
- **품질 차이**: 100~1000 단계 샘플링 대비 미세한 품질 저하 남아 있음.
- **메모리 사용량**: 학생 스코어 모델과 생성기 동시 fine-tuning으로 학습 시 GPU 메모리 소모가 매우 큼.
- **회귀 데이터셋 규모**: 대규모 사전 계산 필요(수백만 쌍), 초기 자원 투자 요구.

***

## 4. 일반화 성능 향상 가능성
- **분포 매칭**은 전체 데이터 분포를 정렬하므로, 단일 모드에 과도하게 최적화되는 현상을 완화하고 다양한 샘플을 생성할 수 있어 **새로운 도메인**에도 적용 가능성이 높다.
- **LPIPS 회귀**는 시각적 인지 거리를 반영해 생성 품질을 안정화시키므로, **다양한 해상도·조건**(예: 텍스트, 클래스)에서도 일관된 성능을 기대할 수 있다.
- **스코어 기반 업데이트**는 GAN과 달리 사전 학습된 diffusion 모델의 강건성을 활용하므로, **노이즈 분포 변화**에도 견고하다.
- 단, fine-tuning 시 Overfitting 방지를 위한 더 강력한 정규화 또는 소수 파라미터 업데이트 기법(e.g., LoRA) 도입이 필요하다.

***

## 5. 향후 연구 및 고려 사항

- **저메모리 증류**: LoRA, 지능형 체크포인팅으로 메모리 발자국 축소.  
- **회귀 손실 대체 실험**: L<sub>2</sub>·SSIM 등 대체 적합성 연구(E: 2.78 vs 2.66 FID).  
- **다양한 조건부 확장**: 텍스트-비디오, 3D 생성 모델 증류로 응용 영역 확장.  
- **샘플 가중치 최적화**: 노이즈 스케줄 및 가이드 스케일 변화에 따른 손실 가중치 자동 조정 연구.  
- **적응적 회귀 샘플링**: 학습 중 동적 샘플링 전략 도입으로 자원 효율성 개선.

***

**결론**  
Distribution Matching Distillation은 **분포 수준의 학습**과 **회귀 정합**을 결합해, **한 단계** 만으로도 원본 diffusion 모델의 고품질 이미지를 실시간 속도로 생성할 수 있게 한다. 메모리 최적화와 손실 구성 개선을 통해 다양한 비전·멀티모달 응용에 대한 **확장 가능성**이 높다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/874f8f30-0a9a-48cb-8ff4-eb36190a31fc/2311.18828v4.pdf)
