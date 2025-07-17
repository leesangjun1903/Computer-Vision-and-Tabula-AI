# Implicit Generation and Modeling with Energy-Based Models | Image generation, Image reconstruction

**핵심 주장 및 주요 기여**  
이 논문은 에너지 기반 모델(Energy-Based Models, EBMs)에 잠재적 세대(implicit generation) 기법을 도입하여 고차원 데이터(이미지, 로봇 궤적 등)에서 우수한 생성 품질과 다양성을 달성할 수 있음을 보인다[1]. 주요 기여는 다음과 같다.

1. **확장 가능한 EBM 학습 알고리즘 제안**  
   - Langevin Dynamics를 이용한 효율적 MCMC 샘플링  
   - 샘플 재생 버퍼(replay buffer)를 활용한 믹싱 시간 단축  
   - 스펙트럴 노멀라이제이션 및 L2 정규화로 학습 안정성 보장  

2. **암시적 세대의 고유 특성 활용**  
   - **합성성(Compositionality)**: 여러 EBMs의 에너지를 합산하여 복합 제약 만족  
   - **잡음 제거 및 인페인팅(inpainting)**: 학습된 분포 모드로부터 직접 복원  
   - **장기 예측 및 연속 학습**: 로봇 궤적 예측, 온라인 클래스 학습 등 다방면 응용  

3. **다양한 응용 영역에서 성능 검증**  
   - 고해상도 이미지 생성(CIFAR-10, ImageNet32×32/128×128)에서 모드 붕괴 없이 GAN에 근접한 품질  
   - 이상치 탐지(OOD detection), 적대적 공격 내성(robustness)에서 확률론적 모델 대비 우수  
   - 로봇 핸드 궤적 예측 및 지속 학습(continual learning)에서 최첨단 성능 달성  

## 1. 해결하고자 하는 문제  
- 전통적 EBM은 MCMC 샘플링의 긴 믹싱 시간과 파티션 함수(partition function) 추정의 어려움으로 고차원 데이터에 적용하기 힘들었다.  
- VAEs와 GANs은 별도 네트워크 설계·학습 균형 문제, 모드 붕괴(mode collapse), 잠재 공간 제약에 따른 표현력 한계를 가짐[1].

## 2. 제안 방법  

### 2.1 에너지 기반 모델과 샘플링  
- **에너지 함수** $$E_\theta(x)$$: 신경망으로 파라미터화하여  

$$
    p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)},\quad Z(\theta)=\int e^{-E_\theta(x)}dx
$$
- **Langevin Dynamics** (Eq.1)를 통해 암시적 샘플 $$x\sim p_\theta$$을 획득:  

$$
    \tilde x_{k} = \tilde x_{k-1} - \tfrac{\lambda}{2}\nabla_x E_\theta(\tilde x_{k-1}) + \omega_k,\quad \omega_k\sim\mathcal{N}(0,\lambda)
$$

### 2.2 최대우도 학습  
- ML 목표 함수 $$\mathcal{L}\_{\rm ML}=\mathbb{E}\_{x\sim p_{\rm data}}[-\log p_\theta(x)]$$  
- 근사 그레이디언트 (Eq.2):  

$$
    \nabla_\theta\mathcal{L}\_{\rm ML}\approx
    \mathbb{E}\_{x^+\sim p_{\rm data}}[\nabla_\theta E_\theta(x^+)]
    -\mathbb{E}\_{x^-\sim q_\theta}[\nabla_\theta E_\theta(x^-)]
$$

### 2.3 학습 안정화 기법  
- **샘플 재생 버퍼**: 과거 샘플을 95% 확률로 초기화, 나머지는 균등 잡음  
- **스펙트럴 노멀라이제이션**: 레이어 Lipschitz 제약  
- **L2 정규화**: 에너지 크기 안정화  

## 3. 모델 구조  
- ResNet 기반 CNN 아키텍처  
  - CIFAR-10: 5M 파라미터, 60단계 Langevin 샘플링  
  - ImageNet128×128: 10M 규모, 100단계 샘플링  
- 모든 컨볼루션 및 FC 레이어에 스펙트럴 노멀라이제이션 적용  
- 조건부 모델에는 클래스별 gain/bias 통합[1]

## 4. 성능 향상 및 한계  

| 적용 분야                    | 성능 지표                                   | 비교 모델 대비                   |
|------------------------------|---------------------------------------------|----------------------------------|
| 이미지 생성(CIFAR-10)    | FID 40.58 (EBM 단일)                        | PixelCNN 65.93, SNGAN 21.7       |
| 이미지 생성(ImageNet32)  | FID 14.31                                    | PixelIQN 22.99                   |
| 이상치 탐지 (OOD)            | AUROC 0.62                                   | PixelCNN++ 0.47, Glow 0.42       |
| 적대적 내성 (PGD, ε>13)      | 정확도 49.6%                                 | Madry et al. 43%                 |
| 로봇 궤적 예측                | Frechet Dist. ↓5.96 (EBM) vs 33.28 (FC)      | –                                |
| 온라인 연속 학습 (Split MNIST)| 정확도 64.99% (EBM) vs 40.04% (VAE)          | EWC 19.8%, LwF 24.2%             |

**한계**  
- 샘플링 비용이 GAN 대비 100배 이상 높음 (수백 초 단위)  
- 대규모 고해상도 모델로 확장 시 계산 부담 증가  
- HMC 등 대체 MCMC 기법 도입 필요성  

## 5. 일반화 성능 향상 가능성  
- **암시적 음성 샘플링**이 모델 경계 바깥(spurious modes)에 과도한 확률 부여를 억제, **이상치 분별** 능력 향상  
- 다중 에너지 합산을 통한 **제약 조합**(compositionality)으로 교차 조건 일반화(zero-shot) 성공  
- 연속 학습 시 **부정 샘플(local forgetting)** 만 영향, 기존 클래스 정보 보존  

## 6. 향후 연구 영향 및 고려 사항  
- **샘플링 효율화**: 적응형 HMC, 메타학습 기반 스텝 크기 조정  
- **스케일 업**: 고해상도 텍스트·비디오·음성 도메인 적용  
- **조합적 표현 학습**: 여러 잠재인자 결합을 통한 구조적 일반화  
- **활용도 확장**: 강화학습 정책 표현, 그래프·시계열 분야  

에너지 기반 잠재 생성은 **모델 단순성**과 **표현 유연성**의 균형을 이루며, 향후 다양한 분야의 **교차 조건 학습**과 **강인한 분포 학습** 연구에 기반이 될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b4f57730-369c-48cb-9f94-5f6aee7c644b/1903.08689v6.pdf
