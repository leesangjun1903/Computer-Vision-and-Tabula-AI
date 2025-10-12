# ADP : Adversarial Purification with Score-based Generative Models | 2021 · 234회 인용, Image classification

**핵심 주장 및 주요 기여**  
딥러닝 기반 이미지 분류기는 미세한 적대적 교란에 취약하며, 기존의 MCMC 기반 에너지 기반 모델(EBM) 정화 방법은 수천 단계의 샘플링이 필요해 실용성이 낮습니다. 본 논문은 Denoising Score-Matching(DSM)으로 학습된 EBM을 사용해 적대적 이미지를 수십 단계 만에 빠르게 정화하고, 무작위 노이즈 주입과 적응형 스텝 크기를 결합해 정화 성능과 강건성을 크게 향상시킨 **Adaptive Denoising Purification(ADP)**을 제안합니다.[1]

***

## 해결하고자 하는 문제  
적대적 공격에 취약한 이미지 분류기에 대해,
- 고가의 재학습 없이 독립된 정화(preprocessing) 모델로 방어
- MCMC 기반 EBMs의 긴 샘플링 체인의 비효율성  
을 극복하고자 합니다.[1]

***

## 제안 방법  
1. **DSM 기반 EBM 학습**  
   - 전통적 최대우도(MLE)+MCMC 대신, 다중 노이즈 수준 $$ \{\sigma_j\}_{j=1}^L $$로 손상된 샘플을 원본으로 복원하는 스코어 네트워크 $$ s_\theta(x,\sigma) $$를 학습  
   - Objective:  

$$
       \min_\theta \sum_{j=1}^L \mathbb{E}_{p_{\mathrm{data}}(x),\,q_{\sigma_j}(\tilde x|x)} \Bigl\|s_\theta(\tilde x,\sigma_j) + \frac{\tilde x - x}{\sigma_j^2}\Bigr\|^2
     $$  
     
  ($$q_{\sigma}(\tilde x|x)=\mathcal{N}(x,\sigma^2I)$$) [1].

2. **Deterministic 정화 업데이트**  
   - Langevin dynamics 대신, 학습된 스코어로 직접 이미지 갱신:  

$$
       x_{t+1} = x_t + \alpha_t\,s_\theta(x_t,\sigma_t)
     $$  
   
   - 소수 단계(10–100)만으로도 빠른 수렴 확인.[1]

3. **무작위 노이즈 주입**  
   - 입력 $$x$$에 Gaussian noise $$\mathcal{N}(0,\sigma^2I)$$를 사전 주입하여 적대적 교란을 가릴 뿐 아니라, DSM 학습 분포와 유사한 영역으로 이동  
   - 여러 번 정화 수행 후 앙상블로 최종 예측  
   - 이는 확률적 평탄화(randomized smoothing) 관점에서 인증된 강건성을 부여.[1]

4. **적응형 스텝 크기**  
   - 매 업데이트 시점에 스코어의 크기 변화를 기반으로  
     $$\alpha_t = 1 / (\|s_\theta(x_t)\|^2 + \epsilon)$$  
   - 추가 밸리데이션 없이도 안정적 수렴 유도.[1]

***

## 모델 구조  
- **스코어 네트워크**: Noise-Conditional Score Network(NCSN) 아키텍처(RefineNet 기반), 파라미터 수 약 30M  
- **분류기**: WideResNet-28-10, 파라미터 수 약 36M  
- 정화 모델과 분류기는 독립 학습되며, 정화 후 분류기 입력만 변경.[1]

***

## 성능 향상  
- CIFAR-10 $$\ell_2\le0.255$$ 위협 모델에서  
  - **표준 정확도**: 86.1% 유지  
  - **강건 정확도**: 80.2–85.5% 획득 (BPDAEOT 등 강력 공격 방어)  
  - MCMC 기반 EBM 대비 수십 배 빠른 정화 단계로 실용성 확보  
- 인증 강건성(certified robustness) 측정에서도 기존 Randomized Smoothing보다 우수한 반경 확보.[1]

***

## 한계  
- 정화 단계의 **무작위성**으로 표준 정확도 일부 희생  
- Common corruptions(CIFAR-10-C) 평가에서는 노이즈 주입이 오히려 성능 저하  
- 대규모 이미지나 고해상도에는 추가 검증 필요  
- 스코어 네트워크 학습 비용 및 메모리 요구량이 큼.[1]

***

## 모델 일반화 성능 관점  
무작위 노이즈 주입은 **적대적 교란뿐 아니라** 자연적 변형(common corruption)에 대한 개선 가능성을 시사하나, 분포 차이로 인해 성능이 일관되지 않았습니다. 향후 기존 DSM 학습에 **다양한 변형 분포(예: AugMix, DCT 변환)**를 통합하여 정화 모델의 일반화 범위를 넓히는 연구가 필요합니다.[1]

***

## 향후 연구 영향 및 고려 사항  
- **통합 방어 프레임워크**: 정화 모델을 분류기 학습에 결합한 공동 최적화 방식으로 전환 가능  
- **효율적 훈련**: 스케일 문제 해결을 위한 경량 스코어 네트워크 및 증류(distillation) 연구  
- **분포 적응적 노이즈 주입**: 공격 유형별 최적 노이즈 분포 학습  
- **다중 모달 방어**: 텍스트·오디오 등 다른 도메인에 확장 검토  

이와 같은 방향성은 적대적 정화 연구의 실용성과 확장성을 크게 제고할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/46cf00ec-1d90-4634-9dde-d418bd5bb385/2106.06041v1.pdf)
