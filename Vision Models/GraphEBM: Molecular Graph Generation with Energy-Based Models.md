# GraphEBM: Molecular Graph Generation with Energy-Based Models | 2021 · 124회 인용, Molecular generation

# 핵심 주장 및 주요 기여 요약

**GraphEBM: Molecular Graph Generation with Energy-Based Models**는 에너지 기반 모델(EBM)을 분자 그래프 생성(molecular graph generation)에 적용한 최초의 연구로, 다음 세 가지 핵심 기여를 제시한다.

1. **EBM 프레임워크 도입**: 분자 그래프 확률 분포를 직접 모델링하기 위해 신경망 기반 스코어 함수 $$s_\theta(G)$$를 학습하고, Langevin Dynamics를 활용해 새로운 분자 그래프를 샘플링한다.  
2. **그래프 구조 특화 아키텍처**: 메시지 패싱 신경망(Message Passing Neural Network; MPNN)을 이용해 에너지 함수 $$E_\theta(G)= -s_\theta(G)$$를 정의함으로써 그래프 구조를 효과적으로 반영한다.  
3. **일관된 화학적 타당성 보장**: 샘플링 시 valency 제약을 손쉽게 포함하여 생성된 분자가 화학적으로 유효하도록 한다.

# 상세 설명

## 해결하고자 하는 문제  
기존 분자 생성 모델은 VAE, GAN, autoregressive 모델 등으로, 주로 이산 구조 공간의 복잡한 분포를 근사하기 어렵다. GraphEBM은 이러한 구조적 복잡성을 직접 다룰 수 있는 **에너지 기반 모델**을 도입함으로써 분자 그래프의 진정한 분포를 학습하고자 한다.

## 제안하는 방법  
### 에너지 함수와 스코어 함수  
- 그래프 $$G$$에 대해 에너지 함수:  

$$
E_\theta(G) = -s_\theta(G)
$$  

- 스코어 함수(Gradient of log-density):  

$$
\nabla_G \log p_\theta(G) \approx \nabla_G s_\theta(G)
$$  

- 학습 목표: 노이즈 임베딩 $$\tilde G = G + \sigma \epsilon$$에 대해  

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tilde G, G}\left[\|\nabla_{\tilde G} \log p_\theta(\tilde G) - \nabla_{\tilde G} \log p(\tilde G|G)\|_2^2\right]
$$  

이를 통해 스코어 매칭(score matching) 방식으로 파라미터 $$\theta$$를 최적화한다.

### 모델 구조  
- **MPNN 기반 인코더**: 각 노드/엣지 특징과 인접 구조를 입력받아 스코어 함수를 산출  
- **Langevin Dynamics 샘플링**: 초기 잡음 그래프에서 반복적으로  

$$
G_{t+1} = G_t + \frac{\alpha}{2} \nabla_{G_t} s_\theta(G_t) + \sqrt{\alpha}\,\epsilon_t
$$  

과정을 수행하여 유효 분자 구조로 수렴  

## 성능 향상  
- **유효성(validity)**: 95% 이상의 생성 분자가 화학적 valency 제약을 만족  
- **다양성(diversity)**: 기존 GraphVAE 대비 1.3배 높은 구조적 다양성  
- **Novelty**: 훈련 세트에 없는 신규 분자 비율 70% 달성  

## 한계  
- **샘플링 속도**: Langevin Dynamics 반복 횟수(수백 단계)로 인해 느린 생성 속도  
- **확장성**: 대형 분자(노드 수 >50) 샘플링 성능 저하  
- **하이퍼파라미터 민감도**: 스텝 사이즈 $$\alpha$$, 노이즈 레벨 $$\sigma$$ 조정이 까다로움  

## 모델 일반화 성능 향상 관점  
GraphEBM은 노이즈 기반 score matching을 통해 **다양한 분포 영역**을 학습함으로써 오버피팅을 줄이고, 적은 데이터로도 **견고한 분포 근사**를 가능하게 한다. 특히 MPNN 구조가 지역적 패턴과 전역 구조를 동시에 포착해, 새로운 화학 구조에도 **일반화 가능한 스코어 함수**를 학습한다.

# 향후 연구 영향 및 고려 사항

이 연구는 **EBM을 구조화된 데이터 생성**에 성공적으로 도입했다는 점에서, 향후 다양한 그래프 생성 분야(소셜 네트워크 합성, 단백질 구조 예측 등)로 확장될 수 있다.  
- **가속화**: Stochastic sampling 대체 기법(ODE 샘플러 등) 연구  
- **스케일 업**: 대형 분자 및 복합 그래프에 대한 효율적 학습  
- **강건성**: 하이퍼파라미터 자동 조정과 메타러닝 결합으로 샘플링 안정화  

이상의 방향을 통해 GraphEBM의 **일반화 성능**과 **실제 응용성**을 더욱 강화할 수 있을 것이다.
