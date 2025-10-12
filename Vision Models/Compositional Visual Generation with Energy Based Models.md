# Compositional Visual Generation with Energy Based Models | 2020 · 197회 인용, Image generation

## 핵심 주장 및 주요 기여  
이 논문은 **에너지 기반 모델**(Energy-Based Models, EBMs)을 활용하여 서로 독립적으로 학습된 개념 분포를 논리 연산(AND, OR, NOT)으로 조합함으로써 복합 개념을 생성하고 추론할 수 있음을 보인다.  
- 독립 학습된 EBM 에너지 함수들의 **합**, **로그-섬-지수(neg-min)**, **차** 연산을 통해 각각 논리적 **합성(conjunction)**, **합집합(disjunction)**, **부정(negation)**을 구현한다.[1]
- 이러한 조합을 통해 자연 이미지(얼굴, 3D 장면 등)에서 복합 속성(예: ‘미소짓는 남성 얼굴’)을 생성하고, 새로운 개념을 지속적으로 학습·통합하며(continual learning), 미관측 복합 조합에 대한 **일반화 능력**을 시연한다.[1]

***

## 1. 해결하고자 하는 문제  
기존 생성 모델은  
1) 사전에 고정된 잠재 요인 공간(vector of factors) 또는  
2) 픽셀 기반 객체 슬롯(segmentation mask)  
방식을 통해 조합성을 지원하지만,  
- 새로운 요인을 동적으로 추가하거나  
- 독립 학습된 분포의 재조합  
이 어려웠다.  

이 논문은 **독립 학습된 확률 분포**로 개념을 표현하고, 이를 논리 연산으로 결합하여  
- 무제한적이고 재귀적(composable)으로 개념을 생성  
- 사전 훈련 없이 새로운 개념을 추가·조합  
- 학습하지 않은 복합 조합에서의 **일반화 및 추론**  
문제를 해결한다.[1]

***

## 2. 제안 방법  
### 2.1 EBM 기본 식  
EBM은 에너지 함수 $$E_\theta(x)$$로 비정규화 확률 분포를 정의한다:  

$$
p(x) = \frac{e^{-E_\theta(x)}}{Z}, 
$$

최대우도학습은 대비발산(Contrastive Divergence)으로 수행된다.[1]
샘플링은 Langevin Dynamics:

$$
x_{k} = x_{k-1} - \frac{\eta}{2}\nabla_x E_\theta(x_{k-1}) + \sqrt{\eta}\,\omega_k,\quad \omega_k\sim\mathcal{N}(0,I).
$$

### 2.2 논리 연산자 정의  
독립 학습된 에너지 함수 $$E_i(x)$$ 들을 다음과 같이 결합한다:[1]
1) **합성(Conjunction)** (논리 AND):  

$$
   E_{\wedge}(x) = \sum_i E_i(x),\quad
   x \leftarrow \text{Langevin}(E_{\wedge})
   $$

2) **합집합(Disjunction)** (논리 OR):  

$$
   E_{\vee}(x) = {-}\log\sum_i\exp\bigl(-E_i(x)\bigr)
   $$

3) **부정(Negation)** (논리 NOT):  

$$
   E_{\neg A,B}(x) = E_B(x) - E_A(x)
   $$

이들을 재귀적으로 중첩하여 복잡한 논리식을 구성할 수 있다.[1]

***

## 3. 모델 구조  
- **얼굴 데이터(CelebA, 128×128)**: ResNet 기반 EBM[1]
- **3D 장면(MuJoCo, 64×64)**: ImageNet 아키텍처 변형 EBM[1]
- MCMC 샘플: CelebA 60단계, MuJoCo 80단계 Langevin Dynamics  
- 학습: Adam(learning rate=3e-4), spectral normalization, Swish 활성화  

***

## 4. 성능 향상 및 한계  
### 4.1 성능 향상  
- **정량 평가** (MuJoCo 씬): 속성 정확도(position/color)  
  - 단일 EBM: 99% 이상의 정확도  
  - 합성(AND) 생성: position 0.872, color 0.801 이상[1]
- **일반화(Generalization)**  
  - Continual learning: 순차 학습 후 위치 정확도 유지(E) vs. GAN은 급감[1]
  - 교차 조합(extrapolation): 적은 관측에도 EBM이 baseline보다 낮은 오차로 추론(사이즈·위치)[1]
- **추론(Inference)**: 여러 관측에서 MAP 추정으로 위치·속성 회귀 시 MSE 감소[1]

### 4.2 한계  
- **MCMC 샘플링 비용**: Langevin Dynamics 단계 수 증가에 따른 연산 부하  
- **Partition function 근사**: Disjunction 성능은 파티션 함수 균등 가정에 의존(실험적으로 유사함이 관찰되나 이론 보장은 미흡)[1]
- **고해상도 확장**: 128×128 이상 해상도 및 복잡 장면에의 확장성 미확인  

***

## 5. 일반화 성능 향상 관점 강조  
EBM의 요인 분리(factor decomposition) 특성 덕분에  
- **독립 학습**된 각 개념 EBM은 다른 개념과 **결합 시 과적합 없이** 재사용됨  
- **새로운 개념** 추가 시 기존 모델 은 고정(freeze)하고 조합만으로도 **일반화** 가능[1]
- 교차 조합 실험에서, 제한적 조합 관측(1–10%)으로도 unseen 조합에 대한 **추론 오차**가 baseline 대비 유의미하게 낮음[1]

***

## 6. 향후 연구 영향 및 고려 사항  
- **모듈러 생성 모델** 연구: 독립 학습 분포의 논리 조합 프레임워크  
- **지속 학습(continual learning)**: 동적 개념 확장 시 catastrophic forgetting 완화 가능성  
- **효율적 샘플링**: MCMC 비용 절감 기법 연구(learned sampler, parallel sampling) 필요  
- **해상도 및 복잡도**: 고해상도·다중 객체·자연 장면 확대로 일반화 검증  
- **파티션 함수 추정**: 이론적 안정성 확보 및 disjunction 연산 개선 과제  

이상으로 Compositional Visual Generation with Energy Based Models의 핵심 기여, 방법론, 성능, 일반화 능력 및 향후 연구 방향을 요약·분석하였다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6e3954db-1d90-4c50-982f-86323354b533/2004.06030v3.pdf)
