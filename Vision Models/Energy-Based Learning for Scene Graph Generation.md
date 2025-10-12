# Energy-Based Learning for Scene Graph Generation | 2021 · 213회 인용, Scene Graph Generation

# 핵심 요약

**Energy-Based Learning for Scene Graph Generation** 논문은 전통적인 교차 엔트로피 기반 학습이 장면 그래프의 구조적 종속성을 무시한다는 문제를 지적하고, 장면 그래프의 구조를 출력 공간에 직접 도입하는 **에너지 기반 학습 프레임워크**를 제안한다. 이 프레임워크는 구조적 제약을 **유도 바이어스(inductive bias)**로 활용하여 적은 라벨에서도 효율적으로 학습할 수 있도록 하며, 기존 최첨단 모델에 적용 시 Visual Genome에서 최대 **21%**, GQA에서 **27%**의 성능 향상을 달성하였다. 또한 데이터가 부족한 **제로샷(zero-shot)** 및 **몇 샷(few-shot)** 상황에서 우수한 성능을 보였다.[1]

# 문제 정의

장면 그래프 생성(Scene Graph Generation)은 이미지 내 객체와 객체 간 관계를 그래프 형태로 표현하는 작업이다.  
기존 방법들은 객체(O)와 관계(R)를 독립적으로 취급하여 교차 엔트로피 손실을 최소화하며 학습하므로,  

$$
\log p(\mathrm{SG}|I) \;=\;\sum_{i\in O}\log p(o_i|I) + \sum_{j\in R}\log p(r_j|I)
$$  

와 같이 요소별 합산만 고려한다.[1]
이로 인해  
- **구조적 일관성 무시**: 예컨대 <사람, riding, 파도>는 다른 관계들과 상충하지만 동일하게 취급  
- **빈도 편향**: 훈련 데이터에 많은 일반적 관계(on 등)을 과도하게 예측  

# 제안 방법

## 에너지 기반 학습 손실

장면 그래프 구성 $$G_{SG}$$와 이미지 그래프 $$G_I$$의 **조인트 에너지** $$E_\theta(G_I, G_{SG})$$를 정의하고,  

$$
L_e = E_\theta(G^+_I, G^+_{SG}) - \min_{G_{SG}} E_\theta(G_I, G_{SG})
$$  

으로 학습한다.[1]
후자는 Stochastic Gradient Langevin Dynamics를 통해  

$$
O_{\tau+1} = O_\tau - \tfrac{\lambda}{2}\nabla_O E_\theta + \epsilon_\tau,\quad
R_{\tau+1} = R_\tau - \tfrac{\lambda}{2}\nabla_R E_\theta + \epsilon_\tau
$$  

로 최적화하며, 추가로 에너지 값 $$L_2$$ 규제 및 기존 과제 손실 $$L_t$$를  

$$
L_{\text{total}}=\lambda_e L_e + \lambda_r L_r + \lambda_t L_t
$$  

로 결합하여 안정적으로 학습한다.[1]

## 모델 구조

1. **이미지 그래프 $$G_I$$**: Faster R-CNN으로 추출한 RoIAlign 특징을 노드 상태로 초기화  
2. **장면 그래프 $$G_{SG}$$**: 기존 모델(예: VCTree 등)의 예측 또는 정답 라벨로 초기화  
3. **에너지 모델**  
   - 이미지 그래프: GNN  
   - 장면 그래프: **Edged Graph Neural Network (EGNN)**  
     
- 노드간 메시지:  

$$
       m_i = \alpha W_{nn}\sum_{j\in N_i}n_j + (1-\alpha)W_{en}\sum_{j\in N_i}e_{j\to i}
       $$  
     
- 엣지 메시지:  

$$
       d_{i\to j} = W_{ee}[n_i\|\;n_j]
       $$  
   
- 각 그래프를 gated pooling하여 벡터화  
   
- MLP로 결합 후 스칼라 에너지 출력[1]

# 성능 개선 및 한계

- **Visual Genome**: VCTree 기준 mR@20이 교차 엔트로피 13.07→14.20 (+21%) 개선  
- **GQA**: Transformer 기반 mR@20이 1.17→1.28 (+27%) 향상  
- **제로샷 리콜(zsR@K)**, **few-shot 리콜**에서도 일관된 성능 우위 확인.[1]

한계로는  
- SGLD 기반 최적화에 따른 **연산 비용 증가**  
- 에너지 값 발산 방지를 위한 정규화·클리핑 필요  
- 대규모 객체·관계 클래스 시 메모리 부담  

# 일반화 성능 향상

에너지 기반 프레임워크는 **출력 공간의 전역 구조**를 모델링함으로써 드문 관계에 대한 학습을 촉진한다.  
제로샷 실험에서 미관측(subject-predicate-object) 삼중항을 성공적으로 예측했으며, few-shot 상황에서도 교차 엔트로피 대비 **데이터 효율**이 크게 향상되었다.[1]

# 향후 연구 영향 및 고려 사항

이 접근법은 구조적 종속성 학습을 필요로 하는 다른 시각·비시각 과제(예: 그래프 임베딩, 텍스트 구조 예측)에도 확장 가능하다.  
향후 연구 시 고려할 점은  
- **계산 효율화**: SGLD 대체 또는 근사 기법 개발  
- **스케일업**: 클래스 수가 많고 그래프가 복잡한 도메인으로의 적용  
- **하이퍼파라미터 안정화**: α, λ 계수 및 클리핑 범위 자동 튜닝  

이를 통해 더욱 견고하고 데이터 효율적인 구조 인지 학습 모델 설계가 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cabe4c92-2a75-46ff-95d0-8216fb5da156/2103.02221v1.pdf)
