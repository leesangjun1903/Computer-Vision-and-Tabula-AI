# Classifier-Free Diffusion Guidance | Data diversity, Optimization

**주요 주장**  
Classifier-Free Diffusion Guidance는 기존의 확산 모델(Classifier Guidance)이 이미지 분류기(classifier)의 기울기를 활용해 샘플 품질과 다양성 간의 절충(trade-off)을 조절하는 것과 달리, 별도의 분류기 없이 순수 생성 모델의 조건부 및 무조건부(score) 추정치를 선형 결합함으로써 동일한 품질·다양성 절충 효과를 달성할 수 있음을 보여준다[1].

**주요 기여**  
1. **간단한 통합 학습**: 조건부(diffusion with label)와 무조건부(diffusion without label) 모델을 단일 네트워크에 통합해, 훈련 시 임의로 레이블을 드롭아웃(puncond 확률)함으로써 두 가지 역할을 동시에 학습[1].  
2. **샘플링 공식 제안**: 조건부 및 무조건부 score 추정치의 가중 결합

$$
\tilde\epsilon_\theta(z_\lambda, c) = (1+w)\,\epsilon_\theta(z_\lambda, c) \;-\; w\,\epsilon_\theta(z_\lambda)
$$

를 통해 guidance 강도 $$w$$를 조절, **별도 분류기 없이** FID·IS 절충 곡선을 실현[1].  
3. **경량화 및 단순성**: 분류기 훈련·유지 비용 제거.  
4. **경쟁력 있는 성능**: ImageNet 64×64·128×128에서 classifier guidance 대비 동등 이상 또는 더 나은 FID·IS 성능 달성[1].  

# 상세 설명

## 1. 해결하고자 하는 문제  
- **기존 분류기 의존성**: Classifier Guidance(Dhariwal & Nichol, 2021)는 샘플 품질 향상을 위해 노이즈에 오염된 데이터에 대해 별도 분류기를 훈련해야 하며, 이는 학습 파이프라인 복잡도 증가 및 사전 학습된 분류기 활용 불가 문제 야기[1].  
- **샘플 품질–다양성 절충**: 낮은 온도 샘플링(truncation)이나 분류기 그래디언트 강도 조절로 FID(Frechet Inception Distance)와 IS(Inception Score) 간의 절충 곡선을 생성할 수 있지만, 기존 확산 모델에는 직접 적용이 어렵다[1].

## 2. 제안하는 방법 및 수식  
- **공동 학습**  
  1. 원본 데이터 $$(x,c)$$ 중 확률 $$p_{\text{uncond}}$$로 레이블 $$c$$를 무시(∅ 토큰 입력).  
  2. 노이즈 레벨 $$\lambda$$를 샘플링하고 $$\epsilon$$-예측 네트워크 $$\epsilon_\theta(z_\lambda, c)$$를 최적화.  
- **Classifier-Free Guidance Sampling**  
  - 각 샘플링 단계 $$t$$에서 다음 결합 score 사용[1]:

$$
    \tilde\epsilon_t = (1+w)\,\epsilon_\theta(z_t, c) \;-\; w\,\epsilon_\theta(z_t)\,
$$
    
여기서 $$w$$는 guidance 강도를 조절하는 하이퍼파라미터.  
  - 이후 일반 ancestral sampler 또는 DDIM으로 $$z_{t+1}$$ 계산.  

## 3. 모델 구조  
- **단일 네트워크**: 입력으로 노이즈 $$z_\lambda$$, 레이블 $$c$$ (또는 ∅) 및 로그-신호대잡음비(λ)를 받아 $$\epsilon_\theta$$를 출력.  
- **무조건부 vs 조건부**: 훈련 시 레이블 드롭아웃을 통해 동일 파라미터로 두 가지 역할 수행.

## 4. 성능 향상  
| 해상도 | 모델        | Guidance $$w$$ | FID ↓   | IS ↑     |
|--------|-------------|----------------|---------|----------|
| 64×64  | Classifier-Free (puncond=0.1) | 0.1            | 1.55[1] | 66.11[1] |
| 64×64  | ADM-G (기존) | –              | 2.07[1] | –        |
| 128×128| Classifier-Free (T=256)       | 0.3            | 2.43[1] | 158.47[1]|
| 128×128| ADM-G (기존) | –              | 2.97[1] | –        |

- **절충 곡선**: $$w$$ 증가 시 FID 감소(품질↑), IS 증가(다양성↓) 추세[1].  
- **샘플링 단계 수**: T 증가 시 전반적 성능 향상. 단, 계산량도 증가[1].

## 5. 한계  
- **추론 속도 저하**: 무조건부·조건부 각각 네트워크 두 번 평가 필요.  
- **다양성 저하**: 높은 $$w$$에서 샘플 다양성 감소.  
- **무조건부 score 부정확**: 과도한 레이블 드롭아웃 시 무조건부 모델 능력 저하 가능.

## 6. 일반화 성능 향상 가능성  
- **Implicit Classifier**: 조건부·무조건부 score 차이로 암묵적 분류기 효과, 분류기 기반 메트릭에 과적합(adversarial) 하지 않으면서 일반화된 특징 학습 유도 가능[1].  
- **레이블 드롭아웃**: 다양한 노이즈 스케일 및 레이블 결손 상황에 견고한 특징 추출을 학습함으로써 적은 레이블 환경에서도 강건성 향상 기대.  
- **응용 확장성**: 텍스트-이미지, 음성 등 다양한 조건부 생성 모델에 동일한 framework 적용 가능.

# 향후 연구 영향 및 고려 사항

**연구 영향**  
- 분류기 필요 없는 단순·효율적 guidance 기법 제시로 확산 모델 연구 가속화.  
- 생성 모델의 **적응적 다양성 제어** 가능성을 열어, 사용자 제어 인터페이스(UI) 및 편집 응용 개발에 기여.

**고려 점**  
1. **추론 최적화**: 조건부 입력 지연(injection) 및 파라미터 공유를 통해 속도 개선 연구 필요.  
2. **다양성 유지**: 고품질과 고다양성 동시 확보를 위한 새로운 loss 또는 정규화 기법 탐색.  
3. **다중 모달 확장**: 텍스트·오디오·3D 등 다양한 조건부 데이터에 대한 일반화 성능 검증.  

**참고문헌**  
[1] Ho, J. & Salimans, T. (2021). Classifier-Free Diffusion Guidance. arXiv:2207.12598v1.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2f108968-20fe-4b44-9c4c-deebe0bfef6c/2207.12598v1.pdf
