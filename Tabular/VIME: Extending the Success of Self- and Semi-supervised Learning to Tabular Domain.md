# VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain

**핵심 주장 및 주요 기여**  
VIME(Value Imputation and Mask Estimation)는 이미지나 언어 도메인에 의존하던 기존 Self-/Semi-Supervised 학습 기법을 일반적인 탭형(tabular) 데이터에 적용 가능하도록 확장한 프레임워크로,  
-  **마스크 벡터 추정(mask estimation)** 과 **값 재구성(reconstruction)** 이라는 두 가지 전이(pretext) 과제를 통해 표현학습을 수행하고,  
-  학습된 인코더를 활용한 **데이터 증강(augmentation)** 및 **일관성(Consistency) 규제**를 도입하여 레이블이 부족한 상황에서도 최첨단 성능을 달성한다.[1]

***

## 1. 해결하고자 하는 문제  
- 탭형 데이터는 이미지의 공간적 관계나 언어의 의미적 관계가 명시적이지 않아, 회전·퍼즐·언어 마스킹 등 기존 전이과제가 적용 불가능하다.  
- 변수 간 상관관계 구조가 데이터마다 다르고 알려지지 않아, 일반적인 Self-/Semi-Supervised 기법이 제대로 동작하지 않는다.[1]

***

## 2. 제안 방법  
### 2.1 Self-Supervised Learning  
- 입력 샘플 $$x\in\mathbb{R}^d$$에 대하여 각 특성(feature)을 Bernoulli 분포($$p_m$$)에 따라 마스킹하여 잡음 샘플 $$\bar x$$로 대체한 변형 $$\tilde x$$를 생성(식 3)  
- **인코더** $$e\colon X\to Z$$가 $$\tilde x$$로부터 잠재 표현 $$z$$를 학습하고,  
- 두 개의 예측기:
  - **마스크 벡터 추정기** $$s_m\colon Z\to^d$$ — 원본에서 어떤 특성이 마스킹되었는지 예측[1]
  - **값 재구성기** $$s_r\colon Z\to X$$ — 마스킹된 특성의 값을 복원  
- 손실함수:  

$$
    \min_{e,s_m,s_r}\; \mathbb{E}_{x,m,\tilde x}\bigl[\ell_m(m,\hat m)+\alpha\,\ell_r(x,\hat x)\bigr]
  $$  
  
$$\ell_m$$은 마스크 예측의 이진 크로스엔트로피(식 5), $$\ell_r$$은 재구성 MSE(식 6).[1]

### 2.2 Semi-Supervised Learning  
- 전이학습된 인코더 $$e$$와 예측모델 $$f$$를 결합하여 예측함수 $$f\circ e$$ 학습  
- **일관성 손실**: 마스킹·복원된 증강 샘플 간 예측 분포 편차 최소화  

$$
    L_{\text{cons}}=\mathbb{E}_{x,m,\tilde x}\bigl\|f(e(\tilde x))-f(e(x))\bigr\|^2
  $$  

- 최종 목적함수:  

$$
    L_{\text{final}}=L_{\text{sup}}+\beta\,L_{\text{cons}}
  $$  
  
$$L_{\text{sup}}$$는 라벨된 데이터에 대한 일반적 분류/회귀 손실.[1]

***

## 3. 모델 구조  
Encoder (여러 층 MLP 또는 Transformer) → Mask Estimator + Feature Estimator (Self-SL) → Predictor (Semi-SL)  
1. Mask Generator로부터 K개의 서로 다른 마스크 샘플 생성  
2. Encoder를 통해 K개의 증강 표현 획득  
3. 예측모델에 일관성 규제 및 지도학습을 동시에 적용[1]

***

## 4. 성능 향상 및 한계  
- **Genomics (UK Biobank SNP, 6개 혈구 지표)**: VIME는 기존 Elastic Net, DAE, Context Encoder, MixUp 대비 훨씬 낮은 MSE 달성(라벨 수 1,000~100,000 구간).[1]
- **Clinical (UK vs US 전립선암 치료)**: XGBoost·로지스틱·MLP 대비 AUROC 0.86/0.84로 우수, 유의미한 분포편차에도 강건.[1]
- **공개 데이터셋(MNIST 탭형 해석, UCI Income, Blog)**: 모든 경우에서 최고 정확도(최대 +1.5%p) 달성.[1]
- **Ablation Study**: Self-SL, Semi-SL 단독 적용보다 통합 시 성능 최대화. Self-SL만 제거 시 더 큰 성능 저하 관찰.[1]
- **한계 및 고려사항**:  
  - 탭형 데이터의 상관관계 구조에 민감하여 하이퍼파라미터($$p_m,\alpha,\beta$$) 튜닝이 필수  
  - 마스크 기반 증강이 모든 도메인에 최적이 아닐 수 있으며, 범주형·연속형 변수 처리 방식 차이에 따른 추가 연구 필요  

***

## 5. 일반화 성능 향상 관점  
- **상관관계 학습**: 마스킹·복원을 통해 변수 간 은밀한 의존관계를 포착하여 복원 및 예측에 활용 → 모델의 과적합 완화  
- **데이터 증강**: 탭형 데이터에 적합한 증강 기법 제시로 모델이 다양한 입력 변형에 견고해짐  
- **일관성 규제**: 지도학습 외에도 불확실성 감소를 유도하여 일반화 성능 강화[1]

***

## 6. 향후 연구 방향  
- **다양한 전이 과제 탐색**: 마스크 추정 외에 순서 예측, 분포왜곡 예측 등 확장  
- **대조 학습(Contrastive Learning) 통합**: VIME 증강과 유사한 샘플 구조 학습  
- **이종 데이터 처리**: 범주형·연속형·텍스트·이미지 혼합 테이블 적용  
- **하이퍼파라미터 자동화**: 메타러닝 기반 최적화  
- **실제 임상·생명과학 적용**: 대규모 병원 데이터셋에 대한 검증 및 규제준수 프레임워크 구축  

VIME는 탭형 데이터에서도 Self-/Semi-Supervised 학습의 이점을 극대화하며, 향후 다양한 도메인과 이종 데이터에의 적용 가능성을 제시한다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/330d1efa-4c37-42b7-80a6-64e72688f02a/NeurIPS-2020-vime-extending-the-success-of-self-and-semi-supervised-learning-to-tabular-domain-Paper.pdf)
