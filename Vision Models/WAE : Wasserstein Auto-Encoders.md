# WAE : Wasserstein Auto-Encoders

**핵심 주장 및 주요 기여**  
Wasserstein Auto-Encoders(WAE)는 **최적수송(Optimal Transport, OT) 관점**에서 생성모델을 제안하며, 데이터 분포 $$P_X$$와 모델 분포 $$P_G$$ 간의 **Wasserstein 거리**를 최소화하는 새로운 알고리즘이다. WAE는 VAE의 변분 바운드 대신 OT 비용에 기반한 다음과 같은 **벌점 형태의 목적함수**를 사용한다:

$$
\mathcal{L}_{\mathrm{WAE}} = \mathbb{E}_{P_X}[c(x, G(z))] + \lambda\,D_Z(Q_Z, P_Z),
$$

여기서 첫 번째 항은 복원 비용, 두 번째 항은 인코더가 생성한 잠재분포 $$Q_Z$$와 사전분포 $$P_Z$$ 간의 불일치 정도를 재는 벌점이다. 이로써[1]
- **AAE(Adversarial Auto-Encoder)**는 WAE-GAN으로 일반화되고,  
- **MMD(Maximum Mean Discrepancy)** 기반의 완전 비적대적 학습(WAE-MMD)을 가능케 한다.[1]

# 1. 해결하고자 하는 문제  
기존 모델의 한계  
- VAE: 안정적인 학습·인코더 구조 장점에도 불구하고 자연 이미지에서 **흐릿한 샘플** 생성 경향.[1]
- GAN: **고품질 샘플** 생성에도 불구하고 인코더 부재, 훈련 불안정성, 모드 붕괴 문제가 있다.[1]
- OT 기반 WGAN: 인코더 부재, $$W_1$$만 지원, 일반 비용 함수 적용 곤란.[1]

WAE 목표  
- OT 비용 $$W_c(P_X,P_G)$$을 **프라이멀(primal) 형태**로 직접 최적화  
- **인코더–디코더** 구조 유지  
- 다양한 비용 함수 $$c(x,y)$$와 분포 벌점 $$D_Z$$ 선택 가능  

# 2. 제안하는 방법 및 수식  
## 2.1 OT 기반 인코더–디코더 모델  
결정론적 디코더 $$G\colon Z\to X$$일 때, Kantorovich식 OT는 다음과 같이 변형된다:[1]

$$
W_c(P_X,P_G)
= \inf_{Q_{Z|X}:\,Q_Z = P_Z} \mathbb{E}_{P_X}\!\big[c\big(x, G(z)\big)\big].
$$

이를 제약 최적화 대신 **벌점 완화(penalized relaxation)** 형태로 바꾼 것이 WAE 목적함수:

$$
\min_{Q_{Z|X}}\,\mathbb{E}_{P_X}[c(x,G(z))] \;+\;\lambda\,D_Z\big(Q_Z,P_Z\big).
$$  

## 2.2 벌점 함수 $$D_Z$$  
- **WAE-GAN**: $$D_Z$$로 Jensen–Shannon Divergence를 채택하고, 잠재공간 판별기(discriminator)를 도입하여 적대적 학습.[1]
- **WAE-MMD**: 잠재공간에서 $$\mathrm{MMD}_k(Q_Z,P_Z)$$ 벌점 사용. RBF 대신 꼬리가 두터운 역다경계역수함수(kernel)로 안정적 추정.[1]

## 2.3 모델 구조  
- **인코더**: 입력 $$x$$를 잠재코드 $$z$$로 매핑(결정론적/확률적 모두 가능)  
- **디코더**: $$z$$를 다시 $$x$$로 복원  
- **잠재 판별기(WAE-GAN)**: 잠재분포 구분기(fully connected DNN)  
- **최적화**: Adam + 배치정규화/컨볼루셔널 네트워크 기반 구조  

# 3. 성능 향상 및 한계  
## 3.1 성능 향상  
- **샘플 품질**: FID 점수(CelebA)에서 VAE(63) 대비 WAE-MMD(55), WAE-GAN(42) 개선.[1]
- **해상도·선명도**: Laplace 필터 기반 sharpness 지표에서도 WAE-GAN 최고.[1]
- **일반화**: 테스트 데이터 복원 및 잠재공간 내 선형 보간(interpolation)에서 VAE보다 **매끄러운 잠재 구조** 확인.[1]

## 3.2 한계  
- **Q_Z–P_Z 불일치 민감성**: 잠재 매칭 오차가 샘플 품질에 직접 영향.[1]
- **적대적 학습 불안정성**: WAE-GAN은 높은 품질 달성에도 훈련 안정성은 WAE-MMD보다 낮음.[1]
- **커널 선택**: MMD 벌점의 경우 커널 대역폭·종류에 따른 성능 편차 존재.[1]

# 4. 일반화 성능 향상 관점  
WAE는 VAE와 달리 **각 샘플이 개별적으로 사전분포에 강제되지 않음**으로써 잠재코드 간 충분한 분산을 허용한다. 이는
- 훈련 시 **복원 능력**(reconstruction)과 **사전 매칭** 간 균형 유지  
- 잠재공간 내 데이터 매니폴드 구조 보존 → **테스트 샘플 복원력** 강화  
- 선형 보간 결과에서 시각적 불연속 감소  

이러한 특성 덕분에 **새로운(미관측) 데이터**에 대한 복원·생성 성능이 VAE 대비 향상된다.

# 5. 향후 연구 방향 및 고려사항  
- **잠재 벌점 개선**: 적대적 양자화(adversarial cost)나 **상호정보(mutual information)** 벌점을 추가하여 표현력 강화  
- **비용 함수 학습**: 입력공간 OT 비용 $$c(x,y)$$를 적대적 학습으로 업데이트  
- **다양한 분포 매칭**: Gauss 이외의 다중 모드 사전분포, 흐름기반(flow-based) 분포 탐색  
- **이론적 분석**: WAE-GAN/WAE-MMD의 쌍대 형식 이론 정교화 및 수렴 분석  
- **응용 확장**: 고해상도 이미지, 시변 데이터(time-series) 등으로 구조·스케일 확장  

WAE는 OT 기반 생성모델 연구에 새로운 틀을 제시하며, **AAE, InfoVAE** 등 다양한 변형 연구로 이어질 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/70559a8d-6cd4-4d81-a693-467e9b59a12e/1711.01558v4.pdf)
