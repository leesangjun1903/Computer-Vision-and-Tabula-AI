# EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations

***

## 1. 논문의 핵심 주장 및 주요 기여 요약

EGSDE 논문은 **Score-based Diffusion Model(SBDM)**을 기반으로 한 새로운 비지도 이미지 투이미지 변환(unpaired image-to-image translation, I2I) 방법을 제안합니다. 기존의 SBDM 기반 비지도 I2I 방식들이 소스 도메인 데이터를 활용하지 않아 현실감(realism)과 충실도(faithfulness)가 제한된다는 한계를 지적하며, **소스 및 타겟 도메인 데이터를 모두 활용한 사전학습 에너지 함수(Energy Function)**를 도입해 이 한계를 극복하는 것을 목표로 합니다. 에너지 함수는 변환 이미지가 **도메인-독립적 특성은 보존하며 도메인-특이적 특성은 변화**하도록 유도합니다. 제안된 EGSDE는 **realism과 faithfulness의 균형을 유연하게 조절**할 수 있으며, 세부 실험에서 SBDM, GAN 기반 최신 방법들보다 우수함을 보였습니다.

***

## 2. 해결 문제, 제안 방법(수식 포함), 모델 구조, 성능, 한계 상세 해설

### 해결하고자 하는 문제

- **Unpaired I2I**에서 소스와 타깃 도메인 간 형태·스타일 변환 시, 현실성(타깃 도메인 적응)과 입력 충실도(소스 이미지의 주요 내용 유지) 간 동시 달성이 어렵고, 기존 SBDM 기반 방법들은 학습시 소스 도메인 데이터의 사용이 미흡해 최적 성능을 못 내는 문제.

***

### 제안 방법: 에너지-가이드 확률미분방정식(Energy-Guided SDE, EGSDE)

#### 기본 SDE/Score-based Diffusion Model 구조

- 순방향 SDE:

$$
  dy = f(y, t)dt + g(t)d w
  $$

- 역방향 SDE: (score 기반)

$$
  dy = [f(y, t) - g(t)^2 \nabla_y \log q_t(y)]dt + g(t)dw
  $$
  
  여기서 score 네트워크 $$s(y, t) \approx \nabla_y \log q_t(y) $$

- 따라서 SBDM의 샘플링 SDE:

$$
  dy = [f(y, t) - g(t)^2 s(y, t)]dt + g(t)dw
  $$

#### EGSDE - 핵심 공식

- 사전 학습된 SDE(score)와 소스/타깃 모두에서 학습된 에너지 함수를 결합:

$$
  dy = [f(y, t) - g(t)^2(s(y, t) - \nabla_y E(y, x_0, t))]dt + g(t)dw
  $$

  - $$s(y,t) $$: target 도메인에서 학습된 score
  - $$E(y,x_0,t) $$: 소스 $$x_0 $$, 생성 $$y $$, 시간 $$t $$에 대한 에너지 함수 (후술)
  - $$\nabla_y E $$: 생성 방향으로의 변화량, 즉 guidance term

#### 에너지 함수(Feature-based Energy Function) 설계

- 에너지 항은 **현실성(Realism) 전문가, 충실도(Faithfulness) 전문가** 두 부분의 로그 potential 합:

$$
  E(y,x,t) = \lambda_s E_s(y,x,t) + \lambda_i E_i(y,x,t)
  $$
  
  - $$E_s $$: 도메인-특이적(변화되어야할) 특성 제거 유도 (cosine similarity of feature extractor)
  - $$E_i $$: 도메인-독립적(보존해야할) 특성 유지 (low-pass filter-based l2 distance)
  - 각각의 가중치 λ_s, λ_i로 두 효과를 트레이드오프 조절

#### 구조적 개념도

- [제품의 전문가(product of experts)] 관점에서, SDE score, 현실성 expert, 충실도 expert가 결합.
- SDE는 타깃 도메인 적합성 보장,
- 현실성 에너지 항은 불필요한 소스 특성 제거,
- 충실도 에너지 항은 중요한 소스 정보 보존,
- 전체적으로 균형 조절이 가능.

***

### 성능 및 한계

#### 성능 지표 및 실험 결과

- **비교 대상**: 기존 SBDMs 기반(Ilvr, SDEdit), GAN 기반(CUT, StarGAN v2 등).
- **벤치마크**: Cat→Dog, Wild→Dog(AFHQ), Male→Female(CelebA-HQ)
- **지표**: FID(현실성), L2, PSNR, SSIM(충실도).
- **결과**: 거의 모든 셋팅에서 SBDM·GAN 계열 최신 방법들 대비 FID, SSIM, PSNR, L2 등에서 우위 확보 (예: Cat→Dog FID 51.04, Wild→Dog FID 50.43, Male→Female FID 30.61 등).
- **트레이드오프 조정 가능**: λ_s, λ_i 하이퍼파라미터로 현실성/충실도 균형 유도 및 서로 보완적 특성.

#### 한계

- domain-independent feature extractor로 **low-pass filter** 사용에 그침 (단순·효과적이지만, disentangled representation learning 등 더 복잡한 구조로 기능 확장 필요성)
- 계산비용↑(SDEit 대비 약 1.4배 느림, Table 5)
- 일부 변환 실패 케이스(눈, 코 미생성, tiger stripe 보존 미흡 등)
- 악용 가능성(조작 이미지 생성 위험) 주의

***

## 3. 모델 일반화 성능 관련 논의

- **소스 및 타겟 양 도메인 학습 활용**으로 기존 대비 domain bias 감소, 변환 robust함 증가 → 다양한 도메인 종류로의 확장/generalization 용이
- 실제로 두 도메인(Cat+Wild → Dog) multi-domain 실험에서도 각종 지표에서 우수한 일반화 성능 시현, 기존 방법 대비 FID, SSIM 모두 개선
- 트레이드오프 및 하이퍼파라미터 유연성, similarity metric 조절 가능성 등으로 실제 다양한 I2I 시나리오에 맞는 세부적 적용·조정이 가능

***

## 4. 앞으로의 영향 및 연구 시 고려 사항

- **영향**: SDE와 energy guidance 결합의 효과를 증명했으므로, 향후 unpaired I2I 분야 및 확장된 공간(텍스트-이미지, multi-modal 등)까지 적용 가능성 클 것. 특히 product-of-experts와 representation learning의 결합 가능성 시사.
- **연구 고려점**: 더 강력한 domain-independent feature extractor(예: Disentangled learning), 빠르고 효율적인 SDE solver 개발, 안전한 활용 및 악용 방지책 마련, 다양한 도메인 적용/테스트, 적응적 하이퍼파라미터 조정 기법 연구 등

***

**정리**: EGSDE는 소스와 타겟 도메인 특징을 조화롭게 반영해 unpaired I2I의 중요한 tradeoff(충실도·현실감)를 높은 수준에서 만족시키는, 확장성·일반화성을 갖춘 방법입니다. 앞으로의 발전 방향은 representation disentangling, 효율성 개선, 안전성·모듈화에 달려 있습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/22c2e88c-8476-4128-96ab-bf64f5b86ee9/2207.06635v5.pdf)
