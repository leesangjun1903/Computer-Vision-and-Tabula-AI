# Improved Contrastive Divergence Training of Energy-Based Model | 2020 · 183회 인용, Image generation

**핵심 주장 및 주요 기여**  
본 논문은 기존 Contrastive Divergence(CD) 학습에서 생략되던 추가적인 그래디언트 항(KL 항)을 효율적으로 추정·포함하여 CD 기반 Energy-Based Model(EBM)의 학습 **안정성**과 **생성 성능**을 크게 향상시킨다. 더불어 데이터 증강(data augmentation)과 다중 해상도(multi-scale) 처리를 도입하여 **모드 탐색**과 **강인성**을 개선하며, 다양한 벤치마크(이미지 생성, OOD 검출, 조합적 생성)에서 우수한 성능을 보인다.[1]

## 1. 해결하고자 하는 문제  
Energy-Based Model은 에너지 함수를 통해 데이터 분포를 표현하나, 최대우도학습 시 파티션 함수의 계산 불가능성으로 인해 MCMC 기반 CD 학습을 사용한다. 그러나 기존 CD 학습은

$$\mathcal{L}\_{\mathrm{CD}} = \mathrm{KL}(p_{\mathrm{data}}\Vert p_{\theta}) - \mathrm{KL}(q_t\Vert p_{\theta}) $$

에서 두 번째 항의 변화에 기인하는 그래디언트를 무시하여 학습 불안정성과 모드 붕괴(mode collapse)를 초래한다.[1]

## 2. 제안 방법  
### 2.1 완전한 CD 손실식  
기존 CD 손실식에 누락된 KL 그래디언트 항을 포함한 전체 손실을 정의한다:  

$$ \mathcal{L}_{\mathrm{Full}} = \mathcal{L}_{\mathrm{CD}} + \mathcal{L}_{\mathrm{KL}}, $$  

여기서  

$$ \mathcal{L}_{\mathrm{CD}} = \mathbb{E}_{p_{\mathrm{data}}}[E(x)] - \mathbb{E}_{q_t}[E(x)], $$  

$$ \mathcal{L}_{\mathrm{KL}} = \mathbb{E}_{q_t}[-\log q_t(x)], $$  

이며, $$\mathcal{L}_{\mathrm{KL}}$$은 샘플 엔트로피를 정규화하여 MCMC 샘플의 다양성을 보장한다.[1]

### 2.2 그래디언트 추정  
- **샘플 에너지 최소화**: Langevin dynamics의 마지막 단계에 대해 자동미분을 적용하여 $$\mathcal{L}_{\mathrm{opt}}$$을 계산한다.  
- **엔트로피 최대화**: 최근접 이웃(K-nearest neighbor) 엔트로피 추정기로 $$\mathcal{L}_{\mathrm{ent}}$$을 계산하여 샘플 다양성을 촉진한다.[1]

### 2.3 데이터 증강 및 다중 해상도 처리  
- **데이터 증강 전이**: MCMC 체인에 색상 변화, 수평 뒤집기, 가우시안 블러 등을 주기적으로 적용하여 모드 전이를 촉진한다.  
- **다중 해상도 에너지 합성**: 원본, 1/2, 1/4 해상도에서 계산된 에너지 함수를 합산하여 전역·국부 정보 모두에 민감한 모델 구조를 설계한다.[1]

## 3. 모델 구조  
Residual block 기반의 CNN 아키텍처에 Self-Attention 및 Layer Normalization을 추가하고, multi-scale 구조를 통해 세 가지 해상도 입력을 동시에 처리한다.[1]

## 4. 성능 향상 및 한계  
- **학습 안정성**: KL 항 추가로 스펙트럴 정규화 없이도 Self-Attention 결합 시 안정적인 수렴을 달성한다.  
- **생성 품질**: CIFAR-10, ImageNet32x32, CelebA-HQ에서 FID 및 Inception Score 개선을 확인하였다. 기존 EBM 대비 FID가 최대 32.48로 낮아지고, SNGAN/SSGAN에 근접하는 성능을 보였다.[1]
- **OOD 검출**: CIFAR-10 학습 모델에 대한 AUROC가 0.88로, 기존 방법 대비 크게 향상되었다.  
- **조합적 생성**: 독립 학습된 여러 EBM을 제로샷(zero-shot)으로 합성하여 고해상도 조건부 이미지를 생성할 수 있음을 입증했다.  
- **한계**: KNN 엔트로피 추정은 고차원에서 샘플 수 증가에 따른 계산 비용 상승, 다중 해상도 처리 시 메모리 사용량 증가 등의 제약이 존재하며, 과도한 MCMC 단계에 대한 계산 부담이 남아 있다.

## 5. 일반화 성능 향상  
KL 항은 샘플 다양성을 보장함으로써 모드 커버리지(mode coverage)를 확대하여 과적합 위험을 줄이고, unseen 데이터에 대한 OOD 감지 성능을 높인다. 데이터 증강 전이 또한 다양한 변형을 학습 중 반영해 일반화 능력을 강화한다.[1]

## 6. 미래 연구에 대한 영향 및 고려사항  
논문 기여는 CD 기반 EBM 연구에 다음과 같은 시사점을 제공한다.  
- **확장성**: 텍스트, 비디오, 단백질 접힘 등 다양한 도메인에 적용 가능하다.  
- **효율성**: 자동미분과 엔트로피 추정기의 결합은 복잡도 증가를 최소화하므로, 더 큰 모델·데이터셋에도 적용할 여지가 있다.  
- **고려사항**: 고차원 데이터에 대한 KNN 엔트로피 계산 최적화, MCMC 단계 수와 증강 전략의 균형, 학습 시간 단축을 위한 병렬화·가속화 기법 연구가 필요하다.

이상의 기법들은 EBM 학습의 새로운 표준이 될 수 있으며, 향후 더욱 강력한 모형 개발과 실제 응용으로 이어질 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4abf5ee0-3b60-42ed-b9ed-53e4ba3e503a/2012.01316v4.pdf)
