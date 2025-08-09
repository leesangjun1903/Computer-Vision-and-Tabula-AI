# Improved Techniques for Training GANs

## 핵심 주장 및 주요 기여  
**Improved Techniques for Training GANs**는 GAN(Generative Adversarial Networks)의 불안정한 학습 문제를 해결하고, 반(半)지도 학습과 고품질 이미지 생성을 동시에 달성하기 위한 여러 기법을 제안한다.  
1. **학습 안정화 기법**  
   - *Feature Matching*: 생성자가 판별기의 중간층 통계(feature)의 기댓값을 일치시키도록 목표를 재정의  
   - *Minibatch Discrimination*: 판별기가 미니배치 내 샘플 간 유사도를 활용해 모드 붕괴(mode collapse)를 방지  
   - *Historical Averaging*: 과거 파라미터 평균을 비용 함수에 추가하여 진동을 완화  
   - *One-sided Label Smoothing*: 긍정 레이블만 스무딩하여 생성 샘플이 데이터 분포 근처로 이동하도록 유도  
   - *Virtual Batch Normalization*: 고정된 참조 배치(reference batch) 통계로 배치 정규화 적용  
2. **평가 지표 제안**  
   - *Inception Score*: 생성 이미지의 다양성과 ‘객체성(objectness)’을 KL 발산 기반에 기초해 수치화  
3. **반지도 학습에서의 성능 향상**  
   - 생성 이미지를 추가 클래스(“fake”)로 간주하여, 레이블이 없는 실제 데이터로부터도 학습 가능토록 손실 함수를 확장  
4. **다양한 데이터셋에서의 실험**  
   - MNIST, CIFAR-10, SVHN에서 반지도 분류 정확도 최첨단(SoTA)을 달성  
   - ImageNet(128×128, 1,000개 클래스) 샘플 생성에서 이전 DCGAN 대비 피사체 인지 가능한 특징(fur, eye, nose 등) 학습

***

## 1. 해결하고자 하는 문제  
기존 GAN 학습은 두 네트워크가 동시에 경쟁하며 파라미터 공간에서 Nash 평형을 찾기 때문에  
- **불안정성**: 경사 하강법이 진동하거나 발산  
- **모드 붕괴**: 생성기가 단일 또는 소수의 샘플만 반복 출력  
- **정량적 평가 부재**: 비주얼 품질을 객관화할 수단 부족  

***

## 2. 제안 방법  
### 2.1 Feature Matching  
생성자를 다음 목표로 학습  

$$
\min_\theta \Big\| \mathbb{E}\_{x\sim p_{\rm data}} f(x) - \mathbb{E}_{z\sim p_z} f\bigl(G(z;\theta)\bigr) \Big\|_2^2
$$  

여기서 $$f(x)$$는 판별기의 중간층 활성화벡터. GAN의 원래 목적 대신 판별기가 학습한 ‘유용한’ 특징 통계를 맞추도록 유도.

### 2.2 Minibatch Discrimination  
판별기가 미니배치 내 샘플 $$x_i$$ 간의 거리를 고려하도록 확장:  
1. $$f(x_i)\in\mathbb{R}^A$$에 텐서 $$T\in\mathbb{R}^{A\times B\times C}$$ 곱해 $$M_i\in\mathbb{R}^{B\times C}$$ 획득  
2. $$c_b(x_i, x_j) = \exp(-\|M_{i,b} - M_{j,b}\|_{1})$$  
3. $$\,o(x_i)_b = \sum_j c_b(x_i, x_j)$$  
4. $$o(x_i)$$를 $$f(x_i)$$에 연결하여 다음 층에 입력  

### 2.3 Historical Averaging  
각 파라미터 $$\theta$$에 과거 평균 $$\frac1t\sum_{i=1}^t \theta^{[i]}$$와의 거리 항을 추가하여  

$$
J(\theta) \to J(\theta)+\lambda\bigl\|\theta - \tfrac1t\textstyle\sum_{i=1}^t \theta^{[i]}\bigr\|_2^2
$$

### 2.4 One-sided Label Smoothing  
판별기 소프트맥스 목표를 “1 → α”로 스무딩(예: α=0.9), “0”은 유지.  

$$
D^*(x) = \frac{\alpha\,p_{\rm data}(x) + 0\cdot p_{\rm model}(x)}{p_{\rm data}(x)+p_{\rm model}(x)}
$$

### 2.5 Virtual Batch Normalization  
생성기 각 입력 $$x$$를 고정 참조 배치 통계를 사용해 정규화. 배치 간 상호 의존성 제거.

***

## 3. 모델 구조  
- **Discriminator**:  
  - MNIST/CIFAR-10: 9층 CNN + 드롭아웃 + 가중치 정규화 + 미니배치 차별화  
  - ImageNet: DCGAN 기반 다중 GPU 확장  
- **Generator**:  
  - 4층 CNN + VBN + Feature Matching 목표  

***

## 4. 성능 향상  
| 데이터셋 | 설정 | 기존 최고 오류율 | 제안 기법 오류율 |
|:---:|:---|:---:|:---:|
| MNIST (20 labels) | permutation-invariant | 106 [라더 네트워크] | 90 ±4.2 |
| CIFAR-10 (4,000 labels) | semi-supervised | 19.58±0.46 [CatGAN] | 18.63±2.32 |
| SVHN (2,000 labels) | semi-supervised | 24.63 [VAT] | 6.16±0.58 |

- **Inception Score**: CIFAR-10 생성 샘플에서 8.09 → 11.24로 개선.  
- **ImageNet**: DCGAN은 단색 패치 학습 불가, 제안 기법은 동물 형태 인지 가능한 샘플 생성  
- **모드 붕괴** 방지 및 **학습 안정성** 대폭 향상  

***

## 5. 한계 및 일반화 성능 개선 가능성  
- **계산 비용**: VBN과 Historical Averaging이 추가 연산 유발  
- **이론적 이해 부족**: 두 네트워크 상호작용의 수렴 보장은 미미  
- **고차원 분포**: ImageNet 해상도·클래스 수 증가 시 해부학적 정확성 부족  
- **일반화**: Feature Matching 기반 반지도 학습이 다른 도메인(텍스트, 음성)에서도 동일한 이득을 보장하는지는 미검증  

***

## 6. 미래 연구에 미치는 영향 및 고려 사항  
- **안정적 GAN 학습**: 본 기법들은 다양한 GAN 변형(Conditional, CycleGAN)에 적용되어 학습 안정화의 표준이 될 전망  
- **평가 메트릭**: Inception Score는 샘플 품질 평가의 기본 지표로 자리잡았으며, 이후 FID(Frechet Inception Distance) 발전에 기여  
- **반지도·무지도 학습 확장**: Feature Matching 및 미니배치 기법을 기반으로 비전 외 분야의 반지도 모델 개발 가능  
- **이론적 연구**: Nash 평형 수렴을 보장하는 최적화 알고리즘 및 이론적 해석이 필요한 과제로 부상  
- **효율화**: VBN 대체 경량 정규화 기법과 Historical Averaging의 비용 최소화 전략 연구 필요  

---  
**주요 인사이트**: 다양한 안정화·평가 기법의 결합이 GAN의 실용적 적용 범위를 크게 확장했으며, 특히 반지도 학습과 고해상도 이미지 생성에서 획기적 성과를 제시하였다. 앞으로는 이론적 수렴 보장과 계산 효율 최적화에 초점을 맞춘 연구가 요구된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/bd752838-f158-426c-be5e-1b7702efe8c1/1606.03498v1.pdf
