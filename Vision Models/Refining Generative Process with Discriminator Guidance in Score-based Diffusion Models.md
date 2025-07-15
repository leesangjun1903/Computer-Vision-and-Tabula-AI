# Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models | Image generation

## 1. 핵심 주장과 주요 기여

이 논문은 **사전 훈련된 확산 모델의 성능을 재훈련 없이 향상시키는 혁신적인 방법**을 제시합니다. 주요 기여는 다음과 같습니다:

**핵심 기여:**
- **Discriminator Guidance (DG) 방법 제안**: 사전 훈련된 점수 모델을 고정한 상태에서 판별자를 통해 생성 성능을 향상시키는 새로운 접근법
- **이론적 근거 제시**: 판별자가 모델 점수를 실제 데이터 점수에 더 가깝게 조정함을 수학적으로 증명
- **범용성 입증**: 다양한 데이터셋(CIFAR-10, CelebA/FFHQ, ImageNet)과 모델(EDM, LSGM, ADM, DiT)에서 일관된 성능 향상
- **SOTA 성능 달성**: 모든 실험 데이터셋에서 최고 성능 기록

## 2. 해결 문제와 제안 방법

### 해결하고자 하는 문제

1. **점수 추정 오차**: 사전 훈련된 모델의 점수 함수 $$s_{\theta_\infty}$$와 실제 데이터 점수 $$\nabla \log p_r^t$$ 간의 차이
2. **재훈련 문제**: 점수 모델 재훈련 시 발생하는 overfitting과 memorization
3. **계산 비용**: 기존 방법들의 높은 계산 비용과 불안정성

### 제안 방법의 핵심

**1. Correction Term 개념 도입**
$$c_{\theta_\infty}(x_t, t) = \nabla \log \frac{p_r^t(x_t)}{p_{\theta_\infty}^t(x_t)}$$

**2. 판별자를 통한 근사**
$$c_\phi(x_t, t) = \nabla \log \frac{d_\phi(x_t, t)}{1 - d_\phi(x_t, t)}$$

**3. 판별자 훈련 (BCE 손실)**

$$L_\phi = \int \lambda(t) \left[ \mathbb{E}\_{p_r^t}[-\log d_\phi] + \mathbb{E}\_{p_{\theta_\infty}^t}[-\log(1-d_\phi)] \right] dt$$

**4. 개선된 생성 과정**
$$dx_t = \left[ f(x_t, t) - g^2(t)(s_{\theta_\infty} + c_\phi)(x_t, t) \right] d\bar{t} + g(t) d\bar{w}_t$$

### 모델 구조

- **사전 훈련된 점수 모델**: 고정된 상태로 유지
- **판별자 구조**: U-Net 기반 encoder
  - ADM classifier의 encoder (고정)
  - 얕은 U-Net encoder (훈련 대상)
- **시간 임베딩**: 모든 noise level에서 실제/생성 데이터 구분
- **클래스 조건부 생성**: 클래스 정보를 포함한 판별자 훈련

## 3. 성능 향상 및 일반화 성능

### 정량적 성능 향상

| 데이터셋 | 모델 | 개선 전 FID | 개선 후 FID | 향상율 |
|----------|------|-------------|-------------|--------|
| CIFAR-10 | EDM | 2.03 | **1.77** | 12.8% |
| CIFAR-10 | LSGM | 2.10 | **1.94** | 7.6% |
| CelebA 64×64 | Soft Truncation | 1.90 | **1.34** | 29.5% |
| FFHQ 64×64 | EDM | 2.39 | **1.98** | 17.2% |
| ImageNet 256×256 | ADM | 4.59 | **3.18** | 30.7% |
| ImageNet 256×256 | DiT | 2.27 | **1.83** | 19.4% |

### 일반화 성능 향상 가능성

**1. 도메인 간 일반화**
- 다양한 이미지 도메인에서 일관된 성능 향상 확인
- 자연 이미지(ImageNet), 얼굴(CelebA/FFHQ), 일반 객체(CIFAR-10) 모두에서 효과적

**2. 모델 간 일반화**
- 서로 다른 아키텍처(EDM, LSGM, ADM, DiT)에서 모두 적용 가능
- 데이터 공간과 잠재 공간 확산 모델 모두에서 효과적

**3. 확장성**
- Classifier Guidance와 결합 가능
- 기존 guidance 기법들과의 통합 가능성 제시

**4. 안정성**
- 판별자 훈련이 안정적이고 빠른 수렴 (Figure 3 참조)
- GAN과 달리 min-max 문제가 아닌 min-min 문제로 더 안정적

## 4. 한계점

1. **추가 계산 비용**: 판별자 훈련을 위한 추가 샘플 생성 필요
2. **NFE 의존성**: 매우 적은 Function Evaluation 횟수에서 효과 감소
3. **Density-chasm 문제**: 극저 noise scale에서의 성능 한계
4. **메모리 요구사항**: 판별자 훈련용 데이터셋 저장 필요

## 5. 앞으로의 연구 영향과 고려사항

### 연구 영향

**1. 패러다임 변화**
- 사전 훈련 모델의 post-hoc 개선 방법론 제시
- 재훈련 없이 기존 모델 성능 향상 가능성 입증

**2. 이론적 기여**
- 점수 기반 생성 모델의 이론적 이해 증진
- correction term의 수학적 정의와 근사 방법 제시

**3. 실용적 가치**
- 계산 효율적인 성능 향상 방법 (점수 훈련 대비 10% 미만의 비용)
- 기존 모델과의 호환성

### 향후 연구 고려사항

**1. 효율성 개선**
- 더 효율적인 correction term 추정 방법 연구
- 극소 NFE 환경에서의 성능 향상 방안

**2. 일반화 확장**
- 다양한 데이터 도메인에서의 검증
- 3D 생성, 비디오 생성 등 다른 모달리티로의 확장

**3. 이론적 발전**
- f-divergence 기반 점수 손실 함수 연구
- 동시 훈련 방법의 이론적 분석

**4. 실용적 고려사항**
- 실시간 생성 환경에서의 적용 가능성
- 판별자 overfitting 방지 기법 개발

이 논문은 **생성 모델의 성능 향상을 위한 새로운 패러다임을 제시**하며, 특히 기존 사전 훈련 모델의 재활용 가능성을 크게 확장시킨 중요한 연구입니다. 이론적 엄밀성과 실용적 효과를 모두 갖춘 방법론으로서, 향후 생성 모델 연구에 상당한 영향을 미칠 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ea1514dc-9d16-4e94-a03d-632b28640c43/2211.17091v4.pdf
