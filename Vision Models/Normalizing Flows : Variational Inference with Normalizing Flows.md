# Normalizing Flows : Variational Inference with Normalizing Flows | Variational Inference, Image generation
# 핵심 주장 및 주요 기여 요약  
본 논문은 **정상화 흐름(Normalizing Flows)** 을 통해 변분 추론의 후방 근사 분포를 단순한 초기 분포에서 일련의 가역 변환을 거쳐 **임의의 복잡도**를 갖는 분포로 확장할 수 있음을 제안한다[1].  
1. 단순 분포 q₀(z)를 일련의 가역 변환 f₁,…,f_K에 적용해 복잡한 분포 q_K(z_K)를 구성  
2. 유한 흐름(finite flows)과 미소 흐름(infinitesimal flows)을 통합한 새로운 후방 근사 범주 제시  
3. 변분 하한(Evidence Lower Bound)에 선형 추가 비용만 부과하는 효율적 학습 알고리즘 제안  
4. MNIST·CIFAR-10 등에서 mean‐field 대비 유의한 대수적 개선 확인  

# 문제 정의, 제안 방법, 모델 구조, 성능 향상, 한계  

문제점  
기존의 mean‐field 근사나 단순 구조 q_φ(z|x)=N(μ,diag(σ²))는 실제 후방 분포의 복잡한 상관관계나 다봉성(multi-modality)을 포착하지 못해 다음과 같은 문제를 겪는다.  
- 분산 과소추정(under‐estimate variance)  
- MAP 추정치 편향(bias)  
- 테스트 데이터 일반화 성능 저하  

제안 방법  

## 1. 기본 원리: 확률 밀도 변환

정상화 흐름의 핵심은 **가역 변환**을 통한 확률 분포의 변형입니다[1]. 단순한 초기 분포를 일련의 가역 함수로 변환하여 복잡한 분포를 생성합니다.

**기본 변환 공식:**
$$ z' = f(z) $$
$$ q(z') = q(z) \left|\det\frac{\partial f}{\partial z}\right|^{-1} $$

여기서:
- $$z$$: 원래 확률변수
- $$z'$$: 변환된 확률변수
- $$q(z)$$: 원래 확률밀도함수
- $$q(z')$$: 변환된 확률밀도함수
- $$\left|\det\frac{\partial f}{\partial z}\right|$$: 야코비안 행렬식

## 2. 연쇄 변환 (Composition of Transformations)

K개의 변환을 순차적으로 적용하여 점진적으로 복잡한 분포를 구성합니다[1]:

$$ z_0 \rightarrow z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow z_K $$

**합성 변환:**

$$ z_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0) $$

**최종 확률밀도:**

$$ \ln q_K(z_K) = \ln q_0(z_0) - \sum_{k=1}^K \ln\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right| $$

이 공식은 연쇄 법칙(chain rule)을 적용하여 각 단계의 야코비안을 누적한 결과입니다[1].

## 3. Planar Flow (평면 흐름)

**변환 함수:**

$$ f(z) = z + u h(w^T z + b) $$

**매개변수:**
- $$u \in \mathbb{R}^d$$: 변환 방향 벡터
- $$w \in \mathbb{R}^d$$: 가중치 벡터  
- $$b \in \mathbb{R}$$: 편향
- $$h(\cdot)$$: 활성화 함수 (예: tanh)

**효율적 야코비안 계산 (O(d) 시간복잡도):**

$$ \psi(z) = h'(w^T z + b)w $$
$$ \left|\det\frac{\partial f}{\partial z}\right| = |1 + u^T\psi(z)| $$

이는 **행렬식 보조정리(matrix determinant lemma)**를 사용하여 $$O(d^3)$$ 계산을 $$O(d)$$로 줄인 핵심 기여입니다[1].

**기하학적 해석:**
평면 흐름은 초평면 $$w^T z + b = 0$$에 수직인 방향으로 수축/팽창을 적용합니다. $$u$$ 벡터 방향으로 비선형 변형이 이루어집니다[1].

## 4. Radial Flow (방사형 흐름)

**변환 함수:**

$$ f(z) = z + \beta h(\alpha,r)(z - z_0) $$

여기서 $$r = \|z - z_0\|$$, $$h(\alpha,r) = \frac{1}{\alpha + r}$$

**매개변수:**
- $$z_0 \in \mathbb{R}^d$$: 기준점
- $$\alpha \in \mathbb{R}_+$$: 스케일 매개변수
- $$\beta \in \mathbb{R}$$: 강도 매개변수

**야코비안 계산:**
$$ \left|\det\frac{\partial f}{\partial z}\right| = [1 + \beta h(\alpha,r)]^{d-1}[1 + \beta h(\alpha,r) + \beta h'(\alpha,r)r] $$

**기하학적 해석:**
방사형 흐름은 기준점 $$z_0$$ 중심의 방사형 수축/팽창을 수행하며, 거리 $$r$$에 따라 비선형 변형을 적용합니다[1].

## 5. 변분 추론에서의 적용

**기존 ELBO (Evidence Lower BOund):**

$$ \mathcal{F}(x) = -\text{KL}[q(z|x)\|p(z)] + \mathbb{E}_q[\log p(x|z)] $$

**정상화 흐름 적용 후 ELBO:**

$$ \mathcal{F}(x) = \mathbb{E}\_{q_0}[\ln q_0(z_0)] - \mathbb{E}\_{q_0}[\log p(x,z_K)] - \mathbb{E}\_{q_0}\left[\sum_{k=1}^K \ln|1 + u_k^T\psi_k(z_{k-1})|\right] $$

**구성 요소 해석:**
1. $$\mathbb{E}_{q_0}[\ln q_0(z_0)]$$: 초기 분포의 엔트로피
2. $$\mathbb{E}_{q_0}[\log p(x,z_K)]$$: 데이터 우도와 사전분포의 결합
3. $$\sum_{k=1}^K \ln|\det(\partial f_k/\partial z_{k-1})|$$: 야코비안 보정 항

## 6. 수치 예제

**2차원 Planar Flow 예제:**

초기 설정: $$z_0 = [1]$$, $$u = [0.5, -0.3]$$, $$w = [1, 0.5]$$, $$b = 0.1$$, $$h(x) = \tanh(x)$$

**단계별 계산:**
1. 선형 결합: $$w^T z_0 + b = 1 \times 1 + 0.5 \times 2 + 0.1 = 2.1$$
2. 활성화: $$h(2.1) = \tanh(2.1) \approx 0.970$$
3. 변환: $$f(z_0) = [1] + [0.5, -0.3] \times 0.970 = [1.485, 1.709]$$
4. 야코비안: $$|\det(\partial f/\partial z_0)| = |1 + u^T\psi(z_0)| = 1.020$$

## 7. 가역성 보장을 위한 제약 조건

**Planar Flow 제약:**

$$ w^T u \geq -1 \quad (\text{when } h(x) = \tanh(x)) $$

수정된 매개변수:

$$ \hat{u} = u + \frac{m(w^T u) - w^T u}{\|w\|^2} w $$

여기서 $$m(x) = -1 + \log(1 + e^x)$$

**Radial Flow 제약:**

$$ \beta \geq -\alpha $$

수정된 매개변수:

$$ \hat{\beta} = -\alpha + \log(1 + e^\beta) $$

## 8. 핵심 장점

**수학적 장점:**
- $$O(d)$$ 시간복잡도로 야코비안 계산 가능[1]
- 적절한 제약 조건 하에서 가역성 보장[1]
- 흐름 길이 $$K$$ 증가를 통한 임의 복잡도 달성[1]

**통계적 장점:**
- 다봉성(multi-modal) 분포 모델링 가능[1]
- 무한 흐름 길이에서 실제 사후분포에 수렴[1]
- 기존 mean-field 근사 대비 유의한 변분 하한 개선[1]

이러한 수학적 정형화를 통해 정상화 흐름은 변분 추론에서 **표현력이 풍부한 사후분포 근사**를 가능하게 하며, **선형 시간 복잡도**로 효율적인 계산을 보장합니다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c5c188e7-5129-4ff9-b474-5e9c449e1671/1505.05770v6.pdf
성능 향상  
- **MNIST**: 대각 공분산     -ln p(x)≤89.9 → NF(k=40) -ln p(x)≤85.7 으로 개선[1]  
- **CIFAR-10**: -ln p(x)(k=0)=-293.7 → (k=10)=-320.7 개선[1]  

한계  
- 흐름 길이 K, 차원 d 증가 시 계산 비용 O(Kd)  
- 고차원 잠재 변수에서 안정적 야코비안 계산 어려움  
- 실제 고차원 응용(예: 고해상도 이미지) 검증 부족  

# 일반화 성능 향상 관점  
정상화 흐름은 근사 분포의 다양성과 표현력을 크게 높여, 학습 시 **과도한 편향**을 완화하고 **테스트 로그우도**를 향상시킨다. 특히 다봉성(posterior multi-modality)을 포착함으로써 과소분산을 줄이고 데이터셋 외 샘플링 시에도 **높은 예측 불확실성**을 유지하여 일반화 성능을 제고한다[1].  

# 향후 연구에 미치는 영향 및 고려 사항  
- **후속 흐름 연구**: Real NVP, IAF, Glow, Sylvester Flows 등 다양한 구조의 정상화 흐름 발전[2][3].  
- **고차원 확장성**: 차원별 야코비안 효율화, 하이브리드 MCMC/변분 기법 통합 필요  
- **자동화**: 흐름 길이·형태 자동 선택, 안정적 학습 스케줄링 연구  
- **응용 다양화**: 딥 생성 모델, 베이지안 최적화, 시계열·그래프 모델 등 확장성 검증  

앞으로는 **계산 안정성**, **자동화된 흐름 설계**, **고차원 잠재 공간 적응**을 중점적으로 연구하여, 정상화 흐름 기반 변분 추론의 **범용성**과 **검증된 일반화 성능**을 확보해야 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c5c188e7-5129-4ff9-b474-5e9c449e1671/1505.05770v6.pdf
[2] https://arxiv.org/pdf/1912.02762.pdf
[3] https://www.auai.org/uai2018/proceedings/papers/156.pdf
[4] https://www.sec.gov/Archives/edgar/data/2073203/0002073203-25-000001-index.htm
[5] https://www.sec.gov/Archives/edgar/data/2073202/0002073202-25-000001-index.htm
[6] https://www.sec.gov/Archives/edgar/data/2042176/000121390025040355/ea0218945-06.htm
[7] https://www.sec.gov/Archives/edgar/data/1901799/000095017024060563/btm-20240331.htm
[8] https://www.sec.gov/Archives/edgar/data/1969169/0001969169-23-000001-index.htm
[9] https://www.sec.gov/Archives/edgar/data/1288847/000128884723000029/fivn-20221231.htm
[10] http://www-odp.tamu.edu/publications/197_SR/synth/synth.htm
[11] https://arxiv.org/pdf/1505.05770.pdf
[12] https://arxiv.org/pdf/2402.16408.pdf
[13] https://arxiv.org/pdf/2108.07089.pdf
[14] https://arxiv.org/pdf/2007.05426.pdf
[15] https://arxiv.org/abs/1505.05770
[16] https://indico.cern.ch/event/939335/contributions/3946863/attachments/2073692/3491279/mpp-jc-23July20.pdf
[17] https://proceedings.mlr.press/v37/rezende15.html
[18] https://www.depthfirstlearning.com/2021/VI-with-NFs
[19] https://paperswithcode.com/paper/variational-inference-with-normalizing-flows
[20] https://cindyyzhang.github.io/papers/6435-project.pdf
[21] http://arxiv.org/pdf/1505.05770.pdf
[22] https://slim.gatech.edu/Publications/Public/Conferences/SEG/2021/siahkoohi2021SEGlbe/abstract.html
[23] https://www.arxiv.org/abs/2505.10466
[24] https://www.semanticscholar.org/paper/d3c69e421cada7a628746e5b762a940313cc9c77
[25] https://www.semanticscholar.org/paper/dc6a8b55d906d78f37de96847b5dd96cb7a4c851
[26] https://arxiv.org/abs/2305.02460
[27] https://joss.theoj.org/papers/10.21105/joss.06309.pdf
[28] https://jmlr.org/papers/volume22/19-1028/19-1028.pdf
