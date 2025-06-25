# Normalizing Flows: An Introduction and Review of Current Methods 

## 1. 핵심 주장 및 주요 기여  
이 논문은 **Normalizing Flows(정규화 흐름)**의 기초 개념을 체계적으로 정리하고, 주요 모델 구조와 학습 기법을 비교·분석하여 향후 연구 방향과 미해결 과제를 제시한다[1]. 특히 다음 세 가지 목표를 갖는다.  
- 독자가 정규화 흐름의 기본 원리와 수학적 배경을 이해하도록 돕는다.  
- 다양한 흐름 모델(affine, coupling, autoregressive, residual, continuous 등)의 구성 방식을 통일된 틀로 설명한다.  
- 현재까지 성능을 입증한 흐름 모델을 비교·고찰하고, 일반화 및 확장 가능성을 논의한다[1].  

## 2. 해결 문제 및 제안 방법  
### 2.1 문제 정의  
주어진 복잡한 분포 $p_Y(y)$를 샘플링과 밀도 평가(density evaluation)가 모두 효율적이고 정확하게 가능한 형태로 모델링하는 것이 목표이다[1].  

### 2.2 정규화 흐름의 원리  
단순한 기저 분포 $p_Z(z)$를 가역 변환 $g:\mathbb{R}^D \! \to \! \mathbb{R}^D$ 의 연쇄로 복잡 분포로 사상(pushforward)한다.  
- 역변환 $f = g^{-1}$과 자코비안 행렬의 행렬식(det)을 이용하여 밀도를 계산한다:

$$
    p_Y(y) \;=\; p_Z\bigl(f(y)\bigr)\,\bigl|\det Df(y)\bigr|
$$
  
[1].  

### 2.3 학습 목표  
- **최대우도 추정**: 관측 데이터 $\{y^{(i)}\}$에 대해

$$
    \max_{\theta,\phi}\sum_i\bigl[\log p_Z\bigl(f(y^{(i)};\theta)\bigr|\phi)+\log|\det Df(y^{(i)};\theta)|\bigr]
$$

[1].  
- **변분추론**: 잠재 변수 모델의 사후분포 근사를 위해 흐름을 재파라미터화 트릭과 결합[1].  

## 3. 모델 구조  

본 논문[1]의 Methods(제3장)에서는 **정규화 흐름(Normalizing Flows)** 모델을 구성하는 다양한 가역 변환(transformations) 기법들을 소개한다. 이들 변환은 단순 분포를 복잡 분포로, 또는 그 역으로 효율적으로 사상하기 위해 고안되며, 크게 다음 여섯 가지 범주로 구분된다.

## 1. 요소별 변환(Elementwise Flows)
가장 단순한 형태로, 각 차원마다 독립적인 1차원 가역 함수 $$h:\mathbb R\to\mathbb R$$를 적용한다.  
수식: 

$$
g(x) = \bigl(h(x_1),\,h(x_2),\dots,h(x_D)\bigr)
$$  

- 장점: 계산 및 역변환이 매우 빠름.  
- 한계: 차원 간 상관관계를 반영할 수 없어 표현력이 부족[1].

## 2. 선형 변환(Linear Flows)
벡터 전체에 대한 선형 사상 $$g(x)=Ax+b$$를 이용한다.  
- 일반적 경우 $$A\in\mathbb R^{D\times D}$$ 의 가역성을 보장하면 가역 변환이 된다.  
- 효율을 위해 대각, 삼각, 순열/직교, LU 분해, 1×1 컨볼루션 등 다양한 특수 구조를 사용한다.  
  - **대각 행렬**: $$\det(A)$$ 계산·역연산 $$\mathcal O(D)$$, 상관관계 표현 불가.  
  - **삼각 행렬**: $$\mathcal O(D^2)$$로 역연산 가능, 부분적 상관정보 포착[1].  
  - **1×1 컨볼루션**: 채널 간 선형 혼합을 구현하며 $$\det$$ 및 역연산 효율 처리[1].

## 3. 플라나 흐름 및 방사 흐름(Planar & Radial Flows)
역변환이 닫힌 형식으로 주어지지 않으나, 자코비안 행렬식 계산은 효율적이다.  
- **Planar Flow**

$$
  g(x)=x + u\,h\bigl(w^\top x + b\bigr),\quad u,w\in\mathbb R^D
$$  

  자코비안 $$\det(I + u\,h'(w^\top x + b)\,w^\top)$$
  
  [1].  
- **Radial Flow**

$$
  g(x)=x + \beta\frac{x-x_0}{\alpha + \|x-x_0\|},\quad x_0\in\mathbb R^D
$$  

  특정 중심 주변 분포를 압축·확장.

## 4. 커플링 흐름 및 자기회귀 흐름(Coupling & Autoregressive Flows)
현실적 모델에서 가장 널리 쓰인다.  
### 4.1 커플링 흐름(Coupling Flows)  
입력 $$x$$를 두 부분 $$(x_A,x_B)$$로 나눠,  

$$
y_A = h\bigl(x_A;\Theta(x_B)\bigr),\quad y_B = x_B
$$  

- $$h$$: 가역 함수(커플링 함수), $$\Theta$$로 파라미터 조정 가능.  
- 자코비안은 삼각 행렬 형태로, $$\det = \det(\partial h/\partial x_A)$$[1].  
- 다층·다차원 커플링으로 높은 표현력 확보.

### 4.2 자기회귀 흐름(Autoregressive Flows)  
출력 각 차원을 이전 입력(또는 이전 출력)에 순차적으로 조건화: 

$$
y_t = h\bigl(x_t;\Theta_t(x_{1:t-1})\bigr).
$$  

- 자코비안은 삼각 행렬, 곱셈으로 $$\det$$ 효율 계산.  
- 역변환은 순차적 계산이 필요해 GPU 병렬화에 제약이 있다[1].  
- **MAF**(Masked AF) vs. **IAF**(Inverse AF): 샘플링/밀도평가 속도 간 trade-off.

### 4.3 유니버설리티(Universality)  
충분한 대수적 조건을 만족하면 어떤 분포도 근사 가능함을 증명.  
- 모노톤 함수의 점별 수렴 집합이 촘촘함을 이용[1].

### 4.4 커플링 함수 종류  
1차원 스칼라 가역 함수 $$h(x;\theta)$$로 요소별 변환. 대표 예:  
- **Affine**: $$h(x)=\alpha x+\beta$$  
- **Mixture CDF**: 물류 혼합 누적분포 적용[1]  
- **Spline**: 단조 스플라인(선형·3차·유리2차)[1]  
- **Neural NAF**: 가중치 양수 MLP로 단조 조건 구현[1]  
- **SOS Polynomial**: 가중치 제약 없이 단조 다항식 합-정사각 표현[1]  
- **Piecewise Bijective**: 비연속 구간별 단조 사상 후 gating 기법으로 invertible[1]

## 5. 잔차 흐름(Residual Flows)
잔차 블록 $$g(x)=x+F(x)$$을 이용한 가역 네트워크.  
- **iResNet**: Lipschitz 상수 \<1 제약으로 역변환 보장, 로그-자코비안은 멱급수 및 해치킨슨 기법으로 근사[1].  
- **Residual Flow**: 러시안 룰렛 추정기로 편향 없는 로그-자코비안 계산[1].

## 6. 연속 흐름(Continuous Flows)
ODE·SDE 기반으로 무한 깊이 흐름을 모델링.  
### 6.1 ODE 기반(Node/FFJORD)  
$$
\frac{dx(t)}{dt} = F\bigl(x(t),\,\theta(t)\bigr)
$$  
- 시간 1맵을 학습, 역전파는 어저인트 민감도(연속 역전파) 사용[1].  
- $$\log p(x(t))$$ 변화는 $$-\mathrm{Tr}(\partial F/\partial x)$$ 적분으로 계산.  

### 6.2 SDE 기반(Langevin Flows)  
$$
dx = b(x,t)\,dt + \sigma(x,t)\,dB_t
$$  
- Fokker–Planck 방정식으로 시간에 따른 밀도 변화 추적.  
- MCMC나 연속-비연속 전이 모델로 근사, 역추론은 VARIATIONAL LOWER BOUND 사용[1].

[1] Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker, “Normalizing Flows: An Introduction and Review of Current Methods,” IEEE TPAMI, arXiv:1908.09257v4, 2020.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/dae79a1b-62cf-4fe3-90e8-05bb1a759f56/1908.09257v4.pdf

## 4. 성능 향상 및 한계  
### 4.1 성능 비교  
- **Tabular 데이터셋**: SOS, Neural AF, Splines 흐름은 전통적 affine 흐름 대비 대규모 우수한 LL 향상[1].  
- **이미지 데이터셋**: Flow++(혼합 CDF+비균등 dequantization) 모델이 최첨단 성능 달성[1].  

### 4.2 한계 및 일반화 성능  
- 모델 구조와 자코비안 계산 효율 사이 **표현력-연산복잡도 트레이드-오프**가 존재한다.  
- **가역성 제약**(Lipschitz, invertibility)은 네트워크 설계 자유도를 제한하며, 과적합 위험을 감소시켜 일반화에 긍정적 역할 가능성 지님[1].  
- **유니버설리티**(보편근사성): Monotone 네트워크, 다항식, 스플라인 변환으로 무한 근사 가능함이 이론적으로 증명되어 모델 일반화 능력의 잠재력을 뒷받침[1].  
- **비유클리드 공간**·**이산 변수** 적용 등 아직 실험적 검증이 미흡하며, 범용 흐름 설계가 필요하다.  

## 5. 향후 연구 영향 및 고려사항  
- **비유클리드·이산 분포**로의 확장 연구: 토러스·구면·리만 다양체 위 흐름, 전체 이산 범위 흐름 설계가 핵심 과제.  
- **조건부 흐름**(Conditional NF): 메모리·데이터 효율적인 conditioner 설계 연구 필요.  
- **손실 함수 다변화**: KL 대신 Wasserstein, JS divergence 등 대체 목표로 일반화 강인성 확보.  
- **모델 정규화**: Jacobian 정규화, 경로 복잡도 페널티 등으로 오버핏 감소 및 안정적 학습 유도.  

이 논문은 정규화 흐름 연구의 **이론적 통합**과 **실용적 가이드라인**을 제공하며, 향후 **모델 일반화**와 **분포 확장**을 위한 로드맵을 제시한다[1].  

[1] Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker, “Normalizing Flows: An Introduction and Review of Current Methods,” IEEE TPAMI, arXiv:1908.09257v4, 2020.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/dae79a1b-62cf-4fe3-90e8-05bb1a759f56/1908.09257v4.pdf
