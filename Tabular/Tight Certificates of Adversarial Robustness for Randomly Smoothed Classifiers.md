# Tight Certificates of Adversarial Robustness for Randomly Smoothed Classifiers

**핵심 주장 및 주요 기여 (간결 요약)**  
이 논문은 **무작위 입력 평활화(randomized smoothing)** 기법을 사용한 분류기들에 대해, Gaussian 노이즈뿐 아니라 다양한 노이즈 분포 및 거리(metric)에 대해 **타이트(tight)** 한 견고성 증명(certificates)을 제시한다. 기존에는 연속 공간에서 ℓ₂ 거리와 정규분포에 국한되었으나, 본 연구는  
- **광범위한 분포**(균등분포, 이산분포 등) 및 **다양한 거리**(ℓ₂, ℓ₀ 등)에 대한 일반적 증명 프레임워크를 제안,  
- **이산 공간**(binary vector)에 대해 ℓ₀ 거리 기반 타이트 증명을 최초로 제시,  
- 결정 트리와 같은 특정 모델군에 추가 가정(함수 클래스 제약)을 도입해 **더욱 강화된 증명**을 도출함으로써,  
- 상태-of-the-art 수준의 **인증된(certified) 정확도**를 이미지·분자 데이터셋에서 달성한다는 점이 주요 기여이다.[1]

***

## 1. 문제 정의  
강력한 분류기라도 입력에 작은 교란(adversarial perturbation)이 가해지면 잘못 분류될 수 있다.  
- 기존의 **찾기 기반(attack-based)** 검증은 반례 미발견이 곧 안전을 보장하지 않으며,  
- **인증 기반(certificate-based)** 연구는 특정 반경 내에 어떠한 교란도 잘못된 예측을 하지 않음을 보장하지만,  
  - 정확한(exact) 인증은 대부분 NP-complete이며,  
  - 실용적 규모의 신경망에 적용 시 너무 보수적(conservative)이다.[1]

이 논문은 **입력 랜덤화(input randomization)** 기법을 활용해, 무작위로 노이즈를 더한 후 투표(ensemble) 예측을 하는 **randomized smoothing** 분류기에 대해, 확률-통계적 분석으로 **타이트한 인증 반경**을 계산하는 방법을 제시한다.

***

## 2. 제안 방법  

### 2.1. 일반 프레임워크  
1. **Point-wise certificate**  
   - 두 지점 $$x, x'$$에 대해 분류 확률 $$p = \Pr[f(x+\delta)=y]$$가 주어지면,  
   - 최소화 문제  

$$
     \min_{f \in \mathcal{F}} \Pr[f(x'+\delta)=y]
     \quad\text{s.t.}\quad
     \Pr[f(x+\delta)=y] = p
     $$
     
  를 풀어 $$x'$$에서의 최저 분류 확률을 얻는다.[1]

2. **Regional certificate**  
   - 주어진 확률 $$p$$를 유지하면서, $$\|x' - x\|_q \le r$$인 모든 이웃점을 보장하는 최대 $$r$$을 찾음:  

$$
     R_{x,p,q} = \sup \{\,r : \min_{\|x'-x\|_q \le r} \Pr[f(x'+\delta)=y] \ge 0.5\}.
     $$

### 2.2. 다양한 분포로의 일반화  
- **균등분포(uniform noise)**: 공간 분할(region partition)을 이용해 타이트 인증 반경 유도(예, Eq.4~Prop.1).[1]
- **이산 분포(discrete noise)**:  
  - 입력을 binary vector $$ \{0,1\}^d $$로 보고, 각 차원을 Bernoulli+uniform 혼합으로 무작위 변환.  
  - likelihood ratio가 일정한 영역 $$L_i$$로 분할하고, 레밸런싱 기법(Lemma 2)으로 효율적 계산.[1]
  - ℓ₀ 거리 반경 $$r$$에 대한 타이트 인증 반경은, 랭크 순 정렬 후 그 역함수를 구해 결정 (Alg. 1).[1]

### 2.3. 모델 제약에 따른 추가 타이트닝  
- 결정 트리(decision tree) 가정하에, 각 노드에서 단일 특성만 분할하도록 제한.  
- 노드별 방문 확률을 재귀적으로 계산(Eq. 8)하여, ℓ₀ adversary를 동적 프로그래밍으로 정확히 찾음.  
- 이를 통해, 분포-무관 인증보다 **더 깐깐한(classifier-aware)** 인증이 가능하다.[1]

***

## 3. 성능 향상 및 한계  

### 3.1. 성능 향상  
- **MNIST** (CNN 모델):  
  - 평균 인증 반경 $$R$$이 Gaussian 기반 대비 약 1.7 증가,  
  - Certified accuracy 갭이 최대 0.4까지 확대.[1]
- **ImageNet** (ResNet50):  
  - ℓ₀ 반경 $$r=1$$에서 0.538 vs. 0.372 (Gaussian)로 대폭 향상.[1]
- **분자 바이너리 특징(BACE)**:  
  - 무작위 평활화 결정 트리 vs. vanilla 결정 트리 비교 시, robust AUC가 전 영역에서 우수.[1]

### 3.2. 한계  
- **계산 복잡도**:  
  - 큰 차원 $$d$$와 높은 $$r$$에서 영역 개수 $$n$$은 $$O(d^2)$$이나, 이산 ℓ₀ 역함수를 대규모 정수 연산으로 처리해야 해, ImageNet 스케일에서는 파라미터별·반경별로 수일의 전처리 시간이 필요.[1]
- **보수성**:  
  - 모델-agnostic 인증은 여전히 충분조건만 제공하여, ℓ₀=2 일 때 실제 Certified accuracy(0.926) 대비 인증값(0.774)이 보수적임.[1]

***

## 4. 일반화 성능 향상 가능성  
- **노이즈 분포 선택**: Gaussian 외 Uniform·Discrete 등 다양한 분포로 확장함에 따라, 실제 데이터 특성·교란 유형에 맞춰 노이즈를 설계하면 더 높은 인증 반경 확보 가능.  
- **모델 제약 활용**: 결정 트리 외에도, *필터 제약* 또는 *스파스 구조* 등을 가진 네트워크에서 유사한 dynamic programming 기법 적용 시, 인증 tightness를 더 높일 수 있음.  
- **하이브리드 접근**: 연속+이산 혼합 분포나, 노이즈 스케일 조절(heteroscedastic noise) 등으로 **일반화 성능** 및 **인증 신뢰도**를 동시에 최적화할 여지 존재.

***

## 5. 향후 연구 영향 및 고려 사항  
- **범용 인증 프레임워크**: 다양한 분포·거리 체계로 확장된 이론은, adversarial defense 연구에서 표준적 기법으로 자리잡을 가능성이 크다.  
- **실용적 가속화**: 대규모 정수 연산과 영역 분할의 계산 비용을 줄이는 **근사 알고리즘** 또는 **병렬화 기술** 개발이 필요하다.  
- **모델-특화 인증**: 네트워크 구조나 훈련 방법(예: self-ensemble, denoising layers)에 대한 사전 지식을 활용해, 더 정밀한 **모델-aware certificates** 연구가 유망하다.  
- **응용 도메인 확장**: 이산 데이터(텍스트, 그래프)나 멀티모달 입력 환경에서도 타이트 인증 적용 가능성을 탐색할 필요가 있다.  

이상의 성과는 adversarial robustness 분야에 **이론적 통일성**과 **실용적 성능** 양쪽을 크게 진전시키는 기틀을 제시한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e8361b43-1003-4141-8392-bbd6a7882443/NeurIPS-2019-tight-certificates-of-adversarial-robustness-for-randomly-smoothed-classifiers-Paper.pdf)
