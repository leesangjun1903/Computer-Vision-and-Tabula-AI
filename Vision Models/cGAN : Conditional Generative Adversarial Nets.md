# Conditional Generative Adversarial Nets | Image generation

## 핵심 주장과 주요 기여

이 논문은 기존 GAN의 한계인 생성 과정의 통제 불가능성을 해결하기 위해 **조건부 정보(conditional information)**를 도입한 cGAN을 제안한다. 주요 기여는 다음과 같다:[1]

```
## 1. 조건부 정보의 예시

**클래스 레이블**: MNIST 실험에서 사용된 숫자 클래스 (0-9)가 대표적인 예시다. 이는 원-핫 벡터로 인코딩되어 Generator와 Discriminator에 추가 입력으로 제공된다.
예를 들어, "7"이라는 숫자를 생성하고 싶을 때 해당 클래스에 대한 원-핫 벡터를 조건으로 사용한다.[1]

**이미지**: 이미지-투-이미지 변환에서 소스 이미지가 조건으로 사용된다. 예를 들어, 흑백 이미지를 컬러 이미지로 변환하는 작업에서 흑백 이미지가 조건부 정보가 된다.[2][3]

**텍스트**: 논문의 다중모달 실험에서 이미지 특성(4096차원 CNN 특성)을 조건으로 하여 텍스트 태그를 생성했다.
또한 최근 연구에서는 자연어 설명을 조건으로 하여 이미지를 생성하는 방식도 활용되고 있다.[1][2]

```

**1. 제어 가능한 생성 메커니즘**: Generator와 Discriminator 모두에 조건 정보 y(클래스 레이블, 이미지, 텍스트 등)를 추가 입력으로 제공하여 원하는 특성을 가진 데이터 생성을 가능하게 했다.[1]

**2. 간단하고 효과적인 구조**: 기존 GAN 아키텍처에 최소한의 수정만으로 조건부 생성이 가능한 프레임워크를 제시했다.[1]

**3. 다중 모달 학습**: 이미지와 텍스트 태그를 연결하는 다중 모달 학습을 통해 학습 데이터에 없던 설명적 태그 생성이 가능함을 보였다.[1]

```
## 2. 다중 모달 학습과 발전 가능성

**다중 모달인 이유**: 논문에서는 이미지와 텍스트라는 서로 다른 두 모달리티를 연결하여 학습한다.
ImageNet으로 사전 훈련된 CNN에서 추출한 4096차원 이미지 특성과 Skip-gram으로 학습한 200차원 단어 벡터를 함께 사용하여, 이미지를 조건으로 해당하는 설명적 태그를 생성한다.[1]

**다른 형태로의 발전 가능성**:
- **음성-텍스트**: 음성 신호를 조건으로 하여 텍스트를 생성하거나, 그 반대 작업[4]
- **3D 형상-2D 이미지**: 3D 모델을 조건으로 하여 다양한 각도의 2D 이미지 생성[5]
- **생체신호-의료 이미지**: ECG 신호를 조건으로 하여 심장 이미지 생성[6][7]
- **분자 구조-물성**: 분자의 거친 입자(coarse-grained) 표현을 조건으로 하여 세밀한 원자 단위 구조 생성[8]
```


## 해결하고자 하는 문제

**무조건부 생성의 한계**: 기존 GAN은 생성되는 데이터의 모드(mode)를 제어할 수 없어, 특정 클래스나 속성을 가진 데이터를 의도적으로 생성하기 어려웠다. 예를 들어, MNIST 데이터셋에서 특정 숫자(예: '7')를 생성하고 싶어도 랜덤하게 생성될 뿐이었다.[1]

**확률적 일대다 매핑의 필요성**: 실제 많은 문제들은 하나의 입력에 대해 여러 가능한 출력이 존재하는 확률적 매핑 특성을 가지고 있다. 특히 이미지 라벨링에서는 동일한 이미지에 대해 서로 다른(하지만 유의미한) 태그들이 붙을 수 있다.[1]

```
## 3. 확률적 일대다 매핑의 활용

논문에서 확률적 일대다 매핑은 **이미지 태깅 작업**에서 구현되었다.
동일한 이미지에 대해 서로 다른 사용자가 다양한(하지만 의미적으로 유사한) 태그를 붙일 수 있는 상황을 모델링했다.[1]

예를 들어, 기차역 이미지에 대해:
- 실제 사용자 태그: "montanha, trem, inverno, frio"
- 생성된 태그: "taxi, passenger, line, transportation, railway station"

이처럼 하나의 이미지 조건에 대해 여러 가능한 태그 조합이 생성될 수 있으며, 각각은 확률적으로 선택된다.[1]
```

## 제안하는 방법 및 수식

**수학적 정식화**: 기존 GAN의 목적함수가 다음과 같았다면:

$$ \min_G \max_D V(D, G) = \mathbb{E}\_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

cGAN은 조건부 확률분포를 학습하도록 확장된다:

$$ \min_G \max_D V(D, G) = \mathbb{E}\_{x \sim p_{data}(x)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|y)))] $$

여기서 y는 조건 정보(클래스 레이블 등)이다.[1]

**구현 방식**: Generator에서는 노이즈 벡터 z와 조건 y를 결합된 숨은 표현(joint hidden representation)으로 처리하고, Discriminator에서는 x와 y를 함께 입력받아 판별을 수행한다.[1]

```
## 4. 숨은 표현(Joint Hidden Representation)의 의미

숨은 표현은 **서로 다른 입력들을 통합하여 만든 중간 특성 표현**을 의미한다. cGAN에서는 노이즈 벡터 z와 조건 정보 y를 각각 별도의 숨은 층으로 매핑한 후, 이들을 결합하여 하나의 통합된 표현을 만든다.[9][10][1]

구체적으로 MNIST 실험에서:
- 노이즈 z: 100차원 → ReLU → 200차원
- 조건 y: 원-핫 벡터 → ReLU → 1000차원  
- **결합된 숨은 표현**: 두 벡터를 연결하여 1200차원 ReLU 층으로 처리

이는 단순히 입력을 연결하는 것보다 더 풍부한 상호작용을 가능하게 한다.[11][12]

```

## 모델 구조

**Generator 구조 (MNIST 실험)**:
- 100차원 균등분포에서 노이즈 z 샘플링
- z는 ReLU를 통해 200차원으로 매핑
- y는 ReLU를 통해 1000차원으로 매핑  
- 두 표현을 연결하여 1200차원 ReLU 층으로 처리
- 최종적으로 784차원(28×28) MNIST 이미지 생성[1]

**Discriminator 구조**:
- x는 240개 유닛, 5개 조각의 maxout 층으로 매핑
- y는 50개 유닛, 5개 조각의 maxout 층으로 매핑
- 결합된 표현은 240개 유닛, 4개 조각의 maxout 층 처리 후 시그모이드로 판별[1]

```
## 5. Maxout 층 사용 이유와 최신 동향

**Maxout 층 사용 이유**:[13][11]
1. **범용 근사기**: Maxout 네트워크는 충분히 많은 유닛이 있으면 어떤 함수든 근사할 수 있다
2. **그래디언트 흐름 개선**: ReLU와 달리 모든 Maxout 유닛을 통해 그래디언트가 흐르므로 vanishing gradient 문제를 완화한다[14]
3. **Dropout과의 호환성**: 모델 평균화(model averaging) 관점에서 Dropout과 잘 작동한다[11]
4. **활성화 다양성**: 여러 선형 함수 중 최댓값을 선택하여 더 복잡한 활성화 패턴을 학습할 수 있다

**최신 연구의 발전**:
- **잔차 연결(Residual Connections)**: ResNet 구조가 더 깊은 네트워크 훈련을 가능하게 함[15][16][17]
- **주의 메커니즘(Attention)**: Self-attention과 multi-head attention이 장거리 의존성 모델링에 효과적[18][2][6]
- **정규화 기법**: Batch normalization, Layer normalization, Instance normalization 등[19][20]
```

**다중모달 실험 구조**:
- ImageNet으로 사전 훈련된 CNN에서 4096차원 이미지 특성 추출
- Skip-gram 모델로 200차원 단어 벡터 생성
- Generator: 가우시안 노이즈(100차원) → 500차원 ReLU → 200차원 선형층
- Discriminator: 단어벡터용 500차원, 이미지특성용 1200차원 ReLU층 → 1000개 유닛 maxout층[1]

## 성능 향상 및 한계

**성능 개선**:
- MNIST에서 Parzen window 로그우도 추정값이 132±1.8로 기존 여러 방법들과 경쟁력 있는 성능을 보였다[2][1]

```
## 6. Parzen Window 로그우도 추정값

Parzen window 추정은 **생성된 샘플들로부터 확률밀도함수를 추정하는 방법**이다.[21][22]

**과정**:
1. 각 클래스에서 1000개 샘플 생성
2. 가우시안 커널을 사용하여 Parzen window 적용
3. 테스트 데이터에 대한 로그우도 계산

**문제점**:[21]
- 고차원 데이터에서는 매우 많은 샘플이 필요
- 실제 로그우도와 큰 차이 발생
- k-means 같은 단순한 방법이 더 높은 점수를 받을 수 있음

논문에서 cGAN의 132±1.8 점수는 "proof-of-concept" 수준으로, 성능 입증보다는 개념 검증 목적이었다.[23][1]
```

- 클래스별 제어 가능한 생성이 성공적으로 구현되었다[1]
- 다중모달 학습에서 학습 데이터에 없던 설명적 태그 생성이 가능했다[1]

**주요 한계**:
1. **모드 붕괴 문제**: cGAN은 기존 GAN보다 모드 붕괴에 더 취약하다. 특히 제한된 데이터에서 클래스 조건부 학습 시 심각한 모드 붕괴가 발생한다[3][4][5]

```
## 7. 모드 붕괴 해결책

**스펙트럴 정규화(Spectral Regularization)**:[24][25]
- 판별자의 가중치 행렬의 특이값 분포를 모니터링
- 스펙트럴 붕괴 현상을 방지하여 모드 붕괴 해결

**동적 클러스터링(Dynamic Clustering)**:[26]
- 생성 과정에서 동적으로 클러스터를 조정
- 다양한 모드의 균등한 탐색 유도

**HingeRLC-GAN**:[27]
- Hinge loss와 RLC 정규화 결합
- FID Score 18, KID Score 0.001 달성

**IID 샘플링 관점**:[28]
- 독립동일분포(IID) 가정을 유지하도록 정규화
- 목표 분포에 대한 IID 특성 보장으로 모드 붕괴 방지
```

2. **훈련 불안정성**: 조건부 정보 추가로 인해 Generator와 Discriminator 간 균형 맞추기가 더 어려워졌다[6][7][3]

3. **아키텍처의 단순함**: 2014년 논문으로서 MLP 구조만 사용하여 현재 기준으로는 제한적이다[8][9]

4. **성능의 한계**: 무조건부 GAN보다 성능이 낮게 나타나는 경우가 있어, 저자들도 "proof-of-concept" 수준이라고 언급했다[1]

## 일반화 성능 향상 가능성

**이론적 근거**: cGAN은 조건부 확률분포 P(X|Y)를 학습하여 보다 구체적이고 제약된 문제를 해결한다[10]. 이는 전체 데이터 분포 P(X)를 학습하는 것보다 각 조건별로 더 정확한 모델링이 가능하다는 장점이 있다[11].

**실제 개선 사례**:
1. **데이터 효율성**: 조건 정보를 활용하여 적은 데이터로도 효과적인 학습이 가능하다[12][13]
2. **전이 학습**: 기존 클래스의 지식을 새로운 클래스로 전파하는 메커니즘이 가능하다[13]
3. **정칙화 효과**: 조건 정보가 추가적인 제약으로 작용하여 과적합을 방지할 수 있다[14]

```
## 8. 정칙화 효과

정칙화는 **모델의 복잡성을 제한하여 과적합을 방지하는 효과**다. cGAN에서 조건부 정보가 추가적인 제약으로 작용한다:[29][30]

**작동 원리**:
- 조건 정보 y가 추가 구조적 제약 제공
- 전체 데이터 분포 P(X) 대신 조건부 분포 P(X|Y) 학습
- 각 조건별로 더 제한된 공간에서 학습

**구체적 효과**:
- **데이터 효율성**: 적은 데이터로도 효과적 학습[31]
- **일반화 성능**: 조건별로 더 focused한 학습으로 과적합 방지
- **안정성**: 조건이 학습 과정에서 guidance 역할
```

**근린 추정(Vicinal Estimation)**: 최근 연구에서는 cGAN의 일반화 오류가 출력 차원과 무관하게 개선될 수 있음을 이론적으로 증명했다.[10]

```
## 9. 근린 추정(Vicinal Estimation)

근린 추정은 **훈련 샘플 주변의 근린 분포를 고려하여 일반화 성능을 향상시키는 방법**이다.[32][33][4]

**핵심 아이디어**:
- 각 훈련 샘플을 점이 아닌 **작은 영역(vicinity)**으로 간주
- 근린 분포에서 가상 샘플들을 생성하여 데이터 증강 효과
- 경험적 위험 최소화 대신 근린 위험 최소화 수행
```

**수학적 표현**:[33]
기존 경험적 위험: $$R\_{emp}(f) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i), y_i) $$

근린 위험: $$R\_{vic}(f) = \frac{1}{n}\sum_{i=1}^n \int \ell(f(x), y_i)dP_{x_i}(x) $$

여기서 $$P\_{x_i}(x) $$는 $$x_i $$ 주변의 근린 분포다.

```
**cGAN에서의 적용**: 근린 추정을 통해 조건부 생성에서 일반화 오류를 출력 차원과 무관하게 개선할 수 있다.[34]

```

## 앞으로의 연구에 미치는 영향

**패러다임 변화**: 이 논문은 생성 모델에서 "제어 가능성"이라는 새로운 패러다임을 제시했다. 이후 StyleGAN, BigGAN, 조건부 확산 모델 등 수많은 후속 연구의 기초가 되었다.[15][16]

**주요 발전 방향**:

1. **아키텍처 혁신**: 
   - **StyleGAN**: 스타일 기반 생성으로 더 정교한 제어 실현[17][15]
   - **BigGAN**: 대규모 배치와 자기주의 메커니즘으로 품질 향상[15]
   - **조건부 정규화 플로우**: 모드 붕괴 문제 해결[18]

```
## 10. 조건부 정규화 플로우 모델 예시

**Noise Flow**: 이미지 노이즈 모델링을 위한 조건부 정규화 플로우[35]
- 이미지 신호에 조건부로 노이즈 분포 학습
- 실제 카메라 노이즈의 복잡한 특성 모델링

**조건부 분자 동역학**:[8]
- 거친 입자(coarse-grained) 표현을 조건으로 함
- 세밀한 원자 단위 분자 구조 생성
- 활성 학습으로 효율적인 배열 공간 탐색

**물리학 응용**:[36]
- 고에너지 물리학에서 재가중(reweighting) 기법 개선
- 몬테카르로 시뮬레이션 보정에 활용
- 조건부 정규화 플로우로 분포 간 매핑 학습
```

2. **응용 분야 확장**:
   - **의료**: 생체신호 데이터 증강[19][20]
   - **패션**: 조건부 의류 이미지 생성[21]
   - **금융**: 사기 탐지용 합성 데이터 생성[22]
   - **우주과학**: 암흑물질 매핑[6]

3. **이론적 발전**: 조건부 적대 네트워크의 수렴성과 일반화 이론이 지속적으로 발전하고 있다[23][24][10]

## 앞으로 연구 시 고려할 점

**1. 모드 붕괴 해결**: 조건부 학습에서 발생하는 모드 붕괴는 여전히 핵심 과제다. 점진적 조건 주입, 동적 클러스터링, 적대적 정규화 등의 기법 연구가 필요하다.[4][5][25][26]

**2. 훈련 안정성**: Wasserstein 거리, spectral normalization, 그래디언트 패널티 등을 활용한 안정적 훈련 기법 개발이 중요하다.[7][6]

**3. 평가 메트릭**: FID, IS 외에도 모드 붕괴를 정량적으로 측정할 수 있는 MCE(Mode Collapse Entropy) 같은 새로운 평가 지표 개발이 필요하다.[27]

```
## 11. MCE(Mode Collapse Entropy) 수식

MCE는 **모드 붕괴 정도를 정량화하는 새로운 메트릭**이다.[37]

```

**정의**:
$$MCE = a = \frac{n\hat{H} - 1}{n - 1} $$

여기서:
- $$n $$: 전체 모드(bin) 개수
- $$\hat{H} $$: 정규화된 섀넌 엔트로피
- $$a \in  $$: 모드 붕괴 정도[1]

**해석**:
- $$a = 1 $$: 모드 붕괴 없음 (모든 모드가 균등한 확률)
- $$a = 0 $$: 완전한 모드 붕괴 (하나의 모드만 존재)

```
**계산 과정**:
1. 생성된 샘플과 실제 데이터 간 1-NN 매칭
2. 각 실제 샘플에 연결된 생성 샘플 개수 계산
3. 정규화된 섀넌 엔트로피 계산
4. MCE 점수 도출

**장점**: 재훈련 불필요, 라벨 데이터 불필요, 해석이 용이함.[37]
```

**4. 확장성**: 대규모 데이터셋과 고해상도 이미지에 대한 효율적인 조건부 생성 방법 연구가 요구된다.[13]

**5. 윤리적 고려사항**: 얼굴 생성, 합성 데이터 오남용 등에 대한 윤리적 지침과 검증 방법이 필요하다.[28]

이 논문은 비록 초기 연구이지만, 생성 AI 분야에서 "조건부 생성"이라는 핵심 개념을 도입하여 현재까지도 활발히 연구되고 있는 기초를 마련했다는 점에서 매우 중요한 의미를 갖는다.

```
## 12. 조건부 생성 모델 발전 아이디어

**1. 계층적 조건부 생성**:
- 거친 조건에서 세밀한 조건으로 점진적 생성
- 다중 해상도 조건 정보 활용
- 예: 스케치 → 윤곽 → 세부사항 순서로 이미지 생성

**2. 동적 조건부 주의 메커니즘**:
- 생성 과정에서 조건 정보의 중요도를 동적 조정[2][18]
- 지역적 조건과 전역적 조건의 적응적 결합
- Cross-attention과 self-attention의 통합

**3. 메타 학습 기반 조건부 생성**:
- 새로운 조건 유형에 빠르게 적응[32]
- Few-shot 조건부 생성
- 조건 임베딩의 메타 학습

**4. 물리 법칙 제약 조건부 생성**:
- 물리적 일관성을 조건으로 하는 생성[8]
- 에너지 보존, 운동량 보존 등의 제약 조건
- 과학 시뮬레이션과의 결합

**5. 대화형 조건부 생성**:
- 사용자 피드백을 실시간으로 조건에 반영
- 점진적 조건 개선을 통한 iterative generation
- 강화 학습과의 결합으로 사용자 선호도 학습

이러한 발전 방향들은 조건부 생성 모델의 제어성, 품질, 다양성을 동시에 향상시킬 수 있는 잠재력을 가지고 있다.
```

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7eb24a31-cf62-4004-99dc-e3f401b42ce8/1411.1784v1.pdf
[2] https://www.scitepress.org/Papers/2024/129379/129379.pdf
[3] https://dl.acm.org/doi/10.1145/3664647.3680772
[4] https://openreview.net/forum?id=7TZeCsNOUB_
[5] https://arxiv.org/abs/2201.06578
[6] https://www.semanticscholar.org/paper/ef7f7dddc465c4d4cb8eedaacabf839b401f3468
[7] https://spotintelligence.com/2023/10/11/mode-collapse-in-gans-explained-how-to-detect-it-practical-solutions/
[8] https://happy-jihye.github.io/gan/gan-3/
[9] https://re-code-cord.tistory.com/entry/Conditional-Generative-Adversarial-Networks
[10] https://openreview.net/forum?id=EX7AxKgc46&noteId=tktTbgcYrT
[11] https://www.numberanalytics.com/blog/conditional-gans-theory-and-practice
[12] https://www.sciendo.com/article/10.2478/jaiscr-2024-0017
[13] https://openaccess.thecvf.com/content/CVPR2021/papers/Shahbazi_Efficient_Conditional_GAN_Transfer_With_Knowledge_Propagation_Across_Classes_CVPR_2021_paper.pdf
[14] https://arxiv.org/abs/2103.14884
[15] https://cvnote.ddlee.cc/2019/09/15/ProGAN-SAGAN-BigGAN-StyleGAN.html
[16] https://apxml.com/courses/generative-adversarial-networks-gans/chapter-2-advanced-gan-architectures
[17] https://www.lunit.io/company/blog/stylegan-a-style-based-generator-architecture-for-gans
[18] https://scipost.org/10.21468/SciPostPhys.16.5.132
[19] https://arxiv.org/abs/2206.13676
[20] https://ieeexplore.ieee.org/document/10678138/
[21] https://journalwjarr.com/sites/default/files/fulltext_pdf/WJARR-2025-0123.pdf
[22] https://ieeexplore.ieee.org/document/10825410/
[23] https://aclanthology.org/2021.adaptnlp-1.3/
[24] https://arxiv.org/abs/2102.10176
[25] https://arxiv.org/abs/1805.08657
[26] https://ieeexplore.ieee.org/document/10440507/
[27] https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Duym_Quantifying_Generative_Stability_Mode_Collapse_Entropy_Score_for_Mode_Diversity_WACVW_2025_paper.pdf
[28] https://ieeexplore.ieee.org/document/10761482/
[29] https://ieeexplore.ieee.org/document/9261705/
[30] https://ieeexplore.ieee.org/document/10575829/
[31] https://ieeexplore.ieee.org/document/10539484/
[32] https://dl.acm.org/doi/10.1145/3490035.3490275
[33] https://ieeexplore.ieee.org/document/10568093/
[34] https://ieeexplore.ieee.org/document/10864577/
[35] https://arxiv.org/pdf/1906.00709.pdf
[36] https://arxiv.org/pdf/2212.13589.pdf
[37] http://arxiv.org/pdf/2410.23108.pdf
[38] http://arxiv.org/pdf/2108.09016.pdf
[39] https://arxiv.org/pdf/2412.03105.pdf
[40] https://arxiv.org/abs/2106.15011
[41] http://arxiv.org/pdf/2003.06472.pdf
[42] https://arxiv.org/pdf/1907.05280.pdf
[43] https://arxiv.org/abs/1703.06029
[44] https://arxiv.org/html/2410.23108
[45] https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf
[46] https://www.diva-portal.org/smash/get/diva2:1722941/FULLTEXT01.pdf
[47] https://arxiv.org/html/2404.13500v1
[48] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002765235
[49] https://stevenkim1217.tistory.com/entry/%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC-Conditional-GAN-Conditional-Generative-Adversarial-Nets
[50] https://hhhhhsk.tistory.com/9
[51] https://www.sciencedirect.com/science/article/pii/S2214509525007855
[52] https://www.sciencedirect.com/science/article/pii/S2046043023000084
[53] https://www.sciencedirect.com/science/article/pii/S0098300422001613
[54] https://www.sciencedirect.com/science/article/pii/S0378778825002610/pdf
[55] https://nowolver.tistory.com/144
[56] https://arxiv.org/abs/2404.13500
[57] https://arxiv.org/abs/2210.07751
[58] https://ieeexplore.ieee.org/document/10859959/
[59] https://www.semanticscholar.org/paper/0c78a3cbf9b7627c1da63909b935ad278a73f750
[60] https://www.mdpi.com/2076-3417/14/2/579
[61] https://arxiv.org/pdf/1911.02996.pdf
[62] http://arxiv.org/pdf/1705.07215.pdf
[63] https://arxiv.org/abs/1804.04391
[64] https://arxiv.org/html/2503.19074v1
[65] https://arxiv.org/pdf/2411.11786.pdf
[66] http://arxiv.org/pdf/1802.02436.pdf
[67] http://arxiv.org/pdf/2405.20987.pdf
[68] https://arxiv.org/abs/1910.00927
[69] https://arxiv.org/pdf/2208.12055v1.pdf
[70] http://arxiv.org/pdf/1910.11626v1.pdf
[71] http://proceedings.mlr.press/v80/pan18c/pan18c.pdf
[72] https://soochanlee.com/publications/mr-gan.html
[73] https://ejournal.csol.or.id/index.php/csol/article/view/170/188
[74] https://openaccess.thecvf.com/content/ICCV2021/papers/Han_Dual_Projection_Generative_Adversarial_Networks_for_Conditional_Image_Generation_ICCV_2021_paper.pdf
[75] https://neptune.ai/blog/gan-failure-modes
[76] https://www.sciencedirect.com/science/article/pii/S0926580524000013
[77] https://arxiv.org/abs/2011.00835
[78] https://www.sciencedirect.com/science/article/abs/pii/S1568494624007774
[79] https://aodr.org/xml/34766/34766.pdf


# Conditional Generative Adversarial Nets: 상세 질문 답변

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7eb24a31-cf62-4004-99dc-e3f401b42ce8/1411.1784v1.pdf
[2] https://peerj.com/articles/cs-2332.pdf
[3] https://dl.acm.org/doi/10.1145/3638584.3638662
[4] https://arxiv.org/abs/2311.02025
[5] https://www.ijcai.org/proceedings/2023/0141.pdf
[6] https://pubmed.ncbi.nlm.nih.gov/38286104/
[7] https://ieeexplore.ieee.org/document/10678138/
[8] https://arxiv.org/html/2402.01195v2
[9] https://deepai.org/machine-learning-glossary-and-terms/hidden-representation
[10] https://aclanthology.org/N19-1046.pdf
[11] https://arxiv.org/pdf/1302.4389.pdf
[12] https://www.sri.com/wp-content/uploads/2021/12/paper_icassp15_wilson_final.pdf
[13] https://www.baeldung.com/cs/maxout-neural-networks
[14] https://github.com/xbeat/Machine-Learning/blob/main/Constructing%20MaxOut%20Neural%20Networks%20in%20Python.md
[15] https://arxiv.org/html/2404.10947v4
[16] https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
[17] https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Dual_Residual_Networks_Leveraging_the_Potential_of_Paired_Operations_for_CVPR_2019_paper.pdf
[18] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0312105
[19] https://linkinghub.elsevier.com/retrieve/pii/S0925231217314509
[20] https://www.sciencedirect.com/science/article/abs/pii/S0925231217314509
[21] https://arxiv.org/pdf/1511.01844.pdf
[22] https://stackoverflow.com/questions/65231792/evaluating-parzen-window-log-likelihood-for-gan
[23] https://hhhhhsk.tistory.com/9
[24] https://arxiv.org/abs/1908.10999
[25] https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Spectral_Regularization_for_Combating_Mode_Collapse_in_GANs_ICCV_2019_paper.pdf
[26] https://ieeexplore.ieee.org/document/10440507/
[27] https://arxiv.org/html/2503.19074v1
[28] https://www.ijcai.org/proceedings/2023/0437.pdf
[29] https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Self-Calibrating_Vicinal_Risk_Minimisation_for_Model_Calibration_CVPR_2024_paper.pdf
[30] http://arxiv.org/pdf/2407.05765.pdf
[31] https://www.sciendo.com/article/10.2478/jaiscr-2024-0017
[32] https://wei-ying.net/pubs/VRM_NeurIPS2022.pdf
[33] http://papers.neurips.cc/paper/1876-vicinal-risk-minimization.pdf
[34] https://openreview.net/forum?id=EX7AxKgc46&noteId=tktTbgcYrT
[35] https://openaccess.thecvf.com/content_ICCV_2019/papers/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.pdf
[36] https://scipost.org/preprints/scipost_202304_00025v1/
[37] https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Duym_Quantifying_Generative_Stability_Mode_Collapse_Entropy_Score_for_Mode_Diversity_WACVW_2025_paper.pdf
[38] https://mesopotamian.press/journals/index.php/bigdata/article/view/225
[39] https://arxiv.org/abs/2406.01766
[40] https://www.semanticscholar.org/paper/01dafc56df4bfe959cf8f6c6bb110e4f8a765249
[41] https://www.science.org/doi/10.1126/science.adi8474
[42] https://www.semanticscholar.org/paper/91e00e8d3ee222c1f608357e2ef5d9f92d6dbd67
[43] https://ieeexplore.ieee.org/document/10761380/
[44] https://ieeexplore.ieee.org/document/10699227/
[45] https://ieeexplore.ieee.org/document/10596957/
[46] https://dl.acm.org/doi/10.1145/3712335.3712397
[47] http://arxiv.org/pdf/1707.06838.pdf
[48] https://pmc.ncbi.nlm.nih.gov/articles/PMC6650051/
[49] https://pmc.ncbi.nlm.nih.gov/articles/PMC5519034/
[50] http://arxiv.org/pdf/1312.1909.pdf
[51] http://arxiv.org/pdf/1511.02583.pdf
[52] http://arxiv.org/pdf/2102.06358.pdf
[53] http://arxiv.org/pdf/2411.03006.pdf
[54] https://arxiv.org/ftp/arxiv/papers/2105/2105.01399.pdf
[55] http://arxiv.org/pdf/2002.04060.pdf
[56] https://arxiv.org/pdf/2002.09024.pdf
[57] https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0233-0
[58] https://proceedings.neurips.cc/paper/2021/file/f2c3b258e9cd8ba16e18f319b3c88c66-Paper.pdf
[59] https://kubig15-suhyeokjang.tistory.com/6
[60] https://spotintelligence.com/2023/10/11/mode-collapse-in-gans-explained-how-to-detect-it-practical-solutions/
[61] https://blog.outta.ai/128
[62] https://dhhwang89.tistory.com/27
[63] https://www.scitepress.org/PublishedPapers/2021/101679/101679.pdf
[64] https://www.youtube.com/watch?v=Y8ve6-ifp9c
[65] https://wikidocs.net/106899
[66] http://ieeexplore.ieee.org/document/5514043/
[67] https://www.semanticscholar.org/paper/cf76e260e4f552d10bcb1474317d4d6571a09353
[68] https://www.semanticscholar.org/paper/2569cb3562a864638d71da5da0ae3008d48f893f
[69] https://www.mdpi.com/2076-3417/13/8/5037
[70] https://arxiv.org/abs/2412.10454
[71] https://www.semanticscholar.org/paper/4dfd46388c8f8dd383daeb2f7e8f524053619671
[72] https://onlinelibrary.wiley.com/doi/10.1111/risa.17679
[73] http://www.aimspress.com/article/doi/10.3934/mbe.2021243
[74] https://www.nature.com/articles/s41598-020-60786-w
[75] http://arxiv.org/pdf/2305.14606.pdf
[76] https://aclanthology.org/2023.emnlp-main.248.pdf
[77] https://arxiv.org/pdf/2003.06566.pdf
[78] https://arxiv.org/pdf/1612.01663.pdf
[79] https://arxiv.org/pdf/2309.16527.pdf
[80] https://arxiv.org/pdf/1706.00182.pdf
[81] http://arxiv.org/pdf/2206.00439.pdf
[82] https://arxiv.org/pdf/2201.06487.pdf
[83] http://arxiv.org/pdf/1703.00593.pdf
[84] https://arxiv.org/pdf/2202.08832v1.pdf
[85] https://github.com/smsharma/jax-conditional-flows
[86] https://github.com/HaozheLiu-ST/MEE
[87] http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/chapelle.pdf
[88] https://neptune.ai/blog/gan-failure-modes
[89] https://dl.acm.org/doi/10.5555/3008751.3008809
[90] https://proceedings.mlr.press/v238/gong24a/gong24a.pdf
[91] https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html
[92] https://openaccess.thecvf.com/content/ICCV2021W/MELEX/papers/Bang_MGGAN_Solving_Mode_Collapse_Using_Manifold-Guided_Training_ICCVW_2021_paper.pdf
[93] https://vds.sogang.ac.kr/wp-content/uploads/2023/01/VDS_2022_%EC%97%AC%EB%A6%84%EC%84%B8%EB%AF%B8%EB%82%98_%EA%B9%80%EC%A4%80%EA%B7%9C.pdf
[94] https://www.sciencedirect.com/science/article/abs/pii/S0167865521003755
[95] https://github.com/VincentStimper/normalizing-flows/blob/master/examples/conditional_flow.ipynb
[96] http://ieeexplore.ieee.org/document/287116/
[97] https://ieeexplore.ieee.org/document/10108069/
[98] https://www.worldscientific.com/doi/10.1142/S0129065723500405
[99] https://ieeexplore.ieee.org/document/10650110/
[100] https://arxiv.org/abs/2406.00048
[101] https://www.ijcai.org/proceedings/2023/275
[102] https://www.mdpi.com/2072-4292/15/13/3357
[103] https://dl.acm.org/doi/10.1145/2911996.2912064
[104] https://www.semanticscholar.org/paper/d0185c37e7f3eea77b76d1dc0745d58678ef841e
[105] https://ieeexplore.ieee.org/document/10684734/
[106] https://arxiv.org/html/2410.03006
[107] https://arxiv.org/abs/2305.10468
[108] http://arxiv.org/pdf/2405.06409.pdf
[109] https://arxiv.org/html/2503.01824v1
[110] http://arxiv.org/pdf/2405.05097.pdf
[111] http://arxiv.org/pdf/2401.13558.pdf
[112] http://arxiv.org/pdf/2211.06137.pdf
[113] https://peerj.com/articles/cs-2140
[114] https://arxiv.org/pdf/2408.08381.pdf
[115] https://arxiv.org/abs/1604.03628
[116] https://michaelauli.github.io/papers/rnn-joint-lm-tm.pdf
[117] https://www.sciencedirect.com/science/article/abs/pii/S1746809424010152
[118] https://www.sciencedirect.com/science/article/abs/pii/S0169023X2200088X
[119] https://wikidocs.net/214487
[120] https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
[121] https://arxiv.org/html/2502.11934v1
[122] https://www.sciencedirect.com/science/article/pii/S2405844024126965
[123] https://www.sciencedirect.com/science/article/pii/S1687850724003418
[124] https://link.springer.com/article/10.1007/s10489-025-06628-6
