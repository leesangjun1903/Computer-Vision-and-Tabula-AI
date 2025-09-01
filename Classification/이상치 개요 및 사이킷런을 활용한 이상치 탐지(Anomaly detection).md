# 이상치 탐지(Anomaly Detection) 완벽 가이드: 사이킷런으로 시작하는 머신러닝

## 이상치 탐지란 무엇인가?

이상치 탐지(Anomaly Detection)는 데이터에서 **일반적인 패턴과 다른 비정상적인 패턴을 찾아내는 기법**입니다. 이런 비정상적인 데이터를 이상치(Anomaly) 또는 아웃라이어(Outlier)라고 부릅니다.[^1]

머신러닝과 딥러닝 모델의 성능은 이런 이상치 데이터에 크게 영향을 받습니다. 때문에 데이터 전처리 과정에서 이상치를 적절히 처리하는 것이 매우 중요합니다.[^1]

## 이상치 탐지의 실제 활용 분야

이상치 탐지는 다양한 실제 상황에서 널리 활용되고 있습니다:

- **사기 탐지**: 비정상적인 금융 거래 패턴 식별[^2][^3]
- **침입 탐지**: 네트워크 보안에서 비정상적인 접근 시도 감지[^3][^4]
- **의료 분야**: 환자 데이터에서 이상 징후 탐지[^4][^2]
- **제조업**: 설비 고장 예측 및 품질 관리[^5][^6]
- **IoT 보안**: 스마트 기기의 비정상적인 동작 감지[^4]


## 사이킷런의 3가지 핵심 이상치 탐지 모델

사이킷런에는 이상치 탐지를 위한 여러 모델이 준비되어 있습니다. 오늘은 가장 널리 사용되는 3가지 모델을 살펴보겠습니다.[^1]

### 1. EllipticEnvelope: 타원 기반 탐지

EllipticEnvelope는 **정규 분포를 가정하고 데이터 분포에 타원을 그려서** 이상치를 탐지합니다. 타원에서 벗어날수록 이상치로 판단하는 방식입니다.[^1]

**특징:**

- 가우스 분산 데이터에 적합
- 정규 분포 가정 하에서 효과적
- 비교적 간단한 구현

**코드 예제:**

```python
from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.1, random_state=42)
clf.fit(X_train)
y_pred_outliers = clf.predict(X_outliers)
```


### 2. LocalOutlierFactor (LOF): 지역 밀도 기반 탐지

LOF는 **해당 관측치의 주변 데이터를 이용하여 국소적 관점으로** 이상치 정도를 파악합니다. 각 데이터 포인트의 지역적 밀도를 비교하여 이상치를 식별하는 방법입니다.[^1][^7][^8]

**특징:**

- 클러스터 밀도가 다른 데이터에 효과적
- 지역적 이상치 탐지에 강점
- 복잡한 데이터 분포에 적합[^7]

**코드 예제:**

```python
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
y_pred_outliers = clf.predict(X_outliers)
```


### 3. IsolationForest: 고립 기반 탐지

IsolationForest는 **의사결정 트리 기반의 이상탐지 기법**으로, 이상치가 정상 데이터보다 쉽게 고립된다는 원리를 활용합니다.[^1][^7][^8]

**특징:**

- 다차원 데이터에서 효율적[^1]
- 큰 데이터셋에서 빠른 처리 속도
- 메모리 효율적[^9]
- 라벨이 없는 비지도 학습 방식[^9]

**코드 예제:**

```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train)
y_pred_outliers = clf.predict(X_outliers)
```


## 실제 구현해보기: 단계별 가이드

### 1단계: 필요한 라이브러리 import

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
```


### 2단계: 실험용 데이터 생성

```python
# 랜덤 시드 설정으로 재현 가능한 결과 생성
rng = np.random.RandomState(42)

# 정상 훈련 데이터 생성 (2개 클러스터)
X_train = 0.2 * rng.randn(1000, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns=['x1', 'x2'])

# 테스트 데이터 생성
X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns=['x1', 'x2'])

# 이상치 데이터 생성
outliers = rng.uniform(low=-1, high=5, size=(50, 2))
outliers = pd.DataFrame(outliers, columns=['x1', 'x2'])
```


### 3단계: 데이터 시각화

```python
plt.scatter(X_train.x1, X_train.x2, c='white', s=80, 
           edgecolor='k', label='training observations')
plt.scatter(outliers.x1, outliers.x2, c='red', s=80, 
           edgecolor='k', label='new abnormal obs.')
plt.legend(loc='upper right')
plt.show()
```


## 모델 성능 비교 및 분석

![세 가지 이상치 탐지 모델의 성능 비교: 테스트 정확도와 이상치 탐지 정확도](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/c15a19f53be5905b295ad6535ffd701b/80ee5f06-71b4-4be0-a0fe-aecdde773b87/13b1d13d.png)

세 가지 이상치 탐지 모델의 성능 비교: 테스트 정확도와 이상치 탐지 정확도

위 차트는 세 모델의 성능을 비교한 결과입니다. **IsolationForest가 이상치 탐지에서 98%로 가장 높은 정확도**를 보였으며, **LocalOutlierFactor가 테스트 데이터에서 93.5%로 가장 좋은 성능**을 나타냈습니다.[^1]

### 성능 분석 결과

| 모델 | 테스트 정확도 | 이상치 탐지 정확도 | 특징 |
| :-- | :-- | :-- | :-- |
| EllipticEnvelope | 90.75%[^1] | 82%[^1] | 정규분포 가정, 빠른 처리 |
| LocalOutlierFactor | 93.5%[^1] | 96%[^1] | 지역 밀도 기반, 복잡한 패턴 |
| IsolationForest | 91.75%[^1] | 98%[^1] | 트리 기반, 대용량 데이터 |

## 실무에서 고려해야 할 중요한 점들

### 1. Contamination 파라미터 설정

모든 모델에서 **contamination 파라미터는 데이터 내 이상치 비율**을 의미합니다. 실제로는 이상치 비율을 모르는 경우가 대부분이므로, 도메인 지식을 바탕으로 적절히 설정해야 합니다.[^1]

### 2. 모델 출력 해석

사이킷런의 이상치 탐지 모델은 다음과 같이 출력합니다:[^1]

- **1**: 정상 데이터
- **-1**: 이상치


### 3. 모델 선택 가이드

**EllipticEnvelope 추천 상황:**

- 데이터가 정규분포를 따르는 경우
- 빠른 처리 속도가 필요한 경우
- 단순한 데이터 구조

**LocalOutlierFactor 추천 상황:**

- 클러스터 밀도가 다양한 경우
- 지역적 이상치 탐지가 중요한 경우
- 복잡한 데이터 분포

**IsolationForest 추천 상황:**

- 대용량 데이터 처리
- 고차원 데이터
- 메모리 효율성이 중요한 경우[^9]


## 최신 트렌드와 발전 방향

### 딥러닝과의 결합

최근에는 **AutoEncoder를 활용한 이상치 탐지**가 주목받고 있습니다. 특히 이미지나 시계열 데이터에서 복잡한 패턴을 학습할 수 있어 전통적인 방법보다 더 정교한 탐지가 가능합니다.[^10][^11]

### 온라인 학습과 실시간 탐지

**OML-AD(Online Machine Learning for Anomaly Detection)** 같은 기법은 데이터 스트림에서 실시간으로 이상치를 탐지할 수 있어, IoT나 금융 거래 모니터링에서 활용도가 높아지고 있습니다.[^11]

### 앙상블 방법

**FusionNet 같은 앙상블 모델**은 여러 알고리즘을 결합하여 98.5% 이상의 높은 정확도를 달성하고 있습니다. 단일 모델의 한계를 극복하는 효과적인 방법으로 주목받고 있습니다.[^3]

## 실습 과제 및 추가 학습 방향

### 실습 과제

1. **다양한 contamination 값으로 실험**: 0.05, 0.1, 0.15로 설정하여 결과 비교
2. **실제 데이터셋 적용**: 공개 데이터셋(예: credit card fraud)에서 성능 비교
3. **하이퍼파라미터 튜닝**: GridSearchCV를 사용한 최적 파라미터 찾기

### 추가 학습 자료

- 사이킷런 공식 문서의 이상치 탐지 섹션[^12]
- 시계열 데이터의 이상치 탐지[^11][^13]
- 딥러닝 기반 이상치 탐지 논문 리뷰


## 마무리

이상치 탐지는 **데이터 품질 개선과 머신러닝 성능 향상의 핵심**입니다. 사이킷런이 제공하는 세 가지 모델은 각각 고유한 장점이 있으므로, 데이터 특성과 비즈니스 요구사항에 맞게 선택하는 것이 중요합니다.

실제 프로젝트에서는 여러 모델을 함께 사용하거나 앙상블 방법을 적용하여 더 robust한 결과를 얻을 수 있습니다. 지속적인 실습과 최신 연구 동향 파악을 통해 이상치 탐지 역량을 키워나가시길 바랍니다.
<span style="display:none">[^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31]</span>

<div style="text-align: center">⁂</div>

[^1]: https://panggu15.github.io/outlier/Anomaly-detection/

[^2]: https://www.jmir.org/2019/5/e11030/

[^3]: https://ieeexplore.ieee.org/document/10415174/

[^4]: https://www.nature.com/articles/s41598-024-56126-x

[^5]: https://journals.riverpublishers.com/index.php/DGAEJ/article/view/17497

[^6]: https://link.springer.com/10.1007/978-3-031-10464-0_11

[^7]: https://spotintelligence.com/2023/08/07/outlier-detection-in-machine-learning/

[^8]: https://www.geeksforgeeks.org/machine-learning/comparing-anomaly-detection-algorithms-for-outlier-detection-on-toy-datasets-in-scikit-learn/

[^9]: https://www.ibm.com/think/topics/machine-learning-for-anomaly-detection

[^10]: https://www.doit.com/anomaly-detection-with-machine-learning-techniques-and-applications/

[^11]: https://arxiv.org/html/2409.09742v1

[^12]: https://scikit-learn.org/stable/modules/outlier_detection.html

[^13]: https://neptune.ai/blog/anomaly-detection-in-time-series

[^14]: https://linkinghub.elsevier.com/retrieve/pii/S0957417424015458

[^15]: https://linkinghub.elsevier.com/retrieve/pii/S1474706524000160

[^16]: https://link.springer.com/10.1007/s41060-024-00593-y

[^17]: https://www.intechopen.com/books/security-and-privacy-from-a-legal-ethical-and-technical-perspective/machine-learning-applications-in-misuse-and-anomaly-detection

[^18]: https://www.mdpi.com/2079-9292/13/11/2148

[^19]: https://arxiv.org/pdf/2201.07284.pdf

[^20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10923980/

[^21]: https://arxiv.org/pdf/2502.18601.pdf

[^22]: http://arxiv.org/pdf/2106.05410v2.pdf

[^23]: https://arxiv.org/pdf/2303.17354.pdf

[^24]: http://arxiv.org/pdf/2302.10753.pdf

[^25]: https://arxiv.org/pdf/2402.08975.pdf

[^26]: https://arxiv.org/html/2503.13195v1

[^27]: https://arxiv.org/pdf/2301.00134.pdf

[^28]: https://www.mdpi.com/2076-3417/12/16/8085/pdf?version=1660750399

[^29]: https://www.sciencedirect.com/science/article/pii/S0031320317303916

[^30]: https://arxiv.org/abs/2106.08779

[^31]: https://www.datrics.ai/articles/anomaly-detection-definition-best-practices-and-use-cases



# EllipticEnvelope
EllipticEnvelope는 통계적 방법 중 하나로, 데이터에서 이상치를 식별하는 데 사용됩니다. 이 기법은 다변량 정규 분포를 기반으로 하여 데이터의 중심을 기준으로 타원형 경계(엽스의 형태)를 생성합니다. 이를 통해 데이터 포인트가 이 경계 외부에 위치할 경우 이상치로 간주할 수 있습니다.

주요 특징은 다음과 같습니다:

정상 분포 가정: 데이터가 정규 분포를 따른다고 가정합니다.
노이즈 제거: 이상치를 감지하여 노이즈를 줄입니다.
다변량 데이터 처리: 여러 변수 간의 관계를 동시에 고려합니다.
이 방법은 클러스터링, 데이터 전처리 등 다양한 분야에서 활용될 수 있습니다. 숫자 데이터 분석에 매우 유용합니다. 

# LocalOutlierFactor
Local Outlier Factor (LOF)는 데이터 포인트의 이상치(outlier) 여부를 판단하는 알고리즘으로, 국소적 밀도 차이를 통해 이상치를 탐지합니다. LOF는 다음과 같은 특징을 가지고 있습니다:

밀도 기반 탐지: 주변 데이터 포인트의 밀도와 비교하여 이상치를 식별합니다.
주변 이웃: 특정 포인트의 k-이웃을 분석하여 그 포인트의 밀도를 평가합니다.
비교적 평가: 다른 데이터 포인트와의 밀도 차이를 통해 이상치 순위를 부여합니다.
이러한 방법을 통해 LOF는 다양한 데이터 세트에서 유용하게 사용될 수 있습니다.

# IsolationForest
Isolation Forest는 이상치 탐지를 위한 기계 학습 알고리즘입니다. 이 알고리즘은 데이터 포인트들이 얼마나 쉽게 "격리"될 수 있는지를 기반으로 이상치를 탐지합니다.

주요 특징은 다음과 같습니다:

격리 개념: 무작위로 선택한 피처와 값을 기준으로 데이터를 분할하여, 이상치는 일반적으로 정상 데이터보다 적은 분할로 격리됩니다.

효율성: 대규모 데이터에도 적합하며, 상대적으로 빠르게 작동합니다.

비모수적: 데이터의 분포에 대한 가정이 필요하지 않아서 다양한 데이터 유형에 적용할 수 있습니다.

이 알고리즘은 특히 신용 카드 사기 탐지, 네트워크 공격 탐지 등에서 유용하게 사용됩니다. 

# Reference
- https://panggu15.github.io/outlier/Anomaly-detection/
