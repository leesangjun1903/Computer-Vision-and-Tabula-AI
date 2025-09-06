# 8 Ways to deal with Continuous Variables in Predictive Modeling


# 머신러닝에서 연속변수를 다루는 8가지 핵심 기법

딥러닝을 공부하다 보면 데이터 전처리의 중요성을 절감하게 됩니다. 특히 연속변수는 범주형 변수와 달리 처리 방법이 다양해서 초보자들이 어려워하는 부분입니다. 오늘은 연속변수를 효과적으로 처리하는 8가지 방법을 실제 예제와 함께 알아보겠습니다.[^1]

## 연속변수란 무엇인가요?

연속변수는 최솟값과 최댓값 사이의 어떤 값이든 가질 수 있는 변수입니다. 나이, 몸무게, 키처럼 우리 주변에서 흔히 볼 수 있는 데이터들이 대부분 연속변수입니다.[^1]

범주형 변수(성별, 혈액형 등)와 달리 연속변수는 처리 방법이 훨씬 복잡합니다. 예를 들어, 성별에 따른 스포츠 참여율은 단순히 남녀 비율만 보면 되지만, 나이에 따른 스포츠 참여율은 구간을 나누거나, 그래프를 그리거나, 변환을 하는 등 다양한 방법이 필요합니다.[^1]

## 1. 데이터 구간화 (Binning): 연속변수를 구간으로 나누기

비닝은 연속변수를 여러 그룹으로 나누는 기법입니다. 복잡한 패턴을 찾기 어려운 연속변수를 분석하기 쉬운 구간으로 만들어줍니다.[^1]

### 비닝의 종류

**Equal Width Binning (동일 구간)**

- 전체 범위를 동일한 크기로 나눕니다[^2][^3]
- 구현이 간단하고 해석이 쉽습니다
- 데이터가 균등하게 분포할 때 효과적입니다

**Equal Frequency Binning (동일 빈도)**

- 각 구간에 동일한 개수의 데이터가 들어가도록 나눕니다[^3][^4]
- 편중된 데이터에 더 효과적입니다
- 이상치에 덜 민감합니다

**Manual Binning (수동 구간화)**

- 도메인 지식을 활용해 직접 구간을 설정합니다[^4]
- 비즈니스 로직이 반영된 의미있는 구간을 만들 수 있습니다


### 실제 비닝 예제

```python
import pandas as pd
import numpy as np

# R의 state.x77 데이터와 유사한 예제
data = np.random.gamma(4, 8, 1000)  # 나이 데이터

# Equal Width Binning
bins_equal = pd.cut(data, bins=5, labels=['청소년', '청년', '중년', '장년', '노년'])

# Equal Frequency Binning  
bins_quantile = pd.qcut(data, q=5, labels=['1분위', '2분위', '3분위', '4분위', '5분위'])

print(bins_equal.value_counts())
print(bins_quantile.value_counts())
```

빈닝을 사용할 때는 정보 손실을 최소화하기 위해 처음에는 작은 구간을 만드는 것이 좋습니다. 하지만 너무 많은 구간은 오히려 복잡성만 증가시킬 수 있으므로 적절한 균형이 필요합니다.[^1]

## 2. 정규화 (Normalization): 동일한 스케일로 만들기

정규화는 서로 다른 스케일의 변수들을 동일한 범위로 맞춰주는 기법입니다. 특히 거리 기반 알고리즘(KNN, k-means 등)에서 필수적입니다.[^1][^5]

### Z-Score 정규화

가장 일반적인 정규화 방법으로, 평균을 빼고 표준편차로 나누어줍니다:[^1][^6]

\$ z = \frac{x - \mu}{\sigma} \$

여기서 x는 관측값, μ는 평균, σ는 표준편차입니다.

### 정규화 예제

Randy가 수학시험에서 76점(평균 70, 표준편차 2), Katie가 과학시험에서 86점(평균 80, 표준편차 3)을 받았다면:[^1]

- Randy의 Z-score: (76-70)/2 = 3
- Katie의 Z-score: (86-80)/3 = 2

Randy가 상대적으로 더 좋은 성적입니다. 이처럼 정규화를 통해 서로 다른 스케일의 데이터를 공정하게 비교할 수 있습니다.[^1]

## 3. 변환 (Transformation): 분포 모양 바꾸기

편중된 데이터는 그대로 사용하면 모델 성능이 떨어집니다. 변환을 통해 데이터 분포를 개선할 수 있습니다.[^1]

### 로그 변환

가장 일반적인 변환 방법으로, 오른쪽으로 편중된 데이터를 정규분포에 가깝게 만들어줍니다:[^1][^7][^8]

```python
# 로그 변환 예제
import numpy as np

# 편중된 수입 데이터
income = np.random.exponential(50000, 1000)

# 로그 변환 적용
income_log = np.log1p(income)  # log(1+x)로 0값 처리

print(f"원본 왜도: {skew(income):.3f}")
print(f"변환 후 왜도: {skew(income_log):.3f}")
```


### 기타 변환 방법

- **제곱근 변환**: 중간 정도의 편중에 효과적입니다[^8][^9]
- **Box-Cox 변환**: 최적의 변환 파라미터를 자동으로 찾아줍니다[^1][^9]
- **역수 변환**: 심각한 편중에 사용됩니다[^9]


## 4. 비즈니스 로직 활용하기

데이터만으로는 알 수 없는 도메인 지식을 활용하는 것이 중요합니다. 예를 들어, 항공업계 데이터를 다룬다면 계절성, 휴가철, 유가 변동 등을 고려해야 합니다.[^1]

### 피처 엔지니어링 예제

키와 몸무게 데이터가 있다면 BMI를 계산할 수 있습니다:[^1]

```python
# BMI 계산 예제
df['bmi'] = df['weight'] / (df['height']/100)**2
df['income_per_age'] = df['income'] / df['age']  # 연령 대비 수입
```

이처럼 기존 변수들을 조합해 더 의미있는 새로운 변수를 만들 수 있습니다.[^1]

## 5. 이상치 처리

이상치는 모델 성능에 큰 영향을 미칠 수 있습니다. 예를 들어, 나이가 200세로 입력되었다면 실제로는 200개월(약 16.7세)일 가능성이 높습니다.[^1]

### IQR 방법을 이용한 이상치 탐지

```python
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75) 
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

outliers = detect_outliers_iqr(df['income'])
print(f"이상치 개수: {outliers.sum()}개")
```

이상치 처리는 비즈니스 이해와 데이터 분석을 결합해서 신중하게 결정해야 합니다.[^1]

## 6. 주성분 분석 (PCA): 차원 축소하기

변수가 너무 많으면(100개, 200개 이상) 모델 훈련이 비효율적입니다. PCA는 많은 변수들을 적은 수의 주요 변수로 압축해주는 기법입니다.[^1]

### PCA의 원리

PCA는 데이터의 분산을 최대한 보존하면서 차원을 축소합니다. 첫 번째 주성분(PC1)이 가장 많은 정보를 담고, 두 번째 주성분(PC2)이 그 다음으로 많은 정보를 담습니다.[^5][^10]

![PCA 스크리 플롯: 주성분별 설명 분산 비율과 누적 분산](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/65bc30fb128e724a36698cb9607f44e6/48fe4531-ae39-462e-af4c-9e1b8bde37d2/36b7d932.png)

PCA 스크리 플롯: 주성분별 설명 분산 비율과 누적 분산

### PCA 적용 예제

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 표준화 후 PCA 적용
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# 설명 분산 비율 확인
explained_variance = pca.explained_variance_ratio_
print("주성분별 설명 분산:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.3f}")
```

일반적으로 고유값(eigenvalue)이 1보다 큰 주성분들을 선택합니다. 95% 이상의 분산을 설명하는 주성분들만 사용하면 효과적인 차원 축소가 가능합니다.[^1]

## 7. 인수분해 (Factor Analysis): 숨겨진 요인 찾기

인수분해는 1904년 Charles Spearman이 개발한 기법으로, 변수들 간의 상관관계를 이용해 숨겨진 공통 요인을 찾아냅니다.[^1]

### PCA vs Factor Analysis

| 특징 | PCA | Factor Analysis |
| :-- | :-- | :-- |
| 목적 | 분산 최대화[^5] | 공통 요인 추출[^1] |
| 적용 | 차원 축소 | 구조 발견 |
| 해석 | 어려움[^5] | 상대적으로 쉬움[^1] |

### R을 이용한 인수분해 예제

```r
# 탐색적 인수분해 (EFA)
pcaFac <- factanal(myData, factors = 3, rotation = 'varimax')
pcaFac$scores[1:10,]
```

VARIMAX 회전은 좌표를 직교로 회전시켜 해석을 더 쉽게 만들어줍니다.[^1]

## 8. 날짜/시간 변수 활용하기

날짜/시간 변수는 연속변수 처리 기법을 연습하기에 최적의 데이터입니다. 다양한 변환과 피처 엔지니어링이 가능하기 때문입니다.[^1]

### 날짜 변수에서 추출 가능한 피처들

일반적인 날짜 형식 (DD-MM-YYYY HH:SS)에서 다음과 같은 변수들을 만들 수 있습니다:[^1]

- **년도, 월, 일, 시간, 분, 초**
- **요일, 주차, 분기**
- **공휴일 여부, 주말 여부**
- **계절 정보**


### 날짜 변수 처리 예제

```r
# R에서 날짜 처리
as.Date("2015-12-1")  # 날짜 생성
Sys.Date() - as.Date("2014-12-01")  # 날짜 차이 계산

# 시간 처리
as.POSIXlt(Sys.time())  # 현재 시간
as.POSIXct("080406 10:11", format = "%y%m%d %H:%M")  # 시간 변환
```

월 데이터에서 분기를 만들거나, 일 데이터에서 평일/주말을 구분하는 등 의미있는 구간화가 가능합니다. 상관관계 분석을 통해 타겟 변수와의 관계를 확인한 후 가장 영향력이 큰 파생 변수를 선택해야 합니다.[^1]

## 정규화 vs 표준화: 언제 무엇을 사용할까?

많은 학생들이 헷갈려하는 부분입니다. 두 방법의 차이를 명확히 알아보겠습니다.[^6][^11]

### 정규화 (Min-Max Scaling)

데이터를 0과 1 사이의 값으로 변환합니다:[^6][^11]

\$X' = \frac{X - X_{min}}{X_{max} - X_{min}} \$

**장점**: 모든 값이 정확히  범위에 들어갑니다[^1][^11]
**단점**: 이상치에 민감합니다[^6]

### 표준화 (Z-Score Standardization)

평균을 0, 표준편차를 1로 만듭니다:[^6][^11]

\$X' = \frac{X - \mu}{\sigma} \$

**장점**: 이상치에 덜 민감하고, 정규분포 가정 하에 효과적입니다[^11]
**단점**: 값의 범위가 고정되지 않습니다[^6]

### 선택 기준

| 상황 | 추천 방법 | 이유 |
| :-- | :-- | :-- |
| 이상치가 많은 경우 | 표준화[^11] | 이상치에 덜 민감 |
| 정규분포 데이터 | 표준화[^11] | 통계적 가정에 부합 |
| 고정 범위 필요 | 정규화[^11] | [^1] 범위 보장 |
| 이미지 처리 | 정규화[^11] | 픽셀 값 정규화 |

실무에서는 두 방법을 모두 시도해보고 성능이 더 좋은 것을 선택하는 것이 일반적입니다.[^6][^11]

![연속변수 전처리 방법 비교: 원본 데이터와 변환된 데이터의 분포](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/65bc30fb128e724a36698cb9607f44e6/b3183392-5857-490b-a40a-f750164d98c8/47f0a203.png)

연속변수 전처리 방법 비교: 원본 데이터와 변환된 데이터의 분포

## 실제 적용 시 주의사항

### 1. 알고리즘별 특성 고려

**거리 기반 알고리즘** (KNN, K-means): 스케일링이 필수입니다[^12]
**트리 기반 알고리즘** (Random Forest, XGBoost): 스케일링의 영향을 덜 받습니다[^12]
**신경망**: 표준화가 학습 속도를 크게 향상시킵니다[^11]

### 2. 데이터 분포 확인

변환 전에 항상 데이터의 분포를 시각화해서 확인해야 합니다. 히스토그램과 Q-Q 플롯을 활용하면 적절한 변환 방법을 선택할 수 있습니다.[^9]

### 3. 정보 손실 최소화

빈닝이나 변환 시 원본 정보가 손실될 수 있습니다. 따라서 변환 전후의 성능을 비교해서 실제로 도움이 되는지 확인해야 합니다.[^1]

## 종합 실습 예제

이론을 실제로 적용해볼 수 있는 완전한 파이프라인을 제공합니다:

```python

# 연속변수 처리 완전 가이드 - 실습 코드

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 샘플 데이터 생성
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.gamma(4, 8, n_samples),  # 나이 (편중된 분포)
    'income': np.random.exponential(50000, n_samples),  # 수입 (매우 편중)
    'height': np.random.normal(170, 10, n_samples),  # 키 (정규분포)
    'weight': np.random.normal(70, 15, n_samples),  # 몸무게 (정규분포)
    'score': np.random.beta(7, 2, n_samples) * 100  # 점수 (베타분포)
}

df = pd.DataFrame(data)

print("=== 1. 원본 데이터 분석 ===")
print(df.describe())

# 2. 빈닝 (Binning) 예시
print("\n=== 2. 빈닝 (Binning) 예시 ===")

# 나이 데이터 빈닝
age_bins_equal = pd.cut(df['age'], bins=5, labels=['청소년', '청년', '중년', '장년', '노년'])
age_bins_quantile = pd.qcut(df['age'], q=5, labels=['하위20%', '20-40%', '40-60%', '60-80%', '상위20%'])

print("Equal Width Binning 결과:")
print(age_bins_equal.value_counts().sort_index())
print("\nQuantile Binning 결과:")
print(age_bins_quantile.value_counts().sort_index())

# 3. 정규화 vs 표준화
print("\n=== 3. 정규화 vs 표준화 ===")

# MinMax Scaler (정규화)
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df), 
    columns=df.columns
)

# Standard Scaler (표준화) 
standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    standard_scaler.fit_transform(df),
    columns=df.columns
)

print("정규화 후 범위:")
print(f"최솟값: {df_normalized.min().min():.3f}")
print(f"최댓값: {df_normalized.max().max():.3f}")

print("\n표준화 후 통계:")
print(f"평균: {df_standardized.mean().mean():.3f}")
print(f"표준편차: {df_standardized.std().mean():.3f}")

# 4. 로그 변환
print("\n=== 4. 로그 변환 ===")
df_log = df.copy()
df_log['income_log'] = np.log1p(df['income'])  # log(1+x)
df_log['age_log'] = np.log1p(df['age'])

print("변환 전후 왜도(skewness) 비교:")
from scipy.stats import skew
print(f"Income 원본 왜도: {skew(df['income']):.3f}")
print(f"Income 로그변환 후 왜도: {skew(df_log['income_log']):.3f}")

# 5. PCA 차원 축소
print("\n=== 5. PCA 차원 축소 ===")
pca = PCA()
pca_result = pca.fit_transform(df_standardized)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("주성분별 설명 분산 비율:")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\n95% 분산 설명에 필요한 주성분: {np.argmax(cumulative_var >= 0.95) + 1}개")

# 6. 이상치 처리
print("\n=== 6. 이상치 처리 ===")
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

income_outliers = detect_outliers_iqr(df['income'])
print(f"Income 이상치 개수: {income_outliers.sum()}개 ({income_outliers.mean()*100:.1f}%)")

# 7. 피처 엔지니어링
print("\n=== 7. 피처 엔지니어링 ===")
df_engineered = df.copy()
df_engineered['bmi'] = df['weight'] / (df['height']/100)**2  # BMI 계산
df_engineered['income_per_age'] = df['income'] / df['age']  # 연령 대비 수입

print("새로 생성된 피처:")
print(f"BMI 평균: {df_engineered['bmi'].mean():.2f}")
print(f"연령 대비 수입 평균: {df_engineered['income_per_age'].mean():.0f}")

print("\n=== 8. 종합 분석 완료 ===")
print("모든 연속변수 처리 기법을 성공적으로 적용했습니다!")


```

이 코드는 실제 데이터에서 자주 마주치는 상황들을 시뮬레이션하고, 8가지 연속변수 처리 기법을 모두 적용해볼 수 있도록 구성되었습니다.

## 마무리

연속변수 처리는 머신러닝 성공의 핵심 요소입니다. 빈닝, 정규화, 변환, 비즈니스 로직 활용, 이상치 처리, PCA, 인수분해, 날짜/시간 처리 등 8가지 방법을 상황에 맞게 조합해서 사용하세요.[^1]

가장 중요한 것은 **데이터의 특성을 이해하고 적절한 방법을 선택하는 것**입니다. 호기심과 인내심을 가지고 다양한 방법을 시도해보세요. 데이터 탐색 없이는 좋은 모델을 만들 수 없습니다.[^1]

각 기법의 장단점을 파악하고, 여러 방법을 조합해서 최적의 전처리 파이프라인을 구축하는 것이 실력 향상의 핵심입니다. 실제 프로젝트에서는 도메인 지식과 데이터 분석을 결합해 창의적인 해결책을 찾는 것이 무엇보다 중요합니다.
<span style="display:none">[^13][^14][^15][^16][^17][^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65]</span>

<div style="text-align: center">⁂</div>

[^1]: https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/

[^2]: https://developers.google.com/machine-learning/crash-course/numerical-data/binning

[^3]: https://www.scaler.com/topics/machine-learning/binning-in-machine-learning/

[^4]: https://rhythmblogs.hashnode.dev/binning-and-binarization-in-machine-learning-techniques-and-applications

[^5]: https://builtin.com/data-science/step-step-explanation-principal-component-analysis

[^6]: https://dine.tistory.com/77

[^7]: https://feature-engine.trainindata.com/en/1.8.x/user_guide/transformation/LogTransformer.html

[^8]: https://www.spsanderson.com/steveondata/posts/2024-12-23/

[^9]: https://www.datanovia.com/en/lessons/transform-data-to-normal-distribution-in-r/

[^10]: https://www.geeksforgeeks.org/data-analysis/principal-component-analysis-pca/

[^11]: https://sungwookoo.tistory.com/35

[^12]: https://eda-ai-lab.tistory.com/647

[^13]: https://ieeexplore.ieee.org/document/10413287/

[^14]: https://pubs.aip.org/jcp/article/159/1/014801/2901354/A-unified-framework-for-machine-learning

[^15]: https://pubs.acs.org/doi/10.1021/acs.est.3c00026

[^16]: https://ieeexplore.ieee.org/document/10427192/

[^17]: https://www.nature.com/articles/s43017-023-00450-9

[^18]: https://www.nature.com/articles/s41598-023-44326-w

[^19]: https://cardio.jmir.org/2023/1/e47736

[^20]: https://www.nature.com/articles/s41467-023-43860-5

[^21]: https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-025-00352-3

[^22]: https://ieeexplore.ieee.org/document/10212832/

[^23]: https://arxiv.org/pdf/2101.01494.pdf

[^24]: http://arxiv.org/pdf/2310.11978.pdf

[^25]: https://arxiv.org/pdf/1902.09615.pdf

[^26]: https://arxiv.org/pdf/2001.08025.pdf

[^27]: https://arxiv.org/pdf/0807.4820.pdf

[^28]: https://arxiv.org/pdf/2202.04348.pdf

[^29]: https://arxiv.org/pdf/2207.07727.pdf

[^30]: https://arxiv.org/pdf/2108.08228.pdf

[^31]: http://arxiv.org/pdf/1703.08619.pdf

[^32]: http://arxiv.org/pdf/2312.15002.pdf

[^33]: https://www.solver.com/bin-continuous-data-example

[^34]: https://www.cs.cmu.edu/~elaw/papers/pca.pdf

[^35]: https://mlpills.substack.com/p/issue-93-binning-in-machine-learning

[^36]: https://www.semanticscholar.org/paper/56a49a1b0dac2d95ad23caa89a31ca0231fe3d70

[^37]: https://www.semanticscholar.org/paper/a8022fd85dd44a473ffbc4ec5d1afee309bb6e0f

[^38]: https://www.semanticscholar.org/paper/248bb137645e73842e7cf34cf850e4f9c91076b3

[^39]: https://www.semanticscholar.org/paper/6710e97bc5613de99c7220501c6e420b4786d4cc

[^40]: http://arxiv.org/pdf/2308.10915.pdf

[^41]: http://arxiv.org/pdf/2502.04554.pdf

[^42]: http://arxiv.org/pdf/2411.08203.pdf

[^43]: https://arxiv.org/pdf/2006.02227.pdf

[^44]: https://arxiv.org/pdf/2310.13818.pdf

[^45]: https://arxiv.org/html/2408.10369v2

[^46]: http://arxiv.org/pdf/2406.06321.pdf

[^47]: http://arxiv.org/pdf/2401.02869.pdf

[^48]: https://www.mdpi.com/1099-4300/22/12/1391/pdf

[^49]: https://arxiv.org/pdf/2106.09164.pdf

[^50]: https://downloads.hindawi.com/journals/mpe/2015/125781.pdf

[^51]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/imt2.107

[^52]: https://arxiv.org/pdf/2412.07093.pdf

[^53]: https://arxiv.org/html/2403.00266v1

[^54]: http://arxiv.org/pdf/2310.14720.pdf

[^55]: https://arxiv.org/pdf/2105.14435.pdf

[^56]: https://changsroad.tistory.com/467

[^57]: https://data-newbie.tistory.com/27

[^58]: https://dsdingdong.tistory.com/23

[^59]: https://subinium.github.io/MLwithPython-4-1/

[^60]: https://deepdata.tistory.com/318

[^61]: https://yuja-k.tistory.com/24

[^62]: https://draw-code-boy.tistory.com/438

[^63]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/65bc30fb128e724a36698cb9607f44e6/d4f2bfda-6798-4b89-86da-f76082bcdaec/d2b7fc9a.csv

[^64]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/65bc30fb128e724a36698cb9607f44e6/fba54e03-7bed-40b3-aded-d6c46e05a4d5/4ee2608a.csv

[^65]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/65bc30fb128e724a36698cb9607f44e6/053a71dc-7700-4c64-b07b-fdff50734569/0e3fe616.py


연속 변수를 어떻게 처리하나요?

연속 변수는 관련시키기 쉽지만 – 어떤 면에서는 자연이 그렇죠. 예측 모델링 관점에서 보면 보통 더 어렵습니다. 왜 그렇게 말할까요? 처리할 수 있는 방법의 수가 많기 때문입니다.

예를 들어, 성별에 따른 스포츠 침투를 분석해 달라고 하면 쉬운 연습입니다. 스포츠를 하는 남성과 여성의 비율을 살펴보고 차이가 있는지 확인할 수 있습니다. 이제 연령에 따른 스포츠 침투를 분석해 달라고 하면 어떨까요? 이를 분석할 수 있는 가능한 방법이 몇 가지나 생각나십니까? 빈/구간 만들기, 플로팅, 변환 등 목록은 계속됩니다!

따라서 연속 변수를 다루는 것은 보통 더 정보에 입각하고 어려운 선택입니다. 따라서 이 글은 초보자에게 매우 유용할 것입니다.

연속 변수를 다루는 방법에는 여러 가지가 있습니다. 다음은 예측 모델링에서 사용할 수 있는 8가지 방법입니다.

정규화(Normalization): 연속 변수를 0과 1 사이로 조정하여 모델의 성능을 향상시킬 수 있습니다.

표준화(Standardization): 데이터의 평균을 0으로, 표준편차를 1로 맞추어 스케일링합니다.

비율 변환(Ratio Transform): 변수 간 비율을 사용하여 비선형 관계를 강조할 수 있습니다.

로그 변환(Log Transformation): 데이터의 분포를 정규화하는 데 유용합니다.

Binning: 연속 변수를 구간으로 나눠 이산형 변수로 변환합니다.

다항 회귀(Polynomial Regression): 변수의 비선형 관계를 포함할 수 있습니다.

상호작용 항(Interaction Terms): 변수 간 상호작용을 모델에 포함시켜 복잡한 관계를 모델링합니다.

특징 선택(Feature Selection): 중요한 변수를 선택하여 모델을 간소화하고 성능을 향상시킵니다.

이러한 방법들을 활용하면 연속 변수를 효과적으로 모델링할 수 있습니다.

# Reference
https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/
- Binning The Variable
- Normalization
- Transformations for Skewed Distribution
- Use of Business Logic
- New Features
- Treating Outliers
- Principal Component Analysis
- Factor Analysis
- Methods to work with Date & Time Variable
