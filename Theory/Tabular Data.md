# Tabular Data 이해와 딥러닝 모델 구성 예시

딥러닝을 공부하는 대학생을 위해, 산업 현장에서 가장 흔히 접하는 **Tabular Data** 개념과 이를 활용한 간단한 딥러닝 모델 구성 과정을 정리했습니다.

***

## 1. Tabular Data란 무엇인가요?

Tabular Data는 엑셀 시트나 데이터베이스 테이블 형태로,

- **컬럼(column)**: 특성(feature)을 나타냅니다.
- **행(row)**: 샘플(sample)을 나타냅니다.

관계형 데이터베이스(RDBMS)는 여러 개의 테이블로 구성되며,
각 테이블은 key–value 관계로 대상을 추상화합니다.
이처럼 표 형태로 정형화된 데이터를 **Tabular Data**라고 부릅니다.

***

## 2. Tabular Data의 특징

1. **정형화된 구조**
    - 모든 샘플이 동일한 컬럼을 가집니다.
2. **다양한 타입**
    - 숫자형(continuous), 범주형(categorical) 데이터를 모두 포함할 수 있습니다.
3. **결측치 처리 필요**
    - 실제 산업 현장 데이터는 누락된 값이 많습니다.
4. **전처리 작업이 핵심**
    - 인코딩, 스케일링, 결측치 대체 등 다양한 전처리가 필요합니다.

***

## 3. 딥러닝 모델 구성 전 준비 단계

1. 데이터 불러오기
2. 결측치 처리
3. 범주형 feature 인코딩
4. 숫자형 feature 스케일링
5. 학습용(train)·검증용(val) 분리
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. 데이터 불러오기
df = pd.read_csv('data.csv')

# 2. 결측치 처리 (간단히 평균 대체)
df.fillna(df.mean(), inplace=True)

# 3. 범주형 인코딩
ohe = OneHotEncoder(sparse=False)
cat_cols = ['category_feature']
encoded = ohe.fit_transform(df[cat_cols])
df_encoded = pd.concat([df.drop(cat_cols, axis=1),
                        pd.DataFrame(encoded, columns=ohe.get_feature_names_out())],
                       axis=1)

# 4. 스케일링
scaler = StandardScaler()
num_cols = ['numeric_feature1', 'numeric_feature2']
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 5. 학습·검증 분리
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
```


***

## 4. 간단한 딥러닝 모델 예시 (Keras)

Tabular Data에 자주 쓰이는 **피드포워드 신경망(MLP)** 구조를 구성해 봅니다.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 모델 구성
model = models.Sequential([
    layers.Input(shape=(X_train.shape[^1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 이진 분류 예시
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)
```

- **Input**: 처리된 피처 벡터 (스케일링·인코딩 완료)
- **Dense 레이어**: 은닉 뉴런 수와 활성화 함수로 모델 용량 조절
- **Dropout**: 과적합 방지
- **Output**: 이진 분류용 sigmoid

***

## 5. 모델 성능 확인

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy Curve')
plt.show()
```

- **Loss Curve**와 **Accuracy Curve**를 통해 학습 상태를 직관적으로 파악할 수 있습니다.

***

## 6. 추가 팁

- 범주형 특성이 많을 때는 임베딩 레이어를 고려하세요.
- 이상치(outlier)가 심한 데이터는 **Robust Scaler**를 사용해 보세요.
- 앙상블 모델(XGBoost, LightGBM)과 비교하여 성능을 검증해 보세요.

***

이제 Tabular Data의 기본 개념부터 딥러닝 모델 구성, 성능 시각화까지의 흐름을 익혔습니다. 실습을 통해 직접 다양한 데이터로 시도해 보시기 바랍니다.

<div style="text-align: center">⁂</div>

[^1]: https://velog.io/@parkchansaem/머신러닝-Tabular-data



Tabular data란 산업 현장의 엑셀 시트에서 아주 흔하게 볼 수 있는 형태로, feature는 컬럼에, sample을 row 방향에 위치한 정형데이터 구조이다.

산업현장에서 주로 사용되는 RDMS(관계형 데이터베이스)는 하나 이상의 데이블로 이루어져 있으며, 각 테이블은 key와 value의 관계를 통해 표현하고 싶은 대상을 추상화한다.

간단히 설명하면, 테이블형태의 정형데이터를 우리는 tabular data라고 이해하면 될 거 같다.

## 머신러닝에서의 Tabular Data 활용
통상적인 머신러닝의 데이터로 tabular data가 제공되며 이것을 이용하여 캐글과 같은 대회에서 트리 계열의 boosting 방법론으로 좋은 성능을 보여주고 있다.
그 중 핵심은 feature selection, feature importance가 중요하다.

## 딥러닝
딥러닝 분야에서는 대부분 이미지, 음성, 언어와 같은 비정형 데이터에서 인상적인 성능을 보여준다.
정형데이터(tabular data)는 머신러닝을 이용한 트리기반의 모델들이 딥러닝만큼 좋은 성능을 내고 있어 주목받지 못하는 경우가 많다고 한다.
그래서 정형데이터를 딥러닝 장점을 활용한 모델들도 계속해서 연구되어 지고 있다. 그 중 Tabnet이라는 모델에 대해서 다음 기회에 포스팅 해보려고 한다.

- https://velog.io/@parkchansaem/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Tabular-data
