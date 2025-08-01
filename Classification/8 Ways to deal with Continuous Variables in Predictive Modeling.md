# 8 Ways to deal with Continuous Variables in Predictive Modeling

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
