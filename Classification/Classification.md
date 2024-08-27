 # CONTENTS

오늘날 많은 산업이 다양한 유형의 매우 큰 데이터 세트를 처리하고 있습니다.  
모든 정보를 수동으로 처리하는 것은 시간이 많이 걸릴 수 있으며 장기적으로 가치를 더하지 못할 수도 있습니다.  
간단한 자동화에서 머신 러닝 기술에 이르기까지 많은 전략이 더 나은 투자 수익을 위해 적용되고 있습니다.  
이 개념적 블로그에서는 가장 중요한 개념 중 하나인 머신 러닝의 분류를 다룰 것입니다.

머신 러닝 에서 분류가 무엇인지 정의한 다음 머신 러닝에서 두 가지 유형의 학습자와 분류와 회귀의 차이점을 명확히 하겠습니다.  
그런 다음 분류를 사용할 수 있는 몇 가지 실제 시나리오를 다루겠습니다.  
그 후 모든 유형의 분류를 소개하고 몇 가지 분류 알고리즘 예를 심층적으로 살펴보겠습니다.  
마지막으로 몇 가지 알고리즘 구현에 대한 실습을 제공합니다.

## What is Classification in Machine Learning?
분류는 모델이 주어진 입력 데이터의 올바른 레이블을 예측하려고 시도하는 지도 학습 방식입니다.  
분류에서 모델은 훈련 데이터를 사용하여 완전히 훈련된 다음, 새로운 보이지 않는 데이터에 대한 예측을 수행하는 데 사용되기 전에 테스트 데이터에서 평가됩니다.

예를 들어, 알고리즘은 주어진 이메일이 스팸인지 아니면 햄(스팸이 아님)인지 예측하는 법을 학습할 수 있습니다.  

머신 러닝에서 분류는 데이터를 별개의 클래스로 분류하는 데 사용됩니다.  
이는 머신 러닝에서 가장 일반적이고 중요한 작업 중 하나이며, 입력 피처를 기반으로 주어진 문제의 결과를 예측하는 데 도움이 됩니다.

간단히 말해서, 분류 머신 러닝 알고리즘을 사용 하면 데이터에 레이블이나 범주를 지정할 수 있습니다.  

이러한 라벨을 클래스라고 합니다.  
이는 분류를 위해 시스템에 입력된 각 인스턴스(또는 객체)의 속성을 분석하여 수행할 수 있습니다.  

전체 프로세스에는 지도 학습이 포함됩니다.  
즉, 알려진 속성(또는 레이블)이 있는 객체를 사용하여 미래 예측을 위한 모델을 훈련합니다.  
훈련이 완료되면 이 모델을 사용하여 새 인스턴스를 분류할 수 있습니다.  

- 회귀 문제	: 자연 속의 숫자적 값(실수, 연속형, 이산형 등)을 예측해야 할 때 숫자형 개체(종속 변수 또는 Y 변수라고도 함)에 미치는 수많은 변수의 영향을 정량화해야 할 필요가 있습니다.
- 분류 문제	: 이 비즈니스 문제는 회귀 문제와 다소 유사합니다. 여기서는 종속 변수에 대한 변수(독립 변수 또는 X 변수라고 함)의 영향을 정량화해야 하는 요구 사항이 있습니다. 그러나 여기서는 다르고 Y 변수는 범주형입니다. 따라서 X 변수를 기반으로 모델을 개발하고 사전 정의된 클래스로 관찰을 예측합니다.
- 예측 문제 :	시간에 따른 값을 예측해야 하며 시간이 예측자 역할을 하는 경우
- 세분화 문제	: 대량의 데이터를 분류해야 하는 상황이 있습니다. 그러나 모델을 감독하는 데 기존 클래스를 사용할 수는 없습니다. 여기서 기본 패턴을 감지하고 관찰을 여러 범주로 나눕니다. 그런 다음 각 특정 클래스에서 발견된 관찰의 특성을 이해하여 이러한 범주를 정의합니다.

머신 러닝에서 분류는 지도 학습 ML 기술입니다.  
데이터 세트의 데이터 인스턴스에 대한 그룹 관계를 예측합니다.  
사전 정의된 범주로 레이블을 사용하여 데이터 객체를 인식, 이해 및 그룹화하는 재귀적 프로세스입니다.  

더 간단히 말해서, 모든 새로운 관찰이 속하는 적절한 세트 또는 클래스를 식별합니다.  
출력 변수는 두 개 이상의 클래스 또는 범주에 속할 수 있습니다.  
예를 들어, Google Photos가 사람, 폭포, 위치 정보가 포함된 사진을 분류하는 경우, 사용 가능한 메트릭으로 인해 동일한 이미지가 언급된 세 가지 범주에 모두 표시됩니다.  

- Also Read: How Confusion Matrix in Machine Learning Helps Solve Classification Issues : https://www.analytixlabs.co.in/blog/confusion-matrix/

이전에 우리는 머신 러닝 측면에서 분류가 무엇인지 이해했습니다.  
또한 분류가 무엇이고 유형, 기본 사항, 속성이 무엇인지 이해해 보겠습니다.  

머신 러닝에서 분류는 관찰 결과가 어느 범주에 속하는지 판단하는 것을 목표로 하며, 이는 종속 변수와 독립 변수 간의 관계를 이해함으로써 이루어집니다.  
여기서 종속 변수는 범주형이고, 독립 변수는 숫자형 또는 범주형일 수 있습니다.  

종속 변수를 사용하면 입력 변수와 범주 간의 관계를 확립할 수 있으며, 분류는 지도 학습 설정에서 작동하는 예측 모델링입니다.  

### 1.1. Classification in Machine Learning: Terminologies 

- Classifier: 입력 변수를 특정 클래스에 매핑하는 알고리즘입니다.
- Feature: 선택된 시나리오의 측정 가능한 속성 또는 지표입니다.
- Initialize: 사용된 분류기를 할당합니다.
- Classification Model : 입력 데이터를 두 개 이상의 개별 그룹으로 분류하는 모델입니다.
- Evaluate: 정확도 점수와 분류 보고서를 찾아 모델을 평가합니다.

 #### 1.1.1. Two-step procedure for classification:  

 - 학습 단계(트레이닝 단계): 초기 단계는 분류 모델을 구성하는 것입니다. 이는 사용 가능한 트레이닝 세트를 사용하여 모델 학습을 가능하게 하여 Classifier를 구축하는 알고리즘을 선택하여 수행됩니다. 모델이 완벽하게 트레이닝되면 정확한 결과를 얻습니다.
- 분류 단계: 모델은 클래스 레이블(부분 모집단)을 예측하고 테스트 데이터에서 구성된 모델을 관찰합니다. 분류 규칙의 정확도를 추정합니다. 결과 분류기는 나중에 예측 기능의 값이 알려진 테스트 데이터 인스턴스에 클래스 레이블을 할당하는 데 사용됩니다. 그래도 클래스 레이블의 값은 알 수 없습니다.

1.2. Why is Classification Used in Machine Learning?  

분류 알고리즘은 머신 러닝에서 주어진 데이터 포인트의 클래스 레이블을 예측하는 데 사용됩니다.  
이를 통해 클래스 레이블이 알려지지 않은 경우에도 머신이 새로운 데이터 포인트를 학습하고 예측할 수 있습니다. 

- 분류 머신 러닝은 지도 학습과 비지도 학습 작업 모두에 사용됩니다.

지도 분류는 훈련 데이터 포인트에 클래스 레이블을 지정하는 것을 포함하는 반면, 비지도 분류는 가능한 클래스 레이블에 대한 사전 지식 없이 데이터 포인트를 클러스터로 그룹화하는 것을 포함합니다.  

- Also Read: Types of Machine Learning : https://www.analytixlabs.co.in/blog/types-of-machine-learning/

일반적인 사용 사례로는 다양한 클래스 레이블 간의 유사성 점수를 기반으로 항목을 추천하는 추천 시스템 과 사기 감지 또는 악의적 활동 식별과 같은 이상 감지 작업이 있습니다.  

레이블이 지정된 인스턴스의 대규모 데이터 세트에 분류 알고리즘을 적용함으로써 머신은 패턴을 학습하고 예측을 할 수 있습니다.  
이러한 알고리즘은 다른 방법으로 발견하거나 식별하기 어려울 수 있는 데이터에서 패턴을 감지합니다.  

## How is Classification Algorithm Implemented?  
2.1. Learners in Classification Problems  
 2.1.1 Lazy learners  
 2.1.2 Eager learners  
2.2. Understanding the Classifiers: Types of Classifications  
 2.2.1. Naive Bayes  
 2.2.2. K-Nearest Neighbour  
 2.2.3. Decision Trees  
 2.2.4. Logistic Regression  
 2.2.5. Random Forest  
 2.2.6. Support Vector Machine (SVM)  
 2.2.7. K-Means Clustering Classification Algorithm  
 2.2.8. XGBoost  
## Predictive Modeling: Classifications  
3.1. Types of Classifications  
 3.1.1. Binary & Multi-Class Classification  
 3.1.2. Multi-label Classification  
 3.1.3. Imbalanced Classification  
3.2. Classification Examples  
 3.2.1. Email Classification  
 3.2.2. Image Classification  
 3.2.3. Anomaly/Fraud Detection  
 3.2.4. Web text classification  
## Evaluation Metrics for Classification Models
4.1. Classification Accuracy:  
4.2. Logarithmic Loss or Log Loss:  
4.3. Confusion Matrix:  
4.4. ROC Curve:  
4.5. AUC:  
## Conclusion
