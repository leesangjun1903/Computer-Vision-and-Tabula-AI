# Machine learning
# AdaBoost
## Adaptive boosting; Discrete AdaBoost

## ◃Deep CNN-Based Face Recognition, ◃Feature Selection, ◃Machine Recognition of Objects

## Definition
The AdaBoost algorithm learns a classifier from data by combining additively a number of weak classifiers. The weak classifiers are incorporated sequentially, one at a time, in order to reduce the empirical exponential classification risk of the combined classifier.

AdaBoost 알고리즘은 여러 약한 분류기를 결합하여 데이터에서 분류기를 학습합니다.  
결합된 분류기의 경험적 지수 분류 위험을 줄이기 위해 약한 분류기를 한 번에 하나씩 순차적으로 통합합니다.

## Background
Boosting [1, 2], introduced by Robert Schapire in [3], is a general technique for combining the response of several predictors with limited accuracy into a single, more accurate predic- tion. AdaBoost is a popular implementation of boosting for binary classification [4]. Soon after its introduction, AdaBoost became one of the most popular learning algorithms; for example, Breiman [1] described AdaBoost with trees as the “best off-the-shelf classifier in the world.” Much of the popularity of AdaBoost was due to its performance, which was comparable to the one of support vector machines [5], and its algorithmic simplicity. In the computer vision community, AdaBoost became very popular due to the work of Viola and Jones in face detec- tion [6], which used it to demonstrate accurate face detection in real time; key to their method is a classifier obtained by boosting (combin- ing) weak classifiers, each incorporating a single Haar wavelet [6, 7]. Haar wavelets can be com- puted very efficiently using integral images. Fur- thermore, the weak classifiers can be computed sequentially, in a cascade, and the computation terminated as soon as sufficient evidence to reject a hypothesis is accumulated.

Robert Shapire가 [3]에서 소개한 부스팅 [1, 2]은 정확도가 제한된 여러 예측기의 응답을 하나의 더 정확한 예측으로 결합하는 일반적인 기법입니다.  
AdaBoost는 이진 분류를 위한 부스팅 [4]의 인기 있는 구현입니다. 도입 직후 AdaBoost는 가장 인기 있는 학습 알고리즘 중 하나가 되었습니다.  
예를 들어, Breiman [1]은 트리 구조가 있는 AdaBoost를 "세계 최고의 기성 분류기"라고 묘사했습니다.  
AdaBoost의 인기는 지원 벡터 머신 [5]와 비교할 수 있는 성능과 알고리즘의 단순성 덕분이었습니다.  
컴퓨터 비전 커뮤니티에서 AdaBoost는 얼굴 감지 분야에서 Viola와 Jones의 연구로 인해 매우 인기를 끌었으며, 이를 통해 실시간으로 정확한 얼굴 감지를 시연했습니다.  
AdaBoost의 핵심은 약한 분류기를 부스팅(결합)하여 얻은 분류기로, 각각 하나의 Haar Wavelet [6, 7]을 통합합니다.  
Haar 웨이블릿은 적분 이미지를 사용하여 매우 효율적으로 계산할 수 있습니다.  
또한 약한 분류기는 순차적으로 단계적 방식으로 순차적으로 계산할 수 있으며, 가설을 기각할 충분한 증거가 축적되면 즉시 계산이 종료됩니다.

The additive boosting framework is fairly gen- eral, and several variants have been proposed, among which are AdaBoost [4]; Real AdaBoost, LogitBoost, and GentleBoost [1]; Regularized AdaBoost [8]; or extensions to multiple classes such as AdaBoost.MH [9].

가산 부스팅 프레임워크는 비교적 일반적이며, 여러 변형이 제안되었습니다.  
여기에는 AdaBoost [4]; Real AdaBoost, LogitBoost 및 GentleBoost [1]; Regularized AdaBoost [8]; 또는 AdaBoost.MH [9]와 같은 여러 클래스로의 확장이 포함됩니다.

## Theory
This section describes the AdaBoost algorithm as originally given by Freund and Schapire [4]. The particular variant below, also known as discrete AdaBoost [1], is summarized in Algorithm 1. The purpose of AdaBoost is to learn a binary classifier that is a function H(x)= y which maps data x ∈ X (e.g., a scrap of text, an image, or a sound wave) to its class label y ∈ {−1,+1}. The classifier H is obtained as the sign of an additive combination of simple classifiers h : X → {−1,+1}, called weak hypotheses. Given coefficients αt ∈ R, the classifier can then be written as:

이 섹션에서는 Freund와 Shapire [4]가 원래 제공한 AdaBoost 알고리즘을 설명합니다.  
아래의 특정 변형, 즉 이산 AdaBoost [1]은 알고리즘 1에 요약되어 있습니다.  

<img width="425" alt="스크린샷 2025-04-14 오후 9 04 11" src="https://github.com/user-attachments/assets/1aec4749-1cd0-451c-9406-d771f1b5a118" />

AdaBoost의 목적은 데이터 x ∈ X(예: 텍스트, 이미지 또는 음파)를 클래스 레이블 y ∈ {-1,+1}에 매핑하는 함수 H(x)= y인 이진 분류기를 학습하는 것입니다.  
분류기 H는 간단한 분류기 h : X → {-1,+1}의 덧셈 조합인 약한 가설의 부호로 얻어집니다.  
계수 αt ∈ R이 주어지면 분류기는 다음과 같이 쓸 수 있습니다:

```math
H(X) \doteq sign(\sum^m_{t=1} \alpha_t h_t(x))
```
The input to AdaBoost is a set H of weak hypotheses and n data-label pairs (x1,y1),...,(xn,yn); the output is a combi- nation H of m weak hypotheses in H and their coefficients α1,...,αm. The algorithm is designed so that the combined classifier closely fits the training data, i.e., H(xi)= yi for most i= 1, . . . , n. Let us denote with Hm the classifier H with m weak hypotheses shown in Eq. (1). AdaBoost operates sequentially by adding to Hm−1 one new weak hypothesis (hm,αm). While any weak hypothesis with performance better than chance can be used, it is more common to select the weak hypothesis hm in the set H that minimizes the weighted empirical error

AdaBoost에 대한 입력은 약한 가설과 n개의 데이터 레이블 쌍(x1, y1),...,(xn, yn)의 집합 H입니다.  
출력은 H의 m개의 약한 가설과 그 계수 α1,..., αm의 조합 H입니다.  
알고리즘은 결합된 분류기가 훈련 데이터, 즉 대부분의 i=1, . . . , n에 대해 H(xi)= yi와 밀접하게 일치하도록 설계되었습니다.  
방정식 (1)에 표시된 m개의 약한 가설을 가진 분류기 H를 Hm으로 나타내 보겠습니다.  
AdaBoost는 Hm-1에 하나의 새로운 약한 가설(hm,αm)을 추가하여 순차적으로 작동합니다.  
성능이 확률보다 우수한 약한 가설은 사용할 수 있지만, 가중 경험적 오류 $\epsilon$ 를 최소화하는 집합 H에서 약한 가설 hm을 선택하는 것이 더 일반적입니다

<img width="295" alt="스크린샷 2025-04-14 오후 9 10 17" src="https://github.com/user-attachments/assets/294721a0-bc3e-4095-9df8-dc4f0cf9d200" />

where w = (w1, . . . ,wn) are non-negative data weights, as given below. Here the term [yi ̸= h(xi)] is equal to 1 if yi ̸= h(xi) and 0 otherwise. Hence, the empirical error ϵ(h;w) is the average number of incorrect classifications of the weak hypothesis h on the weighted training data. 

The selected weak hypothesis hm can be writ- ten as ϵ(h;w), i.e., hm = argminh∈H ϵ(h;w) and is added to the current combination Hm−1 with coefficient

여기서 w = (w1, . . . ,wn)은 아래와 같이 음이 아닌 데이터 가중치입니다.  
여기서 [yi ̸= h(xi))] 항은 yi ̸= h(xi)이면 1이고, 그렇지 않으면 0입니다.  
따라서 경험적 오차 $ϵ(h;w)$은 가중 학습 데이터에서 약한 가설 h의 잘못된 분류의 평균 수입니다. 

선택된 약한 가설 hm은 $ϵ(h;w)$, 즉 hm = argminh ∈H ϵ(h;w)로 쓸 수 있으며, 아래와 같은 현재의 계수는 조합 Hm-1에 추가됩니다.

<img width="327" alt="스크린샷 2025-04-14 오후 9 13 42" src="https://github.com/user-attachments/assets/52ba2641-0019-4dc9-a1b8-09ed88a33828" />

While AdaBoost minimizes the empirical error of the weak hypothesis hm at each iteration, the weights w are chosen so that the empirical error of Hm is reduced as well. AdaBoost starts with uniform weights w = (1, . . . ,1) and updates them according to the rule

AdaBoost는 각 반복마다 약한 가설 hm의 경험적 오류를 최소화하지만, Hm의 경험적 오류도 줄이기 위해 가중치 w를 선택합니다.  
AdaBoost는 균일한 가중치 w = (1, . . . ,1)로 시작하여 아래 규칙에 따라 업데이트합니다.

<img width="397" alt="스크린샷 2025-04-14 오후 9 14 32" src="https://github.com/user-attachments/assets/e2c03e8d-60c1-4277-91a4-0eefeccd10da" />

One intuitive interpretation of this rule is that it gives more importance to examples that are incorrectly classified. A formal justification is given in the next paragraph.

이 규칙에 대한 직관적인 해석 중 하나는 잘못 분류된 예제에 더 많은 중요성을 부여한다는 것입니다. 공식적인 정당화는 다음 단락에서 설명합니다.

### AdaBoost as Stagewise Minimization
Denote by "Fm(x)= m t=1 αtht(x)" the additive combination of the first m weak hypotheses, so that the classifier Hm(x) can be written as sign Fm(x). AdaBoost performs a stagewise minimization of the cost

첫 번째 m개의 약한 가설의 덧셈 조합을 $F_m(x)= \sum^m_{t=1} α_th_t(x)$로 나타내어 분류기 Hm(x)를 기호 Fm(x)로 쓸 수 있도록 합니다. AdaBoost는 단계적으로 아래 비용을 최소화합니다.

<img width="156" alt="스크린샷 2025-04-14 오후 9 17 31" src="https://github.com/user-attachments/assets/d63902e4-b3ce-46c2-abc4-1c345a31d146" />

This cost is known as the empirical exponential loss and is a convex upper bound to the empirical classification error of Hm, in the sense that:

이 비용은 경험적 지수 손실로 알려져 있으며, 아래와 같은 의미에서 Hm의 경험적 분류 오류에 대한 볼록 상한입니다:

<img width="391" alt="스크린샷 2025-04-14 오후 9 18 29" src="https://github.com/user-attachments/assets/dc8260ba-05da-4da2-b2b4-def3c253ce06" />

To understand the effect of the AdaBoost update on the empirical exponential loss, let Fm(x)= Fm−1(x) + αmhm(x) be the updated additive combination at iteration m. As the parameters of Fm−1 are fixed, the empirical exponential loss is a function E of αm and hm:

AdaBoost 업데이트가 경험적 지수 손실에 미치는 영향을 이해하려면, 반복 m에서 $F_m(x)= F_{m-1}(x) + α_mh_m(x)$를 업데이트된 덧셈 조합으로 가정해 보겠습니다.  
Fm-1의 매개변수가 고정되어 있으므로 경험적 지수 손실은 αm과 hm의 함수 E입니다:

<img width="391" alt="스크린샷 2025-04-14 오후 9 23 45" src="https://github.com/user-attachments/assets/d0a26202-4fe5-406a-948d-4e041443d5dd" />

By taking the derivative of E with respect to αm and by setting it to zero, one obtains the optimality condition

αm에 대한 E의 미분을 0으로 설정하면 최적 조건을 얻을 수 있습니다

<img width="391" alt="스크린샷 2025-04-14 오후 9 29 43" src="https://github.com/user-attachments/assets/f742e77f-1035-43c0-a2bd-5182f266bc96" />

which results in the optimal coefficient given in Eq. (2):

방정식 (2)에 주어진 최적 계수를 도출합니다:

<img width="391" alt="스크린샷 2025-04-14 오후 9 30 52" src="https://github.com/user-attachments/assets/b8f88372-1cdd-402b-bf08-519839d316b8" />

By substituting this expression back in the cost (4), one obtains

이 식을 비용 (4)에 다시 대입하면 다음을 얻을 수 있습니다

<img width="391" alt="스크린샷 2025-04-14 오후 9 41 04" src="https://github.com/user-attachments/assets/6e3f389f-f8ca-439c-8145-6a8b69a0ddd1" />

which achieves its smallest value when the empirical classification error ϵ(hm;w) approaches either 0, its minimum, or 1, its maximum. Notice that if the error ϵ(hm;w) > 1/2, then the corresponding weight αm is negative. In other words, when the weak hypothesis hm makes more mistakes than correct classifications, AdaBoost automatically swaps the sign of the output label so that ϵ(−hm;w) < 1/2. Finally, the weight update Eq. (3) follows from

경험적 분류 오류 ϵ(hm;w)가 최소값 0 또는 최대값 1에 가까워질 때 가장 작은 값을 얻습니다.  
오류 ϵ(hm;w)가 1/2를 초과하면 해당 가중치 αm은 음수가 됩니다.  
즉, 약한 가설 hm이 올바른 분류보다 더 많은 실수를 할 때 AdaBoost는 출력 레이블의 부호를 자동으로 전환하여 ϵ(-hm;w)가 1/2 미만이 되도록 합니다.  
마지막으로 가중치 업데이트 방정식 (3)은 다음과 같습니다

<img width="391" alt="스크린샷 2025-04-14 오후 9 43 05" src="https://github.com/user-attachments/assets/c29b2765-b436-4279-b2c7-c5aad62b4018" />

## Applications
One of the main uses of AdaBoost is for the recognition of patterns in data. Recognition can be formulated as a binary classification problem: Find whether data points match the pattern of interest or not. In computer vision, AdaBoost was popularized by its application to object detection, where the task is not only to recognize but also to localize within an image an object of interest (e.g., a face). Most of the ideas summarized in this section were first proposed by Viola and Jones [6].

AdaBoost의 주요 용도 중 하나는 데이터의 패턴을 인식하는 것입니다.  
인식은 이진 분류 문제로 공식화될 수 있습니다: 데이터 포인트가 관심 있는 패턴과 일치하는지 여부를 확인하세요.  
컴퓨터 비전 분야에서 AdaBoost는 관심 있는 객체(예: 얼굴)를 인식할 뿐만 아니라 이미지 내에서 위치를 파악하는 작업에 적용되어 대중화되었습니다.  
이 섹션에 요약된 대부분의 아이디어는 Viola와 Jones에 의해 처음 제안되었습니다[6].

A common technique for object detection is the sliding window detector. This method reduces the object detection problem to the task of clas- sifying all possible image windows (i.e., patches) to find which ones are centered around the object of interest. In practice, windows may be sampled not only at all spatial locations but also at all scales and rotations. This results in a very large number of evaluations of the classifier function for each input image. Therefore, the computa- tional efficiency of the classifier is of paramount importance.

객체 감지를 위한 일반적인 기술은 슬라이딩 윈도우 감지기입니다.  
이 방법은 객체 감지 문제를 모든 가능한 이미지 창(즉, 패치)을 분류하여 관심 객체를 중심으로 어떤 창이 있는지 찾는 작업으로 축소합니다.  
실제로 창은 모든 공간 위치뿐만 아니라 모든 스케일과 회전에서도 샘플링될 수 있습니다.  
그 결과 각 입력 이미지에 대해 분류기 함수에 대한 평가가 매우 많이 이루어집니다. 따라서 분류기의 계산 효율성이 가장 중요합니다.

Classifiers computed with AdaBoost can be made very computationally efficient by using weak hypotheses that are fast to compute and by letting AdaBoost select a small set of hypotheses most useful to the given problem. For example, in the Viola-Jones face detector, a weak hypothesis is computed by thresholding the output of a linear filter that computes averages over rectangular areas of the image. These filters are known as Haar wavelets and, because of their special struc- ture, can be computed in constant time by using the integral image [6].

AdaBoost로 계산된 분류기는 계산 속도가 빠른 약한 가설을 사용하고 AdaBoost가 주어진 문제에 가장 유용한 작은 가설 집합을 선택할 수 있도록 함으로써 매우 계산 효율적으로 만들 수 있습니다.  
예를 들어, Viola-Jones 얼굴 검출기에서는 이미지의 직사각형 영역에서 평균을 계산하는 선형 필터의 출력을 임계값으로 설정하여 약한 가설을 계산합니다.  
이러한 필터는 Haar 웨이블릿으로 알려져 있으며, 특수한 구조 때문에 적분 이미지를 사용하여 일정 시간 내에 계산할 수 있습니다 [6].

```
6. robust real-time object detection, January 2001
https://www.researchgate.net/publication/215721846_Robust_Real-Time_Object_Detection
```

In order to further improve the speed of a sliding window detector, AdaBoost classifiers are often combined in a cascade [6]. A cascade exploits the fact that the vast majority of image windows are not centered around the object of interest and that, furthermore, most of these neg- ative windows are easy to recognize as such. A cascade is built by appending one AdaBoost classifier after another. Classifiers are evaluated sequentially, and an image window is rejected as soon as the response of a classifier is negative. All the classifiers are tuned to almost never reject a window that matches the object of interest (i.e., high recall). However, the first classifiers in the cascade are allowed to return several false positives (i.e., low precision) in exchange for a significantly reduced evaluation cost, obtained, for instance, by limiting the number of weak hypotheses in them. By using this scheme, the computationally costly and highly accurate clas- sifiers are evaluated only on the most challenging cases: windows that resemble the object of inter- est and that therefore contain either the object (i.e., a positive sample) or a visual structure that can be easily confused with it (i.e., a hard negative sample).

슬라이딩 윈도우 검출기의 속도를 더욱 향상시키기 위해 AdaBoost 분류기는 종종 캐스케이드로 결합됩니다 [6].  
캐스케이드는 대부분의 이미지 창이 관심 대상을 중심으로 하지 않으며, 또한 이러한 부정적인 창이 대부분 인식하기 쉽다는 사실을 악용합니다.  
캐스케이드는 하나의 AdaBoost 분류기를 차례로 추가하여 구축됩니다.  
분류기는 순차적으로 평가되며, 분류기의 응답이 부정적인 경우 이미지 창이 거부됩니다.  
모든 분류기는 관심 대상과 일치하는 창(즉, 높은 리콜)을 거의 거부하지 않도록 조정됩니다.  
그러나 캐스케이드의 첫 번째 분류기는 예를 들어 약한 가설의 수를 제한하여 얻은 평가 비용을 크게 줄이는 대가로 여러 개의 오탐(즉, 낮은 정밀도)을 반환할 수 있습니다.  
이 방식을 사용하면 계산 비용이 많이 들고 매우 정확한 분류기는 가장 어려운 경우에만 평가됩니다: 관심 대상을 닮은 창(즉, 긍정적인 샘플) 또는 객체와 쉽게 혼동할 수 있는 시각적 구조(즉, 어려운 부정적 샘플).

Finally, since each weak hypothesis is usually associated with an elementary feature, AdaBoost is also often used for feature selection. In some cases, feature selection improves the interpretability of the classifier. For instance, in the Viola-Jones face detector, the first few Haar wavelets selected by AdaBoost usually capture semantically meaningful anatomical structures such as the eyes and the nose.

마지막으로, 각 약한 가설은 일반적으로 기본 특징과 연관되기 때문에 AdaBoost는 특징 선택에도 자주 사용됩니다.  
경우에 따라 특징 선택은 분류기의 해석 가능성을 향상시킵니다.  
예를 들어, Viola-Jones 얼굴 검출기에서 AdaBoost가 선택한 처음 몇 개의 Haar 웨이블릿은 일반적으로 눈과 코와 같은 의미론적으로 의미 있는 해부학적 구조를 포착합니다.

# Autoencoder
