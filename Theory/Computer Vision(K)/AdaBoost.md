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
## Encoder-decoder architectures
## ◃Deep Generative Models ◃Dimensionality Reduction ◃Generative Adversarial Network (GAN) ◃Manifold Learning ◃Principal Component Analysis (PCA) ◃Recurrent Neural Network

## Definition
An autoencoder is a deep neural architecture comprising two parts, namely, (1) an encoder network that maps each input data point to a point in a different (latent) space and (2) a decoder network that maps the points in the latent space back to the data space. The two components are trained jointly in an unsupervised way, so that their composition approximately preserves points from a given training dataset.

오토인코더는 두 부분으로 구성된 심층 신경망 아키텍처입니다.  
즉, (1) 각 입력 데이터 포인트를 다른 (잠재) 공간의 한 지점으로 매핑하는 인코더 네트워크와 (2) 잠재 공간의 포인트를 다시 데이터 공간으로 매핑하는 디코더 네트워크입니다.  
두 구성 요소는 비지도 학습 방식으로 공동으로 학습되며, 그 구성 요소는 주어진 학습 데이터셋의 포인트를 대략적으로 보존합니다.

## Background
Autoencoders are a very popular deep architec- ture for unsupervised learning going back to at least 1980s [1, 2]. Similar to other unsupervised learning methods such as principal component analysis [3], the objective of autoencoder learning is to find some latent representation of the points in a training dataset that preserves the information contained in the data points, while simplifying the data in a certain way. In the case of autoencoder, the mapping from the data points to their latent representations is parameterized by a deep feedforward network (an encoder). The learning also aims to recover an approximate inverse mapping of the encoder that is also parameterized as a deep network (a decoder). Training the encoder and the decoder in parallel ensures that the discovered latent representation of the data preserves most of the information contained in the data.

오토인코더는 적어도 1980년대로 거슬러 올라가는 비지도 학습을 위한 매우 인기 있는 딥 아키텍처입니다 [1, 2].  
주성분 분석 [3]과 같은 다른 비지도 학습 방법과 마찬가지로 오토인코더 학습의 목적은 데이터 포인트에 포함된 정보를 보존하면서 특정 방식으로 데이터를 단순화하는 학습 데이터셋에서 포인트의 잠재 표현을 찾는 것입니다.  
오토인코더의 경우, 데이터 포인트에서 잠재 표현으로의 매핑은 deep feedforward network (an encoder) 에 의해 매개변수화됩니다.  
이 학습은 또한 인코더의 근사 역 매핑을 복원하는 것을 목표로 하며, 이는 deep network (a decoder) 로도 매개변수화됩니다.  
인코더와 디코더를 병렬로 학습하면 발견된 데이터의 잠재 표현이 데이터에 포함된 대부분의 정보를 보존할 수 있습니다.

To uncover the latent (hidden) structure of the dataset (or the underlying distribution), certain constraints are usually imposed on the architecture of the encoder and/or the decoder. Once the autoencoder is trained, its components or the discovered latent representations can be used in various ways. In particular, the latent representations of data points can often be used as features for different machine learning tasks.

데이터셋의 잠재(숨겨진) 구조(또는 기본 분포)를 밝히기 위해 일반적으로 인코더 및/또는 디코더의 아키텍처에 특정 제약이 가해집니다.  
오토인코더가 학습되면 그 구성 요소나 발견된 잠재 표현을 다양한 방식으로 사용할 수 있습니다.  
특히 데이터 포인트의 잠재 표현은 다양한 머신 러닝 작업의 특징으로 자주 사용될 수 있습니다.

## Theory
Autoencoders are deep architectures trained in an unsupervised way for a dataset x1,x2,.. .,xN in a certain data space X. In a probabilistic setting, we may think of the dataset as a sample from some underlying distribution pX defined on X. Alternatively, we may think of the training dataset as a set of samples from some manifold (the data manifold M) that lies within the data space and forms the support of the distribution pX. Autoencoder learning can then be seen as a way to find a suitable parameterization of the data manifold and is thus related to manifold learning methods.

오토인코더는 특정 데이터 공간 X에서 데이터셋 x1, x2, ..., xN에 대해 비지도 방식으로 훈련된 심층 아키텍처입니다.  
확률적 설정에서는 데이터셋을 X에 정의된 기본 분포 $p_X$ 의 샘플로 생각할 수 있습니다.  
또는, 훈련 데이터셋을 데이터 공간 내에 위치하고 분포 $p_X$ 의 support(도메인(domain)의 부분집합)를 형성하는 일부 매니폴드(데이터 매니폴드 M)의 샘플 집합으로 생각할 수도 있습니다.  
오토인코더 학습은 데이터 매니폴드의 적절한 매개변수화를 찾는 방법으로 볼 수 있으며, 따라서 매니폴드 학습 방법과 관련이 있습니다.

An autoencoder (Fig. 1) includes an encoder eφ with learnable parameters φ that maps each example x to its latent representation z = eφ(x) from a certain latent space Z. The structure of X and Z may be arbitrary, e.g., X may contain images of a certain size, and Z may correspond to a Euclidean space of a certain dimensionality. The second part of the autoencoder, the decoder dθ with learnable parameters θ operates in the reverse direction to the encoder. The decoder thus maps points z ∈ Z to the data space X. The full autoencoder network then corresponds to the composition of the encoder and the decoder:

오토인코더(그림 1)는 각 예제 x를 특정 잠재 공간 Z로부터 잠재 표현 $z = e_φ(x)$ 로 매핑하는 학습 가능한 매개변수 φ를 가진 인코더 $e_φ$ 를 포함합니다.  

<img width="841" alt="스크린샷 2025-04-14 오후 10 23 02" src="https://github.com/user-attachments/assets/529b1048-ba14-4b46-8094-77f305675211" />

X와 Z의 구조는 임의일 수 있으며, 예를 들어 X는 특정 크기의 이미지를 포함할 수 있고 Z는 특정 차원의 유클리드 공간에 해당할 수 있습니다.  
오토인코더의 두 번째 부분인 학습 가능한 매개변수 θ를 가진 디코더 $d_θ$ 는 인코더에 반대 방향으로 작동합니다.  
따라서 디코더는 z ∈ Z를 데이터 공간 X에 매핑합니다. 그런 다음 전체 오토인코더 네트워크는 아래와 같이 인코더와 디코더의 구성에 해당합니다:

<img width="142" alt="스크린샷 2025-04-14 오후 10 22 12" src="https://github.com/user-attachments/assets/894b4168-7b36-4b24-bcc1-a32857661ffd" />

The exact structure of the encoder and the decoder can also be arbitrary. In modern computer vision, it is common to use convolutional architectures [4] with multiple layers for both networks. An autoencoder is trained by approximating the identity mapping on the dataset X. The training loss is therefore defined as:

인코더와 디코더의 정확한 구조는 임의로 설정할 수도 있습니다.  
현대 컴퓨터 비전에서는 두 네트워크 모두에 여러 레이어가 있는 컨볼루션 아키텍처 [4]를 사용하는 것이 일반적입니다.  
오토인코더는 데이터셋 X의 신원 매핑을 근사화하여 학습됩니다. 따라서 학습 손실은 다음과 같이 정의됩니다:

<img width="295" alt="스크린샷 2025-04-14 오후 10 24 44" src="https://github.com/user-attachments/assets/56c06e44-7694-4a7f-9e4d-58612ee17a31" />

where the function Δ measures the dissimilarity between its arguments. In the simplest case, one may use the squared Euclidean distance Δ(x,y) = ∥x− y∥2 2, though more sophisticated variants, such as perceptual dissimilarity [5], are often used in recent works. The choice of Δ plays an important role, since the autoencoders are usually designed in a way that the training loss cannot be minimized to zero, leaving some mismatch between the training data points xi and their reconstructions. The choice of the dissimilarity measure Δ thus determines which kind of dissimilarities between data points and reconstructions are penalized more and which are penalized less.

함수 Δ는 인수 간의 비유사성을 측정합니다.  
가장 간단한 경우에는 유클리드 거리 제곱 Δ(x,y) = ∥x- y ∥2 2(l2 norm)를 사용할 수 있지만, 지각적 비유사성 [5]와 같은 더 정교한 변형이 최근 연구에서 자주 사용됩니다.  
오토인코더는 일반적으로 훈련 손실을 0으로 최소화할 수 없는 방식으로 설계되어 훈련 데이터 포인트 xi와 그 재구성 간에 약간의 불일치가 남기 때문에 Δ 선택이 중요한 역할을 합니다.  
따라서 비유사성 측정 Δ 의 선택은 데이터 포인트와 재구성 간의 비유사성 중 어느 것이 더 많은 불이익을 받고 어떤 것이 덜 불이익을 받는지를 결정합니다.

The training process of an autoencoder cor- responds to the minimization of the loss (2) using some variant of stochastic gradient descent. As with many other unsupervised learning tech- niques, the goal of the training is to learn some latent representation of data that uncovers some latent (hidden) structure in the data. This means that the dataset z1,z2, . . . ,zN , where zi = eφ(xi) (and the underlying distribution pZ = eφ(pX)), should have a form that is simpler in some sense than the original dataset and the original distribution.

오토인코더의 훈련 과정은 확률적 경사 하강법의 변형을 사용하여 손실(2)을 최소화하는 것에 해당합니다.  
다른 많은 비지도 학습 기법들과 마찬가지로, 훈련의 목표는 데이터의 잠재적(숨겨진) 구조를 밝혀내는 데이터의 잠재적 표현을 학습하는 것입니다.  
즉, $z_i = e_φ(x_i)$ (그리고 기본 분포 pZ = e φ(pX))인 데이터셋 z1, z2, . . . .zN은 원래 데이터셋과 원래 분포보다 어떤 의미에서는 더 간단한 형태를 가져야 함을 의미합니다.

From the manifold viewpoint, learning autoencoder essentially amounts to finding the parametrization of the data manifold. One indicator of success here (which is often used by researchers to evaluate the quality of an autoencoder) is the plausibility of interpolation in the latent space. Thus, given two data samples x1 and x2 and their latent representation z1 and z2, the latent points lying on the line segment that connects z1 and z2 in the latent space should be mapped by the decoder onto the data manifold, i.e., ∀λ ∈ [0;1] dθ(λz1 + (1− λ)z2) ∈ M. When data are images of a certain kind, e.g., face images, this condition can be verified by looking at the reconstructions dθ(λz1 + (1− λ)z2) and checking whether they look like face images. Note that a line segment connecting x1 and x2 in the original data space would typically not lie on the data manifold M, as most data manifolds have highly non-convex shape in the embedding space.

매니폴드 관점에서 오토인코더를 학습하는 것은 본질적으로 데이터 매니폴드의 매개변수화를 찾는 것과 같습니다.  
여기서 성공의 한 지표는 (연구자들이 오토인코더의 품질을 평가할 때 자주 사용하는) 잠재 공간에서의 보간 가능성입니다.  
따라서 두 개의 데이터 샘플 x1과 x2, 그리고 그 잠재 표현 z1과 z2가 주어졌을 때, 디코더는 잠재 공간에서 z1과 z2를 연결하는 선분 위의 잠재 지점을 데이터 매니폴드, 즉 ∀λ ∈ [0;1] d θ(z1 + (1- λ)z2) ∈ M에 매핑해야 합니다.  
데이터가 특정 종류의 이미지(예: 얼굴 이미지)인 경우, 이 조건은 재구성된 $d_θ(z_1 + (1- λ)z_2)$을 보고 얼굴 이미지처럼 보이는지 확인함으로써 확인할 수 있습니다.  
원래 데이터 공간에서 x1과 x2를 연결하는 선분은 일반적으로 데이터 매니폴드 M 위에 위치하지 않으며, 대부분의 데이터 매니폴드는 임베딩 공간에서 매우 비볼록적인 모양을 가지고 있기 때문입니다.

### Types of Autoencoders
Given enough capacity of the networks e and d as well as large enough space Z, the autoencoder is likely to learn to copy the data points into the latent space (potentially with some trivial injective transformation T ) and then to copy it back to the original space (while reverting the transformation T ). This is because such solution achieves lowest possible (zero) loss. In this case, the latent representation will not be in any ways simpler than the original data representation, and the data manifold will not be properly parameter- ized. Below, we discuss several ways how autoen- coders can be encouraged to learn simplified data representation.

네트워크 e와 d의 충분한 용량과 충분히 큰 공간 Z가 주어지면, 오토인코더는 데이터 포인트를 잠재 공간으로 복사하는 방법(잠재적으로 사소한 주입 변환 T가 있을 수 있음)을 배우고, 그 후 원래 공간으로 다시 복사하는 방법(변환 T를 되돌리는 방법)을 배울 가능성이 높습니다.  
이는 이러한 솔루션이 가능한 한 낮은 (제로) 손실을 달성하기 때문입니다.  
이 경우, 잠재 표현은 원래 데이터 표현보다 어떤 방식으로든 간단하지 않으며, 데이터 매니폴드는 적절하게 매개변수화되지 않을 것입니다.  
아래에서는 오토인코더가 단순화된 데이터 표현을 학습하도록 권장되는 여러 가지 방법에 대해 논의합니다.

#### Autoencoders as architectures for nonlinear dimensionality reduction. 
In this approach, the dimensionality of Z is taken to be substantially smaller than the dimensionality of X. In this situation, learning an autoencoder effectively amounts to dimensionality reduction, as the learning process seeks to identify the most important factors of variation in the data in order to preserve them within the latent representations.

이 접근 방식에서는 Z의 차원이 X의 차원보다 상당히 작은 것으로 간주됩니다.  
이 상황에서 오토인코더를 학습하는 것은 학습 과정에서 잠재 표현 내에서 데이터를 보존하기 위해 가장 중요한 변동 요인을 식별하려고 하기 때문에 사실상 차원 축소에 해당합니다.

#### Regularized autoencoders
Alternatively, or in addition to having Z of small dimensionality, it is common to impose certain regularization on the learning process. For example, one may penalize a certain norm of the latent vectors zi [6] and/or penalize a certain norm of the network parameters φ and θ [7]. Adding such regular- ization effectively prevents the autoencoder from learning the identity function and forces it to learn a simplified latent representation of data.

또는 작은 차원의 Z를 가지는 것 외에도 학습 과정에 특정 정규화를 부과하는 것이 일반적입니다.  
예를 들어, 잠재 벡터 zi [6]의 특정 노름에 페널티를 주거나 네트워크 매개변수 φ 및 θ [7]의 특정 노름에 페널티를 가할 수 있습니다.  
이러한 정규화를 추가하면 오토인코더가 항등 함수를 학습하는 것을 효과적으로 방지하고 데이터의 단순화된 잠재 표현을 학습하도록 강제할 수 있습니다.

#### Denoising autoencoders
Alternatively, or in addition to the above approaches, one can regularize the training process by injecting (adding) noise or performing some structured corruption process on the training points, so that the autoencoder receives the corrupted versions ˜ xi of the data points xi during training. In this scenario, the autoencoder needs to combine reconstruction with restoration, as the loss function still computes the difference between the reconstruction obtained from the corrupted data and the original data points:

또는 위의 접근 방식 외에도 노이즈를 주입(추가)하거나 훈련 지점에 구조화된 손상 과정을 수행하여 훈련 과정을 정규화할 수 있으며, 이를 통해 오토인코더는 훈련 중에 손상된 데이터 지점 xi의 ˜ xi 버전을 수신할 수 있습니다.  
이 시나리오에서는 손실 함수가 손상된 데이터와 원본 데이터 지점 간의 차이를 여전히 계산하기 때문에 오토인코더는 복원과 재구성을 결합해야 합니다:

<img width="355" alt="스크린샷 2025-04-14 오후 10 36 45" src="https://github.com/user-attachments/assets/0f2843e3-1dba-4afb-a1bb-5d8b80eab6a9" />

Once again, denoising autoencoder cannot attain low training loss by simply copying the data from the input to the output via the latent space, since such copying will not remove the corruption effect. Instead, the autoencoder has to learn to project the corrupted examples onto the manifold containing the original distribution [8] and there- fore has to uncover the latent representation of data, from which the original data points can be reconstructed.

다시 한 번 말씀드리지만, 노이즈 제거 오토인코더는 입력에서 출력으로 데이터를 잠재 공간을 통해 복사하는 것만으로는 낮은 학습 손실을 달성할 수 없습니다.  
왜냐하면 이러한 복사는 손상된 예제를 원래 분포 [8]을 포함하는 매니폴드에 투영하는 방법을 배워야 하기 때문입니다.  
따라서 데이터의 잠재적 표현을 밝혀내야 하며, 이를 통해 원래 데이터 포인트를 재구성할 수 있습니다.

#### Variational autoencoders
Variational autoen- coders (VAEs) [9, 10] combine several of the above ideas and do so in a probabilistic setting. In a variational autoencoder, the encoder and the decoder are thought as stochastic functions. The encoder maps each data point to a distribution in the latent space (which is usually taken to be Gaussian with diagonal covariance matrix, so that both the mean vector and the covariance parameters are predicted by the encoder). The decoder is also designed to map points in the latent space to the (usually Gaussian) distribution in the data space (though the covariance matrix is often fixed and only the mean vector is predicted). The autoencoding process inside VAE corresponds to mapping a training data sample to the distribution in the latent space, sampling from the resulting distribution and mapping the sample back to the data space. The minus log-likelihood of the input data sample w.r.t. the resulting distribution is then minimized. The learning is regularized by penalizing the Kulback-Leibler divergence between the encoding of each data sample and the unit zero-mean Gaussian. When the regularization coefficient equals one, the total learning objective can be interpreted as the maximization of the so-called evidence lower bound on the log-likelihood of the data points.

변분 오토인코더(VAE) [9, 10]는 위의 여러 아이디어를 결합하여 확률적 설정에서 이를 수행합니다.  
변분 오토인코더에서는 인코더와 디코더를 확률 함수로 간주합니다.  
인코더는 각 데이터 포인트를 잠재 공간의 분포(일반적으로 대각선 공분산 행렬이 있는 가우시안으로 간주되므로 인코더에 의해 평균 벡터와 공분산 매개변수가 모두 예측됩니다)로 매핑합니다.  
디코더는 또한 잠재 공간의 포인트를 데이터 공간의 (일반적으로 가우시안) 분포로 매핑하도록 설계되었습니다(공분산 행렬은 종종 고정되어 평균 벡터만 예측되지만).  
VAE 내부의 오토인코딩 프로세스는 훈련 데이터 샘플을 잠재 공간의 분포로 매핑하여 결과 분포에서 샘플링하고 샘플을 다시 데이터 공간으로 매핑하는 것에 해당합니다.  
그런 다음 입력 데이터 샘플의 로그 우도를 최소화합니다.  
학습은 각 데이터 샘플의 인코딩과 표준 제로 평균 가우시안 사이의 쿨백-라이블러(KL) 발산에 페널티를 부여하여 정규화됩니다.  
정규화 계수가 1이면 총 학습 목표는 데이터 포인트의 로그 우도에 대한 evidence lower bound의 최대화로 해석할 수 있습니다.

### Related Models
Autoencoders have deep connections to a number of models and tasks. Some of these connections are discussed below.

오토인코더는 여러 모델 및 작업과 깊은 연관성을 가지고 있습니다. 이러한 연결 중 일부는 아래에서 설명합니다.

#### Principal component analysis (PCA)
PCA [3] can be regarded as a particular type of an autoencoder with the encoder and the decoder having the fully connected single-layer architecture under additional constraints on the matrices and biases of the layers and more efficient training algorithms (namely, singular value and eigenvalue decompositions). The usage patterns discussed below are thus common to autoencoders and PCA, though autoencoders with multiple convolutional layers in most circumstances attain much better performance for image datasets.

PCA [3]은 인코더와 디코더가 완전히 연결된 단일 레이어 아키텍처를 가진 특정 유형의 오토인코더로 간주될 수 있으며, 이는 레이어의 행렬과 바이어스, 그리고 더 효율적인 학습 알고리즘(즉, 특이값 및 고유값 분해)에 대한 추가적인 제약 하에 이루어집니다.  
따라서 아래에서 설명하는 사용 패턴은 오토인코더와 PCA에 공통적으로 적용되지만, 대부분의 상황에서 다중 컨볼루션 레이어를 가진 오토인코더는 이미지 데이터셋에서 훨씬 더 나은 성능을 발휘합니다.

#### Data compression
When the dimensionality of the latent space is lower than the original space or if the latent representation of the dataset is more amenable for compression algorithms, training an autoencoder can be used for lossy data compression [11].

잠재 공간의 차원이 원래 공간보다 낮거나 데이터셋의 잠재 표현이 압축 알고리즘에 더 적합한 경우, 오토인코더를 훈련하여 손실 데이터 압축을 수행할 수 있습니다 [11].

#### Generative adversarial networks (GANs)
GANs [12] are another kind of latent models learned to approximate the data manifold/distribution. In their original form, GANs allow to map points from latent space to data space (as do decoders within autoencoders), but not vice versa. Multiple hybrid models that combine autoencoders with GANs exist [13, 14].

GAN [12]은 데이터 매니폴드/분포를 근사하는 방법을 배우는 또 다른 종류의 잠재 모델입니다.  
원래 형태의 GAN은 잠재 공간에서 데이터 공간으로 포인트를 매핑할 수 있게 해주지만(자동 인코더 내의 디코더와 마찬가지로), 그 반대는 아닙니다. 자동 인코더와 GAN을 결합한 여러 하이브리드 모델이 존재합니다[13, 14].

#### Generative latent optimization (GLO)
The recently proposed GLO model [15] is another deep latent model learned to parameterize the data manifold. GLO can be regarded as a simplification of the autoencoder model, where only the decoder network is trained and reused after training.

최근 제안된 GLO 모델 [15]은 데이터 매니폴드를 매개변수화하는 방법을 학습한 또 다른 심층 잠재 모델입니다.  
GLO는 디코더 네트워크만 학습하고 학습 후 재사용하는 오토인코더 모델의 단순화로 간주할 수 있습니다.

```
15.
Optimizing the latent space of generative networks.
```

## Applications
Autoencoders have a number of applications, and below we provide examples of the most common patterns of usage.

오토인코더에는 여러 가지 응용 프로그램이 있으며, 아래에서는 가장 일반적인 사용 패턴의 예를 제공합니다.

### Feature extraction
Once an autoencoder is trained, its encoder part eφ can be used as a feature extractor for various machine learning tasks including supervised learning. In the semi-supervised training scenario, a large amount of unlabeled data {x′ i} can be used to train an autoencoder, and a small amount of labeled data {xi,yi} (where yi is a label of xi) can then be used to train a predictor (e.g., a classifier) cψ with learnable parameters ψ in the latent space, so that yi ≈ cψ(eφ(xi)). Assuming that the training of the autoencoder is successful, and the distribution in the latent space has a simpler form than in the data space, training such predictor in the latent space Z will lead to better generalization than training a predictor in the original data space X. Note that the decoder dθ is effectively discarded in such a scenario.

오토인코더가 훈련되면, 인코더 부분 $e_φ$ 는 지도 학습을 포함한 다양한 기계 학습 작업의 특징 추출기로 사용될 수 있습니다.  
반지도 학습 시나리오에서는 대량의 라벨이 없는 데이터 ${x′_i}$를 사용하여 오토인코더를 훈련시킬 수 있으며, 소량의 라벨이 붙은 데이터 {xi,yi}(여기서 yi는 xi의 라벨)를 사용하여 잠재 공간에서 학습 가능한 매개변수 ψ를 가진 예측기(예: 분류기) $c_ψ$ 를 훈련시켜 $yi ≈ c(e_φ(x_i))$를 ψ로 만들 수 있습니다.  
오토인코더의 훈련이 성공적이고 잠재 공간에서의 분포가 데이터 공간보다 더 간단한 형태를 가진다고 가정하면, 잠재 공간 Z에서 이러한 예측기를 훈련시키는 것이 원래 데이터 공간 X에서 예측기를 훈련시키는 것보다 더 나은 일반화로 이어질 것입니다.  
이러한 시나리오에서는 디코더 $d_θ$가 효과적으로 폐기된다는 점에 유의하세요.

### Pretraining generator networks
It is common to use the decoder part of a pretrained autoencoder for conditional sampling/synthesis. In more detail, consider a scenario when a limited amount of aligned data {(xi,yi)} ⊂ X ⊗ Y (e.g., images from the image space X and their text descriptions from the description space Y) and a large amount of unaligned samples {x′ j } ⊂ X (e.g., images without descriptions) are given. In order to learn image synthesis conditioned on text description, one may first learn an autoencoder on unaligned data {x′ j } with latent space Z and then learn a mapping fτ from Y to the latent space Z with learnable parameters τ, such that its composition with the pretrained decoder dθ maps xi close to yi (i.e., xi ≈ dθ(fτ (yi))). The composition of the mapping fτ and the decoder will thus provide a mapping from Y to X (text- to-image in our example) that is trained both on aligned and unaligned data. Such mapping is likely to generalize to unseen data better than the mapping learned solely on aligned data.

조건부 샘플링/합성을 위해 사전 학습된 오토인코더의 디코더 부분을 사용하는 것이 일반적입니다.  
더 자세히 설명하자면, 제한된 양의 정렬된 데이터 {(xi,yi)} ⊂ X ⊗ Y(예: 이미지 공간 X의 이미지 및 설명 공간 Y의 텍스트 설명)와 많은 양의 정렬되지 않은 샘플 {x′ j ⊂ X(예: 설명이 없는 이미지)가 주어졌을 때의 시나리오를 고려해 보세요.  
텍스트 설명에 조건을 둔 이미지 합성을 학습하기 위해 먼저 잠재 공간 Z를 가진 정렬되지 않은 데이터 {x′ j τ}에 대한 오토인코더를 학습한 다음, 학습 가능한 매개변수 τ를 가진 Y에서 잠재 공간 Z로의 매핑 f θ를 학습하여 사전 학습된 디코더 d θ(xi ≈ d(yi))와의 구성이 xi에 가깝게 매핑되도록 할 수 있습니다.  
따라서 매핑 f τ와 디코더의 구성은 정렬된 데이터와 정렬되지 않은 데이터 모두에 대해 학습된 Y에서 X로의 매핑(예시 텍스트에서 이미지로)을 제공할 것입니다.  
이러한 매핑은 정렬된 데이터만으로 학습된 매핑보다 보이지 않는 데이터에 더 잘 일반화될 가능성이 높습니다.

### Disentangling of factors
Under certain circumstances, the latent distribution learned by an autoencoder has some factor disentangling properties so that a certain factor of variation affects all or almost all dimensions in the original space X but only a small subset of dimensions in the latent space Z. For example, when autoencoder is trained for face images, certain latent dimensions may correspond to person identity, while being invariant to expression (and vice versa). Such disentagling is most common to observe in a variational autoencoder (VAE) due to the diagonal covariance structure imposed on the latent distributions within VAE.

특정 상황에서 오토인코더가 학습한 잠재 분포는 원래 공간 X의 모든 또는 거의 모든 차원에 영향을 미치지만 잠재 공간 Z의 일부 차원에만 영향을 미치는 특정 변동 요인의 분리 특성을 가지고 있습니다.  
예를 들어, 얼굴 이미지에 대해 오토인코더를 학습할 때 특정 잠재 차원은 표현에 불변하면서도 사람의 정체성에 해당할 수 있습니다(그리고 그 반대의 경우도 마찬가지입니다).  
이러한 분리는 VAE 내의 잠재 분포에 부과되는 대각선 공분산 구조로 인해 변분 오토인코더(VAE)에서 관찰되는 것이 가장 일반적입니다.

### Data manipulation in latent space
Even if fac- tors of variation are not disentangled in the latent space, it often happens that high-level (semantic) editing is easier in the latent space than in the data space. For example, given a dataset of face images, it often happens that a change of a certain attribute (e.g., changing neutral face expression to smiling expression) can be easily modeled in the latent space. Sometimes, such transformation that is very complex in the data space can be well approximated by a simple translation by a certain vector in the latent space. The parameters of the transformation in the latent space can be learned from a small amount of extra annotation (e.g., in the example above, the translation vector can be learned as a difference between the mean of the latent representations of several smiling faces and the mean of the latent representations of several neutral faces).

변동의 요인들이 잠재 공간에서 분리되지 않더라도, 데이터 공간보다 잠재 공간에서 고수준(의미론적) 편집이 더 쉬운 경우가 종종 발생합니다.  
예를 들어, 얼굴 이미지 데이터셋이 주어졌을 때, 특정 속성의 변경(예: 중립적인 얼굴 표정을 웃는 표정으로 바꾸는 것)이 잠재 공간에서 쉽게 모델링될 수 있습니다.  
때때로 데이터 공간에서 매우 복잡한 이러한 변환은 잠재 공간에서 특정 벡터에 의한 간단한 번역으로 잘 근사될 수 있습니다.  
잠재 공간에서의 변환 매개변수는 소량의 추가 주석(예: 위의 예시에서 번역 벡터는 여러 웃는 얼굴의 잠재 표현 평균과 여러 중립 얼굴의 잠재 표현 평균 간의 차이로 학습될 수 있습니다)을 통해 학습될 수 있습니다.

### Unsupervised restoration and anomaly detection
The ability of autoencoders to project data on the data manifold can be used to restore corrupted data as well as to identify outlier samples that do not belong to the data manifold [16, 17].

오토인코더가 데이터 매니폴드에 데이터를 투영하는 능력은 손상된 데이터를 복원하고 데이터 매니폴드에 속하지 않는 이상값 샘플을 식별하는 데 사용할 수 있습니다 [16, 17].

```
16 : Electric power system anomaly detection using neural networks
17 : Structured Denoising Autoencoder for Fault Detection and Analysis
```




 
