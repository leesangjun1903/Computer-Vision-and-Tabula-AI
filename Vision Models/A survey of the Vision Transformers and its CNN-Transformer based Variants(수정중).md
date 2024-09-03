# A survey of the Vision Transformers and its CNN-Transformer based Variants

최근에는 다양한 컴퓨터 비전 응용 분야에서 CNN(컨볼루션 신경망)을 대체할 수 있는 대안으로 비전 트랜스포머가 인기를 얻고 있습니다.  
이미지의 글로벌 관계에 초점을 맞추는 능력으로 인해 이러한 비전 트랜스포머는 용량이 크지만 CNN에 비해 일반화가 좋지 않을 수 있습니다.  
최근에는 비전 트랜스포머의 컨볼루션 및 셀프 어텐션 메커니즘의 하이브리드화가 로컬 및 글로벌 이미지 표현을 모두 활용하는 능력으로 인해 인기를 얻고 있습니다.  
하이브리드 비전 트랜스포머라고도 알려진 이러한 CNN-Transformer 아키텍처는 비전 애플리케이션에 대한 놀라운 결과를 보여주었습니다.  
최근 이러한 하이브리드 비전 트랜스포머의 수가 급격히 증가함에 따라 이러한 아키텍처에 대한 분류 및 설명이 필요합니다.  
이 Survey는 최근 비전 트랜스포머 아키텍처의 분류 체계, 특히 하이브리드 비전 트랜스포머의 분류 체계를 제시합니다.  
또한 Attention 메커니즘, 위치 임베딩, 다중 규모 처리, 컨볼루션 등 각 아키텍처의 주요 기능도 논의합니다.  
이 설문 조사는 다양한 컴퓨터 비전 작업에서 탁월한 성능을 제공할 수 있는 하이브리드 비전 트랜스포머의 잠재력을 강조합니다.  
또한 이는 빠르게 발전하는 이 분야의 미래 방향을 제시합니다.

# Introduction

디지털 이미지는 본질적으로 복잡하며 사물, 장면, 패턴과 같은 높은 수준의 정보를 나타냅니다.  
이 정보는 컴퓨터 비전 알고리즘에 의해 분석되고 해석되어 객체 인식, 움직임 추적, 특성 추출 등 이미지 내용에 대한 의미 있는 통찰력을 추출할 수 있습니다.  
컴퓨터 비전은 다양한 분야에 적용되어 활발하게 연구되고 있는 분야입니다.  
그러나 이미지 데이터에서 높은 수준의 정보를 추출하는 것은 밝기, 포즈, 배경의 혼잡함 등의 변화로 인해 어려울 수 있습니다.

CNN(Convolutional Neural Network)의 출현은 컴퓨터 비전 영역에 혁명적인 변화를 가져왔습니다.  
이러한 네트워크는 다양한 범위의 컴퓨터 비전 작업, 특히 이미지 인식, 객체 감지 및 분할에 성공적으로 적용되었습니다.  
CNN은 이미지에서 특성과 패턴을 자동으로 학습하는 기능으로 인해 인기를 얻었습니다.  
일반적으로 특성 모티프로 알려진 지역적 패턴은 이미지 전체에 체계적으로 분포되어 있습니다.  
컨볼루션 레이어의 다양한 필터는 다양한 특성 모티프를 포착하도록 설계되었습니다.  
CNN의 풀링 레이어는 차원 축소와 변형에 대한 견고성을 통합하는 데 활용됩니다.  
CNN의 로컬 수준 처리로 인해 공간 정보가 손실될 수 있으며, 이는 더 크고 복잡한 패턴을 처리할 때 성능에 영향을 줄 수 있습니다.

최근 Vaswani et al.이 처음 소개한 이후 트랜스포머로 전환이 이루어졌습니다.  
2017년에는 텍스트 처리 애플리케이션용으로 출시되었습니다.  
2018년에 Parmer 등은 이미지 인식 작업을 위해 트랜스포머를 활용하여 뛰어난 결과를 보여주었습니다.  
이후 다양한 비전 관련 애플리케이션에 트랜스포머를 적용하는 것에 대한 관심이 높아지고 있습니다.  
2020년 Dosovitskiy et al.은 이미지 분석을 위해 특별히 설계된 트랜스포머 아키텍처인 ViT(Vision Transformer)를 출시하여 경쟁력 있는 결과를 보여주었습니다.

ViT 모델은 입력 이미지를 특정 수의 패치로 분할하는 방식으로 작동하며, 각 패치는 이후에 평면화되어 일련의 트랜스포머 레이어에 공급됩니다.  
트랜스포머 레이어를 사용하면 모델이 패치와 해당 기능 간의 관계를 학습하여 이미지의 전역 규모에서 기능 모티프를 식별할 수 있습니다.  
로컬 수용 필드가 있는 CNN과 달리 ViT는 self-attention 모듈을 활용하여 장거리 관계를 모델링하므로 이미지의 전체 보기를 캡처할 수 있습니다.  
ViT의 전역 수용 필드는 공간 정보를 유지하여 이미지 전체에 분산된 복잡한 시각적 패턴을 식별하는 데 도움이 됩니다.

![](https://wikidocs.net/images/page/236673/Fig_TR_CV_Survey_01.png)

- multi-attention mechanism and convolution operation

CNN과 ViT는 디자인과 시각적 패턴을 캡처하는 방식의 차이 외에도(그림 1 참조) 귀납적 편향도 다릅니다.  
CNN은 인접 픽셀의 상관 관계에 크게 의존하는 반면 ViT는 최소한의 사전 지식을 가정하므로 레이블이 지정된 데이터에 덜 의존합니다.  
ViT 모델은 객체 인식, 분류, 의미론적 분할 및 기타 컴퓨터 비전 작업에서 탁월한 결과를 얻었습니다.

그러나 ViT의 대용량에도 불구하고 제한된 훈련 데이터의 경우 CNN에 비해 여전히 낮은 성능을 나타냅니다.  
게다가 수용 영역이 넓기 때문에 훨씬 더 많은 계산이 필요합니다.  
따라서 CNN-Transformers라고도 알려진 HVT(Hybrid Vision Transformers) 개념이 CNN과 ViT의 성능을 결합하기 위해 도입되었습니다.  
이러한 하이브리드 모델은 CNN의 컨볼루셔널 레이어를 활용하여 로컬 특성을 캡처한 다음 ViT에 공급되어 self-attention 메커니즘을 사용하여 글로벌 컨텍스트를 얻습니다.  
HVT는 많은 이미지 인식 작업에서 향상된 성능을 보여주었습니다.

최근에는 ViT의 최근 아키텍처 및 구현 발전을 논의하기 위해 다양하고 흥미로운 Survey가 실시되었습니다.  
이러한 문서의 대부분은 주로 ViT와 컴퓨터 비전의 응용 프로그램에 중점을 두고 있지만 NLP 응용 프로그램용으로 개발된 Transformer 모델을 기반으로 하는 자세한 논의도 제공합니다.  
한편, 본 survey는 CNN의 아이디어와 트랜스포머(CNN-Transformer) 아키텍처, 이들의 분류 및 응용을 결합한 최근의 하이브리드 비전 트랜스포머에 주로 중점을 둡니다.  
이 외에도 본 survey는 일반적인 ViT의 분류를 제시하고 핵심 아이디어(건축 설계)를 기반으로 새로 등장하는 접근 방식을 철저하게 분류하려고 합니다.  
이와 관련하여 먼저 ViT 네트워크의 필수 구성 요소를 소개한 다음 다양한 최신 ViT 아키텍처에 대해 논의합니다.

본 문서에서 논의된 보고된 ViT 모델은 기본 두드러진 특성에 따라 크게 6가지 범주로 분류됩니다.  
ViT에 대해 논의한 후에는 컨볼루션 작업과 다중 어텐션 메커니즘(multi-attention mechanism)의 이점을 모두 활용하는 모델인 HVT에 대한 자세한 논의가 이어집니다.  
또한, HVT의 최근 아키텍처와 다양한 컴퓨터 비전 작업에서의 응용에 대해 자세히 설명합니다.  
또한 아키텍처에서 self-attention과 결합하여 컨볼루션 작업을 통합하는 방법을 기반으로 HVT에 대한 분류를 제시합니다.  
우리의 분류 체계는 HVT를 7개의 주요 그룹으로 나누며, 각 그룹은 컨볼루셔널 연산자와 다중 주의 연산자를 모두 활용하는 다양한 방법을 반영합니다.

논문은 다음과 같이 구성됩니다. (그림 2 참조) 

![](https://wikidocs.net/images/page/236673/Fig_TR_CV_Survey_02.png)

섹션 1에서는 ViT(Vision Transformers)에 대한 체계적인 이해를 제시하고 CNN과의 차이점 및 하이브리드 비전 트랜스포머의 출현을 강조합니다.  
계속해서 섹션 2에서는 다양한 ViT 변형에 사용되는 기본 개념을 다루고, 섹션 3과 섹션 4에서는 각각 최근 ViT 및 HVT 아키텍처에 대한 분류를 제공합니다.  
섹션 5에서는 특히 컴퓨터 비전 분야에서 HVT의 사용에 중점을 두고 있으며 섹션 6에서는 현재 과제와 미래 방향을 제시합니다.  
마지막으로 7장에서는 Survey를 마무리한다.

# Fundamental Concepts in ViTs
Transformer의 기본 아키텍처 레이아웃은 그림 3에 나와 있습니다.  

![](https://wikidocs.net/images/page/236673/Fig_TR_CV_Survey_03.png)

처음에는 입력 이미지가 분할되고 평면화되어 패치 임베딩(Patch Embedding)이라고 알려진 저차원 선형 임베딩으로 변환됩니다.  
그런 다음 위치 임베딩과 클래스 토큰이 이러한 임베딩에 첨부되고 클래스 레이블을 생성하기 위해 트랜스포머의 인코더 블록에 공급됩니다.  
인코더 블록에는 MSA(Multi-Head Attention) 계층 외에도 FFN(Feed-Forward Neural) 네트워크, 정규화 계층 및 잔여 연결이 포함되어 있습니다.  
마지막으로 마지막 헤드(MLP 계층 또는 디코더 블록)는 최종 출력을 예측합니다.  
이러한 각 구성 요소는 다음 하위 섹션에서 자세히 설명됩니다.

## Patch embedding
패치 임베딩은 ViT 아키텍처에서 중요한 개념입니다.  
여기에는 이미지 패치를 벡터 표현으로 변환하는 작업이 포함되며, 이를 통해 ViT는 트랜스포머 기반 접근 방식을 사용하여 이미지를 토큰 시퀀스로 처리할 수 있습니다.  
입력 이미지는 고정된 크기의 겹치지 않는 부분으로 분할되고, 1차원 벡터로 평면화되고, D 임베딩 차원(방정식 1)이 있는 선형 레이어를 사용하여 고차원 특성 공간에 투영됩니다.  

$𝑿_{𝑝𝑎𝑡𝑐ℎ}^{𝑁×𝐷} =𝑅(𝑰_{𝑖𝑚𝑎𝑔𝑒}^{𝐴×𝐵×𝐶})$

이 접근 방식을 통해 ViT는 다양한 패치 간의 장기적인 종속성을 학습하여 이미지와 관련된 작업에서 유망한 결과를 얻을 수 있습니다.  
$𝑰_{𝑖𝑚𝑎𝑔𝑒}$ : 입력 이미지, 크기 : ${𝐴×𝐵×𝐶}$  
$𝑅(.)$ : $𝑿_{𝑝𝑎𝑡𝑐ℎ}$ 크기의 N 개수 패치를 D만큼 생성하는 재구성 함수  , ($𝑁= A/P × B/P$, $𝐷= 𝑃× 𝑃$)  
P : 패치 크기, C : 채널 수

## Positional embedding
ViT는 위치 인코딩을 활용하여 위치 정보를 입력 시퀀스에 추가하고 이를 네트워크 전체에 유지합니다.  
패치 간의 순차 정보는 패치 임베딩 내에 통합된 위치 임베딩을 통해 캡처됩니다.  
ViT 개발 이후 순차 데이터 학습을 위해 수많은 위치 임베딩 기술이 제안되었습니다. 이러한 기술은 세 가지 범주로 분류됩니다:

### Absolute Position Embedding (APE)
위치 임베딩은 인코더 블록 전에 APE를 사용하여 패치 임베딩에 통합됩니다.  

$𝑿=𝑿_{𝑝𝑎𝑡𝑐ℎ}+𝑿_{𝑝𝑜𝑠} \tag{2}$

X : Transformer 입력  
$𝑿_{𝑝𝑎𝑡𝑐ℎ}$ : 패치 임베딩 , $(𝑁+1)×𝐷$ 차원  
$𝑿_{𝑝𝑜𝑠}$ : 학습 가능한 위치 임베딩 , $(𝑁+1)×𝐷$ 차원 , D : 임베딩 차원  

학습 가능한 단일 또는 두 세트의 위치 임베딩에 해당하는 $𝑿_{𝑝𝑜𝑠}$을 학습하는 것이 가능합니다.

### Relative Position Embedding (RPE)
RPE(Relative Position Embedding) 기술은 상대 위치와 관련된 정보를 Attention 모듈에 통합하는 데 주로 사용됩니다.  
이 기술은 패치 간의 공간적 관계가 절대 위치보다 더 큰 비중을 갖는다는 아이디어에 기반을 두고 있습니다.  
RPE 값을 계산하기 위해 학습 가능한 매개변수를 기반으로 하는 조회 테이블이 사용됩니다.  
조회 프로세스는 패치 간의 상대적 거리에 따라 결정됩니다.  
RPE 기술은 다양한 길이의 시퀀스로 확장 가능하지만 훈련 및 테스트 시간이 늘어날 수 있습니다.  

### Convolution Position Embedding (CPE)
CPE(Convolutional Position Embedding) 방법은 입력 시퀀스의 2D 특성을 고려합니다.  
2D 컨볼루션은 2D 특성을 활용하기 위해 제로 패딩을 사용하여 위치 정보를 수집하는 데 사용됩니다.  
CPE(컨볼루션 위치 임베딩)를 사용하여 Vision Transformer의 여러 단계에서 위치 데이터를 통합할 수 있습니다.  
CPE는 특히 self-attention 모듈, FFN(Feed-Forward Network) 또는 두 인코더 레이어 사이에 도입될 수 있습니다.

## Attention Mechanism
ViT 아키텍처의 self-attention 메커니즘은 시퀀스의 엔터티 간의 관계를 명시적으로 표현하는 기능으로 인해 핵심 구성 요소입니다.  
이는 글로벌 상황 정보 측면에서 각 엔터티를 나타내고 이들 간의 상호 작용을 포착하여 다른 항목에 대한 한 항목의 중요성을 계산합니다.  
self-attention 모듈은 입력 시퀀스를 쿼리, 키, 값이라는 세 가지 다른 임베딩 공간으로 변환합니다.  
쿼리 벡터가 포함된 키-값 쌍 집합이 입력으로 사용되며 출력 벡터는 값의 가중 합계와 소프트맥스 연산자를 사용하여 계산됩니다.  
여기서 가중치는 채점 함수(방정식 3)로 계산됩니다.

$𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑸,𝑲,𝑽)=𝑠𝑜𝑓𝑡𝑚𝑎𝑥 \left( \frac{𝑸 ⋅ 𝑲^𝑇}{ \sqrt{𝑑_𝑘} } \right)⋅𝑽 \tag{3}$

$𝑸$ : Query, $𝑲^𝑇$ : 전치된 key, $𝑽$ : Value matrix  
$sqrt{𝑑_𝑘}$ : 배율 인수, $𝑑_𝑘$ : key matrix 크기  

### Multi-Head Self-Attention (MSA)
제한된 용량으로 인해 단일 헤드 self-attention 모듈은 몇 가지 위치에만 초점을 맞추고 다른 중요한 위치는 무시할 수 있습니다.  
이는 Self-Attention 블록의 병렬 스택을 사용하여 Self-Attention 레이어의 효율성을 높이는 MSA에 의해 해결됩니다.  
다양한 표현 하위 공간(쿼리, 키 및 값)을 attention 계층에 할당함으로써 MSA는 다양한 시퀀스 요소 간의 다양하고 복잡한 상호 작용을 포착할 수 있습니다.  
MSA는 여러 개의 self-attention 블록으로 구성됩니다.  
각 self-attention 블록과 관련된 쿼리, 키 및 값 하위 공간에 대한 학습 가능한 가중치 행렬이 있습니다.  
그 후 출력은 학습 가능한 매개변수 $𝑊^𝑂$를 사용하여 연결되고 출력 공간에 투영됩니다.  
attention 과정의 수학적 표현은 다음과 같습니다:  

$MSA(𝑸,𝑲,𝑽)$ $=𝐶𝑜𝑛𝑐𝑎𝑡(ℎ𝑒𝑎𝑑_1,ℎ𝑒𝑎𝑑_2,⋅⋅⋅,ℎ𝑒𝑎𝑑_ℎ)⋅𝑾^𝑂$  
$ℎ𝑒𝑎𝑑_𝑖 =𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑸_𝑖,𝑲_𝑖,𝑽_𝑖), \text{ where } 𝑖=1,2,...,ℎ$

모든 입력 시퀀스에 대해 필터를 동적으로 계산하는 Self-attention의 능력은 컨볼루션 프로세스에 비해 상당한 이점을 제공합니다.  
종종 정적인 컨볼루셔널 필터와 달리 self-attention은 입력 데이터의 특정 컨텍스트에 맞게 조정될 수 있습니다.  
Self-attention은 입력 포인트 수나 순열의 변화에도 강력하므로 불규칙한 입력을 처리하는 데 적합합니다.  
반면에 기존의 컨볼루션 절차는 가변 개체가 포함된 입력을 처리하는 데 적합하지 않으며 2D 이미지와 같은 격자형 구조가 필요합니다.  
Self-attention은 순차 데이터를 모델링하는 강력한 도구이며 자연어 처리와 관련된 작업에 효과적인 것으로 나타났습니다.

## Transformer layers
ViT 인코더는 입력 시퀀스를 처리하는 여러 레이어로 구성됩니다.  
이러한 계층은 계층 정규화, 잔여 연결, FFN(피드포워드 신경망) 및 MSA 메커니즘으로 구성됩니다.  
이러한 레이어는 입력 시퀀스의 복잡한 표현을 학습하기 위해 여러 번 반복되는 통합 블록을 생성하도록 배열됩니다.

### Residual connection
인코더/디코더 블록의 하위 계층은 잔여 링크를 활용하여 성능을 향상하고 정보 흐름을 강화합니다.  
추가 정보로 MSA의 출력 벡터에 원본 입력 위치 임베딩이 추가됩니다.  
그런 다음 잔여 연결 뒤에는 계층 정규화 작업이 수행됩니다(방정식 6).

$𝑿_{𝑜𝑢𝑡𝑝𝑢𝑡} = 𝐿𝑎𝑦𝑒𝑟𝑁𝑜𝑟𝑚(𝑿 + 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑿)) \tag{6}$

### Normalization layer
레이어 정규화에는 Pre-LN(Pre-LN)과 같은 다양한 방법이 있는데, 이 방법은 자주 활용되며 정규화 레이어는 MSA 또는 FFN 이전에 잔여 연결 내부에 배치됩니다.  
트랜스포머 모델의 훈련을 향상시키기 위해 배치 정규화를 포함한 다른 정규화 절차가 제안되었지만 특성 값의 변경으로 인해 효율적이지 않을 수 있습니다.

### Feed-forward network
입력 데이터에서 더 복잡한 속성을 얻기 위해 모델에 트랜스포머별 피드포워드 네트워크(FFN)가 사용됩니다.  
여기에는 두 개의 완전히 연결된 레이어와 레이어 사이의 GELU와 같은 비선형 활성화 함수가 포함되어 있습니다(방정식 7).  

$𝐹𝐹𝑁(𝑿)= 𝑏^{[2]}+𝑾^{[2]}∗𝜎(𝑏^{[1]}+𝑾^{[1]}∗𝑿) \tag{7}$

FFN은 self-attention 모듈 이후의 모든 인코더 블록에서 활용됩니다.  
FFN의 숨겨진 레이어는 일반적으로 2048 차원을 갖습니다.  
이러한 FFN 또는 MLP 레이어는 로컬이며 전역 self-attention 레이어와 병진적으로 동일합니다.

식 7에서 비선형 활성화 함수 GELU는 𝜎로 표현됩니다.  
네트워크의 가중치는 $𝑾^{[1]}$ 및 $𝑾^{[2]}$로 표시되는 반면, $𝑏^{[1]}$ 및 $𝑏^{[2]}$는 레이어별 편향에 해당합니다.  

## Hybrid Vision Transformers (CNN-Transformer Architectures)
컴퓨터 비전 작업 영역에서는 비전 트랜스포머가 인기를 얻었지만 CNN에 비해 여전히 이미지별 유도 편향이 부족합니다.  
CNN에서는 지역성, 병진 등분산 및 2차원 이웃 구조가 전체 모델의 모든 레이어에 뿌리내려 있습니다.  
또한 커널은 인접한 픽셀 간의 상관 관계를 활용하여 좋은 특성을 빠르게 추출할 수 있습니다.  
반면 ViT에서는 이미지가 선형 레이어를 통해 인코더 블록에 공급되는 선형 패치(토큰)로 분할됩니다.  
선형 레이어의 특성상 지역 정보를 추출하는 데 그다지 효과적이지 않습니다.

많은 HVT 설계는 특히 패치 및 토큰화를 위한 이미지 처리 워크플로 시작 시 이미지의 로컬 특성을 캡처할 때 컨볼루션의 효율성에 중점을 두었습니다.  
예를 들어 CvT(Convolutional Vision Transformer)는 컨볼루션 투영을 사용하여 이미지 패치의 공간 정보와 하위 수준 정보를 학습합니다.  
또한 CNN의 공간 다운샘플링 효과를 모방하기 위해 토큰 수를 점진적으로 줄이고 토큰 너비를 늘리는 계층적 레이아웃을 활용합니다.  
마찬가지로 CEIT(Convolution-enhanced Image Transformers)는 컨볼루션 작업을 활용하여 이미지-토큰 모듈을 통해 하위 수준 특성을 추출합니다.  
새로운 시퀀스 풀링 기술은 토큰화를 수행하기 위해 conv-pool-reshape 블록을 통합하는 Compact Convolutional Transformer(CCT)에 의해 제공됩니다.  
또한 처음부터 훈련했을 때 CIFAR10과 같은 소규모 데이터세트에서 약 95%의 정확도를 보여줬는데, 이는 일반적으로 다른 기존 ViT로는 달성하기 어렵습니다.

최근 여러 연구에서는 ViT(Vision Transformers)의 로컬 특성 모델링 능력을 향상시키는 방법을 조사했습니다.  
LocalViT는 깊이별 컨볼루션을 사용하여 로컬 특성을 모델링하는 능력을 향상합니다.  
LeViT는 ViT 아키텍처 초기에 4개의 레이어로 구성된 CNN 블록을 사용하여 추론 시 채널을 점진적으로 늘리고 효율성을 향상시킵니다.  
ResT에서도 유사한 방법을 사용하지만 변동하는 이미지 크기를 관리하기 위해 깊이별 컨볼루션 및 적응형 위치 인코딩이 사용됩니다.

추가 데이터가 없으면 CoAtNets의 고유한 심도 컨볼루션 아키텍처와 상대적 Self-Attention은 뛰어난 ImageNet 상위 1 정확도를 달성합니다.  
더 강력한 크로스 패치 연결을 생성하기 위해 Shuffle Transformer는 셔플 작업을 제공하고 CoaT는 다양한 규모의 토큰 간의 관계를 인코딩하기 위해 깊이별 컨볼루션과 교차 주의를 통합하는 하이브리드 접근 방식입니다.  
또 다른 방법인 "Twins"는 분리 가능한 깊이별 컨볼루션과 상대 조건부 위치 임베딩을 통합하여 PVT를 기반으로 합니다.  
최근에는 하이브리드 아키텍처인 MaxVit이 다축 주목이라는 아이디어를 선보였습니다.  
하이브리드 블록은 MBConv 기반 컨볼루션과 블록별 self-attention 및 그리드별 self-attention으로 구성되며, 이 블록이 여러 번 반복되면 계층적 표현을 생성하고 이미지 생성 및 분할과 같은 작업이 가능합니다.  
블록별 및 그리드별 Attention 레이어는 각각 로컬 및 전역 특성을 추출할 수 있습니다.  
컨볼루션 및 트랜스포머 모델의 장점은 이러한 하이브리드 설계에 결합되도록 고안되었습니다.

Architectural level modifications in vision transformers
3.1. Patch-based approaches
 3.1.1. T2T-ViT (Tokens-to-Token Vision Transformer)
 3.1.2. TNT-ViT (Transformer in Transformer)
 3.1.3. DPT (Deformable Patch-based Transformer)
 3.1.4. CrowdFormer
3.2. Knowledge transfer-based approaches
 3.2.1. DeiT (Data-efficient Image Transformers)
 3.2.2. TaT (Target aware Transformer)
 3.2.3. TinyViT (Tiny Vision Transformer)
3.3. Shifted window-based approaches
3.4. Attention-based approaches
 3.4.1. CaiT (Class attention layer)
 3.4.2. DAT (Deformable attention transformer)
 3.4.3. SeT (patch-based Separable Transformer)
3.5. Multi-transformer-based approaches
 3.5.1. CrossViT (Cross Vision Transformer)
 3.5.2. Dual-ViT (Dual Vision Transformer)
 3.5.3. MMViT (Multiscale Multiview Vision Transformer)
 3.5.4. MPViT (Multi-Path Vision Transformer)
3.6. Details and taxonomy of HVTs (CNN-Transformer architectures)
 3.6.1. Early-layer integration
 3.6.2. Lateral-layer integration
 3.6.3. Sequential integration
 3.6.4. Parallel integration
 3.6.5. Hierarchical integration
 3.6.6. Attention-based integration
 3.6.7. Channel boosting-based integration
Applications of HVTs
4.1. Image/video recognition
4.2. Image generation
4.3. Image segmentation
4.4. Image Restoration
4.5. Feature extraction
4.6. Medical image analysis
4.7. Object Detection
4.8. Pose Estimation
Challenges
Future directions
Conclusion
