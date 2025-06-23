# DenseNet(2017) : Densely Connected Convolutional Networks

모든 레이어들을 서로 연결해 레이어간 정보의 흐름량을 최대한으로 한다.  
feed-forward nature을 보존하기 위해 이전까지 연산을 하며 얻은 모든 feature map을 입력값으로 받는다.  

많은 parameter를 요구하지 않습니다.  
ResNet 대비 정보와 그레디언트의 흐름이 향상되었습니다.  
그래서 학습시키기 쉬워졌습니다.  
왜냐하면 이는 곧 역전파를 계속해도 그레디언트의 손실이 적다는 뜻이기 때문입니다.  
그레디언트가 더 멀리, 더 확실히 전달된다면 손실함수가 원하는 방향으로 모델이 쉽게 학습됩니다.
regularizing effect도 있습니다. 그래서 학습에 사용할 데이터셋이 작아도 과적합(overfitting)이 일어나지 않습니다.  

하나의 값으로 합쳐진게 아니라 서로 연결되어 하나의 데이터가 된 것이죠.  
이러한 dense connectivity을 모든 레이어에서 거치기 때문에 저자는 자신들이 만든 네트워크를 Dense Convolutional Network (DenseNet)이라 불렀습니다.

## Densely Connected Convolutional Networks (DenseNet)

**DenseNet(덴스넷)**은 기존의 합성곱 신경망(CNN)과 다르게, 네트워크의 각 레이어를 모든 이전 레이어와 직접 연결하는 구조를 가진 혁신적인 딥러닝 모델입니다. 
---

### **DenseNet의 기본 아이디어**

- **기존 CNN 구조**: 일반적인 CNN에서는 각 레이어가 바로 앞의 레이어에서만 정보를 받아옵니다. 즉, 정보가 순차적으로 한 단계씩 전달됩니다.
- **DenseNet 구조**: DenseNet에서는 한 레이어가 이전의 모든 레이어로부터 정보를 받아옵니다. 예를 들어, 5번째 레이어는 1~4번째 레이어의 출력(특징 맵)을 모두 입력으로 사용합니다[1][2][3][4].

#### **비유**
- 기존 CNN: 릴레이 경주처럼 다음 사람에게만 바통을 넘김
- DenseNet: 모든 선수에게 동시에 바통을 넘기는 것과 비슷

---

### **DenseNet의 구조**

- **Dense Block**: 여러 개의 레이어가 모여 있는 블록으로, 블록 내의 모든 레이어가 서로 연결됩니다.
- **Transition Layer**: Dense Block 사이에 위치하며, 특징 맵의 크기를 줄이고(풀링), 채널 수를 줄여줍니다. 이를 통해 네트워크가 너무 커지는 것을 방지합니다[2][4].

#### **연결 방식**
- 각 레이어는 이전 레이어의 출력들을 모두 이어붙여(concatenate)서 입력으로 사용합니다.
- 예를 들어, 4번째 레이어의 입력 = 1, 2, 3번째 레이어의 출력 + 원본 입력

---

### **DenseNet의 장점**

- **특징 재사용**: 이전 레이어에서 추출한 특징을 계속 활용하므로, 불필요하게 같은 특징을 여러 번 학습하지 않아도 됩니다. 이는 파라미터(가중치) 수를 줄여주고, 모델이 더 효율적으로 학습할 수 있게 합니다[2][3][4].
- **그래디언트 소실 문제 완화**: 각 레이어가 손실 함수로부터 직접적으로 그래디언트를 전달받을 수 있어, 깊은 네트워크에서도 학습이 잘 됩니다[1][2][3][4].
- **효율적인 학습**: 파라미터 수가 적고, 정보가 잘 전달되어서 적은 데이터로도 좋은 성능을 낼 수 있습니다.
- **과적합 방지**: 파라미터가 적고, 특징을 효율적으로 재사용하므로 과적합 위험이 줄어듭니다[2][4].

---

### **DenseNet과 기존 CNN, ResNet 비교**

| 구조      | 연결 방식                          | 장점                       | 단점/특징                    |
|-----------|-----------------------------------|----------------------------|------------------------------|
| CNN       | 바로 이전 레이어만 연결            | 단순, 직관적               | 깊어질수록 학습 어려움        |
| ResNet    | 바로 이전 + skip connection(더하기)| 깊은 네트워크 학습 가능     | 일부 정보만 skip             |
| **DenseNet** | 모든 이전 레이어와 직접 연결(이어붙이기) | 특징 재사용, 그래디언트 전달 우수 | 메모리 사용량 증가 가능성     |

---

### **DenseNet의 실제 활용 예시**

- 이미지 분류(예: ImageNet)
- 객체 탐지
- 의료 영상 분석
- 자연어 처리, 음성 인식 등 다양한 분야[2][4]

---

### **DenseNet의 직관적 요약**

> DenseNet은 "모든 레이어가 서로 친구처럼 직접 소통하는" 구조입니다.  
> 이 덕분에 정보가 빠짐없이 잘 전달되고, 효율적으로 학습할 수 있습니다.

---

### **수식으로 보는 DenseNet의 연결 수**

- 레이어가 $$ L $$개라면, DenseNet의 연결 수는  
$$
\frac{L(L+1)}{2}
$$
  입니다. (예: 4개의 레이어 → 10개의 연결)[1][3][4]

---

DenseNet은 깊은 신경망에서 발생하는 문제(정보 소실, 파라미터 낭비 등)를 효과적으로 해결하면서도, 다양한 분야에서 뛰어난 성능을 보여주는 대표적인 딥러닝 모델입니다.

[1] https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
[2] https://www.giskard.ai/glossary/densenet
[3] https://romanakchurin.com/deep-learning/densenet/
[4] https://serp.ai/posts/densenet/
[5] https://www.sciencedirect.com/science/article/abs/pii/S0031320320304131
[6] https://www.studocu.com/in/document/indian-institute-of-technology-goa/computer-science/densely-connected-convolutional-networks/75784003
[7] https://www.youtube.com/watch?v=-W6y8xnd--U
[8] https://community.wolfram.com/groups/-/m/t/2166844
[9] https://paperswithcode.com/paper/densely-connected-convolutional-networks
[10] https://python.plainenglish.io/densenet-densely-connected-convolutional-neural-networks-0fd219379138?gi=30cdbd1794b2

## Densenet 논문리뷰 및 구현
- https://csm-kr.tistory.com/10

# Reference
- https://velog.io/@minkyu4506/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Densely-Connected-Convolutional-NetworksDenseNet
- https://deep-learning-study.tistory.com/528
