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

# Reference
https://velog.io/@minkyu4506/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Densely-Connected-Convolutional-NetworksDenseNet
