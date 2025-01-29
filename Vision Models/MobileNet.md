# MobileNet
- https://ctkim.tistory.com/entry/%EB%AA%A8%EB%B0%94%EC%9D%BC-%EB%84%B7

# MobileNetV2
mobilenetv2는 ReLU 함수를 거치게 되면 정보가 손실된다는 것에 영감을 받아 이를 최소화하기 위해 Inverted Residuals와 Linear Bottlenecks를 제안함

depthwise convolution연산시 채널별로 쪼개서 계산하는데 relu함수 적용시 0으로 처리될때가 많음. 그래서 채널수가 적을때는 리니어하게해야함

linear bottenecks는 레이어에 채널 수가 적다면 linear activation을 사용합니다. 비선형 함수인 relu를 사용하게 되면 정보가 손실되기 때문입니다. 

inverted residuals는 기존의 BottleNeck 구조는 첫 번째 1x1 conv layer에서 채널 수를 감소시키고 3x3 conv로 전달합니다. 채널 수가 감소된 레이어에서 ReLU 함수를 사용하면 정보 손실이 발생하게 됩니다. 따라서 첫 번째 레이어에서 입력값의 채널 수를 증가시키고 3x3conv layer로 전달합니다. 

# Reference
- 주요 CNN알고리즘 구현 : MobileNet v2 https://velog.io/@tbvjvsladla/26.-%EC%A3%BC%EC%9A%94-CNN%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B5%AC%ED%98%84-MobileNet-v2-1-%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-%EA%B3%A0%EA%B8%89%EC%8B%9C%EA%B0%81-%EA%B0%95%EC%9D%98-%EB%B3%B5%EC%8A%B5
- https://m.blog.naver.com/phj8498/222689054103
