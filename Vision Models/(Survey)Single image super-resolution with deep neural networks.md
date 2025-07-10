https://krasserm.github.io/2019/09/04/super-resolution/

이 글은 단일 이미지 초고해상도에 대한 소개입니다.  
최근 몇 년 동안의 몇 가지 중요한 개발 사항을 다루고 Tensorflow 2.0에서 구현한 내용을 보여줍니다.  
주로 초고해상도 모델을 미세 조정하기 위한 특수 잔여 네트워크 아키텍처와 생성적 적대 네트워크(GAN)에 중점을 둡니다.  

초고해상도는 저해상도(LR) 이미지에서 고해상도(HR) 이미지를 복구하는 프로세스입니다.  
복구된 HR 이미지를 초고해상도 이미지 또는 SR 이미지 라고 합니다 .  
초고해상도는 LR 이미지의 단일 픽셀에 대한 솔루션이 매우 많기 때문에 ill-posed 문제입니다.  
bilinear or bicubic interpolation과 같은 간단한 접근 방식은 LR 이미지의 로컬 정보만 사용하여 해당 SR 이미지의 픽셀 값을 계산합니다.  

반면, 지도 학습 머신 러닝 접근법은 많은 수의 예제에서 LR 이미지에서 HR 이미지로의 매핑 함수를 학습합니다.  
초고해상도 모델은 LR 이미지를 입력으로, HR 이미지를 대상으로 학습합니다.  
이러한 모델에서 학습한 매핑 함수는 HR 이미지를 LR 이미지로 변환하는 downgrade function의 역수입니다.  
downgrade function은 알려지거나 알려지지 않을 수 있습니다.

알려진 downgrade function은 예를 들어 bicubic downsampling과 같은 이미지 처리 파이프라인에서 사용됩니다.  
알려진 downgrade function을 사용하면 HR 이미지에서 LR 이미지를 자동으로 얻을 수 있습니다.  
이를 통해 방대한 양의 HR 이미지에서 대규모 교육 데이터 세트를 생성할 수 있어 self-supervised learning이 가능합니다 .  

https://hackernoon.com/self-supervised-learning-gets-us-closer-to-autonomous-learning-be77e6c86b5a

downgrade function이 알려지지 않은 경우, 지도 학습 모델 학습에는 기존 LR 및 HR 이미지 쌍이 있어야 하며, 이는 수집하기 어려울 수 있습니다.  
또는 비지도 학습 방법을 사용하여 페어되지 않은 LR 및 HR 이미지에서 다운그레이드 함수를 근사하는 방법을 학습할 수 있습니다.  
그러나 이 문서에서는 알려진 downgrade function (bicubic downsampling)를 사용하고 지도 학습 접근 방식을 따릅니다.  

# High-level architecture
![](https://krasserm.github.io/img/2019-09-04/figure_1.png)

최신 초고해상도 모델 중 다수는 LR 공간에서 대부분의 mapping functio을 학습한 다음 네트워크 끝에 하나 이상의 upsampling layer을 학습합니다.  
이를 그림 1에서 post-upsampling SR 이라고 합니다 .  
upsampling layer은 학습 가능하며 이전 convolution layer와 함께 end-to-end 방식으로 함께 훈련됩니다.

이전 접근 방식은 먼저 pre-defined upsampling 작업으로 LR 이미지를 업샘플링한 다음 HR 공간에서 매핑을 학습했습니다( pre-defined upsampling SR ).  
이 접근 방식의 단점은 레이어당 더 많은 매개변수가 필요하여 더 높은 계산 비용이 발생하고 더 깊은 신경망의 구성이 제한된다는 것입니다.  

## Residual design
초고해상도는 LR 이미지에 포함된 대부분의 정보가 SR 이미지에 보존되어야 함을 요구합니다.  
따라서 초고해상도 모델은 주로 LR과 HR 이미지 간의 잔차를 학습합니다.  
따라서 residual 네트워크 설계는 매우 중요합니다.  
Identity 정보는 스킵 연결을 통해 전달되는 반면 고주파 콘텐츠의 재구성은 네트워크의 주 경로에서 수행됩니다.  

![](https://krasserm.github.io/img/2019-09-04/figure_2.png)

그림 2는 여러 layer에 걸친 global skip connection을 보여줍니다.  
이러한 layer는 종종 ResNet 또는 특수 변형(섹션 EDSR 및 WDSR 참조 )에서와 같이 residual blocks입니다.  
잔여 블록의 ㅣocal skip connection은 네트워크를 최적화하기 쉽게 만들어 더 깊은 네트워크의 구성을 지원합니다.  

## Upsampling layer
이 글에서 사용하는 upsampling layer는 sub-pixel convolution layer입니다.  
주어진 크기의 입력 H * W * C, upsampling factor s, sub-pixel convolution layer는 H×W×s^2C 를 만들고 convolution 연산을 거쳐 sH×sW×C 모양으로 바꿔 upsampling 연산을 마칩니다.

![](https://krasserm.github.io/img/2019-09-04/figure_3.png)

결과는 s로 인해 공간적으로 확장된 출력입니다.  

https://arxiv.org/abs/1609.05158

다른 대안은 transposed convolution입니다.  

https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

transposed convolution도 학습할 수 있지만, sub-pixel convolution보다 receptive field가 작아서 맥락적 정보를 적게 처리하여 덜 정확한 예측을 초래할 수 있다는 단점이 있습니다.  

# Super-resolution models
## EDSR
이 고수준 아키텍처를 따르는 초고해상도 모델 중 하나는 논문 Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)에 설명되어 있습니다.  

https://arxiv.org/abs/1707.02921

이 모델은 NTIRE 2017 초고해상도 챌린지 에서 우승했습니다 .  
EDSR 아키텍처 개요는 다음과 같습니다.  

![](https://krasserm.github.io/img/2019-09-04/figure_4.png)

residual block design은 ResNet과 다릅니다.   
Batch normalization layer는 그림 5의 오른쪽에 표시된 대로 마지막 ReLU 활성화와 함께 제거되었습니다.  

![](https://krasserm.github.io/img/2019-09-04/figure_5.png)

EDSR 저자들은 Batch normalization이 이미지의 스케일 정보를 잃고 활성화 범위의 유연성을 감소시킨다고 주장합니다.  
Batch normalization 레이어를 제거하면 초고해상도 성능이 향상될 뿐만 아니라 GPU 메모리도 최대 40%까지 감소하여 상당히 큰 모델을 학습할 수 있습니다.  

EDSR은 초고해상도 스케일(즉, upsampling factor)에 대해 single sub-pixel upsampling 레이어 ×2, ×3나 ×4의 스케일이 적용된 두 개의 upsampling 레이어를 사용합니다.  
edsr함수는 Tensorflow 2.0으로 EDSR 모델을 구현합니다.  

## WDSR
또 다른 초고해상도 모델은 EDSR의 파생 모델이며, NTIRE 2018 초고해상도 챌린지 의 현실적인 트랙에서 우승한 논문 Wide Activation for Efficient and Accurate Image Super-Resolution 에 설명되어 있습니다 .  
이 모델은 전체 매개변수 수를 늘리지 않고 Identity 매핑 경로의 채널 수를 줄이고 각 잔여 블록의 채널 수를 늘려 잔여 블록 설계를 추가로 변경합니다.  
WDSR-A 및 WDSR-B 모델의 잔여 블록 설계는 그림 6에 나와 있습니다 .

![](https://krasserm.github.io/img/2019-09-04/figure_6.png)

저자들은 ReLU 이전의 채널 수를 잔여 블록에서 늘리면 더 많은 정보가 활성화 함수를 통과할 수 있어 모델 성능이 더욱 향상될 것이라고 추측합니다.  
또한 weight normalization를 구현 하면 더 깊은 모델의 훈련과 수렴이 더욱 용이해져 EDSR 훈련에 사용된 것보다 훨씬 더 높은 학습률을 사용할 수 있다는 사실도 발견했습니다.  

https://arxiv.org/abs/1602.07868

이는 EDSR에서 weight normalization layer를 제거하면 더 깊은 모델을 훈련하기가 더 어려워지기 때문에 놀라운 일이 아닙니다.  
weight normalization은 weight vector의 방향을 크기에서 분리하는 neural network weight의 재매개변수화일 뿐이며, 이는 최적화 문제의 조건을 개선하고 수렴 속도를 높입니다.  

하지만 weight normalization layer 파라미터의 데이터에 의존된 초기화는 수행되지 않습니다.  
이는 Batch normalization과 유사하게 feature를 재조정하여 EDSR 논문에서 보여주고 WDSR 논문에서 확인한 것처럼 모델 성능을 저하시킵니다.  
반면, 데이터에 외존된 초기화 없이 가중치 정규화만 사용하면 더 깊은 WDSR 모델의 정확도가 더 높아집니다.  

# Model training
이 섹션에서 훈련된 코드의 실행을 건너뛰려면 여기에서 사전 훈련된 모델을 다운로드하여 다음 섹션 에서 LR 이미지에서 SR 이미지를 생성하는 데 사용할 수 있습니다.

## Data
EDSR 및 WDSR 모델을 훈련하기 위해 DIV2K 데이터 세트를 사용합니다 .  
이는 다양한 콘텐츠가 포함된 LR 및 HR 이미지 쌍의 데이터 세트입니다.  
LR 이미지는 다양한 다운그레이드 기능에 사용할 수 있습니다.  
bicubic여기서 다운샘플링을 사용합니다.  
800개의 훈련 HR 이미지와 100개의 검증 HR 이미지가 있습니다.  

데이터 증강을 위해, 무작위 자르기, 뒤집기 및 회전을 수행하여 다양한 수의 훈련 이미지를 얻습니다.  

## Pixel loss
pixel-wise L2 loss 와 pixel-wise L1 loss는 초고해상도 모델을 훈련하는 데 자주 사용되는 손실 함수입니다.  
이들은 각각 HR 이미지 간의 pixel-wise mean squared error와 pixel-wise mean absolute error를 측정합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-24%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.12.30.png)

H, W, C 는 이미지의 Height, Width, Channel 수이며 pixel-wise L2 loss는 초고해상도 경쟁에서 자주 사용되는 평가 지표인 PSNR을 직접 최적화합니다 .  
실험 결과 pixel-wise L1 loss는 때로 더 나은 성능을 달성할 수 있으므로 EDSR 및 WDSR 훈련에 사용됩니다.  

픽셀 단위 손실 함수의 주요 문제는 지각적 품질이 떨어진다는 것입니다.  
생성된 SR 이미지는 종종 고주파 콘텐츠, 사실적인 텍스처가 부족하고 흐릿하게 인식됩니다.  
이 문제는 Perceptual loss 함수로 해결됩니다.  

## Perceptual loss
더 나은 지각적 품질을 가진 SR 이미지를 생성하기 위한 이정표 논문은 생성적 적대적 네트워크를 사용한 사진적 사실적 단일 이미지 초고해상도 (SRGAN)입니다.  
저자는 content loss 과 adversarial loss 로 구성된 지각적 손실 함수를 사용합니다 . 

https://arxiv.org/abs/1609.04802

콘텐츠 손실은 SR 및 HR 이미지에서 추출한 deep feature들을 사전 훈련된 VGG 네트워크 ϕ 와 비교합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-24%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.29.47.png)

ϕl(I) : feature map at layer l, Hl , Wl and Cl : the height, width and number of channels of that feature map

Generator로 초고해상도 모델을 학습합니다.  
G : 생성적 적대 네트워크 (GAN) 에서 . GAN Discriminator D 는 SR과 HR 이미지를 구별하기 위해 최적화된 반면 생성기는 판별기를 속이기 위해 더욱 사실적인 SR 이미지를 생성하기 위해 최적화되었습니다.  
이들은 generator loss을 content loss와 결합합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-24%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.34.37.png)

그래서 perceptual loss 를 만들어내어 초고해상도 모델에 최적화 대상으로 적용됩니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-24%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%203.34.44.png)

GAN에서 초고해상도 모델 즉 생성기를 처음부터 훈련하는 대신, pixel-wise loss로 사전 훈련하고 perceptual loss로 모델을 미세 조정합니다.  
SRGAN 논문은 EDSR 의 전신인 초고해상도 모델로 SRResNet을 사용합니다 .  

실험에서 SRGAN 접근 방식이 EDSR 및 WDSR 모델의 미세 조정에도 매우 효과적이라는 것을 발견했습니다.  

# Results
학습된 EDSR 모델을 사용하여 이제 LR 이미지 에서 SR 이미지를 만들 수 있습니다.  
perceptual loss로 미세 조정하면 pixel-wise loss만으로 학습하는 것보다 SR 이미지에서 더 사실적인 텍스처가 생성되는 것을 명확하게 볼 수 있습니다.  

![](https://krasserm.github.io/img/2019-09-04/output_13_0.png)

또한, 미세조정된 WDSR-B 모델은 더욱 사실적인 텍스처를 가진 SR 이미지를 생성합니다.

![](https://krasserm.github.io/img/2019-09-04/output_15_0.png)

