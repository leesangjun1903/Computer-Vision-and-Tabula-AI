# SRGAN : Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network | Super resolution
# SRGAN 소개

## Abs
더 빠르고 깊은 컨볼루션 신경망을 사용하여 단일 이미지 초해상도의 정확도와 속도에 있어서 획기적인 발전에도 불구하고, 한 가지 핵심 문제는 크게 해결되지 않은 채로 남아 있습니다.   
대규모 업스케일링 정도(large upscaling factors)가 큰 경우 초해상도로 처리할 때 더 미세한 질감(finer texture details)을 어떻게 복구할 수 있을까요?  
최적화 기반 초해상도 방법의 동작은 주로 목적 함수(objective function)의 선택에 의해 주도됩니다.  
최근 연구는 주로 평균 제곱 재구성 오류(mean squared reconstruction error, MSE)를 최소화하는 데 중점을 두고 있습니다.  
결과는 피크 신호 대 잡음비(PSNR)가 높지만 고주파 세부 정보(high-frequency details)가 부족한 경우가 많고 더 높은 해상도에서 예상되는 눈으로 보이는 고화질이라고 생각하는 부분과 일치하지 않는다는 점에서 지각적으로 만족스럽지 않습니다.  

이 논문에서는 이미지 초해상도(SR)를 위한 생성적 적대 신경망(GAN)인 SRGAN을 제시합니다.  
저희가 아는 한, 이 프레임워크는 4배 업스케일링 정도에 대한 사실적 이미지(photo-realistic natural images)를 추론할 수 있는 최초의 프레임워크입니다.   
이를 위해 적대적 손실(adversarial loss)과 콘텐츠 손실(content loss)로 구성된 지각적 손실 함수(perceptual loss function)를 제안합니다.  
적대적 손실은 초해상도 이미지와 원본의 사실적 이미지를 구별하도록 훈련된 판별기 네트워크(discriminator network)를 사용하여 자연 이미지 다양체(natural image manifold)에 대한 솔루션을 추진합니다.  
또한 픽셀 공간의 유사성 대신 지각적 유사성에 의해 동기 부여된 콘텐츠 손실(content loss)을 사용합니다.  
저희의 심층 잔차 네트워크는 공개 벤치마크에서 크게 다운샘플링된 이미지(downsampled images)에서 사실적 질감(photo-realistic textures)을 복구할 수 있습니다.  
PSNR, SSIM 은 MSE 기반 계산방식이라 이미지 성능 평가에 적합하지 않습니다. 
따라서 광범위한 평균 의견 점수(MOS) 테스트를 사용하여 평가하고 SRGAN을 이용했을 때 지각 품질이 크게 향상되었음을 보여줍니다.  
SRGAN으로 얻은 MOS 점수는 최첨단 방법으로 얻은 MOS 점수보다 원래 고해상도 이미지의 MOS 점수에 더 가깝습니다. 

# Introduction
저해상도(LR) 대응물로부터 고해상도(HR) 이미지를 추정하는 매우 어려운 작업을 초해상도(SR)라고 합니다. SR은 컴퓨터 비전 연구 커뮤니티 내에서 상당한 관심을 받았으며 광범위한 응용 분야를 가지고 있습니다[63, 71, 43].   
초해상도 문제의 잘못된 특성은 재구성된 SR 이미지에 텍스처 세부 정보가 일반적으로 없는 높은 업스케일링 요인에서 특히 두드러집니다.   
지도 SR 알고리즘의 최적화 대상은 일반적으로 복구된 HR 이미지와 실측 데이터 사이의 평균 제곱 오차(MSE)를 최소화하는 것입니다.   
MSE를 최소화하면 SR 알고리즘을 평가하고 비교하는 데 사용되는 일반적인 척도인 피크 신호 대 잡음비(PSNR)도 최대화되기 때문에 편리합니다[61].   
그러나 높은 텍스처 세부 정보와 같이 지각적으로 관련된 차이를 캡처하는 MSE(및 PSNR)의 능력은 픽셀 단위의 이미지 차이를 기반으로 정의되기 때문에 매우 제한적입니다[60, 58, 26].   
이는 그림 2에 나와 있으며, 여기서 가장 높은 PSNR이 지각적으로 더 나은 초해상도 결과를 반드시 반영하지는 않습니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.17.04.png)
기존 SR 모델 중 하나인 SRResNet이 생성한 이미지를 매우 확대해보면, original HR image와 비교했을 때 texture detail이 떨어지는 것을 확인할 수 있습니다.  
저자들은 이 원인이 기존 SR 모델들의 loss function에 있다고 보았습니다.   
기존 SR 모델들의 목표는 보통 복구된 HR 이미지와 원본 이미지의 pixel 값을 비교하여 pixel-wise MSE를 최소화하는 것입니다.   
그러나 pixel-wise loss를 사용하면 high texture detail을 제대로 잡아내지 못하는 한계가 있습니다.  
저자들은 이전 연구와는 다르게 VGG network의 high-level feature map을 이용한 perceptual loss를 제시하여 이런 문제를 해결하였다고 합니다. 
초해상도 이미지와 원본 이미지의 지각적 차이(perceptual loss)는 복구된 이미지가 Ferwerda[16]에 의해 정의된 것처럼 사실적이지 않다는 것을 의미합니다.  

이 작업에서는 스킵 연결이 있는 심층 잔차 네트워크(ResNet)를 사용하고 MSE에서 벗어난 초해상도 생성 적대 네트워크(SRGAN)를 제안합니다.   
이전 작업과 달리 고해상도 이미지와 지각적으로 구별하기 어려운 솔루션을 장려하는 판별기(discriminator)와 결합된 VGG 네트워크[49, 33, 5]의 고수준 특징 맵(high-level feature maps)을 사용하여 새로운 지각 손실(perceptual loss)을 정의합니다.   
4배 업스케일링 팩터의 초해상도의 이미지가 그림 1에 나와 있습니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.17.32.png)

# Method
## Adversarial network architecture
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.18.00.png)

1. Generator network, G  
이미지에 나와있는 것처럼 똑같은 layout을 지닌 B개의 residual block으로 구성되어있습니다.  

Residual block의 구성:
- kernel size: 3 x 3
- kernel 개수: 64
- stride: 1
- Batch normalization layer
- Activation function: ParametricReLU (PReLU)

일반적으로 convolution layer를 이용하면 그 image의 차원은 작아지거나 동일하게 유지됩니다.   
초해상도(super resolution)를 위해 image의 차원을 증가시켜야 하는데 여기서 이용된 방식이 sub-pixel convolution이라고 합니다.   

2. Discriminator Network, D  
LeakyReLU(α=0.2)를 사용했고, max-pooling은 이미지 크기를 줄이므로 사용하지 않았습니다.  
- 3 × 3 kernel을 사용하는 conv layer 8개로 구성
- feature map의 수는 VGG network처럼 64부터 512까지 커짐.

마지막 feature maps 뒤에는 dense layer 두 개, 그리고 classification을 위한 sigmoid가 붙습니다.

## Perceptual loss function

Loss function으로 Perceptual loss를 사용하며 content loss와 adversarial loss로 구성되어 있습니다.  
이중 adversarial loss는 우리가 일반적으로 알고 있는 GAN의 loss와 비슷합니다. 조금 특별한 부분은 Content loss입니다.  

![](https://media.vlpt.us/images/cha-suyeon/post/623efc08-b6a6-4cf8-b29f-6aa1283ee629/image.png)

### Content loss

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.44.23.png)

- Φ_i,j = Feature map obtained by the jth convolution (after activation) before the ith maxpooling layer within the VGG 19 network
- 시그마 안에 값 = Generator가 생성한 이미지와 original HR 이미지로 부터 얻은 Feature map 사이의 Euclidean distance
- Wi,j & Hi,j = the dimensions of the respective feature maps within the VGG network. VGG 앞 feature maps 차원

Generator을 이용해 얻어낸 가짜 고해상도 이미지를 진짜 고해상도 이미지와 Pixel by pixel로 비교하는 것을 Per-pixel loss라고 하고,
각 이미지를 pre-trained CNN 모델에 통과시켜 얻어낸 feature map을 비교하는 것을 Perceptual loss라고 합니다. 
동일한 이미지이나 한 pixel씩 오른쪽으로 밀려있는 두 이미지가 있다고 가정해보겠습니다.  
이런 경우 loss는 0 이어야하겠지만 per-pixel loss를 구하면 절대 0이 될 수 없습니다.  
per-pixel loss의 이러한 단점은 super resolution의 고질적인 문제인 Ill-posed problem 때문에 더 부각됩니다.  

Ill-posed problem이란 저해상도 이미지를 고해상도로 복원을 해야 하는데, 가능한 고해상도의 이미지가 여러 개 존재하는 것을 말합니다.  
![](https://hoya012.github.io/assets/img/deep_learning_super_resolution/2.PNG)

GAN 모델을 이용하여 여러 개의 가능한 고해상도 이미지 (아래 그림상 Possible solutions)를 구하여도 MSE based Per-pixel loss를 사용하면 possible solutions 들을 평균내는 결과를 취하게 되므로, GAN이 생성한 다양한 high texture detail들이 smoothing 되는 결과를 초래합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.17.49.png)

이런 단점을 해결하기 위해 저자들은 GAN이 생성한 HR 이미지와 Original HR 이미지를 Pretrained VGG 19에 통과시켜 얻은 Feature map 사이의 Euclidean distance를 구하여 content loss를 구하였습니다.

### Adversarial loss

D_theta_D 는 Generator가 생성한 이미지를 진짜라고 판단할 확률로 앞에 - 가 붙어있으므로 이를 최소화하는 방향으로 학습합니다.  
기존 GAN loss는 log (1-x)의 형태로 되어잇으나 이러면 training 초반 부에 학습이 느리다는 단점이 있다고 합니다.  
이를 -log (x) 형태로 바꾸어주면 학습 속도가 훨씬 빨라진다고 하네요.

# Experiments

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2010.18.13.png)
SR GAN이 생성한 이미지를 매우 확대해보면, SRResNet이 만든 이미지와 비교했을 때 texture detail이 좋아졌음을 확인할 수 있습니다.  

또한 MOS (Mean Opinion score) testing을 진행하였을 때 SRGAN의 엄청난 성능을 확인했습니다.  
MOS (Mean Opinion score) testing은 26명의 사람에세 1점(bad) 부터 5점 (excellent)까지 점수를 매기도록 한 것입니다.  

(기존 Super Resolution rating에서 흔히 사용하던 PSNR이나 SSIM과 같은 점수를 사용하지 않은 이유는 해당 점수들이 MSE를 이용하여 기계적으로 점수를 산출할 뿐, 실제 사람의 평가를 제대로 반영하지 못하는 한계를 보였기 때문이라고 합니다.)

# Conclusion
저희는 널리 사용되는 PSNR 측정으로 평가할 때 공개 벤치마크 데이터 세트에서 새로운 최신 기술을 설정하는 심층 잔차 네트워크 SRResNet에 대해 설명했습니다.  
저희는 이 PSNR에 초점을 맞춘 이미지 초해상도 구현의 몇 가지 한계를 강조하고 GAN을 훈련하여 적대적 손실로 콘텐츠 손실 기능을 강화하는 SRGAN을 도입했습니다.  
광범위한 MOS 테스트를 통해 대규모 업스케일링 정도(4배)에 대한 SRGAN 재구성이 최첨단 참조 방법으로 얻은 재구성보다 상당한 차이로 더 사실적임을 확인했습니다.

### Reference
https://kevinitcoding.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-SRGAN-%EB%85%BC%EB%AC%B8-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC-Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-Network#1
https://wikidocs.net/146367
