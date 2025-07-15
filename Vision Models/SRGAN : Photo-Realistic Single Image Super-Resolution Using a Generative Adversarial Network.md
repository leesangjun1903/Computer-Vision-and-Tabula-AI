# SRGAN : Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network | Super resolution

**Main Takeaway:** SRGAN pioneers the use of adversarial training combined with perceptual loss to recover **photo-realistic textures** in 4× single-image super-resolution, setting a new benchmark for perceptual quality even when traditional metrics (PSNR/SSIM) may decrease[1].

## 1. 핵심 주장 및 주요 기여

SRGAN은 다음 세 가지 주요 기여를 제시한다[1]:

1. **고배율(4×)에서의 실질적 질감 복원**  
   – 기존 MSE 기반 모델이 생성한 과도하게 매끄러운(high-PSNR) 이미지는 실제 질감(detail)을 잃는 반면, SRGAN은 자연 이미지 매니폴드로의 이동을 유도하여 사진처럼 사실적인 텍스처를 재현한다.  

2. **Perceptual Loss**  
   – Content Loss: 단순 픽셀 MSE 대신 VGG19 네트워크의 중·고수준 특징 맵 φ5,4 사이의 유클리디안 거리 사용  
   – Adversarial Loss: 판별기 D가 생성 이미지 G(ILR)를 실제 HR로 잘못 분류하도록 −log D(G(ILR))를 최소화  
   – 이 두 손실을 결합하여 지각(perceptual) 기준에 부합하는 초해상도 학습을 수행[1].

3. **Deep ResNet 기반 Generator & Discriminator**  
   – Generator: 16개 residual block, 각 블록에 Conv(3×3,64)–BatchNorm–PReLU, 마지막에 sub-pixel convolution으로 4× 업스케일링[1].  
   – Discriminator: 3×3 스트라이드 합성곱으로 해상도를 점진 축소하며 LeakyReLU 활성화, 최종 Sigmoid 출력[1].

## 2. 해결 과제 및 제안 기법

### 2.1. 문제 정의  
저해상도 LR 이미지를 입력받아 고해상도 HR 이미지를 예측하는 SISR(single-image super-resolution)은 고배율에서 손실된 **고주파 텍스처** 복원이 핵심 과제이다[1].

### 2.2. Perceptual Loss 수식  

$$
l_\text{SR} \;=\; l_\text{content} \;+\; 10^{-3}\,l_\text{Gen}
$$  

– Content Loss (VGG):  

$$
l_\text{VGG/i,j} = \frac{1}{W_{i,j}H_{i,j}}
\sum_{x,y} \bigl[\phi_{i,j}(I^\text{HR})\_{x,y} - \phi_{i,j}(G( I^\text{LR}))_{x,y}\bigr]^2
$$  

– Adversarial Loss:  

$$
l_\text{Gen} = \sum_n -\log D\bigl(G(I^\text{LR}_n)\bigr)
$$

### 2.3. 네트워크 구조  
– Generator G: 피쳐 복원을 위한 깊은 ResNet  
– Discriminator D: 진위 판별을 위한 CNN  
– 학습 순서: SRResNet-MSE로 초기화 후 GAN 학습[1]

## 3. 성능 향상 및 한계

| 모델            | PSNR (Set5) | SSIM (Set5) | MOS       |
|-----------------|-------------|-------------|-----------|
| Bicubic         | 28.43 dB    | 0.8211      | 1.97      |
| SRCNN           | 30.07 dB    | 0.8627      | 2.65      |
| SRResNet-MSE    | **32.05 dB**| **0.9019**  | 3.37      |
| SRGAN-VGG5,4 | 29.40 dB    | 0.8472      | **3.58**[2] |
  
- **정량적 PSNR/SSIM**에서는 SRResNet-MSE가 우수하지만,  
- **주관적 MOS(Mean Opinion Score)** 평가에서 SRGAN-VGG5,4가 최고(3.58)로, 원본 HR에 가장 근접[2].

**한계:**  
– PSNR/SSIM과 같은 전통 지표는 인간 지각 품질과 상관 낮음  
– 고주파 잡음·아티팩트 발생 가능  
– 의료·감시 분야처럼 사실성보다 정확도가 중요한 응용에는 부적합할 수 있음[1].

## 4. 모델 일반화 성능 향상 가능성

1. **Content Loss 다변화**  
   – 더 다양한 VGG 계층(φi,j) 조합 실험으로 일반화 강화  
2. **Adversarial 안정화 기법**  
   – Relativistic GAN, WGAN-GP 등의 손실 대체  
3. **도메인 적응(Domain Adaptation)**  
   – 의료·위성·문자 등 특수 도메인별 판별기 미세조정  
4. **경량화·효율화**  
   – MobileNet 계열·지식 증류로 실시간 적용 확대  
5. **다중 스케일 학습**  
   – 다양한 업스케일 배율을 동시에 학습하여 강건성 확보  

## 5. 향후 연구 영향 및 고려 사항

- **지각 기준 중심 SR 연구 확대:** 전통적 MSE 추종에서 벗어나 “사람이 보기 좋은” 화질 복원으로 패러다임 전환.  
- **응용별 Loss 설계:** 의료·문자·인물 분야 등, hallucination이 허용되지 않는 도메인에 맞는 content loss 연구 필요.  
- **객관적 평가 지표 개발:** MOS를 대신할 자동화된 perceptual 품질 측정법 부재.  
- **안정적 GAN 훈련 기법:** 고주파 아티팩트 저감 및 훈련 수렴성 개선.  
- **다중 모달 SR:** 텍스트, 깊이(depth), 열화상 등을 함께 복원하는 연구의 단초 제공.

SRGAN은 “수치” 성능보다 “지각(visual perception)” 성능을 우선시하는 **차세대 SR 연구**의 토대를 마련했으며, 향후 다양한 도메인과 배율, 효율성 요구를 충족하는 SR 모델 개발에 지속적으로 영감을 줄 것이다.

[1] https://arxiv.org/abs/1609.04802
[2] https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/284160fb-d788-4a51-88e5-14ef94a6bd7d/1609.04802v5.pdf
[4] https://www.sec.gov/Archives/edgar/data/1899830/000189983025000015/perf-20241231.htm
[5] https://www.sec.gov/Archives/edgar/data/1799332/000149315225010404/form10-k.htm
[6] https://www.sec.gov/Archives/edgar/data/1799332/000164117225012422/form8-k.htm
[7] https://www.sec.gov/Archives/edgar/data/1799332/000164117225011492/form8-k.htm
[8] https://www.sec.gov/Archives/edgar/data/1799332/000164117225003721/form8-k.htm
[9] https://www.sec.gov/Archives/edgar/data/1799332/000164117225009440/form8-k.htm
[10] https://www.sec.gov/Archives/edgar/data/1799332/000149315225010396/form8-k.htm
[11] https://pubs.acs.org/doi/10.1021/acs.jafc.3c09458
[12] https://www.cambridge.org/core/product/identifier/S2056467823000439/type/journal_article
[13] http://www.thieme-connect.de/DOI/DOI?10.1055/s-0043-1768745
[14] https://link.springer.com/10.1007/s12325-024-02877-y
[15] http://www.thieme-connect.de/DOI/DOI?10.1055/s-0041-1726514
[16] https://link.springer.com/10.1007/s12325-023-02502-4
[17] https://www.informingscience.org/Publications/5330
[18] https://link.springer.com/10.1007/s12325-022-02136-y
[19] https://dydeeplearning.tistory.com/2
[20] https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/
[21] https://blog.outta.ai/168
[22] https://www.sciencedirect.com/science/article/pii/S0010482520304704
[23] https://www.bmvc2021-virtualconference.com/assets/papers/0747.pdf
[24] https://iamseungjun.tistory.com/27
[25] https://www.digitalocean.com/community/tutorials/super-resolution-generative-adversarial-networks
[26] https://www.sciencedirect.com/topics/computer-science/the-super-resolution-generative-adversarial-network
[27] https://kevinitcoding.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-SRGAN-%EB%85%BC%EB%AC%B8-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC-Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-Network
[28] https://arxiv.org/html/2505.10589v3
[29] https://arxiv.org/abs/1809.00219
[30] https://hi-guten-tag.tistory.com/203
[31] https://www.geeksforgeeks.org/machine-learning/super-resolution-gan-srgan/
[32] https://velog.io/@pabiya/Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-AdversarialNetwork
[33] https://scholarworks.gvsu.edu/jcppubs/110/
[34] https://www.semanticscholar.org/paper/1d056c5e0267a41016e469a1d55dc11218eeb966
[35] https://arxiv.org/pdf/2005.12597.pdf
[36] https://arxiv.org/pdf/2308.15730.pdf
[37] http://arxiv.org/pdf/2107.09427.pdf
[38] https://arxiv.org/pdf/1903.09922.pdf
[39] https://arxiv.org/pdf/2209.03355.pdf
[40] https://arxiv.org/pdf/2208.03008.pdf
[41] https://arxiv.org/html/2407.15604v1
[42] https://www.mdpi.com/2072-4292/16/8/1460/pdf?version=1713605832
[43] https://arxiv.org/abs/2307.16169
[44] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2018.6570
[45] https://paka96.tistory.com/29
[46] https://github.com/tensorlayer/SRGAN
[47] https://d-tail.tistory.com/25

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
