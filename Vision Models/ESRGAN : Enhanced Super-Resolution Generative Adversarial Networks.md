# ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks | Super resolution

## 1. 핵심 주장 및 주요 기여  
Enhanced SRGAN(ESRGAN)은 단일 이미지 초해상도(SISR)에서 현실감 있고 자연스러운 텍스처를 생성하는 것을 목표로 한 SRGAN의 한계를 극복한다.  
- Residual-in-Residual Dense Block(RRDB) 기반 네트워크 구조 도입으로 용량과 학습 안정성 향상  
- Relativistic average GAN(RaGAN)을 discriminator에 적용해 “상대적 실재성” 학습  
- VGG 특징의 활성화 전(feature-before-activation) 기반 지각 손실(perceptual loss)로 밝기 일관성과 세부 묘사 강화  

이 세 가지 개선을 통해 ESRGAN은 SRGAN 대비 선명도와 텍스처 자연성이 모두 향상되며, PIRM2018-SR 챌린지(Region 3)에서 최고 지각 지수(perceptual index)를 기록했다.

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- PSNR 최적화 방식은 픽셀 단위 평균 제곱 오차를 줄이나 주관적 화질(고주파 디테일)을 희생하여 과도하게 부드러운 결과만 생성  
- SRGAN은 GAN 기반으로 자연스러운 텍스처를 생성하지만 잔여(replica) 노이즈 및 아티팩트 발생  

### 2.2 네트워크 구조  
ESRGAN의 Generator는 SRResNet 기본 틀을 따르되, 기본 블록을 RRDB로 교체하고 모든 Batch Normalization(BN)층을 제거한다.  
- RRDB: 다중 수준 잔차 학습(residual-in-residual) + Dense connections  
- 잔차 스케일링(residual scaling)과 작은 초기화(small-scale initialization)로 안정적 학습  

모델 전체 구조:  
LR → Conv → 23× RRDB → Conv → Upsampling × 2 → Conv → SR 출력  

### 2.3 손실 함수  
총 손실:  

$$
L_G = L_{\text{percep}} + \lambda\,L_{G}^{\text{RaGAN}} + \eta\,L_{1}
$$  

- $$L_{1} = \mathbb{E}\_{x,y}\|G(x)-y\|_{1}$$  
- Relativistic adversarial loss :  

$$
    D_{\text{Ra}}(x_r,x_f) = \sigma\bigl(C(x_r)-\mathbb{E}_{x_f}[C(x_f)]\bigr),
  $$  

$$
    L_{D}^{\text{Ra}} = -\mathbb{E}\_{x_r}[\log D_{\text{Ra}}(x_r,x_f)] - \mathbb{E}\_{x_f}[\log(1-D_{\text{Ra}}(x_f,x_r))],
  $$  

$$
    L_{G}^{\text{RaGAN}} = -\mathbb{E}\_{x_r}[\log(1-D_{\text{Ra}}(x_r,x_f))] - \mathbb{E}\_{x_f}[\log D_{\text{Ra}}(x_f,x_r)].
  $$  

- 지각 손실(perceptual loss): VGG 특징망에서 **활성화 전** feature 차이를 최소화해 밝기 왜곡 완화 및 경계 선명도 강화  

### 2.4 성능 향상  
- PIRM2018-SR 챌린지 Region 3에서 최저 지각 지수 달성  
- 기존 SRGAN 대비 평균 지각 지수 개선 및 PSNR ≈ 0.2 dB 소폭 손실로 밸런스 최적화  
- 정성 평가에서 동물 모피·건축 구조 등 복합 텍스처 재현력 대폭 향상  

### 2.5 한계  
- PSNR/SSIM 등 왜곡도 척도의 손실이 불가피하여 화질 지표 상 소폭 후퇴  
- GAN 기반 특유의 불안정 학습 문제(아티팩트, 모드 붕괴) 완전 제거는 미흡  
- 대규모 고해상도 데이터 및 연산 자원 소모 큼  

## 3. 일반화 성능 향상 관련 고찰  
- **BN 제거**: 테스트 시 배치 통계 불일치로 인한 아티팩트 억제, 도메인 전이력 강화  
- **작은 초기화·잔차 스케일링**: 매우 깊은 RRDB 네트워크 학습 안정화  
- **다양한 텍스처 데이터 활용**: DIV2K·Flickr2K·OST 병합 훈련으로 복합 도메인에 대한 복원력 증대  
- **상대적 판별자**: 더 일반적인 ‘실재성 비교’ 학습으로 과적합 억제 및 세부 텍스처 학습 강화  

이들 기법은 도메인 차이나 아웃오브분포(OOD) LR 입력에 대해 보다 견고한 초해상도 모델을 가능케 한다.

## 4. 향후 연구 영향 및 고려 사항  
- **새로운 perceptual loss 설계**: MINC 손실 등 텍스처 특화 특징망으로의 확장 가능  
- **경량화·효율화**: 모바일·엣지 환경용 경량 RRDB 및 지각 손실 근사 연구 필요  
- **안정적 GAN 학습**: 모드 붕괴 방지 및 학습 안정성 개선을 위한 advanced regularization  
- **다중 도메인 SISR**: 의료·위성·과학 이미지 등 범용화 및 분할별 맞춤 손실함수 탐색  
- **신뢰성·해석성 확보**: 생성 결과에 대한 uncertainty estimation과 왜곡 분석 기법  

ESRGAN은 GAN 기반 SR 연구의 새로운 표준으로 자리매김했으며, 텍스처 재현과 학습 안정성 측면에서 후속 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f28c012c-e72d-4920-8949-5c771bcec94e/1809.00219v2.pdf

# ESRGAN

# Abs
초해상도 생성 적대적 네트워크(SR-GAN)[1]는 단일 이미지 초해상도를 얻기 위해 현실적인 텍스처를 생성할 수 있는 중요한 작업입니다.  
그러나 환각에 빠진 세부 사항은 종종 불쾌한 결함(artifact)를 동반합니다.  
시각적 품질을 더욱 향상시키기 위해 저희는 SRGAN의 세 가지 주요 구성 요소인 네트워크 아키텍처, 적대적 손실 및 지각 손실을 철저히 연구하고 각각을 개선하여 향상된 SRGAN(ESRGAN)을 도출합니다.  
특히, 저희는 기본 네트워크 구축 단위로 배치 정규화 없이 Residual-in-Residual Dense Block(RRDB)을 소개합니다.  
또한, 저희는 Relativistic GAN의 아이디어로 판별자가 절대값 대신 상대값을 예측하도록 하였다.  
마지막으로, 활성화 전 기능을 사용하여 지각 손실을 개선하여 밝기 일관성과 텍스처 복구에 대한 더 강력한 감독을 제공할 수 있습니다.  
이러한 개선의 혜택을 받아 제안된 ESRGAN은 SRGAN보다 더 현실적이고 자연스러운 텍스처로 일관되게 더 나은 시각적 품질을 달성했습니다.

# Introduction
근본적인 컴퓨터 비전 문제인 단일 이미지 초해상도(SISR)는 연구 커뮤니티와 AI 기업에서 점점 더 많은 관심을 받고 있습니다. SISR은 단일 저해상도(LR) 이미지에서 고해상도(HR) 이미지로 복구하는 것을 목표로 합니다.  
Dong 등이 제안한 SRCNN의 선구적인 작업 이후 심층 컨볼루션 신경망(CNN) 접근 방식은 성공적인 발전을 가져왔습니다.    
다양한 네트워크 아키텍처 설계 및 훈련 전략은 SR 성능, 특히 PSNR(PSNR) 값[5,6,7,1,8,9,10,11,12]을 지속적으로 향상시켰습니다.  
그러나 이러한 PSNR 접근 방식은 PSNR metric이 인간의 주관적인 평가와 근본적으로 일치하지 않기 때문에 충분한 고주파수의 세부 정보 없이는 과도하게 평활된 결과를 출력하는 경향이 있습니다[1].  

SR 결과의 시각적 품질을 향상시키기 위해 몇 가지 지각 기반 방법이 제안되었습니다.  
예를 들어, pixel space 대신 feature space에서 SR 모델을 최적화하기 위해 perceptual loss[13,14]가 제안됩니다.  
네트워크가 원래 이미지와 더 유사한 솔루션을 선호하도록 장려하기 위해 [1,16]에 의해 Generative adversarial network(GAN)[15]이 SR에 도입됩니다.  
이전의 semantic image은 복구된 텍스처(질감)의 세부 정보를 개선하기 위해 추가로 통합됩니다[17].  
시각적으로 만족스러운 결과를 추구하는 방식의 이정표 중 하나는 SRGAN[1]입니다.  
기본 모델은 residual blocks[18]로 구축되며 GAN 프레임워크에서 perceptual loss을 사용하여 최적화됩니다.   
이러한 모든 기술을 사용하여 SRGAN은 PSNR 지향 방법에 비해 재구성의 전반적인 시각적 품질을 크게 향상시킵니다.  

그러나 그림 1과 같이 SRGAN 결과와 실제 (ground-truth) 이미지 사이에는 여전히 명확한 격차가 존재합니다.   
이 연구에서는 SRGAN의 주요 구성 요소를 다시 살펴보고 세 가지 측면에서 모델을 개선합니다.  
첫째, 용량이 더 크고 훈련하기 쉬운 Residual-in-Residual Dense Block (RDDB) 을 도입하여 네트워크 구조를 개선합니다.  
또한 [20]에서와 같이 Batch Normalization (BN) [19] 레이어를 제거하고 residual scaling [21,20] 및 더 작은 initialization를 사용하여 매우 심층적인 네트워크를 쉽게 훈련합니다.  
둘째, 저희는 "한 이미지가 실제인지 가짜인지"가 아닌 "한 이미지가 다른 이미지보다 더 현실적인지" 판단하는 방법을 배우는 Relativistic average GAN (RaGAN) [2) 를 사용하여 판별기(discriminator)를 개선합니다.  
저희의 실험은 이러한 개선이 생성기가 더 현실적인 텍스처 세부 사항을 복구하는 데 도움이 된다는 것을 보여줍니다.  
셋째, 저희는 SRGAN에서와 같이 activation 전에 VGG feature들을 사용하여 개선된 perceptual loss을 제안합니다.  
저희는 4.4절에서 볼 수 있듯이 조정된 perceptual loss이 더 선명한 가장자리와 더 시각적으로 만족스러운 결과를 제공한다는 것을 경험적으로 발견했습니다.  
광범위한 실험은 ESRGAN이라고 하는 향상된 SRGAN이 선명도와 세부 사항 모두에서 SOTA 방법을 능가한다는 것을 보여줍니다 (그림 1 및 그림 7 참조).  

저희는 PIRM-SR 챌린지[3]에 참여하기 위해 ESRGAN의 변형을 취했습니다.  
이 챌린지는 [22]를 기반으로 지각 품질 인식 방식으로 성능을 평가하는 최초의 SR 대회로, 저자는 왜곡과 perceptual quality가 서로 상충된다고 주장합니다.   
perceptual quality는 Ma의 점수[23]와 NIQE[24], 즉 지각 지수 = 1/2((10-Ma)+NIQE의 비 참조 측정으로 판단됩니다.  
perceptual quality가 낮을수록 더 나은 지각 품질을 나타냅니다.

그림 2와 같이 perception-distortion plane은 RMSE(Root-Mean-Square Error)의 임계값으로 정의된 세 개의 영역으로 나뉘며, 각 영역에서 가장 낮은 지각 지수를 달성하는 알고리즘이 지역 챔피언이 됩니다.  
우리는 지각 품질을 새로운 최고치로 끌어올리는 것을 목표로 하기 때문에 주로 R 3에 초점을 맞추고 있습니다.  
앞서 언급한 개선 사항과 4.6절에서 논의한 몇 가지 다른 조정 덕분에 제안한 ESRGAN은 PIRM-SR 챌린지(R3)에서 최고의 지각 지수로 1위를 차지했습니다.
시각적 품질과 RMSE/PSNR의 균형을 맞추기 위해 재구성 스타일과 평활도를 지속적으로 조정할 수 있는 network interpolation 전략을 추가로 제안합니다.  
또 다른 대안은 픽셀 단위로 이미지를 직접 보간하는 image interpolation입니다.  
우리는 이 전략을 사용하여 이미지 보간 방식을 R1과 R2에 참여시킵니다.  
네트워크 보간 및 이미지 보간 전략과 그 차이점에 대해서는 3.4절에서 설명합니다.

# Related Works

저희는 SR 문제를 해결하기 위한 심층 신경망 접근 방식에 중점을 둡니다. 선구적인 작업으로, Dong et al. [4,25]는 LR에서 HR 이미지로의 매핑을 end-to-end방식으로 학습하여 이전 작업에 비해 우수한 성능을 달성하는 SRCNN을 제안합니다.  
이후 이 분야에서는 deeper network with residual learning [5], Laplacian pyramid structure [6], residual blocks [1], recursive learning [7,8], densely connected network [9], deep back projection [10] and residual dense network [11]와 같은 다양한 네트워크 아키텍처를 목격했습니다.  
특히, Lim et al. [20]은 잔차 블록에서 불필요한 BN 레이어를 제거하고 모델 크기를 확장하여 EDSR 모델을 제안하여 상당한 개선을 달성했습니다.  
Zhang et al. [11]은 SR에서 효과적인 residual dense block을 사용할 것을 제안하고, channel attention[12]를 통해 더 깊은 네트워크를 탐색하여 최첨단 PSNR 성능을 달성합니다.  
지도 학습 외에도 일반적인 이미지 복원 문제를 해결하기 위해 reinforcement learning[26] 및 unsupervised learning[27]과 같은 다른 방법도 도입되었습니다.  

- end-to-end란?
딥러닝에서 end-to-end의 의미는 입력에서 출력까지 파이프라인 네트워크 없이 신경망으로 한 번에 처리하는 것을 의미합니다.  
즉, 복잡한 파이프라인 없이 하나의 신경망으로 입력받은 것을 출력할 수 있다는 것입니다.


# Proposed methods
- SRGAN과의 차이
SRGAN의 성능향상을 위해 논문에서는 크게 세가지를 바꾼다.
1. Network Architecture
2. Relativistic Discriminator
3. Perceptual Loss

## Network Architecture

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F81527999-ad80-48cc-94d7-767698293599%2Fimage.png)

SRGAN에서는 Batch Normalization(BN)을 사용하지만, train 데이터셋과 test 데이터셋의 통계치가 달라 BN을 사용하면 artifact 현상이 생기게 되며, 일반화 성능이 저하된다.  
따라서 저자들은 안정적인 학습과 일관된 성능을 위해 BN을 제거하였다.  
BN을 제거함으로 계산 복잡도, 메모리 사용량에서 이점이 생긴다.  

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F4640cc7b-9842-46e1-b128-9e812a4f139c%2Fimage.png)

기존의 SRGAN의 구조는 그대로 사용하며 Block만 교체한 모습이다.  
RRDB는 기존 SRGAN의 Residual Block 보다 더 깊고 복잡한 구조로, 주 경로에서 dense block을 사용하는데, 이로 인해 네트워크 용량은 커진다.

이것들 외에 Residual scaling : 불안정성을 방지하기 위해 주 경로에 추가하기 전에 0과 1 사이의 상수를 곱해 residuals 스케일을 축소한다.  
Smaller initialization : residual 구조는 초기 매개변수 분산이 작을 수록 더욱 쉽게 학습시킬 수 있다.  
등의 기술도 이용하였다.

## Relativistic Discriminator
기존의 SRGAN의 판별자는 하나의 input 이미지(x)가 진짜이고 자연스러운 것일 확률을 추정했다.  
relativistic discriminator는 실제 이미지(Xr)가 가짜 이미지(Xf) 보다 상대적으로 더 현실적일 확률을 예측한다.  

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F5a978bc2-f9d3-4601-8c2a-397b74f8856b%2Fimage.png)

## Perceptual Loss
기존에는 activation 이후에 feature map을 사용했지만, activation 전에 feature map을 사용함으로써 SRGAN보다 더 효과적인 perceptual loss(Lpercep)를 개발하였고, 이를 통해 기존에 있던 2가지의 문제점을 해결하였다.  
매우 깊은 네트워크 activation 이후에 활성화된 features들은 매우 sparse함으로, 낮은 성능으로 이어진다.

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F50658d2f-d28b-4a04-8d98-1349c1f64005%2Fimage.png)

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F04da0e1b-6eef-4c41-9fab-27602258da24%2Fimage.png)

활성화 후 feature들을 사용하는 것은 ground-truth 이미지와 비교했을 때 일관성이 없는 복원된 밝기를 유발한다.  
(왼쪽 그래프 빨간색 gt, 파란색 after activation, 초록색 before activation)

## Total Loss
최종적으로 loss는 perceptual loss, RaGan loss, L1 loss가 사용된다.

![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F382f8dc8-abe4-4484-9b68-e9e23c6b3719%2Fimage.png)

## Network Interpolation
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Fbfaf8cd4-ac28-46b9-a6bf-a8ce01c949cf%2Fimage.png)

PSNR-oriented network(Gpsnr)을 학습한 후, 미세 조정을 통해 GAN-based network(Ggan)를 얻었다.  
이것으로 기존 GAN 방식의 학습이 진행되면서 perceptual quality가 좋아져도 artifact가 생기는 문제를 어느정도 해결한다.  
또 모델을 재학습시킬 필요없이 지속적으로 지각 품질과 정확도의 균형을 유지할 수 있게 해준다.

# Experiments
## The PIRM-SR Challenge
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Ff0c0349a-d9e2-430d-823a-c8a0a5eedea7%2Fimage.png)

## Qualitative Results
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2F45980c30-2efd-4af5-87a2-e2ac398cd562%2Fimage.png)

## Benchmark
![](https://velog.velcdn.com/images%2Fkanghoon12%2Fpost%2Fd51374fa-b979-46e8-8900-84fa21121b98%2Fimage.png)

# Conclusion
저희는 이전 SR 방법보다 지속적으로 더 나은 지각별 품질을 달성하는 ESRGAN 모델을 제시했습니다.  
이 방법은 지각 지수 측면에서 PIRM-SR 챌린지에서 1위를 차지했습니다.  
저희는 BN 레이어가 없는 여러 RDDB 블록을 포함하는 새로운 아키텍처를 공식화했습니다.  
또한 제안된 심층 모델의 훈련을 용이하게 하기 위해 잔여 스케일링 및 더 작은 초기화를 포함한 유용한 기술이 사용됩니다.  
또한 한 이미지가 다른 이미지보다 더 현실적인지 여부를 판단하는 방법을 배우는 상대론적 GAN을 판별기로 사용하여 생성기가 더 자세한 텍스처를 복구하도록 안내했습니다.  
또한 활성화 전 기능을 사용하여 지각 손실을 개선하여 더 강력한 감독을 제공하고 더 정확한 밝기와 현실적인 텍스처를 복원했습니다.

# Reference
https://velog.io/@kanghoon12/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-ESRGAN
https://velog.io/@joon6093/%EC%9D%BC%EB%8B%A8-%EB%B0%95%EC%A3%A0-ESRGAN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%95%B4%EC%83%81%EB%8F%84-%ED%96%A5%EC%83%81
https://velog.io/@danielseo/Computer-Vision-ESRGAN
https://velog.io/@jeewoo1025/What-is-end-to-end-deep-learning
https://limitsinx.tistory.com/290
https://zeroact.tistory.com/9

