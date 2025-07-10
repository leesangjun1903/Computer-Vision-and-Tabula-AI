# Designing a Practical Degradation Model for Deep Blind Image Super-Resolution 논문 분석

## 1. 핵심 주장과 주요 기여

### **핵심 주장**
기존 Single Image Super-Resolution (SISR) 방법들은 실제 이미지의 복잡한 degradation을 제대로 모델링하지 못해 실용성이 떨어진다. 이를 해결하기 위해 **실제 이미지의 다양한 degradation을 포괄하는 새로운 practical degradation model**을 설계하고, 이를 통해 훈련된 blind SISR 모델이 실제 이미지에서 우수한 성능을 달성할 수 있다[1].

### **주요 기여**
1. **실제 이미지를 위한 practical degradation model 설계**: blur, downsampling, noise의 세 가지 핵심 요소를 각각 확장하고 복잡화
2. **Random shuffle strategy 도입**: degradation 순서를 무작위로 섞어 degradation space를 대폭 확장
3. **범용 blind SISR 모델 개발**: 다양한 degradation에 robust한 BSRGAN 모델 제안
4. **Hand-designed degradation model의 중요성 입증**: 정확한 degradation modeling이 실용적 SISR의 핵심임을 보여줌[1]

## 2. 문제 정의 및 제안 방법

### **해결하고자 하는 문제**
- **기존 degradation model의 한계**: Bicubic degradation과 traditional degradation $$y = (x \otimes k) \downarrow s + n$$은 실제 이미지의 복잡한 degradation을 충분히 반영하지 못함
- **Domain gap 문제**: 합성 데이터로 훈련된 모델이 실제 이미지에서 성능 저하
- **범용 blind SISR 모델 부재**: 다양한 실제 degradation에 적용 가능한 모델의 부족[1]

### **제안하는 방법**

#### **2.1 새로운 Degradation Model 구성요소**

**Blur 모델링**:
- **Isotropic Gaussian blur** $$B_{iso}$$: kernel width $$\sigma \in [0.1, 2.4]$$ (scale factor 2), $$[0.1, 2.8]$$ (scale factor 4)
- **Anisotropic Gaussian blur** $$B_{aniso}$$: rotation angle $$\theta \in [0, \pi]$$, axis length $$[0.5, 6]$$ (scale factor 2), $$[0.5, 8]$$ (scale factor 4)
- 커널 크기: $$7 \times 7$$부터 $$21 \times 21$$까지 uniform sampling[1]

**Downsampling 방법**:
- $$D_s^{nearest}$$: nearest neighbor + shift compensation
- $$D_s^{bilinear}$$: bilinear interpolation  
- $$D_s^{bicubic}$$: bicubic interpolation
- $$D_s^{down-up}$$: down-up sampling with scale factor $$s/a \rightarrow a$$, where $$a \in [1/2, s]$$[1]

**Noise 모델링**:
1. **Gaussian noise** $$N_G$$: 3D zero-mean Gaussian $$N(0, \Sigma)$$
   - AWGN: $$\Sigma = \sigma^2 I$$
   - Gray-scale AWGN: $$\Sigma = \sigma^2 \mathbf{1}$$
   - $$\sigma$$ uniformly sampled from $$[1/255, 25/255]$$

2. **JPEG compression noise** $$N_{JPEG}$$: quality factor $$[0, 100]$$

3. **Processed camera sensor noise** $$N_S$$: reverse-forward ISP pipeline with 5 camera models[1]

#### **2.2 Random Shuffle Strategy**
핵심 혁신인 random shuffle strategy는 degradation sequence $$\{B_{iso}, B_{aniso}, D_s, N_G, N_{JPEG}, N_S\}$$를 무작위로 섞어 다양한 degradation 조합을 생성한다[1].

### **모델 구조**
**BSRGAN**: ESRGAN 구조를 기반으로 한 2단계 훈련
1. **PSNR-oriented BSRNet** 훈련
2. **Perceptual quality-oriented BSRGAN** 훈련

**훈련 설정**:
- 손실 함수: L1 loss + VGG perceptual loss + PatchGAN loss (가중치 1:1:0.1)
- 데이터셋: DIV2K, Flick2K, WED, FFHQ (2,000 face images)
- LR patch size: 72×72 (기존 대비 증가)
- Optimizer: Adam, learning rate: 1×10⁻⁵, batch size: 48[1]

## 3. 성능 향상 및 일반화 성능

### **성능 향상 결과**
**DIV2K4D 데이터셋**에서 4가지 degradation type에 대한 평가:
- **BSRNet**: 전체적으로 최고 PSNR 성능 달성
- **BSRGAN**: 전체적으로 최고 LPIPS 성능 달성 (perceptual quality)
- 기존 bicubic 훈련 모델들(RRDB, ESRGAN)은 non-bicubic degradation에서 현저한 성능 저하[1]

**RealSRSet 데이터셋**에서 실제 이미지 평가:
- 복잡한 노이즈 제거 및 세부 디테일 복원 능력 우수
- 다양한 실제 degradation에 대한 robust한 성능 입증[1]

### **일반화 성능 향상의 핵심 요소**

#### **3.1 Degradation Space 대폭 확장**
- **기존**: 제한적인 degradation space (bicubic, simple blur)
- **제안**: 다양한 blur 조합, 4가지 downsampling 방법, 3가지 noise 타입, random shuffle strategy를 통한 대폭 확장된 degradation space[1]

#### **3.2 실제 이미지 Degradation 커버리지**
- 카메라 센서 노이즈 모델링 (5개 카메라 모델)
- JPEG 압축 아티팩트 모델링
- ISP 파이프라인 시뮬레이션을 통한 실제 이미지 처리 과정 반영[1]

#### **3.3 Training Data Diversity**
- 무한한 양의 paired LR/HR 데이터 생성 가능
- 완벽한 pixel-level alignment
- 다양한 degradation 조합으로 robust한 학습 가능[1]

#### **3.4 Domain Gap 해결**
- 기존 synthetic data와 real data 간 domain gap 문제 해결
- 실제 이미지 degradation을 더 정확히 모델링하여 real image에서 우수한 성능 달성[1]

## 4. 한계 및 제약사항

### **모델의 한계**
1. **복잡성**: 매우 많은 degradation parameter로 인한 모델 복잡성 증가
2. **비현실적 케이스**: 실제로 드물게 발생하는 degradation case도 포함
3. **Artifacts**: texture 영역에서 'bubble' artifacts 발생
4. **Trade-off**: bicubic degradation에서 일부 성능 저하 발생[1]

### **계산 복잡성**
- 복잡한 degradation model로 인한 데이터 생성 비용 증가
- 대용량 patch size (72×72)로 인한 메모리 요구량 증가
- 다양한 degradation 조합으로 인한 훈련 시간 증가[1]

## 5. 향후 연구에 미치는 영향

### **연구 패러다임 변화**
- **Degradation modeling의 중요성 재조명**: 정확한 degradation 모델링이 실용적 SISR의 핵심임을 입증
- **실제 이미지 적용을 위한 practical approach 강조**: 이론적 완벽성보다 실용성에 중점
- **Hand-designed vs. learned degradation model 논의 촉발**: 수작업 설계와 학습 기반 방법의 장단점 비교 필요[1]

### **후속 연구 방향**
1. **더 정교한 degradation model 개발**: 특정 도메인에 특화된 degradation 모델링
2. **적응적 degradation 조절**: 입력 이미지에 따른 동적 degradation parameter 조절
3. **효율성 개선**: 경량화된 degradation model 및 실시간 처리 최적화
4. **평가 방법론 개선**: 실제 이미지에 적합한 새로운 평가 지표 개발[1]

### **앞으로 연구 시 고려할 점**
1. **정확성 vs. 효율성**: 복잡한 degradation model의 정확성과 계산 효율성 간 균형
2. **도메인 특화**: 의료 영상, 위성 이미지 등 특정 도메인에 맞는 degradation 모델 개발
3. **평가 기준**: perceptual quality와 일치하는 새로운 IQA metric 개발
4. **확장성**: video super-resolution, multi-modal degradation 등으로의 확장 가능성 고려[1]

이 연구는 **실용적 blind image super-resolution**의 새로운 패러다임을 제시하며, 실제 이미지 처리 응용에서 중요한 돌파구를 마련했다는 점에서 향후 연구에 지속적인 영향을 미칠 것으로 예상된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b32398ec-ca3c-430e-80ea-7bd4ff720be3/2103.14006v2.pdf

# BSRGAN : Designing a Practical Degradation Model for Deep Blind Image Super-Resolution | Super resolution
![image](https://velog.velcdn.com/images/danielseo/post/362a2f69-c318-4e4c-8206-8389faf6ac12/BSRGAN%20%EC%8D%B8%EB%84%A4%EC%9D%BC.PNG)

# Abs

본 논문은 무작위로 섞은 blur, downsampling, noise degradations로 구성된, 더 복잡하지만 실용적인 degradation 모델을 설계하였다.

- blur
  - isotropic와 anisotropic Gaussian kerenels를 사용한 두 개의 convolutions에 의해 생성된다.
- downsampling
  - nearest, bilinear, bicubic interpolations 중 무작위로 선택된다.
- noise
  - 각기 다른 noise level들을 사용한 Gaussian noise를 더함으로써 통합된다.
 
이 새로운 degradation 모델의 효과를 확인하기 위해 저자들은 deep blind ESRGAN super-resolver를 학습시켰고, 다양한 degradation들로 종합 및 실제 이미지 모두 super-resolve에 적용해 보았다.

# Introduction

# Related Work

# A Practical Degradation Model
새롭고 실용적인 SISR degradation에 대해 이야기하기 전에 bicubic과 전통적인 degradation 모델에 대해 다음과 같은 사실을 언급하면 도움이 될 것이다.

1. 기존의 degradation 모델에 따르면, blur, downsampling, noise 이렇게 3가지 요소가 있는데 실제 이미지의 degradation에 영향을 미친다.  
2. 저화질 이미지와 고화질 이미지 둘 다 noisy와 blurry가 될 수 있기 때문에 저화질 이미지를 만들기 위한 기존의 degradation 모델로 blur, downsampling, noise를 추가한 pipeline을 사용하는 것은 불필요하다.  
3. 기존의 degradation 모델의 blur kernel 공간은 규모에 따라 달라야 하므로 실제로는 매우 큰 scale factor를 파악하기가 까다롭다.  
4. bicubic degradation이 실제 저화질 이미지에는 적절하지 않지만, data augmentation에 사용될 수 있으며, 깨끗하고 선명한 이미지 super resolution에는 좋은 선택이다.

## Blur
Blur는 이미지 degradation에 흔히 사용된다.  
저자들은 고화질 공간과 저화질 공간으로부터 blur를 모델링하는 것을 제안했다.

한편으로는 기존 SISR degradation 모델의 경우, 고화질 이미지는 먼저 blur kernel을 사용한 convolution에 의해 blur 처리가 되었다.  
사실 이 고화질 blur의 목적은 aliasing 되는 것을 방지하는 것과 그 다음에 있는 downsampling 후 더 많은 공간적 정보들을 보존하는 것이다.  
또 다른 한편으로, 실제 저화질 이미지는 흐릿하게 될 수 있어서 저화질 공간에서 이러한 흐림을 모델링하는 것은 실현 가능한 방법이다.

저자들은 Gaussian kerenls이 SISR 작업을 수행하기에 충분하다는 것을 고려하여, isotropic Gaussian kerenls($B_{iso}$)와 anisotropic Gaussian kernel($B{aniso}$) 이 2개의 Gaussian blur를 수행하였는데, 이를 통해 blur의 degradation 공간이 매우 확장될 수 있었다.  

- Blur kernel setting

 - size = [7x7, 9x9, ... , 21x21] 중에서 균등하게 추출됨
 - isotropic Gaussian kerenl
   - kernel width
      - scale이 2일 경우, [0.1, 2.4] 중에서 추출됨
      - scale이 4일 경우, [0.1, 2.8] 중에서 추출됨
 - anisotropic
   - rotation angle
    - [0, π] 중에서 추출됨
    - scale이 2일 경우, 각 축(axis)의 길이는 [0.5, 6] 중에서 추출됨
    - scale이 4일 경우, 각 축(axis)의 길이는 [0.5, 8] 중에서 추출됨

## Downsampling
고화질을 저화질로 downsample하기 위한 직접적인 방법은 nearest neighbor interpolation이다.  
하지만 이렇게 만들어진 저화질의 경우 왼쪽 위의 모서리쪽에 0.5x(s-1) 픽셀의 조정불량(misalignment) 문제가 생길 것이다.  
이것에 대한 해결책으로써 저자들은 2D linear grid interpolation 방법을 통해 21x21 isotropic Gaussian kerenl 중심을 0.5x(s-1) 픽셀만큼 이동시켰고, 이것을 nearest neighbour downsampling 전 convolution에 적용하였다.  
Gaussian kernel의 넓이는 Baniso kernel의 넓이로 설정하였다.

- nearest downsampling => $D^S_{nearest}$
- bicubic downsampling => $D^S_{bicubic}$
- bilinear downsampling => $D^S_{bilinear}$
- down-up smapling => $D^S_{down-up}$
저자들은 고화질을 downscale하기 위해 네 가지 downsampling 중에서 균등하게 추출하였다.

## Noise
### Gaussian noise $N_{G}$
- ($Σ = σ^2I$) => widely-used channel independent AWGN model
- ($Σ = σ^21$) => widely-used gray-scale AWGN model
- 3D zero-mean Gaussian noise model

I = identity matrix  
1 = 3x3 matrix with all elements equal to one

### JPEG compression noise $N_{JPEG}$
JPEG은 대역폭과 저장량 감소를 위해 가장 널리 쓰이는 이미지 압축 기준이다.

- JPEG 품질 요소 = [30, 95]

JPEG 품질 요소는 0~100의 값으로, 0에 가까울수록 high compression & low quality이며, 100에 가까울수록 low compression & high quality이다.

### Processed camera sensor noise $N_{S}$
![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F4a09464a-8a2d-472a-b66f-50c0edcc0615%2Fprocessed%20camera%20sensor%20noise.PNG)

image signal processing (ISP) pipeline

먼저 reverse ISP pipeline을 통해 RGB 이미지로부터 raw 이미지를 얻은 후, camera sensor noise를 만들어진 raw 이미지에 추가한 후 ISP pipleine을 통해 noisy RGB 이미지를 복원한다.  

ISP pipeline은 5가지 종류로 구성되어 있다.

1. demosaicing
- matlab의 demosaic fuction과 같은 방법으로 사용되었다.
2. exposure compensation
- global scaling은 [2^-0.2, 2^0.3] 중에서 선택되었다.
3. white balance
- red gain과 blur gain은 [1.2, 2.4] 중에서 선택되었다.
4. camera to XYZ(D50) color space conversion
- 3x3 color correction matrix는 raw 이미지 파일의 metadata에서 ForwardMatrix1과 ForwardMatrix2의 무작위 가중치 조합이다.
5. tone mapping and gamma correction
- 쌍으로 구성된 raw 이미지 파일과 RGB output을 기반으로 각 카메라에 대해 논문 What is the Space of Camera Response Functions?에서 가장 적합한 tone curve를 수동으로 선택했다.

### Random Shuffle
![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F0de397d0-47c1-42bd-97d6-de2f1c50a6d9%2F%EC%BA%A1%EC%B2%98.PNG)

위의 도식에 있는 제안된 degradation 모델은 scale factor 2에 대한 것이다.  
고화질 이미지에서 무작위로 섞은 degration sequence(Biso, Baniso, D2, Ng, Njpeg, Ns)가 먼저 진행된다.  
그 다음, 저화질 이미지를 JPEG 형식으로 저장하기 위해 JPEG 압축 degration(Njpeg)가 적용된다.  
scale factor 2를 사용한 downscaling operation($D^2$)는 $D^2_{nearest}$, $D^2_{bilinear}$, $D^2_{bicubic}$, $D^2_{down-up}$ 중에서 선택된다.

# Discussion
제안한 새로운 degradation 모델을 더욱 잘 이해하기 위해 일부 논의를 추가할 필요가 있다.

첫 번째, 해당 degradation 모델은 주로 저하된(degraded) 저화질 이미지들을 종합하도록 설계되었다.  
이것의 가장 직접적인 용도는 저화질 및 고화질 이미지를 포함하는 deep blind super-resolver를 학습하는 것이다.  
특히, 본 degradation 모델은 제한된 데이터나 정렬되지 않은 방대한 고화질 이미지 데이터셋을 기반으로 완벽하기 정렬된 학습 데이터를 무한히 생성할 수 있다.

두 번째, 너무 많은 degradation 매개변수와 무작위로 섞는 전략을 포함하고 있기 때문에 제안된 degradation 모델은 저하된 저화질 이미지에 적용되지 않습니다.

세 번째, degradation 모델이 특정 실제 장면(scene)의 극단적인 degradation을 유발할 수 있지만, 이는 여전히 deep blind 이미지의 일반화 성능을 개선하는 데 기여한다.

네 번째, DNCNN과 같이 다양한 degradation을 처리할 수 있는 단일 모델을 갖춘 대용량 DNN은 다양한 확대, JPEG 압축 정도, 다양한 noise level을 처리할 수 있으며 VDSR에서 상당한 성능을 발휘한다.

다섯 번째, degradatio 매개변수들을 조정함으로써 특정 애플리케이션에 대한 실용성을 향상시키기 위해 더 합리적인 degradation 유형들을 추가할 수 있다.

# Deep Blind SISR Model Training
이 논문의 색다른 점은 새로운 degradation 모델과 ESRGAN과 같은 기존 네트워크 구조를 차용하여 deep blind 모델을 학습시키는 것에 있다.  
제안된 degradation 모델의 장점을 보기 위해, 저자들은 널리 사용되는 ESRGAN 네트워크를 채택하고, 새로운 degradation 모델에서 생성된 합쳐진 두 저화질 및 고화질 이미지를 사용하여 학습시켰다.

저자들은 먼저 PSNR 지향적인 BSRNet 모델을 학습시키고, 지각 품질 지향적인 BSRGAN 모델을 학습시켰다.  
PSNR 지향적인 BSRNet 모델은 pixel-wise 평균 문제로 인해 과도하게 매끄러운(oversmooth) 결과를 만드는 경향이 있기 때문에 실제 적용에서는 지각 품질 지향적인 모델인 BSRGAN이 선호된다.  
따라서 저자들은 BSRGAN 모델에 집중하였다.

ESRGAN과 비교하여 BSRGAN은 몇 가지 방법으로 수정되었다.

1. 이미지 prior를 캡쳐하기 위해 먼저 약간 다른 고화질 이미지 데이터셋을 사용했다.(DIV2K, Flick2K, WED, FFHQ의 2,000개 얼굴 이미지)  
그 이유는 BSRGAN의 목표는 다용도의 blind 이미지 super resolution 문제를 해결하는 것이며, degradation prior 외에도 한 이미지 prior는 super-resolver의 성공에 기여할 수 있기 때문이다.
2. BSRGAN은 72x72의 큰 저화질 patch size를 사용한다.  
그 이유는 본 degradation 모델이 bicubic degradation을 통해서 만들어진 저화질 이미지 보다 더 심하게 저하된 저화질 이미지를 생성할 수 있기 때문이다.
그리고 큰 patch는 더 나은 복원을 위해 더 많은 정보를 캡쳐할 수 있다.
4. L1 loss, VGG percepture loss, PatchGan loss의 가중치 조합을 최소화하여 BSRGAN을 학습시켰다. (L1 loss weight = 1, VGG percepture loss weight = 1, PatchGAN loss weight = 0.1)  
특히 VGG percepture loss는 super-resolved 이미지의 색상 변화 문제를 방지하는 것이 더 안정적이기 때문에 미리 학습된 19-layer VGG 모델의 네 번째 maxpooling layer 전, 네 번째 convolusion에서 실행된다.

저자들은 learning rate를 0.00001로 고정시켰고, batch size는 48로 하여 Adam optimizer와 함께 BSRGAN을 학습시켰다.

optimizer = Adam, 
learning rate = 0.00001, 
batch size = 48

# Experimental Results
![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F4d0d48e7-088c-46ca-95f5-66035bfacf20%2F%EC%BA%A1%EC%B2%98.PNG)

![](https://velog.velcdn.com/images%2Fdanielseo%2Fpost%2F33e87ced-dd70-47f9-b024-c164b8664469%2F%EC%BA%A1%EC%B2%98.PNG)

