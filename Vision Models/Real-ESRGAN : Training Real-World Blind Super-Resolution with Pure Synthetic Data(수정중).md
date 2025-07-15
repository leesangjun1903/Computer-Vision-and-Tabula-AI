# Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

# Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

## 핵심 주장 및 주요 기여
Real-ESRGAN은 **순수 합성 데이터만으로 실제 저화질 이미지의 복원을 가능**하게 하는 실용적 블라인드 초해상화 모델을 제안한다.  
1. **고차(higher-order) 열화 모델링**: 기존의 단일(1차) 열화 과정을 여러 번 반복하여 실제 세계의 복합 열화를 보다 충실히 모사.  
2. **Sinc 필터 활용**: 과도 샤프닝에 따른 링잉(ringing) 및 오버슈트(overshoot) 아티팩트를 합성 과정에 포함.  
3. **U-Net 판별자 + 스펙트럴 노멀라이제이션**: 국소 질감 복원에 강인하고 안정적인 학습을 위한 판별자 구조 및 정규화 도입.  
4. **모두 합성된 훈련 페어**만으로 실제 이미지에 뛰어난 시각 품질을 달성, 기존 실세계 SR 방식 대비 실용성과 성능 우위 입증.  

## 문제 정의
실세계 저해상도(LR) 이미지는 **알 수 없는 복합 열화(degradations)**를 겪는다.  
– 카메라 블러, 센서 노이즈, 샤프닝／리사이즈 아티팩트, JPEG 압축, 인터넷 전송 중 반복 압축·노이즈 누적 등  
– 기존 SR 연구는 이상적인 비큐빅 다운샘플링만 다루거나, 1차 열화 모델로 제한  

## 제안 방법

### 1. 고차 열화 모델 (High-Order Degradation)

$$x = D_n \circ D_{n-1} \circ \cdots \circ D_1(y), \quad D_i(y) = \bigl[(y \otimes k_i)\downarrow_{r_i} + n_i \bigr]_{\text{JPEG},q_i} $$

- $$n=2$$를 기본으로, 각 단계의 블러 커널 $$k_i$$, 노이즈 $$n_i$$, 리사이즈 배율 $$r_i$$, JPEG 품질 $$q_i$$를 독립 샘플링  
- 블러: 이방성／등방성 가우시안, 일반화 가우시안, 플래토 커널 + 확률적 Sinc 필터(0.1)  
- 노이즈: 가우시안／포아송, 컬러／그레이, 세기 범위 혼합  
- 리사이즈: area, bilinear, bicubic 중 무작위  
- JPEG: 품질 $$q\in$$, 마지막 Sinc 필터는 JPEG 전·후 무작위 순서  

### 2. 네트워크 아키텍처
- **Generator**: ESRGAN의 RRDB(Residual-in-Residual Dense Block) 기반, 스케일×4／2／1 지원  
  - 입력 전처리로 Pixel-Unshuffle 적용  
- **Discriminator**: U-Net 구조 + Spectral Normalization  
  - 픽셀 단위 실체 판단(realness) 맵 출력→세밀한 그래디언트 피드백  

### 3. 손실 함수 및 학습
1. PSNR 지향 L1 손실로 예비 학습(Real-ESRNet)  
2. L1 + Perceptual Loss + GAN Loss 조합으로 Real-ESRGAN 정교화  

$$
   \mathcal{L}\_\text{total} = \mathcal{L}\_{L1} + \lambda_p\,\mathcal{L}\_\text{perc} + \lambda_g\,\mathcal{L}_\text{GAN},
$$

   where $$\lambda_p=1, \lambda_g=0.1$$.  
- VGG19의 conv1–conv5 특징맵 사용  
- 총 1.4M iteration, Adam optimizer, EMA 적용  

## 성능 향상
- **정성 비교**: RealSR, BSRGAN 등 대비 링잉·오버슈트 제거, 자연스러운 질감 복원 우수[그림7].  
- **NIQE 지표**: 다양한 실세계 데이터셋에서 기존 기법 대비 낮은 비참가형 품질 점수 달성(낮을수록 우수).  

| Dataset         | Bicubic | ESRGAN  | BSRGAN  | Real-ESRGAN |
|-----------------|---------|---------|---------|-------------|
| RealSR-Canon    | 6.13    | 6.77    | 6.15    | **5.75**    |
| OST300          | 4.44    | 3.52    | 4.74    | **4.17**    |

## 일반화 성능 향상 가능성
- **고차 열화**: 훈련 시 다양한 복합 열화 사례를 합성하여, 모델이 실제 분포 밖의 변형에도 강건해짐.  
- **Sinc 필터**: 과도 샤프닝 아티팩트도 학습되어, 실제 세계의 예측 불가능한 링잉 제거 능력 강화.  
- **U-Net 판별자**: 지역적 질감 및 경계 복원에 상세한 피드백 제공, 데이터 분포 전반에 대한 일반화 촉진.  
- **제한**: 여전히 **알 수 없는 극단적 열화**나 **반복 압축으로 인한 에일리어싱**에는 취약.  

## 향후 연구 영향 및 고려사항
- **데이터 합성 전략**: 더 높은 차수(n>2) 혹은 비정형 순서 조합 연구  
- **열화 예측 모듈 결합**: 명시적 열화 예측과 고차 합성의 하이브리드  
- **비참가형 평가 개선**: NIQE 외 인간 주관 평가와 가시화 기반 메트릭 개발  
- **실시간 적용**: 경량화 모델 및 효율적 온디바이스 학습  
- **안정성 이슈**: GAN 유도 왜곡 방지, 과도 샤프닝·에일리어싱 대응  

Real-ESRGAN은 **합성 데이터만으로 실세계 SR** 연구에 새로운 패러다임을 제시하며, 복합 열화 모사 및 GAN 기반 복원 기술의 발전 방향을 이끈다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/efd8ec27-7490-478e-8112-49deb800b575/2107.10833v2.pdf

![](https://velog.velcdn.com/images/heaseo/post/862022ac-4820-4de5-94c9-f2b1521c2fc6/Real-ESRGAN%20results.png)

# Abs
Blind super-resolution을 통해 복잡하고 알 수 없는 열화가 있는 저해상도 이미지를 복원하려는 수 차례 시도가 많았지만 완벽하게 복원하기는 힘들었었다.  
이 논문은 기존의 강력한 ESRGAN super-resolution 모델을 바탕으로 다양한 전처리 열화기법 단계를 추가해서 Real-ESRGAN 모델로 확장한다.  
실생활에서 생기는 열화를 구현하기 위해 high-order 열화기법이 도입됬다.  
게다가, 논문의 연구진은 rining과 overshoot 현상도 고려해서 새로운 열화 전처리 단계를 만들었다.  
추가적으로, U-Net discriminator에 spectral normalization를 추가해서 판별능력을 올려 안정적인 학습을 할 수 있도록 도왔다.  
결과적으로, 광범위한 실제 데이터셋을 이용한 비교를 통해 Real-ESRGAN이 기존 ESRGAN보다 시각적으로 뛰어난 성능을 입증했다.

# Introduction
단일 이미지 초해상화(SISR)은 저해상도 이미지로부터 고해상도 이미지를 재구성하는 것이 목표이다.  
신경망 네트워크를 통해 처음으로 선보인 SRCNN 모델은 SR 연구분야 발전에 큰 기여를 했다.  
하지만 이전까지 SR 모델의 열화기법은 단순히 bicubic으로 저해상도 이미지를 생성했기 때문에 실생활에서 생기는 저해상도 이미지와는 많이 달랐다.

Blind super-resolution은 알 수 없는 경로로 생긴 열화와 복잡한 열화가 합친 저해상도의 이미지를 고해상도 이미지로 복원하는 것이 목표이다.  
기존의 모델들을 명시적과 암시적 두 가지로 열화 기법으로 분류할 수 있다.  
고전적 열화 기법들은 blur, downsampling, noise and JPEG compression을 포함하고 있고 명시적 모델에 많이 사용됬다.  
하지만 이러한 열화 기법은 실생활에서 생기는 복잡하고 알 수 없는 열화를 복원할 수 없었다.  
암시적 모델은 GAN을 이용해서 열화 모델을 만들었다.  
하지만 암시적 모델로 저해상도 이미지를 만드는데 한계가 있었고 무엇보다 저해상도 이미지를 일반화하기 어려웠다.

본 연구는 기존에 강력한 ESRGAN 모델에 실생활에 생기는 열화를 최대한 비슷하게 만든 저해상도 이미지와 고해상도 이미지를 쌍으로 묶어 학습을 진행한다.  
실생활에서 생기는 열화는 대부분 한가지 과정에서 생기는 것이 아니라 여러가지 다양한 과정을 통해 열화가 생긴다.  
예를 들어 카메라의 시스템, 이미지 편집, 인터넷 전송 과정에서 생긴다.  
일반적으로 우리는 핸드폰으로 사진을 찍으면 여러가지 camera blur, sensor noise, sharpening artifacts 그리고 JPEG compression가 생긴다.  
그리고 우리는 이미지 편집 이후 소셜미디어에 올린다. 이때 한번 더 압축과 알 수 없는 노이즈가 생긴다.  
이처럼 여러가지 과정을 통해 알 수 없고 복잡한 열화가 생긴다.  
문제는 실생활에서 생기는 열화는 앞의 과정이 여러번 진행된다.

위의 프로세스를 통해 본 연구는 first-order 열화 기법을 high-order 열화 기법으로 확장했다.  
쉽게 말하면 여러번의 열화 과정을 추가해서 저해상도 이미지를 생성하는 것이다.  
본 연구 second-order 열화 과정을 추가해서 단순성과 효율성 두 마리 토끼를 잡았다.  
최근에 많은 연구들은 임의적으로 열화 기법을 섞은 전략을 많이 내세우는데 여전히 고정된 열화 과정과 불분명한 결과가 생긴다.  
반면에 본 연구가 주장하는 high-order 열화 모델은 유연하고 실생활에 생기는 프로세스를 최대한 비슷하게 모방하려고 한다.  
게다가 본 연구는 sinc filters를 추가해서 ringing 그리고 overshoot artifacts 현상을 구현하여 저해상도 이미지를 생성한다.

본 연구는 새로운 열화 기법 프로세스가 추가한 Real-ESRGAN 모델을 학습할 때 몇 가지 어려운 점을 아래와 같이 극복했다.  
1. Generator의 output을 HR 이미지와 정확한 차이점을 구별하기 위해 새로운 discriminator가 필요했다. 새로운 discriminator를 VGG-style에서 U-Net 디자인으로 변경했다.
2. 기존 U-Net design의 discriminator는 복잡한 열화 이미지를 제데로 구별할 수 없었기 때문에 spectral normalization regularization을 추가해서 안정적인 학습을 구현했다. 그 결과로 local detail을 복원하는 동시에 더 많은 열화를 지울 수 있었다.

# Related Work
SRCNN 등장이후 SR분야에 상당한 발전이 있었다.  
또한 시각적인 만족감을 위해 GAN 모델이 등장했다.  
많은 연구는 단순한 bicubic downsampling으로 저해상도 이미지를 만들어 고해상도 이미지와 쌍으로 학습했다.  
그 결과로 실생활 열화가 포함 된 저해상도 이미지를 고해상도로 복원하는 것은 실패했었다.  
최근에는 GAN 또는 전이학습을 통해 이러한 저해상도 열화 이미지를 복원하려고 노력하고 있다.

Introduction에서도 언급을 했었지만 기존에도 blind super-resolution 모델들이 등장했었다.  
명시적 열화 모델은 기본적으로 2개의 요소를 포함하고 있는데, 열화 예측 및 조건부 복원이다.  
상기 2개의 요소는 별도로 수행하거나 반복적으로 수행된다.  
이 접근방식들은 사전에 정의된 열화 정보에 의존하며 단순한 열화에 대해서만 고려를 하고있다.  
또한 정확한 열화 정보를 알 수 없을 경우 artifacts가 생기는 단점이 있다.

다른 blind super-resolution 모델인 암시적 모델은 실생활 열화가 포함된 저해상도 이미지와 고해상도 이미지를 쌍으로 묶어 학습한다.  
여기서 생성하는 저해상도 열화 이미지는 보통 특정 카메라로 생성한 이미지, GAN으로 생성한 이미지 또는 blur와 noise를 섞은 이미지로 만든다.  
하지만 특정 카메라로 생성한 열화 이미지는 말 그대로 특정 카메라에서 생성한 저해상도 이미지이기 때문에 일반화하기 어렵고 GAN 모델로 열화 이미지를 만드는 것은 쉽지않다.  
그러므로 위의 데이터셋으로 학습한 모델의 결과는 만족스럽지 못하다.

고전적인 열화 모델은 blind Super resolution에 많이 쓰인다.  
하지만 실생활에서 생기는 열화는 매우 복잡해서 명시적인 모델링을 할 수가 없다.  
그러므로 본 연구는 암시적 모델과 high-order 열화 과정을 추가해서 학습을 진행한다.

# Methodology
## Classical Degradation Model
Blind super resolution은 알 수 없고 복잡한 열화가 첨가된 저해상도 이미지를 고해상도로 복원하는 것인 목표이다.  
고전적인 열화기법은 input에 열화를 더한다.  
일반적으로 ground-truth image y에 blur kenel k를 합성곱해서 이미지를 흐릿하게 만든다.  
그 다음 scale factor r 만큼 해상도의 크기를 줄인다(downsampling).  
그리고 해상도 크기가 줄어든 이미지 x에 노이즈 n을 추가한 뒤 마지막으로 JPEG 압축 노이즈를 섞어 마무리한다.

$x = D(y) = [(y ~ k) ↓r +n]_{JPEG}$

D는 degradation process 이다.

### Blur
보통 blur degradation은 linear blur filter에 합성곱을 한 것이라고 한다.  
Isotropic과 anisotropic Gaussian filters가 일반적인 선택지다.

$k(i,j)=\frac{1}{N}exp(−\frac{1}{2}C^TΣ^{−1}C), C=[i,j]^T$

Σ : 공분산 행렬, C는 공간 좌표, N은 정규화 상수.

### Noise
1) addictive Gaussian noise
- 가우스 분포와 동일한 확률 밀도 함수를 갖는다.
- 노이즈 강도는 가우스 분포의 표준편차(즉, 시그마 값)에 의해 제어된다.
- RGB 채널에 동일한 샘플링 노이즈를 적용해서 회색 노이즈를 합성한다.
2) Poisson noise (or shot noise)
- 포아송 분포를 따르는 노이즈
- 이 기법은 양자 변동을 통해 특정 노출 수준에서 감지된 광자 수의 변동에 의해 발생하는 센서 노이즈를 모델링할 수 있다.
- 포아송 노이즈는 이미지 강도에 비례하는 강도를 가지며 서로 다른 픽셀의 노이즈는 독립적이다.

### Resize (Downsampling)
다운샘플링은 저해상도 이미지를 만드는 가장 기본적인 방법이다.  
Resize 알고리즘은 nearest-neighbor, area, bilinear, bicubic interpolation이 있다.  
각 리사이즈 알고리즘은 다른 효과를 나타내고 특히 특정 부분에서 overshoot artifacts가 생성된다.

다양하고 복잡한 리사이즈 효과를 내기 위해 본 연구는 임의적으로 보간법을 선택해 리사이즈를 했다.  
하지만 nearest-neighbor 알고리즘은 작은 이슈로 제외되었다.

### JPEG compression
JPEG compression은 디지털 이미지에 보편적으로 사용되는 손실 압축 기술이다.  
첫째 이미지를 YCbCr 색공간으로 바꾸고 chroma channels을 다운샘플링한다.  
그 후 이미지는 8x8 블럭으로 나눠지고 각 블록은 2차원 이산 코사인 변환 (DCT)을 사용해서 변환된다.  
일반적으로 저해상도 이미지에 생기는 블럭 artifacts는 JPEG compression의해 생성된다.

## High-order Degradation Model
고전적인 열화 기법으로 학습한 blind super resolutoin 모델을 이용해서 실제 열화 저해상도 이미지들을 고해상도로 복원했지만 효과는 미비하다.  
특히 복잡하고 알 수 없는 열화가 들어간 이미지는 제대로 복원할 수 없었다.

![](https://velog.velcdn.com/images%2Fheaseo%2Fpost%2F000c6fe2-227b-4ff3-930f-83a09fec26b3%2FReal-ESRGAN%20degradation.png)

이유는 실생활 열화 이미지와 고전적인 열화 모델로 생성된 이미지의 특성은 매우 달랐기 때문이다.  
그러므로 본 연구는 고전적인 열화 기법을 high-order 열화 기법으로 확장시켜 조금 더 실용적인 전처리 기법을 만들었다.

고전적인 열화 기법은 first-order는 고정적인 열화 기법수로 이루어진 단순한 열화가 포함되어 있다.  
하지만 실생활에서 생기는 열화는 다양한 열화 과정을 통해 생성된다.  
예를 들어 카메라의 시스템, 이미지 편집, 인터넷상에서 전송되는 과정에서 열화가 생긴다.

위의 이유로 본 연구는 high-order 열화 기법을 고안했고 매번 다른 하이퍼파라미터를 적용해서 n번 만큼 열화 과정 진행해서 저해상도 이미지를 생성한다.

$x=D^n(y)=(D_n ◦···◦D_2 ◦D_1)(y)$

특히 second-order 열화 과정이 실생활 열화 저해상도 이미지를 고해상도로 잘 복원시킬 수 있는 키 포인트라고 한다.  
각 degradation 전략은 random shuffling 전략을 통해 반복적인 degradation을 적용한다.

## Ringing and overshoot artifacts
Ringing artifacts는 종종 띠 또는 마치 영혼이 물체에서 빠져 나가는것 같은 형태로 모서리 부분에 나타난다.  
Overshooting artifacts는 보통 ringing artifacts 결합한 열화 형태로 생성된다.

![](https://velog.velcdn.com/images%2Fheaseo%2Fpost%2F50b98bdb-0cb9-431f-8b07-d751ba256c49%2Fringing%20%26%20overshot.png)

이러한 아티팩트는 일반적으로 JPEG 압축에 sharp 알고리즘을 추가한 후 생성된다.  
Figure 5에서 윗줄의 이미지들은 실제로 ringing과 overshoot artifacts가 있는 사진들이다.

본 연구는 sinc filter를 통해 고주파 영역을 제한하고 ringing과 overshoot artifacts를 생성한다.

$k(i,j)=\frac{ω_c}{2\pi{\sqrt{i^2+j^2}}}J_1(ω_c\sqrt{i^2+j^2})$

(i,j)는 커널 좌표, $ω_c$ : 주파수를 자르는 용도, $J_1$는 first order 함수

Sinc 필터를 blur 커널 프로세스와 마지막 JPEG compression과정 두 곳에 적용했다.  
일부 이미지는 먼저 강한 sharp을 적용해 overshoot를 생성하고 JPEG compression을 한다.

## Networks and Training
![](https://velog.velcdn.com/images%2Fheaseo%2Fpost%2Fbc601e9b-96e4-4801-839f-979d9ddceada%2FESRGAN%20network.png)

### ESRGAN generator:
본 연구는 ESRGAN과 동일한 super resolution 모델을 사용했다. 또한 기존에 4배 ESRGAN 네트워크 대신 1배와 2배도 적용할 수 있도록 수정했다.  
ESRGAN의 1배 그리고 2배 모델에는 pixel-unshuffle을 적용해 공간적 사이즈를 줄이고 channel의 수를 늘려 input으로 사용했다.  
결과적으로 작아진 해상도에서 연산을 진행하기 때문에 GPU memory 소모량을 줄일 수 있었고 연산량 또한 적어졌다.

### U-Net discriminator with spectral normalization (SN):
Real-ESRGAN은 폭 넓은 열화를 가진 저해상도 이미지를 복원해야 되기 때문에 기존에 ESRGAN에서 사용한 discriminator는 부적합하다.  
특히 Real-ESRGAN은 정확한 local texture까지 파라미터 학습이 필요하기 때문에 조금 더 무거운 discriminator를 요구한다.  
그러므로 VGG-style discriminator보다는 skip-connetion을 가지고 있는 U-Net 디자인 discriminator를 채택했다.  
U-Net은 각 픽셀에 대한 실제 값을 출력하며 픽셀당 자세한 피드백을 Generator에 제공할 수 있다.  
하지만 기본적인 U-Net 구조는 복잡한 열화를 가지고 있는 이미지를 잘 구별할 수 없으므로 spectral normalization regularization을 추가해서 안정된 학습을 할 수 있도록 했다.  
또한, spectral normalizaion은 GAN 훈련에 의해 지나치게 날카롭고 부자연스러운 아티팩트를 완화하는 데에도 도움이 된다.

### The training process:
학습 과정은 2개의 스테이지로 나누어지게 된다.  
첫째, L1 loss를 사용한 PSNR-oriented 모델을 학습한다.  
그리고 PSNR-oriented 모델을 Real-ESRNet으로 정의한다.  
둘째, 학습이 완료된 Real-ESRNet을 L1 loss, perceptual loss 그리고 GAN loss를 포함해서 Real-ESRGAN을 학습시킨다.

# Experiments
## Training details
- Datasets: DIV2K, Flickr2K, OutdoorSceneTraining
- Patch size: 256
- Batch size: 48
- Optimizer: Adam
- Fine-tunning: Real-ESRNet은 사전에 학습 된 ESRGAN을 사용한다.
- Iterations: Real-ESRNet (1000K), Real-ESRGAN (400K)
- Learning rate: $1×10^{−4}$
- Loss: Real-ESRNet (L1 loss), Real-ESRGAN (L1 loss, perceptual loss, GAN loss)

## Degradation details
- Gaussian kernels: {0.7, 0.15, 0.15}
- Generalized Gaussian kernels: {0.7, 0.15, 0.15}
- Plateau-shaped kernels: {0.7, 0.15, 0.15}
- Blur kernel size: randomly {7,9, ...21}
- Blur standard deviation σ: first order ({0.2, 3}), second order ({0.2, 1.5})
- Shape parameter β: generalized Gaussian({0.5, 4}), plateau-shaped kernels({1, 2})
- Sinc kernel: first order (0.1), second order (0.2)  
----------------------------------
- Gaussian noises: {0.5, 0.5}
- Poisson noises: {0.5, 0.5}
- Noise sigma range: first order ({1, 30}), second order ({1, 25})
- Poisson noise scale: first order ({0.05, 3}), second order ({0.05, 2.5})
- Gray noise: 0.4
- JPEG compression: {30, 95}
- Sinc filter: 0.8

# Results
![](https://velog.velcdn.com/images%2Fheaseo%2Fpost%2F1808c539-6646-4ca5-aede-7a3c08affd12%2Fteaser.jpg)


# Reference
- https://velog.io/@heaseo/Real-ESRGAN-Training-Real-World-Blind-Super-Resolution-with-Pure-Synthetic-Data-%ED%95%B4%EC%84%9D
