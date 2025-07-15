# PGGAN : Progressive Growing of GANs for Improved Quality, Stability, and Variation | Image generation

**핵심 주장 및 주요 기여**  
본 논문은 고해상도 이미지 생성에서 GAN 학습의 불안정성과 낮은 다양성 문제를 해결하기 위해, 생성기(Generator)와 판별기(Discriminator)를 학습 초기에 저해상도로 시작하여 점진적으로 해상도를 높여가는 방식(Progressive Growing)을 제안한다. 이를 통해  
- 학습 속도가 2–6× 향상됨  
- 고해상도(최대 1024×1024)의 안정적 생성이 가능  
- 생성 이미지의 품질 및 다양성이 크게 개선  

더불어 미니배치 간 특징 표준편차(Minibatch Standard Deviation) 층, 픽셀 단위 정규화(Pixelwise Feature Vector Normalization), 동등 학습률(Equalized Learning Rate) 등의 기법을 결합하여 훈련을 더욱 안정화하고 과도한 신호 확장을 방지한다.

## 1. 문제 정의  
기존 GAN은  
1) 고해상도에서 판별기가 실제와 생성물을 쉽게 구분하여 그라디언트가 불안정  
2) 대용량 이미지 처리 시 메모리 제약으로 배치 크기가 작아져 학습이 불안정  
3) 모드 붕괴(mode collapse)로 다양성이 떨어짐  

등의 문제를 안고 있다.

## 2. 제안 방법  
### 2.1 Progressive Growing  
- 학습 초기 4×4 해상도로 시작  
- 일정 단계마다 새로운 합성곱 블록을 추가하며 해상도 2배씩 증가  
- 새로 추가된 레이어의 영향은 α로 선형 보간(fade-in)  

$$
    \text{출력} = (1-α)\,G_{\text{old}} + α\,G_{\text{new}},\quad α\to1
$$

### 2.2 미니배치 표준편차 층  
- 배치 내 각 특성(feature)의 표준편차를 계산해 평균  
- 모든 위치에 동일 값을 특징 맵으로 추가해 판별기가 배치 통계 활용토록 유도  

### 2.3 픽셀 단위 정규화  
- 생성기 내부 각 픽셀의 피처 벡터 $$a_{x,y}$$를 다음과 같이 정규화  

$$
    b_{x,y} = \frac{a_{x,y}}{\sqrt{\frac{1}{N}\sum_j (a_{x,y}^j)^2 + \epsilon}}
$$

### 2.4 동등 학습률  
- He 초기화 상수 $$c$$로 실시간 가중치 스케일링  

$$\hat w_i = w_i / c$$
  
- 파라미터별 동적 범위 정규화로 학습 속도 균일화  

## 3. 모델 구조  
- 생성기·판별기 모두 3×3 합성곱 블록을 반복적으로 쌓은 대칭 구조  
- 해상도 단계마다 업샘플/다운샘플(Nearest-Neighbor/평균 풀링)  
- 마지막 toRGB(1×1 Conv) 및 fromRGB 적용  
- 전체 파라미터 수 약 23M

## 4. 성능 향상  
| 해상도 | 기존(WGAN-GP) SWD×10³ | 제안법 SWD×10³ | 속도 향상 |
|:----:|:---------------------:|:-------------:|:-------:|
| 128×128 | 9.28 | 4.28 | 2× |
| 1024×1024 | — | FID 7.30, SWD avg 5.44 | 5.4× |

- **이미지 품질**: CELEBA-HQ(1024×1024)에서 자연스러운 고해상도 얼굴 생성  
- **다양성**: CIFAR-10 무감독 설정에서 Inception Score 8.80 기록  
- **안정성**: 낮은 배치 크기에서도 모드 붕괴 억제  

## 5. 한계 및 일반화 가능성  
- 복잡한 객체·장면에 대한 의미적 이해는 부족  
- 미세 구조(fine details) 개선 여지  
- Progressive Growing 자체는 모델-손실 함수 무관하게 적용 가능  
- 대규모 분산 학습 환경에서도 효율적 사용 기대  

## 6. 연구에 미치는 영향 및 고려 사항  
- **Curriculum Learning 관점 확장**: 점진적 난이도 증가 기법을 다른 생성 모델에도 적용 가능  
- **고해상도 비전 응용**: 의료·위성·예술 분야의 이미지 합성에 직접 활용  
- **후속 연구 제안**:  
  - 의미적 제어(Semantic Control) 결합  
  - 고해상도 비디오 생성을 위한 시간적 프로그레시브 기법  
  - Transformer 기반 Generator와의 하이브리드 구조  

향후 연구에서는 Progressive Growing의 단계별 학습 스케줄, 레이어 페이딩 전략, 그리고 다양한 정규화 기법의 조합이 모델 일반화 성능에 어떤 영향을 미치는지 심층 분석할 필요가 있다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f6f7e829-3675-44d8-8e72-a19506bf5b11/1710.10196v3.pdf

PGGAN은 현재 GAN 분야에서 sota를 달성하고 있는 StyleGAN 시리즈의 기반이 되었습니다.

# Abs
우리는 적대적 생성형 네트워크를 위한 새로운 훈련 방법론을 설명합니다.  
핵심 아이디어는 생성기와 판별기를 점진적으로 성장시키는 것입니다.  
저해상도에서 시작하여 훈련이 진행됨에 따라 점점 더 미세한 세부 사항을 모델링하는 새로운 레이어를 추가합니다.  
이를 통해 훈련 속도를 높이고 크게 안정화하여 $1024^2$에서 전례 없는 품질의 이미지(예: CELEBA 이미지)를 생성할 수 있습니다.  
또한 생성된 이미지의 변화를 높이고 비지도 CIFAR10에서 8.80의 기록적인 시작 점수를 달성하는 간단한 방법을 제안합니다.  
또한 생성기와 판별기 간의 건전하지 않은 경쟁을 방지하는 데 중요한 몇 가지 구현 세부 사항을 설명합니다.  
마지막으로, 이미지 품질과 변화 측면에서 GAN 결과를 평가하기 위한 새로운 메트릭을 제안합니다.  
추가 기여로 CELEBA 데이터 세트의 고품질 버전을 구축합니다.

# Introduction
GAN을 이용하여 고해상도의 이미지를 생성하는 것은 아주 어려운 태스크입니다.  
고해상도의 이미지를 생성하도록 generator를 학습시키는 경우 학습 이미지의 distribution과 학습 결과 생성된 이미지의 distribution의 차이가 커집니다.  
또한 고해상도의 이미지는 같은 메모리에서 저해상도의 이미지보다 적은 배치사이즈를 가져가게 하는데, 이는 불안정한 학습을 야기합니다.  
이러한 상황에서 본 논문에서는 generator와 discriminator를 저해상도의 이미지로부터 고해상도의 이미지로까지 layer들을 추가하면서 점진적으로 커지게합니다.  
이를 통해 학습 속도를 향상시키고 고해상도에서도 안정적인 학습을 가능케 했습니다.

논문이 발표된 당시, 생성된 이미지의 quality 뿐만 아니라 variation(diversity)까지도 함께 고려하고 측정하고자하는 많은 시도가 있었습니다.  
본 논문에서도 또한 3장에서 variation을 향상시키기 위한 방법들을 제시하고, 5장에서 quality와 variation을 측정하기 위한 새로운 metric을 제시합니다.

4.1절에서는 네트워크를 초기화할때의 약간의 변화에대해 서술하고, 다른 layer들 사이에서 균형잡힌 학습 속도를 확보합니다.  
게다가, mode collapse는 discriminator가 overshoot하기에 발생하는데, 이를 해결하기 위한 방법을 4.2절에서 제시합니다.

기존 연구들에서 사용되었던 데이터셋 (CelebA, LSUN, CIFAR10)은 모두 저해상도의 이미지에 해당합니다.  
그래서 본 논문에서는 1024x1024로 고해상도의 데이터셋 CelebA-HQ를 만들어서 공개했습니다.  

# PROGRESSIVE GROWING OF GANS
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrOGEI%2FbtrIlxP564W%2FhWtShqoXeXlnJCcYpEcmK1%2Fimg.png)

본 논문의 주요한 contribution은 GAN을 학습시킬때 저해상도의 이미지부터 시작하여 위의 사진처럼 layer를 추가해가면서 고해상도에 도달하게 하는 것입니다.  
이를 통해 image distribution에서 큰 구조의 (coarse-grained) 특징들을 우선 학습하고, 점차 세밀한 (fine-grained) 특징들을 이어서 학습하는 것입니다.  

Generator와 discriminator는 서로 반대되는 구조를 갖고 있습니다.  
모든 layer들은 학습하는 동안 고정되어 있고, layer가 추가되면 아래 그림처럼 부드럽게 흐려지게 합니다.  
이를 통해 새로운 layer가 추가되었을 때 기존 layer에 대한 충격을 완화할 수 있습니다.  

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdI3KXC%2FbtrItmlpOJA%2FwgUpFLjhLQi6unjvD8qB4k%2Fimg.png)

본 논문에서는 이러한 progressive training이 몇가지 장점을 갖고있다고 합니다.  
초기 학습에서 저해상도의 이미지를 학습하는것은 훨씬 안정적입니다.  
해상도를 조금씩 늘려가면서 학습하는 것은 1024x1024의 바로 학습하는 것보다 훨씬 쉬운 문제로 변환됩니다.  
그리고 마지막으로 학습 시간이 단축됩니다.  
한번에 고해상도의 이미지를 학습하는 것 보다 최종 해상도에 따라 최대 6배정도 학습 속도가 향상되었다고 합니다.

# Increasing Variation using Minibatch Standard Deviation
GAN은 학습 이미지에서의 variation만 포착하는 경향이 있습니다.  
이에 Salimans et al.은 minibatch discrimination을 제안했는데, 각각의 이미지에서 뿐만 아니라 minibatch에서도 feature statistics를 계산하여 실제 이미지와 생성된 이미지를 비슷한 statistic을 가지도록 했습니다.  
본 논문에서는 이를 새로운 파라미터나 하이퍼파라미터의 도입 없이 더욱 간소화시켰습니다.  
우선, minibatch에 대해 각각의 spatial location의 feature의 std.를 계산합니다.  
그 다음에 이를 모든 feature와 spatial location에 대해 평균을 내어 하나의 single value로 만들고, 이를 minibatch의 모든 spatial location에 대해 복제하고 concatenate하여 하나의 constant feature map을 만듭니다.  
이 map은 discriminator에 어디에 넣어도 좋지만, 가장 마지막에 넣는 것이 좋다고 합니다. 

# Normalization in Generator and Discriminator
GAN은 generator와 discriminator의 비정상적인 경쟁에 의해 발생하는 gradient에 취약합니다.  
이를 다시 얘기하자면 두 network의 학습 속도가 다르다는 뜻으로 해석 될 수 있습니다.  
기존 연구에서는 batch normalization을 추가하는 등의 조치를 취했지만, 본 논문에서는 이러한 signal을 직접적으로 규제하는 방법을 제시합니다.

## Equalized Learning Rate
본 논문에서는 단순한 standard normal distribution으로 weight을 initialize합니다.  
그 다음에 실행중에 weight를 scaling 하는데, $\hat{w}_i=w_i/c$ 이고, c는 He initialization에서 사용된 per-layer normalization costant입니다.

$\frac{1}{c}=\sqrt{\frac{2}{n_{in}}}$

이를 통해 weight의 update가 파라미터의 scale에 영향받지 않고 진행됩니다.  
따라서, dynamic range와 학습 속도가 모든 weight에 똑같이 적용됩니다.

## Pixelwise Feature Vector Normalization in Generator
Generator와 discriminator의 gradient가 통제를 벗어나는 (spiral out of control) 경우를 방지하기 위해서, 본 논문에서는 feature vector의 단위 길이 만큼 각 pixel을 normalize합니다.  
본 논문에서는 이를 AlexNet에서 소개되었는 local response normalization의 변형으로 구현했습니다.

```math
b_{x,y}=a_{x,y}/\sqrt{\frac{1}{N}\sum^{N-1}_{j=0}{a^j_{x,y}}^2+\epsilon}
```

# Multi-scale Statistical Similarity for Assessing GAN Results
GAN을 평가할때 사용되던 MS-SSIM은 큰 규모의 mode collapse는 잘 포착하지만 variation (diversity)의 손실이라던가 하는 작은 변화는 잘 포착하지 못한다고 합니다.  
따라서 본 논문에서는 Laplcian pyramid를 활용합니다.  
$16,384(2^{14})$개의 이미지를 샘플링하고 pyramid의 각 level에서 $128(2^7)$개의 descriptor를 추출하여 layer마다 총 $2^{21}$개의 descriptor가 생깁니다.  
각 descriptor는 7x7에 RGB 3 채널로 이루어져 있어 총 dimension이 147입니다.  
Pyramid의 l번째 feature에서 실제 이미지와 생성된 이미지에서의 패치를 각각 $\{x^l_i\}^{2^{21}}_{i=1}, \{y^l_i\}^{2^{21}}_{i=1}$이라고 할 때, 둘을 각각의 채널별로 normalize하고 그 둘의 sliced Wassertein distace (SWD)를 구합니다.  
SWD가 적게 나오면 두 패치의 distribution이 비슷하다는 것이고, 해당 resolution에서 appearance와 variation 두 측면에서 모두 비슷하다고 볼 수 있습니다.

# Experiments
## Importance of Individual Contributions in terms of Statistical Similarity
본 논문에서는 SWD와 MS-SSIM을 각각의 contribution의 성능을 측정하기 위해 사용했습니다.  
Baseline은 WGAN-GP를 적용한 Gularajani et al.의 training configuration입니다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcfjHnx%2FbtrIsdo3Qdk%2FR8fvk1Jb3ErkiVN7UMby91%2Fimg.png)

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fz7Z6r%2FbtrIuYETjQX%2F9GayybGPHojflKyOIotPTK%2Fimg.png)

(a)에서가 (h)보다 더 좋지 않은 이미지를 생성하고 있지만, MS-SSIM으로 측정한 결과 둘의 차이가 크지 않았습니다. 이에 반해 SWD는 큰 차이를 보이고 있습니다. 따라서, SWD가 MS-SSIM과는 달리 color, texture, viewpoints 등의 variation을 잘 포착하고 있음을 알수 있습니다.

 고해상도의 이미지를 다루기 위해서는 배치 사이즈를 줄여야 하므로 (c)에서는 배치를 64에서 16으로 줄였습니다.  
 그러나 생성 결과가 매우 불안정해졌고 (d)에서 BatchNorm이나 LinearNorm을 제거하는 등, training parameter를 수정하고 나니 학습이 안정적으로 진행되었습니다.  
 (e*)에서는 Salimans et al.의 minibatch discrimination을 적용시켰는데 성능향상을 보이진 못했고, (e)에서는 본 논문의 minibatch standard deviation을 적용시켰더니 SWD에서 성능 향상을 볼 수 있었습니다.  
 나머지 (f),(g)에서도 성능 향상이 나타났습니다.  
 마지막으로 (h)에서는 학습을 충분히 시켜 수렴시킨 결과입니다.

 ## Convergence and Training Speed
 ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcbLBha%2FbtrIpurty6R%2Fk2nka5Nbkl13wtrac8WuI0%2Fimg.png)

 (a)와 (b)의 비교를 통해 progressive growing을 통해 더 나은 최적값에서 모델이 수렴되고, 2배정도 학습 시간이 단축되는 것을 확인 할 수 있었습니다.  
 (c)에서는 해상도가 증가함에 따라, progressive growing의 방식이 더 빨라짐을 보여주고 있습니다.
