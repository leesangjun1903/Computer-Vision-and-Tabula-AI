# Generative Adversarial Networks for Image Super-Resolution: A Survey

## 1. 논문의 핵심 주장과 주요 기여

이 서베이 논문은 **단일 이미지 초해상도(Single Image Super-Resolution, SISR)** 분야에서 생성적 적대 신경망(GANs)의 포괄적 분석을 제공합니다[1]. 논문의 핵심 주장과 기여는 다음과 같습니다:

### 주요 기여
- **197개 논문의 종합적 검토**: 기존 문헌에서 다루어지지 않았던 GANs 기반 SISR 방법들에 대한 최초의 포괄적 서베이 제공[1]
- **체계적 분류 체계**: GANs를 지도학습, 반지도학습, 비지도학습 방식으로 분류하여 각각의 특성과 성능을 분석[1]
- **성능 비교 및 분석**: 정량적(PSNR, SSIM) 및 정성적 분석을 통한 다양한 GAN 모델들의 비교 평가[1]
- **미래 연구 방향 제시**: SISR 분야에서 GANs의 한계와 도전과제, 그리고 향후 연구 방향을 명확히 제시[1]

## 2. 해결하고자 하는 문제

### 주요 문제점
1. **작은 샘플 데이터셋 문제**: 기존 딥러닝 방법들이 대용량 데이터셋에서는 우수한 성능을 보이지만, **작은 샘플에서는 제한적 성능**을 보임[1][2]

2. **실제 환경에서의 적용 한계**: 기존 방법들이 실제 카메라로 촬영한 손상된 이미지에서는 충분하지 않은 성능을 보임[1]

3. **기존 서베이의 부족**: GANs 기반 SISR 방법들을 종합적으로 정리한 문헌이 부족함[1]

## 3. 제안하는 방법 및 모델 구조

### 3.1 GAN 기본 구조
논문에서 제시하는 기본적인 GAN 구조는 다음과 같습니다:

**수식 1 - GAN의 목적 함수**:

$$ \min_G \max_D V(D,G) = \mathbb{E}\_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}\_{z \sim p_z(z)}[\log(1-D(G(z)))] $$

여기서 G는 생성자(Generator), D는 판별자(Discriminator)를 의미합니다[1].

### 3.2 분류 체계에 따른 방법들

# 3 Popular GANs for Image Applications

## 개요  
Section 3에서는 **데이터 샘플 크기**에 따라 **Big-Sample GANs**와 **Small-Sample GANs** 두 가지 관점으로, 이미지 생성·변환·검출·보정 등 다양한 컴퓨터 비전 과제에 활용된 대표적인 GAN(Generative Adversarial Network) 변형들을 체계적으로 정리합니다.

## 3.1 Big-Sample GANs for Image Applications  
“Big-Sample”이란 대규모 학습 데이터를 충분히 확보한 상태에서 GAN을 훈련하는 경우를 뜻합니다. 이 영역의 주요 응용은 **이미지 생성(Generation)**과 **객체 검출(Object Detection)** 두 가지입니다.

### 3.1.1 이미지 생성 (Image Generation)  
- **목적**: 풍부한 데이터 분포를 학습해 실제 같은 고해상도·고품질 이미지를 합성  
- **대표 모델**  
  - **StyleGAN**  
    - *특징*: 잠재 공간을 스타일 레이어별로 조작해 얼굴·사물 등 특정 영역의 세부 질감·형태 제어 가능.  
  - **BEGAN**  
    - *특징*: Wasserstein 거리 기반 오토인코더 디스크리미네이터 사용, 생성자·판별자 균형 유지.  
  - **MGAN / PSGAN / SGAN**  
    - *특징*: Markovian 패치 기반 실시간 텍스처 합성(PSGAN), 주기성 텍스처 학습(MGAN), 공간 텐서 사용(SGAN).  

| 모델     | 키워드                           |
|---------|---------------------------------|
| StyleGAN | 스타일 제어, 잠재 공간 분리            |
| BEGAN     | 경계 평형 오토인코더                   |
| MGAN     | 마코프 패치 텍스처 합성               |
| PSGAN     | 주기적 텍스처 학습                   |
| SGAN     | 공간 텐서 기반 텍스처 합성            |

### 3.1.2 객체 검출 (Object Detection)  
- **목적**: GAN을 통해 부족한 학습 데이터를 보강하거나, 작은 객체의 해상도·특징을 향상시켜 검출 성능 개선  
- **대표 모델**  
  - **SeGAN**  
    - *특징*: 물체의 보이지 않는(occluded) 부분을 세그멘터와 생성 네트워크로 복원.  
  - **Perceptual GAN**  
    - *특징*: 저해상도 소형 객체 특성을 고해상도 대형 객체 특성으로 변환해 검출기 성능 향상.  
  - **SOD-MTGAN**  
    - *특징*: 다중 태스크 GAN으로 SR(초해상도)와 검출을 통합 학습.  

| 모델           | 키워드                                |
|---------------|--------------------------------------|
| SeGAN       | 세그멘테이션 기반 occluded 객체 복원      |
| Perceptual GAN | 소형→대형 객체 표현 변환               |
| SOD-MTGAN    | SR+검출 다중 태스크 학습              |

## 3.2 Small-Sample GANs for Image Applications  
“Small-Sample”은 학습 데이터가 제한적일 때 GAN을 활용하는 분야로, 대표적으로 **스타일 전이(Style Transfer)**와 **이미지 복원·인페인팅(Inpainting)** 과제가 있습니다.

### 3.2.1 이미지 스타일 전이 (Style Transfer)  
- **목적**: 페어링(쌍)된 고·저해상도 데이터 없이도 서로 다른 스타일(예: 메이크업, 색감, 질감)을 전이  
- **대표 모델**  
  - **CycleGAN**  
    - *특징*: 두 개의 GAN을 ‘순환 일관성(cycle consistency)’으로 연결, 페어링 데이터 없이 학습.  
  - **RAMT-GAN**  
    - *특징*: 메이크업 데이터셋에 특화된 스타일 전이.  
  - **CATVGAN / ITCGAN / ArCycleGAN / URCycleGAN / ECycleGAN**  
    - *특징*: 상관 정렬, U-net 구조, 속성 등록, CBAM(attention) 등 다양한 변형.  

| 모델             | 키워드                              |
|------------------|------------------------------------|
| CycleGAN      | 비페어 학습, 순환 일관성               |
| RAMT-GAN      | 메이크업 스타일 전이                  |
| CATVGAN        | 상관 정렬 스타일 전이                 |
| ITCGAN         | U-net 기반 조건부 GAN                |
| ArCycleGAN     | 속성 기반 순환 GAN                   |
| URCycleGAN     | U-net + CycleGAN                   |
| ECycleGAN     | CBAM(attention) 적용 CycleGAN        |

### 3.2.2 이미지 인페인팅 (Inpainting)  
- **목적**: 파손·결측된 이미지 영역을 주변 정보와 GAN으로 보강하여 자연스럽게 복원  
- **대표 모델**  
  - **PGGAN**  
    - *특징*: 패치 기반 인페인팅, 국소적 세부 복원.  
  - **DE-GAN**  
    - *특징*: 사전 지식(prior) 활용, 얼굴 영역 보정 최적화.  
  - **GFC / GIICA**  
    - *특징*: 오토인코더, 문맥적(attention) 메커니즘 사용.  

| 모델          | 키워드                             |
|--------------|-----------------------------------|
| PGGAN      | 패치 기반 종합 인페인팅             |
| DE-GAN    | prior 지식 기반 얼굴 인페인팅       |
| GFC        | 오토인코더 + 의미론적 파싱 손실      |
| GIICA     | 컨텍스추얼 어텐션 기반 복원         |

## 정리  
1. **Big-Sample GANs**  
   - 방대한 데이터로 고품질 이미지 생성(StyleGAN, BEGAN 등)  
   - 객체 검출 성능 향상(SeGAN, Perceptual GAN 등)  
2. **Small-Sample GANs**  
   - 비페어 학습 스타일 전이(CycleGAN 및 변형)  
   - 손상 이미지 인페인팅(PGGAN, DE-GAN, GFC, GIICA 등)  

Section 3는 이처럼 **샘플 규모**와 **응용 과제**에 따라 GAN 변형을 체계적으로 분류·비교하며, 각 모델이 해결한 **핵심 문제**와 **핵심 기법(네트워크 구조, 손실 함수, attention, prior 지식 등)** 을 상세히 설명합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2f9036fc-c08f-4e94-88a8-aeb55db44761/2204.13620v4.pdf

#### 지도학습 GANs (Supervised GANs)
- **SRGAN**: 잔차 네트워크와 지각 손실을 결합한 최초의 GAN 기반 SISR 방법[3]
- **ESRGAN**: 잔차 밀집 블록(Residual-in-Residual Dense Block)을 사용하여 더 자세한 정보 복원[1]
- **ESRGAN+**: 인접 레이어 융합과 노이즈 추가를 통한 성능 향상[1]

#### 반지도학습 GANs (Semi-supervised GANs)
- **GAN-CIRCLE**: 와서스타인 거리의 순환 일치성을 유지하여 노이즈가 있는 저해상도 이미지를 고해상도로 변환[1]
- **MSSR**: 소프트 다중 라벨을 사용한 반지도 초해상도 방법[1]

#### 비지도학습 GANs (Unsupervised GANs)
- **CycleGAN 기반 방법들**: 쌍을 이루지 않은 데이터셋에서 학습할 수 있는 순환 일치성 기반 방법들[1]
- **KernelGAN**: 내부 GAN을 사용한 블라인드 초해상도 방법[1]

# 4 GANs for Image Super-Resolution: 이해하기 쉬운 상세 설명

이 절에서는 GAN(Generative Adversarial Network)을 *이미지 초해상도(Super-Resolution, SR)* 목적으로 활용한 연구를 네 가지 관점으로 정리한다. 먼저, **감독학습(supervised)**, **반감독학습(semi-supervised)**, **비감독학습(unsupervised)** 방식별로 GAN 변형을 분류하고, 각 방식 내부에서  
1. 네트워크 구조(architecture) 개선  
2. 사전 지식(prior knowledge) 활용  
3. 손실 함수(loss function) 개선  
4. 다중 과제(multi-task) 통합  
관점으로 주요 모델을 설명한다.  

## 4.1 감독학습 GANs for SR  
### 4.1.1 구조 개선 기반(Supervised + Improved Architectures)  
- **LAPGAN**: Laplacian 피라미드 구조로 여러 단계(coarse-to-fine)에서 점진적으로 해상도 복원.  
- **PGGAN(ProGAN)**: 생성자·판별자를 점진적으로 확장하여 안정적 고해상도 생성.  
- **ESRGAN**: Residual-in-Residual Dense Block(RRDB)과 상대적 판별자(relativistic discriminator)를 도입해 질감 복원력 강화.  
- **ESRGAN+**: 인접 레이어 융합 및 노이즈 입력으로 세부 묘사 극대화.  
- **Multi-Discriminator GAN**: 관점·경계·주파수별 판별자 추가로 checkerboard artifact 및 고주파 왜곡 완화.

### 4.1.2 사전 지식 활용(Supervised + Prior Knowledge)  
- **SRDGAN**: Dual-GAN으로 고→저(LR)와 저→고(HR) 변환망 학습, 노이즈·블러 커널 추정 전초 작업 포함.  
- **GLEAN**: 대규모 사전학습된 네트워크 특징을 텍스처 복원에 활용.  
- **I-SRGAN**: 적외선(IR) 영상 특성(gradient prior) 반영해 열화 커널 불확실성 감소.

### 4.1.3 손실 함수 개선(Supervised + Loss)  
- **RankSRGAN**: 순위 학습(ranking) 기반 손실로 인간 지각 품질에 부합하도록 학습.  
- **GMGAN**: 새로운 화질 평가 지표(quality loss) 통합으로 시각적 품질 제고.  
- **FSLSR**: Fourier 공간 손실로 고주파 정보 보존 및 학습 가속.  
- **I-WAGAN**: 개선된 Wasserstein penalty와 지각 손실 결합.

### 4.1.4 다중 과제 통합(Supervised + Multi-Task)  
- **MSSRGAN**: SR과 노이즈 제거 모듈 결합 후 인물 재식별(re-ID) 네트워크 학습[1].  
- **RSISRGAN**: SR과 위성 영상 물체 검출 결합해 탐지 성능 향상.  
- **JPLSRGAN**: 차량 번호판 SR과 판독(task) 동시 최적화.  
- **MRD-GAN**, **MESRGAN+**: SR과 의료 영상 denoising 동시 수행.

## 4.2 반감독학습 GANs for SR  
- **GAN-CIRCLE**: Wasserstein 기반 cycle-consistency로 CT 영상 노이즈 억제 및 SR.  
- **MSSR**: soft multi-label 그래프 컨볼루션과 결합해 제한적 라벨에서 SR 학습.  
- **CTGAN**: 준지도 방식으로 adversarial + cycle + identity + sparsifying transform 손실 결합.  
- **Gemini-GAN**: 3D SR과 분할(segmentation)을 혼합 도메인 적응으로 동시 해결.

## 4.3 비감독학습 GANs for SR  
### 4.3.1 구조 개선 기반(Unsup. + Arch.)  
- **CinCGAN**: Cycle-in-Cycle GAN으로 blind SR 단계별(denoise→upsample→fine-tune) 학습.  
- **DNSR**: 양방향 구조 일관성(bidirectional structural consistency)으로 블라인드 SR.  
- **MCinCGAN**: 다중 Cycle-GAN 쌓기로 표현력 극대화.  
- **RWSR-CycleGAN**: checkerboard artifact 완화용 복합 업샘플 모듈 결합.  
- **KernelGAN**, **InGAN**: 내부 패치 통계로 낮은 해상도 커널 추정, 자체-생성 이미지로 blind SR.

### 4.3.2 사전 지식 활용(Unsup. + Prior)  
- **DULGAN**: 데이터 일관성(data fidelity) + 정규화(regularizer) + adversarial loss로 픽셀 충실도 보장.  
- **USROCTGAN**: cycle + identity prior로 OCT 영상 구조·색·텍스처 복원.  
- **EIPGAN**: 위성 영상 통계 prior로 고해상도 구조 정보 재구성.

### 4.3.3 손실 함수 개선(Unsup. + Loss)  
- **URSGAN**: IQA 기반 손실(image quality assessment)로 원격탐사 SR 품질 강화.  
- **MADGAN**: Self-attention GAN(SAGAN) + L₁ 손실로 뇌 MRI anomaly detection과 SR 동시.  
- **DLGAN**: content loss 강화로 하이퍼스펙트럴 SR 최적화.

### 4.3.4 다중 과제 통합(Unsup. + Multi-Task)  
- **VAEGAN**: VAE + GAN + IQA 아이디어로 노이즈 제거 및 SR 동시.  
- **ASLGAN**: 저역통과 필터 손실과 다중 MR 시퀀스 가중치로 MRI denoise + SR.  
- **Pix2NeRF**, **Pi-GAN**: 3D implicit representation 기반 SR과 3D-aware 이미지 합성 병합.

위 분류를 통해 **GAN 기반 이미지 초해상도** 기법이  
- *감독 / 반감독 / 비감독* 방식별,  
- *구조 / 사전 지식 / 손실 함수 / 다중 과제* 관점으로  
어떻게 발전해 왔는지 쉽게 파악할 수 있다. GAN 특유의 대립학습(adversarial training)을 활용해 **질감 디테일 복원**, **블라인드 SR**, **다중 손상 복원** 등 다양한 현실 과제를 해결하는 연구들이 활발히 진행 중이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2f9036fc-c08f-4e94-88a8-aeb55db44761/2204.13620v4.pdf

### 3.3 주요 손실 함수

**수식 2 - 지각 손실(Perceptual Loss)**:

$$ L_{perceptual} = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}} (\phi\_{i,j}(I^{HR})\_{x,y} - \phi_{i,j}(G_{\theta_G}(I^{LR})))^2 $$

여기서 $$\phi_{i,j}$$는 VGG 네트워크의 특징 맵을 의미합니다[3].

## 4. 성능 향상 및 한계

### 4.1 성능 향상
논문에서 제시된 주요 성능 지표는 다음과 같습니다:

| 방법 | 데이터셋 | PSNR | SSIM |
|------|----------|------|------|
| ESRGAN | Set14 (×4) | 30.50 | 0.7620 |
| SRGAN | Set14 (×4) | 26.02 | 0.7379 |
| KernelGAN | Set14 (×2) | 30.36 | 0.8669 |

[1]

### 4.2 주요 한계점

1. **불안정한 훈련**: 생성자와 판별자 간의 대립적 최적화로 인한 훈련 불안정성[1][2]

2. **높은 계산 비용**: GAN은 생성자와 판별자로 구성되어 계산 비용과 메모리 소모가 큼[1]

3. **참조 이미지 필요**: 대부분의 기존 GANs는 쌍을 이룬 고해상도-저해상도 이미지가 필요[1]

4. **복잡한 이미지 초해상도 처리 한계**: 실제 환경의 복합적 손상(저해상도 + 노이즈 + 어둠 등)을 효과적으로 처리하지 못함[1]

5. **평가 지표의 한계**: PSNR과 SSIM만으로는 복원된 이미지의 품질을 완전히 측정할 수 없음[1]

## 5. 일반화 성능 향상 가능성

논문에서 제시한 일반화 성능 향상 방안들:

### 5.1 구조적 개선
- **어텐션 메커니즘 도입**: Transformer 기반 어텐션을 활용한 중요 특징 추출[1][4]
- **경량화 모델 설계**: 그룹 컨볼루션, 사전 지식과 얕은 네트워크 구조의 결합[1]

### 5.2 학습 방법론 개선  
- **자기지도학습 활용**: 고품질 참조 이미지 확보를 위한 자기지도학습 방법[1]
- **멀티태스크 학습**: 다양한 저수준 태스크의 속성을 결합한 복합적 문제 해결[1]

### 5.3 최신 연구 동향
최근 연구들에서는 **확산 모델(Diffusion Models)과의 결합**이 주목받고 있습니다[2][5]. 이는 GAN의 빠른 추론 속도와 확산 모델의 높은 품질을 결합하여 더 나은 일반화 성능을 달성할 가능성을 제시합니다[2].

## 6. 향후 연구에 미치는 영향과 고려사항

### 6.1 연구에 미치는 영향

1. **표준화된 분류 체계 제공**: 지도/반지도/비지도 학습 기반 분류가 향후 연구의 기준점 역할[1]

2. **벤치마크 설정**: 197개 논문의 성능 비교를 통한 연구 벤치마크 제공[1]

3. **연구 방향성 명확화**: 실제 환경 적용을 위한 구체적 연구 방향 제시[1]

### 6.2 향후 연구 시 고려사항

#### 즉시 고려해야 할 점들:
1. **하이브리드 모델 개발**: GAN과 확산 모델, Transformer 등의 결합 연구 필요[2][6]
2. **실시간 처리 가능한 경량화**: 모바일 및 엣지 디바이스에서의 실시간 처리를 위한 경량화 연구[7]
3. **다중 도메인 적응**: 의료, 위성, 감시 등 다양한 도메인에 특화된 방법론 개발[8][9]

#### 장기적 연구 방향:
1. **새로운 평가 지표 개발**: PSNR, SSIM을 넘어서는 인간 지각에 기반한 평가 지표 개발[1]
2. **자기지도학습 기반 방법론**: 라벨이 필요 없는 완전 자기지도 방법론 개발[1]
3. **해석 가능한 AI**: GAN의 블랙박스 특성을 극복하는 해석 가능한 초해상도 모델 개발[6]

이 서베이는 GANs 기반 이미지 초해상도 연구의 현재 상태를 종합적으로 정리하고, 향후 연구가 나아가야 할 명확한 방향을 제시함으로써 이 분야의 발전에 중요한 기여를 하고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2f9036fc-c08f-4e94-88a8-aeb55db44761/2204.13620v4.pdf
[2] https://www.nature.com/articles/s41598-024-52370-3
[3] https://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html
[4] https://ieeexplore.ieee.org/document/10657802/
[5] https://arxiv.org/html/2504.13622v1
[6] https://ieeexplore.ieee.org/document/10737883/
[7] https://ieeexplore.ieee.org/document/10534055/
[8] https://ieeexplore.ieee.org/document/10714352/
[9] https://www.mdpi.com/2079-9292/12/20/4235
[10] https://ieeexplore.ieee.org/document/10678355/
[11] https://ieeexplore.ieee.org/document/10704176/
[12] http://www.cjig.cn/zh/article/doi/10.11834/jig.230747/
[13] https://ieeexplore.ieee.org/document/10674442/
[14] https://ieeexplore.ieee.org/document/10872526/
[15] https://arxiv.org/pdf/2204.13620.pdf
[16] https://ojs.aaai.org/index.php/AAAI/article/view/28201/28398
[17] https://arxiv.org/abs/2204.13620
[18] http://www.diva-portal.org/smash/get/diva2:1216797/FULLTEXT01.pdf
[19] https://openaccess.thecvf.com/content/WACV2021/papers/Chen_Hierarchical_Generative_Adversarial_Networks_for_Single_Image_Super-Resolution_WACV_2021_paper.pdf
[20] https://openreview.net/forum?id=46mbA3vu25
[21] https://github.com/tensorlayer/SRGAN
[22] https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf
[23] https://velog.io/@pabiya/Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-AdversarialNetwork
[24] https://arxiv.org/html/2311.18508v1
[25] https://www.sciencedirect.com/science/article/pii/S0301932224002295
[26] https://kevinitcoding.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-SRGAN-%EB%85%BC%EB%AC%B8-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC-Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-Network
[27] https://www.sciencedirect.com/science/article/abs/pii/S0030402622009032
[28] https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.13192
[29] https://www.sciencedirect.com/science/article/abs/pii/S0167865522002756
[30] http://thesai.org/Publications/ViewPaper?Volume=15&Issue=11&Code=ijacsa&SerialNo=17
[31] https://ieeexplore.ieee.org/document/11065600/
[32] https://www.ijraset.com/best-journal/gan-based-super-resolution-algorithm-for-high-quality-image-enhancement
[33] https://ieeexplore.ieee.org/document/10678491/
[34] http://lib.physcon.ru/doc?id=95a2c6e82233
[35] https://www.ijraset.com/research-paper/gan-based-super-resolution-algorithm-for-high-quality-image-enhancement
[36] https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Chen_Unsupervised_Image_Super-Resolution_With_an_Indirect_Supervised_Path_CVPRW_2020_paper.pdf
[37] https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1578321/full
[38] https://www.mdpi.com/2072-4292/15/20/5062
[39] https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Kim_Unsupervised_Real-World_Super_Resolution_With_Cycle_Generative_Adversarial_Network_and_CVPRW_2020_paper.pdf
[40] https://arxiv.org/pdf/2312.16471.pdf
[41] https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Cheon_Generative_Adversarial_Network-based_Image_Super-Resolution_using_Perceptual_Content_Losses_ECCVW_2018_paper.pdf
[42] https://arxiv.org/html/2204.13620v4
[43] https://www.sciencedirect.com/science/article/abs/pii/S0031320321003873
[44] https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf
[45] https://www.sciencedirect.com/science/article/abs/pii/S1566253522001762
[46] https://www.mdpi.com/2072-4292/15/5/1391
[47] https://www.mdpi.com/2079-9292/12/13/2975
[48] https://dl.acm.org/doi/10.1145/3700035.3700037
[49] https://arxiv.org/abs/2410.17966
[50] https://arxiv.org/pdf/2107.12679.pdf
[51] https://arxiv.org/pdf/1902.06068.pdf
[52] https://arxiv.org/pdf/2404.06294.pdf
[53] https://arxiv.org/pdf/1902.02144.pdf
[54] https://arxiv.org/pdf/1908.06382.pdf
[55] https://arxiv.org/pdf/1904.07523.pdf
[56] https://arxiv.org/pdf/1810.06611.pdf
[57] https://ace.ewapublishing.org/media/ca132dcc66ca4c2c8b9137e7ef6fcf45.marked_4x2GS6p.pdf
[58] https://arxiv.org/html/2505.10589v4
[59] https://sofar-sogood.tistory.com/entry/SRGAN%EB%A6%AC%EB%B7%B0-Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-Network-CVPR-17
[60] https://consensus.app/search/what-are-the-key-differences-between-single-image-/VYytUYqORgKXbML_zgDdDQ/
[61] https://arxiv.org/abs/2411.09512
[62] https://downloads.spj.sciencemag.org/remotesensing/2021/9829706.pdf
[63] https://arxiv.org/pdf/2211.13524.pdf
[64] https://arxiv.org/abs/2001.08126v2
[65] https://www.mdpi.com/1099-4300/24/8/1030/pdf?version=1659429633
[66] https://www.matec-conferences.org/articles/matecconf/pdf/2022/17/matecconf_rapdasa2022_07011.pdf
[67] https://arxiv.org/pdf/1901.03419.pdf
[68] https://onlinelibrary.wiley.com/doi/pdfdirect/10.1049/iet-ipr.2018.5767
[69] https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf_icacc2022_03054.pdf
