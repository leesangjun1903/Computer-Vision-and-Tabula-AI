# Frequency-Aware Deepfake Detection (FreqNet) 논문 요약 및 설명

## 1. 배경 및 문제 정의  
GAN(Generative Adversarial Network)의 발전으로 인해 합성 이미지 생성이 쉬워지면서, 눈으로는 진짜와 가짜를 구분하기 어려운 Deepfake 이미지가 증가하고 있다[1].  
기존의 Deepfake 탐지기는 주로 spatial domain(공간 영역) 정보를 활용하거나, 특정 주파수 대역의 artifacts(인공 패턴)를 학습하여 훈련 데이터에 과도하게 의존함으로써, 새로운 GAN 모델로 생성된 가짜 이미지를 잘 탐지하지 못하는 한계를 보였다[2].

**목표**  
- 제한된 훈련 데이터로부터도 다양한 GAN 모델에서 생성된 Deepfake를 일반화하여 탐지할 수 있는 범용 탐지기 개발[1].

## 2. FreqNet 개요  
FreqNet은 주파수 도메인(frequency domain) 학습을 핵심으로 하는 경량화된 CNN 기반 Deepfake 탐지 네트워크이다[1].  
주요 아이디어는 입력 이미지와 중간 특성(feature) 맵을 Fast Fourier Transform(FFT)을 통해 주파수 영역으로 변환하고, 고주파 정보(high-frequency)를 지속적으로 강조하며 source-agnostic(생성 모델 특성 비의존) 학습을 수행하는 것이다[1].

### 2.1 전체 구조  
1. **High-Frequency Representation (HFR)**  
   - 입력 이미지 $$x$$를 FFT로 변환 후, 중앙 저주파 영역만 제거하는 high-pass filter $$\mathcal{B}_h$$를 적용해 고주파 성분 $$f_h$$을 추출하고, iFFT로 다시 이미지 공간 $$x_h$$로 복원하여 네트워크 입력으로 사용[1].  
2. **High-Frequency Representation of Feature**  
   - CNN의 각 컨볼루션 레이어 출력 $$M^k$$에 대해 공간 및 채널 차원별로 FFT → $$\mathcal{B}_h$$ → iFFT를 적용하여 고주파 특성 $$M^k_h$$를 강조[1].  
3. **Frequency Convolutional Layer (FCL)**  
   - 특성 맵 $$M^k$$를 FFT로 변환하여 진폭(amplitude)과 위상(phase) 스펙트럼을 분리한 뒤, 각각에 컨볼루션 레이어 $$L_{conv}$$를 적용하여 학습한 후 iFFT로 복원함으로써 주파수 공간에서 직접 학습하도록 함[1].  

이 세 모듈을 결합한 경량 CNN(FreqNet, 약 1.9M 파라미터)은 spatial 정보만 이용한 대규모 모델(304M 파라미터)보다도 높은 일반화 성능을 보인다[3].

## 3. 핵심 모듈 상세설명  

### 3.1 High-Frequency Representation (HFR)  
- **과정**:  
  1) 이미지 $$x \in \mathbb{R}^{W\times H \times 3}$$에 FFT 적용 → $$\mathcal{F}(x)$$[1]  
  2) 중심 영역 $$\lvert i\rvert < W/4, \lvert j\rvert < H/4$$를 0으로 하는 $$\mathcal{B}_h$$로 고주파 $$f_h$$ 추출[1]  
  3) iFFT($$\mathcal{IF}(f_h)$$)로 $$x_h$$ 생성 → 네트워크 입력으로 사용[1]  

- **효과**: GAN 업샘플링 과정에서 발생하는 미세한 고주파 artifacts에 민감도를 높임[1].

### 3.2 High-Frequency Feature Representation  
- **과정**:  
  각 레이어 출력 $$M^k \in \mathbb{R}^{W\times H \times C}$$에 대해,  
  - 공간 FFT $$\mathcal{F}\_{W,H}$$, 채널 FFT $$\mathcal{F}_{C}$$ → $$\mathcal{B}_h$$ 적용 → iFFT 복원[1].  
- **효과**: 중간 특징에서도 고주파 정보를 강조하여, 특정 GAN의 빈도 패턴에 과도하게 종속되지 않도록 함[1].

### 3.3 Frequency Convolutional Layer (FCL)  
- **과정**:  
  1) $$M^k$$를 FFT($$\mathcal{F}\_{W,H}$$)로 변환 → amplitude $$f_{am}$$, phase $$f_{ph}$$ 분리  
  2) $$f_{am}, f_{ph}$$에 CNN 레이어 $$L_{conv}$$ 적용→ 학습된 $$\widetilde f_{am}, \widetilde f_{ph}$$ 획득  
  3) iFFT($$\widetilde f_{am} + \mathrm{i}\widetilde f_{ph}$$)로 $$\widetilde M^k$$ 복원[1].  
- **효과**: 주파수 공간에서 직접 컨볼루션 학습을 실시하여 다양한 artifact 패턴을 포괄적으로 학습함[1].

## 4. 실험 및 성능  
- **데이터셋**: ForenSynths(8 GAN), Self-Synthesis(9 GAN) 총 17 GAN 모델[1]  
- **훈련 설정**: 1-class, 2-class, 4-class (ProGAN 기반)  
- **비교 모델**: F3Net, BiHPF, FreGAN, LGrad, Ojha 등  
- **결과**:  
  | 모델       | 파라미터 수 | 평균 Accuracy(17 GAN) |
  |------------|-------------|-----------------------|
  | Ojha (SOTA)| 304.0 M     | 83.0%                 |
  | **FreqNet**| **1.9 M**   | **92.8% (+9.8%)**     |[3]  

- **Cross-Model 테스트**: 4-class 설정에서 FreqNet은 91.5%로 최고 성능 기록[1].  
- **추가 평가**: CelebA-HQ 얼굴 이미지 테스트에서 ProGAN·StyleGAN·StyleGAN2 각각 98.7%·99.0%·99.5% 정확도 달성[1].

## 5. 결론  
FreqNet은 입력 이미지 및 중간 특징의 주파수 도메인 학습을 통해, GAN 모델별 특화된 artifacts에 과도하게 의존하지 않고 **범용 Deepfake 탐지**를 가능하게 한다[1].  
경량 구조에도 불구하고 대규모 모델 대비 높은 일반화 성능을 보이며, 실제 환경에서의 적용 가능성을 크게 높였다[3].  

---

[1] arXiv:2403.07240 “Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning”  
[2] AAAI’24 “Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning”  
[3] Table 3 in arXiv:2403.07240 (Parameters vs. mAcc.)

[1] https://ojs.aaai.org/index.php/AAAI/article/view/28310
[2] https://arxiv.org/abs/2403.07240
[3] https://ieeexplore.ieee.org/document/10286049/
[4] https://arxiv.org/abs/2504.00454
[5] https://ieeexplore.ieee.org/document/10203558/
[6] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13276
[7] https://www.mdpi.com/2079-9292/13/9/1749
[8] https://www.mdpi.com/1424-8220/23/21/8763
[9] https://ojs.aaai.org/index.php/AAAI/article/view/28310/28609
[10] https://www.themoonlight.io/en/review/frequency-aware-deepfake-detection-improving-generalizability-through-frequency-space-learning
[11] https://paperswithcode.com/paper/frequency-aware-deepfake-detection-improving
[12] https://arxiv.org/html/2403.07240v1
[13] https://sinoxiv.napstic.cn/article/6244102
[14] https://www.semanticscholar.org/paper/Frequency-Aware-Deepfake-Detection:-Improving-Space-Tan-Zhao/40fc0bf0309db53e60d5b2f656991d72631d8481
[15] http://arxiv.org/pdf/2403.07240v1.pdf
[16] https://www.themoonlight.io/es/review/frequency-aware-deepfake-detection-improving-generalizability-through-frequency-space-learning
[17] https://ieeexplore.ieee.org/document/10240940/
[18] https://ieeexplore.ieee.org/document/10657197/
[19] https://dl.acm.org/doi/10.1609/aaai.v38i5.28310
