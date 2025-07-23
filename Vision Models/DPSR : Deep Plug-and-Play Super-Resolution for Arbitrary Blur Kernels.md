# DPSR : Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels | Super resolution

## 1. 핵심 주장 및 주요 기여

이 논문의 핵심 주장은 기존의 딥러닝 기반 단일 이미지 초해상도(SISR) 방법들이 주로 bicubic degradation에 설계되어 있어 임의의 blur kernel을 가진 저해상도 이미지에 대해서는 성능이 제한적이라는 문제를 지적하고, 이를 해결하기 위한 새로운 접근법을 제시한다는 것입니다[1].

### 주요 기여는 다음과 같습니다:

**1) 새로운 열화 모델 제안**: 기존의 일반적인 열화 모델 $$y = (x \otimes k) \downarrow_s + n $$ 과 단순한 bicubic 모델 $$y = x \downarrow_s $$ 대신, $$y = (x \downarrow_s) \otimes k + n $$ 이라는 새로운 열화 모델을 제안했습니다[1][2].

**2) 딥 플러그 앤 플레이 프레임워크**: 기존의 plug-and-play 방식이 가우시안 디노이저를 prior로 사용하는 것과 달리, 본 연구는 super-resolver prior를 사용하는 방식을 도입했습니다[1][2].

**3) 원리적 최적화 알고리즘**: Half Quadratic Splitting (HQS) 알고리즘을 통해 변수 분할 기법을 사용하여 에너지 함수를 최적화하는 체계적인 접근법을 제시했습니다[1][2].

## 2. 해결하고자 하는 문제

### 핵심 문제점:
- **기존 DNN 기반 SISR 방법들의 한계**: 대부분의 딥러닝 기반 초해상도 방법들이 bicubic degradation에만 최적화되어 있어, 실제 환경에서 발생하는 다양한 blur kernel에 대해서는 성능이 크게 저하됨[1][2]
- **Blur kernel 추정의 어려움**: 일반적인 열화 모델에서는 blur kernel 추정이 매우 어려움[1]
- **기존 방법들의 제약**: SRMD는 가우시안 blur kernel에만 제한적이고, ZSSR은 심하게 블러된 이미지에 대해서는 효과가 제한적임[1][2]

## 3. 제안하는 방법 및 수식

### 3.1 새로운 열화 모델
논문에서 제안하는 핵심 열화 모델은 다음과 같습니다:

$$ y = (x \downarrow_s) \otimes k + n $$

여기서:
- $$x $$: 원본 고해상도 이미지
- $$y $$: 관측된 저해상도 이미지  
- $$\downarrow_s $$: bicubic downsampler (scale factor s)
- $$k $$: blur kernel
- $$n $$: 가우시안 노이즈[1]

### 3.2 에너지 함수
Maximum A Posteriori (MAP) 추정에 따른 에너지 함수:

$$ \min_x \frac{1}{2\sigma^2} \|y - (x \downarrow_s) \otimes k\|^2 + \lambda\Phi(x) $$

여기서 $$\Phi(x) $$ 는 정규화 항, $$\lambda $$ 는 정규화 파라미터입니다[1].

### 3.3 변수 분할 기법
보조 변수 $$z$$를 도입하여 제약 최적화 문제로 변환:

$$ \hat{x} = \arg\min_x \frac{1}{2\sigma^2} \|y - z \otimes k\|^2 + \lambda\Phi(x) $$
$$ \text{subject to } z = x \downarrow_s $$

### 3.4 HQS 알고리즘
Half Quadratic Splitting을 통한 반복적 해법:

$$ L_\mu(x,z) = \frac{1}{2\sigma^2} \|y - z \otimes k\|^2 + \lambda\Phi(x) + \frac{\mu}{2} \|z - x \downarrow_s\|^2 $$

이를 두 개의 하위 문제로 분해:

**Z-step (Eqn. 7)**: 

$$ z^{k+1} = \arg\min_z \|y - z \otimes k\|^2 + \mu\sigma^2 \|z - x^k \downarrow_s\|^2 $$

**X-step (Eqn. 8)**:

$$ x^{k+1} = \arg\min_x \frac{\mu}{2} \|z^{k+1} - x \downarrow_s\|^2 + \lambda\Phi(x) $$

### 3.5 폐쇄형 해법
Z-step은 FFT를 이용한 폐쇄형 해법을 가집니다:

$$z^{k+1} = \mathcal{F}^{-1}\left(\frac{\mathcal{F}(k)^*\mathcal{F}(y) + \mu\sigma^2\mathcal{F}(x^k \downarrow_s)}{|\mathcal{F}(k)|^2 + \mu\sigma^2}\right)$$

X-step은 super-resolver prior로 해결됩니다:

$$x^{k+1} = SR(z^{k+1}, s, \sqrt{1/\mu})$$

## 4. 모델 구조

### 4.1 SRResNet+ 구조
논문에서는 기존 SRResNet을 수정한 SRResNet+를 제안합니다:

- **입력**: 저해상도 이미지 + 노이즈 레벨 맵
- **특징 맵 수**: 64개에서 96개로 증가
- **배치 정규화 제거**: 더 나은 성능을 위해 제거
- **훈련 데이터**: DIV2K 데이터셋의 800개 이미지 사용[1]

### 4.2 반복적 최적화 구조
1. **Blur 처리**: FFT 기반 폐쇄형 해법으로 blur 왜곡 제거
2. **Super-resolution**: 딥 네트워크를 통한 고해상도 복원
3. **교대 반복**: 두 단계를 반복하여 최적해 도출[1]

## 5. 성능 향상 및 실험 결과

### 5.1 정량적 성능
**BSD68 데이터셋 결과** (Scale factor ×4)[1]:
- **Gaussian blur**: DPSR 24.04dB vs RCAN 21.27dB
- **Motion blur**: DPSR 25.69dB vs RCAN 18.08dB  
- **Disk blur**: DPSR 24.84dB vs RCAN 19.85dB

### 5.2 비교 방법 대비 우수성
- **VDSR, RCAN**: bicubic degradation에만 최적화되어 복잡한 blur에서 성능 저하
- **IRCNN+RCAN**: 단계별 오류 누적 문제
- **DeblurGAN+RCAN, GFN**: 대형 복잡 blur kernel 처리 능력 제한
- **ZSSR**: 심한 blur에 대한 효과 제한[1]

### 5.3 수렴성 및 효율성
- **빠른 수렴**: 약 15번의 반복으로 수렴
- **실행 시간**: 256×256 이미지에 대해 약 1.8초 (단일 GPU)
- **ZSSR 대비**: 12-18초 vs 1.8초로 크게 향상[1]

## 6. 모델의 한계

### 6.1 방법론적 한계
1. **Non-blind SISR 한정**: 논문은 blur kernel이 알려진 상황에서만 작동하며, 완전한 blind SISR은 다루지 않음[1]
2. **Uniform blur kernel**: Non-uniform blur kernel은 고려하지 않아 실제 환경에서의 적용성이 제한적[1]
3. **별도 훈련 필요**: 각 scale factor별로 별도의 모델 훈련이 필요[1]

### 6.2 실용적 한계
1. **Blur kernel 추정 의존성**: 실제 환경에서는 정확한 blur kernel 추정이 어려울 수 있음
2. **계산 복잡도**: 반복적 최적화로 인한 추가적인 계산 비용
3. **메모리 요구량**: FFT 연산과 반복 과정에서 상당한 메모리 사용[1]

## 7. 일반화 성능 향상 가능성

### 7.1 플러그 앤 플레이 프레임워크의 유연성
**모듈러 구조의 장점**:
- **Super-resolver 교체 가능**: 다양한 최신 네트워크 아키텍처로 쉽게 교체 가능[3][4]
- **다양한 prior 적용**: Denoiser prior 외에 super-resolver prior 사용으로 새로운 가능성 제시[1][2]

### 7.2 다양한 응용 분야로의 확장
**실제 적용 사례들**:
- **원격 감지 이미지**: DPSRResNet으로 위성 이미지 초해상도에 적용[5]
- **적외선 이미지**: DMSR 알고리즘으로 적외선 이미지 품질 향상[4]
- **의료 영상**: 감마선 영상 시스템의 해상도 향상[6]

### 7.3 최신 기술과의 융합 가능성
**발전 방향**:
- **Diffusion 모델 결합**: DiffPIR과 같이 diffusion 모델을 prior로 활용[7][8]
- **Flow Matching 적용**: 생성 모델의 장점을 활용한 PnP-Flow 방법[9][10]
- **Attention 메커니즘**: RCAN 등의 attention 기반 네트워크와의 결합[11][12]

### 7.4 이론적 확장성
**수학적 프레임워크의 견고성**:
- **다른 분할 알고리즘**: ADMM, FISTA 등 다양한 변수 분할 방법 적용 가능[1]
- **수렴성 보장**: 이론적 수렴성 분석을 통한 안정적인 최적화[13][14]

## 8. 미래 연구에 미치는 영향 및 고려사항

### 8.1 연구 방향에 미치는 영향

**1) Plug-and-Play 패러다임의 확산**:
- **Prior의 다양화**: Denoiser에서 super-resolver, 그리고 최근 diffusion 모델까지 prior의 범위가 확장[7][9]
- **모듈러 접근법**: 물리적 모델과 학습 기반 모델의 효과적 결합 방법론 제시[15][16]

**2) 실제 환경 적합성 강조**:
- **현실적 열화 모델**: Bicubic을 넘어선 다양한 열화 상황 고려의 중요성 부각[1][17]
- **Domain Gap 해결**: 합성 데이터와 실제 데이터 간의 차이 극복 필요성[18][17]

### 8.2 후속 연구 동향

**1) 방법론적 발전**:
- **개선된 DPSR 변형들**: IDPSR(Residual-in-Residual Dense Block 활용)[3], Enhanced DPSR with RCAN[11][12] 등
- **수렴성 및 안정성 향상**: Equivariant PnP[19], Convergent Unrolled Networks[14] 등

**2) 응용 분야 확장**:
- **비디오 초해상도**: PnP 방법의 시간적 일관성 고려[20]
- **3D 및 의료 영상**: CT, MRI 등 의료 영상 복원[15][21][22]

### 8.3 향후 연구 시 고려사항

**1) 기술적 도전과제**:
```
• End-to-End 학습: 반복적 최적화와 딥러닝의 완전한 통합
• Blur Kernel 추정: 더 정확하고 강건한 kernel 추정 방법
• 계산 효율성: 실시간 처리를 위한 최적화
• Non-uniform Blur: 공간적으로 변하는 blur 처리
```

**2) 실용성 관련 고려사항**:
```
• 사용자 친화성: 복잡한 파라미터 튜닝 없는 자동화
• 하드웨어 요구사항: 모바일 기기에서의 구현 가능성  
• 데이터셋 다양성: 더 다양한 실제 환경 데이터 필요
• 평가 메트릭: PSNR/SSIM을 넘어선 지각적 품질 평가
```

**3) 이론적 발전 방향**:
```
• 수렴성 이론: 더 일반적인 조건에서의 수렴성 보장
• 최적성 분석: Global optimum 도달 조건 분석
• 불확실성 정량화: 복원 결과의 신뢰도 측정
• 다중 목적 최적화: 여러 품질 기준의 동시 고려
```

**4) 실제 응용을 위한 고려사항**:
```
• 도메인 적응: 특정 응용 분야별 최적화 방법
• 사용자 상호작용: 사용자 피드백을 활용한 적응적 복원
• 윤리적 고려: 이미지 조작 및 deepfake 방지
• 표준화: 성능 평가 및 비교를 위한 표준 벤치마크
```

## 결론

DPSR 논문은 단순히 새로운 알고리즘을 제시한 것을 넘어서, plug-and-play 프레임워크의 가능성을 크게 확장하고 실제 환경에서의 초해상도 문제 해결에 대한 새로운 패러다임을 제시했습니다. 특히 super-resolver prior의 도입과 체계적인 최적화 방법론은 후속 연구들의 기반이 되고 있으며, 최근의 diffusion 모델이나 flow matching 등 최신 생성 모델과의 결합 연구로 이어지고 있습니다[7][9][10]. 

향후 연구에서는 완전한 blind SISR, 실시간 처리, 그리고 더 다양한 실제 환경에서의 적용성을 고려한 발전이 필요할 것으로 보입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/91b5fb45-fcd8-40ee-a07d-27b3cb81710c/1903.12529v1.pdf
[2] https://ieeexplore.ieee.org/document/8953757/
[3] https://link.springer.com/10.1007/s10044-023-01192-6
[4] https://dl.acm.org/doi/10.1145/3408127.3408162
[5] https://ieeexplore.ieee.org/document/9324647/
[6] https://pubs.aip.org/mre/article/10/2/027402/3335835/Single-image-super-resolution-of-gamma-ray-imaging
[7] https://ieeexplore.ieee.org/document/10208800/
[8] https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.html
[9] https://arxiv.org/abs/2410.02423
[10] https://openreview.net/forum?id=5AtHrq3B5R
[11] https://journals.sagepub.com/doi/full/10.3233/JIFS-202696
[12] https://openreview.net/forum?id=WOMveNqCCW
[13] https://math.stackexchange.com/questions/2674263/half-quadratic-splitting-alternating-optimization-with-penalty
[14] https://arxiv.org/abs/2402.12872
[15] https://ieeexplore.ieee.org/document/10289892/
[16] https://link.springer.com/10.1007/978-3-030-58601-0_27
[17] https://www.sciencedirect.com/science/article/abs/pii/S0893608025005015
[18] https://github.com/cszn/DPSR/blob/master/demo_test_dpsr.py
[19] https://openaccess.thecvf.com/content/CVPR2024/papers/Terris_Equivariant_Plug-and-Play_Image_Reconstruction_CVPR_2024_paper.pdf
[20] https://ieeexplore.ieee.org/document/10401873/
[21] https://proceedings.mlr.press/v172/xin22a/xin22a.pdf
[22] https://openreview.net/forum?id=h7rXUbALijU
[23] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11878/2601705/Automatic-detection-and-counting-of-small-yellow-thrips-on-lotus/10.1117/12.2601705.full
[24] https://arxiv.org/abs/1903.12529
[25] https://patents.google.com/patent/CN112070669A/en
[26] https://scholar.google.com/citations?user=0RycFIIAAAAJ
[27] https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.pdf
[28] https://dblp.org/rec/conf/cvpr/0008Z019
[29] https://sci-hub.se/downloads/2020-10-07/ce/tian2020.pdf
[30] https://openreview.net/forum?id=J7eQGL9jcY
[31] https://www.computer.org/csdl/proceedings-article/cvpr/2019/329300b671/1gyrRGoXUVG
[32] https://www.sciencedirect.com/science/article/abs/pii/S0893608020302665
[33] https://velog.io/@datapro9207/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-DPSRDeep-Plug-and-play-Super-Resolution-for-Arbitrary-Blur-Kernels
[34] https://scholar.google.fi/citations?user=0RycFIIAAAAJ
[35] https://github.com/cszn/DPSR
[36] https://paperswithcode.com/paper/deep-plug-and-play-super-resolution-for
[37] https://link.springer.com/10.1007/s11277-022-09490-8
[38] https://arxiv.org/abs/2402.01779
[39] https://arxiv.org/abs/2403.01144
[40] https://www.mdpi.com/2313-433X/10/2/50
[41] https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/ICIP-Turning-Denoiser-to-Super-Resolver.pdf
[42] https://stanford.edu/class/ee367/slides/lecture10.pdf
[43] https://arxiv.org/abs/2008.13751
[44] https://stanford.edu/class/ee367/reading/ee367_notes_deconvolution.pdf
[45] https://www.reddit.com/r/MachineLearning/comments/7gls3j/r_deep_image_prior_deep_superresolution/
[46] https://engineering.purdue.edu/ChanGroup/project_PnP.html
[47] https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.html
[48] https://perso.telecom-paristech.fr/aleclaire/mva/9_plug_and_play.pdf
[49] https://www.sciencedirect.com/science/article/abs/pii/S135044951930667X
[50] https://www.degruyter.com/document/doi/10.1515/jiip-2019-0054/html
[51] https://link.springer.com/10.1007/978-3-030-66823-5_7
[52] https://arxiv.org/pdf/2303.08999.pdf
[53] https://arxiv.org/pdf/2306.00386.pdf
[54] http://arxiv.org/pdf/2206.07281.pdf
[55] https://www.mdpi.com/1424-8220/21/14/4892/pdf
[56] http://arxiv.org/pdf/2107.08717.pdf
[57] https://arxiv.org/html/2306.15244
[58] http://arxiv.org/pdf/2410.11666.pdf
[59] https://pmc.ncbi.nlm.nih.gov/articles/PMC11195205/
[60] https://pmc.ncbi.nlm.nih.gov/articles/PMC8309932/
[61] https://www.mdpi.com/2076-3417/11/7/3285/pdf
[62] https://dl.acm.org/doi/abs/10.1007/s10044-023-01192-6
[63] https://iopscience.iop.org/article/10.1088/1361-6560/ad8c98
[64] https://www.semanticscholar.org/paper/d4bfe991fab1a66be502eb3700ce6ff47d342d65
[65] http://arxiv.org/pdf/2501.03780.pdf
[66] http://arxiv.org/pdf/2207.12056.pdf
[67] http://arxiv.org/pdf/2408.08091.pdf
[68] https://arxiv.org/html/2402.01779v2
[69] https://arxiv.org/abs/2312.01831
[70] https://www.mdpi.com/2313-433X/10/2/50/pdf?version=1708353353
[71] https://arxiv.org/abs/2209.08240
[72] https://arxiv.org/abs/2302.14736
[73] https://arxiv.org/html/2407.04621
[74] https://arxiv.org/html/2410.09529v1
