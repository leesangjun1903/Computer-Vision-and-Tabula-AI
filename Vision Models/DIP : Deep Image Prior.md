# Deep Image Prior | Image denoising, Image generation, Super resolution

Deep Image Prior 논문은 컴퓨터 비전 분야에서 중요한 패러다임 변화를 제시한 연구로, 사전 훈련 없이도 단일 이미지를 사용하여 다양한 역문제를 해결할 수 있다는 혁신적인 접근법을 제안했습니다[1].

## 핵심 주장 및 주요 기여

### 1. 핵심 주장

**네트워크 구조 자체가 강력한 이미지 사전(Prior)을 제공한다**는 것이 논문의 핵심 주장입니다. 저자들은 딥 컨볼루션 네트워크의 성공이 단순히 대량의 훈련 데이터에서 학습한 사전 지식 때문이 아니라, **네트워크 구조 자체가 자연 이미지의 저수준 통계적 특성을 포착하는 강력한 귀납적 편향(Inductive Bias)을 제공하기 때문**임을 보였습니다[1].

### 2. 주요 기여

**무감독 이미지 복원 프레임워크**: 무작위로 초기화된 네트워크를 단일 손상된 이미지에 최적화하는 것만으로도 디노이징, 초해상도, 인페인팅 등의 역문제를 해결할 수 있음을 입증했습니다[1].

**기존 패러다임의 재해석**: 학습 기반 방법과 수작업 사전 기반 방법 사이의 간극을 메우며, 네트워크 구조 자체가 수작업으로 제작된 사전과 같은 역할을 할 수 있음을 보였습니다[1].

## 해결하고자 하는 문제와 제안 방법

### 문제 정의

기존의 이미지 복원 방법들은 크게 두 가지 한계를 가지고 있었습니다:

1. **학습 기반 방법**: 대량의 쌍을 이룬 훈련 데이터가 필요하며, 특정 도메인에 국한됨
2. **전통적 사전 기반 방법**: Total Variation 등의 간단한 정규화 항만으로는 복잡한 자연 이미지의 특성을 충분히 모델링하기 어려움

### 제안 방법 - 수학적 공식화

기존의 정규화된 역문제 해결 방식:

$$ x^* = \arg\min_x E(x; x_0) + R(x) $$

여기서 $$E(x; x_0)$$는 데이터 항, $$R(x)$$는 정규화 항입니다.

**Deep Image Prior 접근법**:

$$ \theta^* = \arg\min_\theta E(f_\theta(z); x_0) $$

$$ x^* = f_{\theta^*}(z) $$

여기서:
- $$f_\theta$$: 무작위로 초기화된 컨볼루션 네트워크
- $$z$$: 고정된 무작위 입력 (일반적으로 노이즈)
- $$\theta$$: 네트워크 매개변수
- $$x_0$$: 관측된 손상 이미지

**다양한 응용의 데이터 항**:

- **디노이징**: $$E(x; x_0) = \|x - x_0\|^2$$
- **초해상도**: $$E(x; x_0) = \|d(x) - x_0\|^2$$ (여기서 $$d(\cdot)$$는 다운샘플링 연산자)
- **인페인팅**: $$E(x; x_0) = \|(x - x_0) \odot m\|^2$$ (여기서 $$m$$은 마스크)

## 모델 구조

### 주요 구조적 특징

**U-Net 계열 인코더-디코더 구조**: 스킵 연결을 가진 "hourglass" 형태의 구조를 주로 사용하며, 입력과 출력이 동일한 공간 해상도를 가집니다[1].

**핵심 구조 요소들**:
- **다운샘플링**: 컨볼루션의 스트라이드를 통해 구현
- **업샘플링**: 이중선형 또는 최근접 이웃 업샘플링 사용 (전치 컨볼루션은 성능이 떨어짐)
- **스킵 연결**: 다양한 스케일의 구조를 결합하여 자연 이미지 모델링에 유리
- **활성화 함수**: LeakyReLU 사용
- **패딩**: 반사 패딩 (reflection padding) 사용

### 구조 선택의 중요성

논문에서는 **네트워크 구조가 성능에 미치는 결정적 영향**을 강조합니다[1]:

- **U-Net**: 매우 빠른 수렴을 보이지만 너무 강력한 스킵 연결로 인해 노이즈 과적합이 발생할 수 있음
- **ResNet**: 스킵 연결 부족으로 느린 수렴과 과도하게 강한 사전을 보임
- **최적 구조**: 적절한 수준의 스킵 연결을 가진 인코더-디코더 구조

## 성능 향상 및 한계

### 성능 향상

**디노이징 성능**: 표준 벤치마크에서 PSNR 29.22-31.00을 달성하여, 사전 훈련이 없음에도 CBM3D (31.42)와 비교할 만한 성능을 보였습니다[1].

**초해상도**: 4배 및 8배 초해상도에서 학습 기반 방법에 근접한 성능을 달성했으며, 특히 비학습 방법들 중에서는 최고 성능을 보였습니다[1].

**다양한 응용**: 인페인팅, 신경망 역변환, 활성화 최대화 등 다양한 역문제에서 효과적임을 입증했습니다[1].

### 주요 한계

**계산 비용**: 각 이미지마다 수천 번의 반복 최적화가 필요하여 실시간 응용에는 부적절합니다[1].

**조기 중단의 필요성**: 최적화 과정에서 성능이 피크에 도달한 후 노이즈에 과적합되어 성능이 저하되므로, 적절한 조기 중단이 필수적입니다[1].

**특정 작업별 성능 한계**: 학습 기반 방법들보다는 전반적으로 성능이 떨어지며, 특히 SRGAN 등의 GAN 기반 방법들이 할루시네이션을 통해 생성하는 세밀한 디테일은 재현할 수 없습니다[1].

## 모델의 일반화 성능 향상 가능성

### 스펙트럼 편향 특성

Deep Image Prior의 핵심 메커니즘은 **스펙트럼 편향(Spectral Bias)**에 있습니다. 네트워크는 **저주파 성분을 고주파 성분보다 먼저 그리고 더 잘 학습**하는 특성을 보입니다[2][3][4].

**주파수 대역 학습 순서**:
1. 저주파 구조적 정보 (전체적인 형태, 윤곽)
2. 중간 주파수 세부사항
3. 고주파 노이즈 및 세밀한 디테일

### 네트워크 구조와 일반화 성능

**구조별 일반화 특성**[5][6]:
- **깊이**: 더 깊은 네트워크는 더 복잡한 이미지 사전을 학습할 수 있음
- **스킵 연결**: 적절한 수준의 스킵 연결이 다양한 스케일의 정보를 효과적으로 결합
- **업샘플링 방법**: 이중선형 업샘플링이 전치 컨볼루션보다 일반적으로 더 나은 성능

**일반화 성능 향상 전략들**:

1. **Neural Architecture Search (NAS)**: 자동으로 최적의 네트워크 구조를 탐색하여 특정 작업에 맞는 구조를 찾을 수 있습니다[5][6].

2. **메타학습 접근법**: MetaDIP와 같은 방법으로 적절한 초기화를 학습하여 수렴 속도를 10배 향상시킬 수 있습니다[7].

3. **자기지도 사전훈련**: 레이블이 없는 데이터로 사전훈련하여 DIP의 성능을 향상시킬 수 있습니다[8][9].

## 후속 연구에 미치는 영향과 고려사항

### 연구 분야에 미친 영향

**패러다임 전환**: 이미지 복원에서 "훈련 데이터 없는 딥러닝"이라는 새로운 패러다임을 제시했으며, 이는 무감독 학습 연구의 새로운 방향을 열었습니다[10][11][12].

**의료 영상 분야**: PET, CT, MRI 등 의료 영상에서 훈련 데이터 확보가 어려운 상황에서 특히 유용한 접근법으로 널리 채택되었습니다[10][8][13][14].

**다양한 응용 확장**: 하이퍼스펙트럼 언믹싱[15], 지진파 디노이징[16], 3D 재구성[17] 등 다양한 분야로 확장되었습니다.

### 향후 연구 시 고려사항

**구조 최적화**: Neural Architecture Search를 통한 자동 구조 탐색이 중요한 연구 방향이며, 이미지별 최적 구조가 다를 수 있음을 고려해야 합니다[6][18].

**조기 중단 전략**: 신뢰할 수 있는 자동 조기 중단 기준 개발이 실용성 확보에 필수적입니다[19].

**스펙트럼 편향 제어**: 주파수 기반 분석을 통한 편향 측정 및 제어 방법의 개발이 성능 향상의 핵심입니다[20][2].

**계산 효율성**: 실시간 응용을 위한 가속화 방법 (메타학습, 2단계 최적화 등)의 추가 연구가 필요합니다[13][7].

**일반화 성능**: 도메인 간 일반화 및 다양한 손상 유형에 대한 강건성 향상이 중요한 과제입니다[21][22].

Deep Image Prior는 딥러닝의 성공이 단순히 대량의 데이터에서 비롯되는 것이 아니라, **네트워크 구조 자체의 귀납적 편향**에서 오는 것임을 명확히 보여준 중요한 연구입니다. 이는 향후 더 효율적이고 범용적인 이미지 복원 방법 개발의 토대가 되고 있으며, 특히 훈련 데이터가 제한적인 상황에서의 딥러닝 응용에 새로운 가능성을 제시했습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/12fe99a8-8a13-460f-b2d7-e7acb0cad81e/1711.10925v4.pdf
[2] https://arxiv.org/abs/2107.01125
[3] https://www.bayesiandeeplearning.org/2019/papers/33.pdf
[4] https://arxiv.org/abs/1912.08905
[5] https://link.springer.com/10.1007/978-3-030-58523-5_26
[6] https://ieeexplore.ieee.org/document/9878799/
[7] https://arxiv.org/abs/2209.08452
[8] https://ieeexplore.ieee.org/document/10138706/
[9] https://arxiv.org/abs/2302.13546
[10] https://iopscience.iop.org/article/10.1088/1361-6560/abcd1a
[11] https://arxiv.org/abs/2110.09490
[12] https://openreview.net/forum?id=K1EG2ABzNE&noteId=KLbtyQ08BC
[13] https://ieeexplore.ieee.org/document/10338248/
[14] https://iopscience.iop.org/article/10.1088/1361-6420/aba415
[15] https://ieeexplore.ieee.org/document/10281765/
[16] https://library.seg.org/doi/10.1190/geo2024-0236.1
[17] https://ieeexplore.ieee.org/document/10103560/
[18] https://openreview.net/forum?id=_k0CnK5V7F
[19] https://openreview.net/forum?id=JIl_kij_aov
[20] https://github.com/shizenglin/Measure-and-Control-Spectral-Bias
[21] https://www.mdpi.com/1424-8220/22/15/5593
[22] https://ieeexplore.ieee.org/document/9706178/
[23] https://www.sec.gov/Archives/edgar/data/1833498/000121390023089245/fs12023a1_spectral.htm
[24] https://www.sec.gov/Archives/edgar/data/1833498/000121390023081580/fs12023_spectralalinc.htm
[25] https://www.sec.gov/Archives/edgar/data/1833498/000121390023096535/fs12023a3_spectral.htm
[26] https://www.sec.gov/Archives/edgar/data/1833498/000121390023094062/fs12023a2_spectral.htm
[27] https://www.sec.gov/Archives/edgar/data/1833498/000121390023035113/fs42023_rosecliffacq1.htm
[28] https://www.sec.gov/Archives/edgar/data/1806310/000095017023042089/tsha-20230630.htm
[29] https://www.sec.gov/Archives/edgar/data/1372514/000110465922129237/tm2233181d1_s1.htm
[30] https://www.sec.gov/Archives/edgar/data/1833498/000121390024005801/fs12024a1_spectral.htm
[31] https://www.mdpi.com/1424-8220/20/16/4505
[32] https://www.sciendo.com/article/10.2478/amcs-2018-0056
[33] https://link.springer.com/10.1007/s00500-022-07083-y
[34] https://ieeexplore.ieee.org/document/10743078/
[35] https://ieeexplore.ieee.org/document/8911386/
[36] https://www.tj-es.com/ojs/index.php/tjes/article/view/1165
[37] https://ieeexplore.ieee.org/document/9879465/
[38] https://ieeexplore.ieee.org/document/10483728/
[39] https://aistudy9314.tistory.com/47
[40] https://www.themoonlight.io/ko/review/analysis-of-deep-image-prior-and-exploiting-self-guidance-for-image-reconstruction
[41] https://www.themoonlight.io/ko/review/deep-priors-for-satellite-image-restoration-with-accurate-uncertainties
[42] https://en.wikipedia.org/wiki/Deep_image_prior
[43] https://2bdbest-ds.tistory.com/entry/GAN-Exploiting-Deep-Generative-Prior-for-Versatile-Image-Restoration-and-ManipulationECCV2020oral-paper
[44] https://www.themoonlight.io/ko/review/dipli-deep-image-prior-lucky-imaging-for-blind-astronomical-image-restoration
[45] https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf
[46] https://openbackyard.tistory.com/127
[47] https://s-space.snu.ac.kr/handle/10371/210176
[48] https://arxiv.org/abs/2402.04097
[49] https://doinghun.com/deep-image-prior-dip-2018-cvpr/
[50] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11742658
[51] https://arxiv.org/abs/2306.14209
[52] https://aistudy9314.tistory.com/48
[53] http://jksmrt.or.kr/journal/article.php?code=92153
[54] https://www.mdpi.com/2079-9292/10/16/2014
[55] https://www.sec.gov/Archives/edgar/data/1677897/000157587225000377/upyy-20250228.htm
[56] https://www.sec.gov/Archives/edgar/data/1164863/000116486325000009/npo-20241231.htm
[57] https://www.sec.gov/Archives/edgar/data/719413/000095017024015673/hl-20231231.htm
[58] https://www.sec.gov/Archives/edgar/data/1084765/000108476523000016/rgp-20230527x10k.htm
[59] https://www.sec.gov/Archives/edgar/data/353184/000035318423000071/airt-20230331.htm
[60] https://www.sec.gov/Archives/edgar/data/1839175/000121390023030226/f10k2022_m3brigade2.htm
[61] https://www.sec.gov/Archives/edgar/data/1588272/000119312523079332/d437941d10k.htm
[62] https://ieeexplore.ieee.org/document/10629196/
[63] https://ieeexplore.ieee.org/document/10338041/
[64] https://ieeexplore.ieee.org/document/10323916/
[65] https://arxiv.org/abs/2111.11926
[66] https://www.youtube.com/watch?v=FPzi8cUhNNY
[67] http://iccvm.org/2022/papers/poster-4.pdf
[68] https://arxiv.org/abs/1711.10925
[69] https://pmc.ncbi.nlm.nih.gov/articles/PMC10210546/
[70] https://www.macnica.co.jp/en/business/ai/blog/142025/
[71] https://arxiv.org/abs/2008.11713
[72] https://github.com/DmitryUlyanov/deep-image-prior
[73] https://yunchunchen.github.io/NAS-DIP/paper.pdf
[74] https://nate9389.tistory.com/2164
[75] https://www.sciencedirect.com/science/article/abs/pii/S0031320325004467
[76] https://velog.io/@bluein/paper-34
[77] https://www.sec.gov/Archives/edgar/data/1853138/000119312525119920/d948047ds4.htm
[78] https://www.sec.gov/Archives/edgar/data/1747286/000119312525119920/d948047ds4.htm
[79] https://www.sec.gov/Archives/edgar/data/1477960/000147793225002922/cbbb_10k.htm
[80] https://www.sec.gov/Archives/edgar/data/1280263/000095017025046499/amba-20250131.htm
[81] https://www.sec.gov/Archives/edgar/data/1477960/000147793225000304/cbbb_s1a.htm
[82] https://www.sec.gov/Archives/edgar/data/1477960/000147793225000119/cbbb_s1.htm
[83] https://www.sec.gov/Archives/edgar/data/1973239/000197323925000016/arm-20250331.htm
[84] https://sic.ici.ro/vol-31-no-2-2022/enhancing-the-generalization-performance-of-few-shot-image-classification-with-self-knowledge-distillation/
[85] https://ieeexplore.ieee.org/document/10603548/
[86] https://pmc.ncbi.nlm.nih.gov/articles/PMC12082789/
[87] https://arxiv.org/html/2402.04097v2
[88] https://www.sciencedirect.com/science/article/abs/pii/S0097849321001126
[89] https://par.nsf.gov/biblio/10162414-spectral-bias-deep-image-prior
[90] https://dl.acm.org/doi/10.1007/s11263-021-01572-7
[91] https://www.sciencedirect.com/science/article/abs/pii/S105120042500257X
[92] https://scispace.com/papers/the-spectral-bias-of-the-deep-image-prior-zas3ualcf6
[93] https://www.sec.gov/Archives/edgar/data/1958489/000175392624001542/g084415_s1a.htm
[94] https://www.sec.gov/Archives/edgar/data/1958489/000175392624002059/g084589_s1a.htm
[95] https://www.sec.gov/Archives/edgar/data/1958489/000175392624001978/g084521_s1a.htm
[96] https://www.sec.gov/Archives/edgar/data/1958489/000175392624001447/g084359_s1a.htm
[97] https://www.sec.gov/Archives/edgar/data/1958489/000175392624001089/g084274_s1a.htm
[98] https://www.sec.gov/Archives/edgar/data/749660/000119312522086983/d289381d10k.htm
[99] https://www.sec.gov/Archives/edgar/data/1833498/000121390024027863/ea0202419-10k_spectral.htm
[100] https://www.sec.gov/Archives/edgar/data/1958489/000175392624000432/g083998_s1a.htm
[101] https://www.sec.gov/Archives/edgar/data/1958489/000175392623000582/g083520_s1a.htm
[102] https://www.sec.gov/Archives/edgar/data/355019/000035501921000014/fonar-def14a_fiscal2020.htm
[103] https://www.sec.gov/Archives/edgar/data/1958489/000175392624000605/g084143_s1a.htm
[104] https://www.sec.gov/Archives/edgar/data/1696396/000156459021017843/mito-20f_20201231.htm
[105] https://ieeexplore.ieee.org/document/10205101/
[106] https://ieeexplore.ieee.org/document/9856970/
[107] https://arxiv.org/pdf/2310.00894.pdf
[108] http://arxiv.org/pdf/1908.02995.pdf
[109] http://arxiv.org/pdf/1705.08041.pdf
[110] https://arxiv.org/abs/2112.06074
[111] https://arxiv.org/pdf/2206.02070.pdf
[112] https://arxiv.org/pdf/2201.08625.pdf
[113] https://www.mdpi.com/2227-7390/11/14/3201/pdf?version=1689928340
[114] https://arxiv.org/pdf/2402.04097.pdf
[115] https://arxiv.org/html/2410.15971v1
[116] https://pmc.ncbi.nlm.nih.gov/articles/PMC11078028/
[117] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11132476
[118] https://www.themoonlight.io/ko/review/bagged-deep-image-prior-for-recovering-images-in-the-presence-of-speckle-noise
[119] https://link.springer.com/10.1007/s00500-025-10642-8
[120] https://www.mdpi.com/2075-4418/13/9/1651
[121] http://arxiv.org/pdf/2209.08452.pdf
[122] https://arxiv.org/pdf/2001.04776.pdf
[123] http://arxiv.org/pdf/2008.11713.pdf
[124] https://pmc.ncbi.nlm.nih.gov/articles/PMC9326425/
[125] https://arxiv.org/pdf/2211.14298.pdf
[126] http://arxiv.org/pdf/2304.11409.pdf
[127] https://arxiv.org/abs/2111.15362
[128] http://arxiv.org/pdf/2310.19477.pdf
[129] http://arxiv.org/pdf/1711.10925.pdf
[130] https://www.sciencedirect.com/science/article/abs/pii/S0893608021002379
[131] https://www.semanticscholar.org/paper/58ea57580b9cde6958e3e88e49ce70070ddb20ee
[132] https://ieeexplore.ieee.org/document/10423584/
[133] https://arxiv.org/pdf/2302.13546.pdf
[134] https://arxiv.org/abs/1904.07457v1
[135] https://arxiv.org/html/2404.12142v1
