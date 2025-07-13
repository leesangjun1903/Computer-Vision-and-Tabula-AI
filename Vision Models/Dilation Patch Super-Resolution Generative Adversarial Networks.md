# DPSRGAN: Dilation Patch Super-Resolution Generative Adversarial Networks | Super resolution

## 핵심 주장과 주요 기여

**DPSRGAN (Dilation Patch Super-Resolution Generative Adversarial Networks)**은 Single Image Super-Resolution (SISR) 문제를 해결하기 위한 새로운 GAN 기반 접근법으로, 기존 방법들의 한계를 극복하고 경량화된 모델을 통한 고품질 이미지 복원을 목표로 합니다[1].

### 주요 기여점

1. **Dilated Convolution을 통한 생성자 개선**: 기존 SRResNet의 교대 잔차 블록에 dilated convolution을 추가하여 더 큰 수용장(receptive field)을 확보하고 전역 구조 정보를 효과적으로 활용합니다[1][2][3].

2. **조건부 GAN 훈련**: 저해상도 입력 이미지를 고해상도/복원 이미지와 연결하여 판별자에 입력함으로써 입력-출력 관계를 고려한 조건부 훈련을 수행합니다[1][4].

3. **Markovian 판별자 (PatchGAN)**: 전체 이미지가 아닌 N×N 패치 단위로 진위 여부를 판별하는 PatchGAN을 사용하여 더 선명한 세부사항을 생성하고 크기에 무관한 처리를 가능하게 합니다[1][5][4].

## 해결하고자 하는 문제

### SISR의 근본적 도전과제

SISR은 본질적으로 **ill-posed 문제**로, 하나의 저해상도 이미지가 여러 고해상도 이미지로 매핑될 수 있는 불확실성을 가지고 있습니다[1]. 특히 고해상도 타겟에서는 모든 텍스처 디테일이 모델에 의해 생성되어야 하는 어려움이 있습니다.

### 기존 방법들의 한계

- **흐릿한 출력 생성**: 대부분의 기존 방법들은 세밀한 디테일이 부족한 흐릿한 결과를 생성합니다[1][6].
- **과도한 모델 복잡성**: 더 나은 결과를 위해 매우 무거운 모델을 사용하여 실용성이 떨어집니다[1].
- **제한적인 수용장**: 기존 네트워크는 지역적 특징에만 집중하여 전역적 구조 정보를 충분히 활용하지 못합니다[1].

## 제안하는 방법

### 모델 구조

**생성자 (Generator)**: 
- 기본 구조: 16개의 잔차 블록을 가진 SRResNet 기반
- 개선사항: 교대 잔차 블록에 3×3 커널, 확장률 1의 dilated convolution 추가[1]

**판별자 (Discriminator)**:
- Markovian 판별자 (PatchGAN): N×N 행렬로 출력하여 각 패치의 진위 여부 판별
- 완전 합성곱 구조로 크기 독립적 처리 가능[1]

### 손실 함수

**전체 GAN 손실**:

$$ l_{GAN} = l_p + 10^{-3} \cdot l_{adv} $$

**지각적 손실 (Perceptual Loss)**:

$$ l_p = \frac{1}{W_{i,j}H_{i,j}} \sum_{x=1}^{W_{i,j}} \sum_{y=1}^{H_{i,j}} (\Phi_{i,j}(I_{HR})\_{x,y} - \Phi_{i,j}(I_{SR})_{x,y})^2 $$

**적대적 손실 (Adversarial Loss)**:

$$ l_{adv} = -\log D(I_{HR} \oplus G(I_{LR})) $$

**판별자 손실**:

$$ l_{BCE} = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} -[y_{i,j} \log(p_{i,j}) + (1-y_{i,j}) \log(1-p_{i,j})] $$

여기서 ⊕는 이미지 연결, Φ는 VGG19 네트워크 특징맵, N은 패치 차원을 나타냅니다[1].

### 훈련 과정

1. **MSE 사전 훈련**: 생성자를 MSE 손실로 먼저 훈련하여 GAN 훈련의 초기화 제공
2. **교대 훈련**: 생성자와 판별자를 번갈아가며 훈련
3. **레이블 스무딩**: 가짜 레이블 0.0-0.3, 진짜 레이블 0.7-1.0 사용[1]

## 성능 향상 및 결과

### 정량적 성능
- **PSNR**: 32.24 (DPSRGAN) vs 30.98 (SRGAN) vs 32.37 (SRResNet)
- **SSIM**: 0.86 (DPSRGAN) vs 0.82 (SRGAN) vs 0.88 (SRResNet)  
- **MOS**: 3.91 (DPSRGAN) vs 3.62 (SRGAN) vs 3.02 (SRResNet)
- **모델 크기**: 약 5M 매개변수 vs SRGAN의 약 17M 매개변수[1]

### 질적 개선
- **텍스처 일관성**: Markovian 판별자를 통한 더 일관된 텍스처 생성
- **전역 구조 보존**: Dilated convolution으로 인한 향상된 전역 특징 유지
- **크기 무관성**: 동일한 모델로 다양한 크기의 이미지 처리 가능[1]

## 모델의 일반화 성능 향상 가능성

### 기술적 장점

1. **크기 무관성**: PatchGAN의 완전 합성곱 구조로 인해 모델이 다양한 크기의 이미지에 적응할 수 있습니다[1][5].

2. **경량화**: 기존 SRGAN 대비 약 70% 감소한 매개변수로 더 빠른 훈련과 추론이 가능합니다[1].

3. **다중 스케일 처리**: Dilated convolution을 통한 더 큰 수용장으로 다양한 스케일의 특징을 효과적으로 처리할 수 있습니다[2][3].

### 한계점

1. **GAN 고유의 불안정성**: 훈련 과정에서의 모드 붕괴(mode collapse)와 불안정성 문제가 여전히 존재합니다[1][7].

2. **특정 데이터셋 편향**: CelebA 데이터셋으로만 훈련되어 다른 도메인의 이미지에 대한 일반화 성능이 제한적일 수 있습니다[1].

3. **실제 환경 적용**: 실제 환경의 복잡한 저하 패턴에 대한 적응력이 충분히 검증되지 않았습니다[8][9].

## 한계 및 향후 연구 방향

### 현재 한계

1. **GAN 안정성**: 저자들은 GAN 훈련의 본질적인 불안정성을 인정하며, Wasserstein GAN이나 Spectral Normalization 등의 안정화 기법 도입을 제안합니다[1].

2. **사전 훈련 의존성**: 생성자의 사전 훈련이 필요하여 전체 훈련 과정이 복잡합니다[1].

3. **실제 데이터 적응**: 합성 데이터로 훈련된 모델이 실제 환경의 다양한 저하 패턴에 적응하는 능력이 제한적입니다[10][11].

### 향후 연구 방향

1. **안정화 기법**: Wasserstein GAN, Spectral Normalization 등을 통한 훈련 안정성 개선[1].

2. **실제 데이터 적응**: 자가 지도 학습이나 도메인 적응 기법을 통한 실제 환경 적응력 향상[10][11].

3. **다중 도메인 일반화**: 다양한 이미지 도메인에서의 일반화 성능 향상을 위한 연구[8][9].

## 향후 연구에 미치는 영향

### 긍정적 영향

1. **경량화 추세**: DPSRGAN의 성공적인 경량화는 실용적인 SR 모델 개발에 중요한 방향성을 제시합니다[1].

2. **PatchGAN 활용**: 이미지 복원 분야에서 PatchGAN의 효과적인 활용 방안을 제시하여 관련 연구에 영향을 미칩니다[5][4].

3. **조건부 훈련**: 입력-출력 관계를 고려한 조건부 GAN 훈련의 중요성을 입증합니다[1].

### 고려할 점

1. **일반화 평가**: 다양한 데이터셋과 실제 환경에서의 더 광범위한 평가가 필요합니다[12][13].

2. **비교 연구**: 최신 Transformer 기반 방법들과의 체계적인 비교 연구가 요구됩니다[14][15].

3. **안정성 개선**: GAN 훈련의 안정성을 높이는 추가적인 연구가 필요합니다[16][7].

이 논문은 SISR 분야에서 경량화와 성능 개선을 동시에 달성한 의미 있는 연구로, 향후 실용적인 초해상도 모델 개발에 중요한 기여를 하였습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/981b729e-8f67-4a83-a8e6-7216230fed8a/DPSRGAN_Dilation_Patch_Super-Resolution_Generative_Adversarial_Networks.pdf
[2] https://arxiv.org/abs/2505.21262
[3] https://ieeexplore.ieee.org/document/8502129/
[4] https://cl2020.tistory.com/15
[5] https://brstar96.github.io/devlog/mldlstudy/2019-05-13-what-is-patchgan-D/
[6] http://ieeexplore.ieee.org/document/8099502/
[7] https://arxiv.org/abs/2110.10915
[8] https://www.mdpi.com/2079-9292/12/13/2975
[9] https://ieeexplore.ieee.org/document/10328742/
[10] https://arxiv.org/html/2403.02601v1
[11] https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Low-Res_Leads_the_Way_Improving_Generalization_for_Super-Resolution_by_Self-Supervised_CVPR_2024_paper.pdf
[12] https://github.com/lyh-18/SRGA
[13] https://arxiv.org/abs/2205.07019
[14] https://www.mdpi.com/2079-9292/13/1/194
[15] https://openreview.net/forum?id=owziuM1nsR
[16] https://arxiv.org/abs/2311.18508
[17] https://www.sec.gov/Archives/edgar/data/1621672/000143774925017257/slgg20250331_10q.htm
[18] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000014/smci-20250331.htm
[19] https://www.sec.gov/Archives/edgar/data/1621672/000143774925010233/slgg20241231_10k.htm
[20] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000004/smci-20240630.htm
[21] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000020/smci-20250623.htm
[22] https://www.sec.gov/Archives/edgar/data/1621672/000143774925017611/slgg20250514_def14a.htm
[23] https://www.sec.gov/Archives/edgar/data/1878057/000162828025016393/sghc-20241231.htm
[24] https://linkinghub.elsevier.com/retrieve/pii/S0923596521000229
[25] https://www.mdpi.com/2072-4292/15/22/5272
[26] https://arxiv.org/abs/2505.00374
[27] https://ieeexplore.ieee.org/document/8967055/
[28] https://ieeexplore.ieee.org/document/10252045/
[29] http://link.springer.com/10.1007/s10489-020-01670-y
[30] https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1436052/full
[31] https://arxiv.org/abs/2204.13620
[32] https://pmc.ncbi.nlm.nih.gov/articles/PMC11363189/
[33] https://www.techscience.com/cmc/v64n3/39471
[34] https://arxiv.org/abs/1707.07128
[35] https://arxiv.org/html/2505.16310v1
[36] https://velog.io/@wilko97/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Image-to-Image-Translation-with-Conditional-Adversarial-Networks-2017-CVPR
[37] https://thesai.org/Publications/ViewPaper?Volume=14&Issue=2&Code=IJACSA&SerialNo=100
[38] https://www.mdpi.com/2079-9292/13/12/2281
[39] https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf
[40] https://www.sciencedirect.com/science/article/pii/S0925231217315813
[41] https://aijyh0725.tistory.com/17
[42] https://dcollection.sogang.ac.kr/dcollection/srch/srchDetail/000000063135
[43] https://daebaq27.tistory.com/99
[44] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000006/smci-20241231.htm
[45] https://www.sec.gov/Archives/edgar/data/1621672/000143774925019131/slgg20250601_8k.htm
[46] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000018/smci-20250623.htm
[47] https://ojs.aaai.org/index.php/AAAI/article/view/28201
[48] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.13192
[49] https://ieeexplore.ieee.org/document/10449395/
[50] https://www.nature.com/articles/s41598-024-52370-3
[51] https://ieeexplore.ieee.org/document/10149876/
[52] https://dl.acm.org/doi/10.1145/3456726
[53] https://arxiv.org/abs/1609.04802
[54] https://gdsc-university-of-seoul.github.io/cv_fall_srgan/
[55] https://arxiv.org/html/2504.13622v1
[56] https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-NetworkSRGAN
[57] https://velog.io/@hyebbly/Deep-Learning-Loss-%EC%A0%95%EB%A6%AC-2-VGG-loss
[58] https://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf
[59] https://blog.outta.ai/168
[60] https://hi-guten-tag.tistory.com/203
[61] https://arxiv.org/html/2402.19387v1
[62] https://github.com/tensorlayer/SRGAN
[63] https://hwanny-yy.tistory.com/18
[64] https://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf
[65] https://kubig15-suhyeokjang.tistory.com/5
[66] https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/
[67] https://kevinitcoding.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-SRGAN-%EB%85%BC%EB%AC%B8-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC-Photo-Realistic-Single-Image-Super-Resolution-Using-a-Generative-Adversarial-Network
[68] https://blog.chungjungsoo.dev/dev-posts/srgan-animation-upscaling/
[69] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000005/smci-20240930.htm
[70] https://ieeexplore.ieee.org/document/10675360/
[71] https://ieeexplore.ieee.org/document/10509697/
[72] https://ieeexplore.ieee.org/document/10737353/
[73] https://www.mdpi.com/1424-8220/24/11/3560
[74] https://ieeexplore.ieee.org/document/10688339/
[75] https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Wang_NTIRE_2023_Challenge_on_Stereo_Image_Super-Resolution_Methods_and_Results_CVPRW_2023_paper.pdf
[76] https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wang_NTIRE_2022_Challenge_on_Stereo_Image_Super-Resolution_Methods_and_Results_CVPRW_2022_paper.pdf
[77] https://simons.berkeley.edu/news/research-vignette-promise-limitations-generative-adversarial-nets-gans
[78] https://eitca.org/artificial-intelligence/eitc-ai-adl-advanced-deep-learning/advanced-generative-models/modern-latent-variable-models/examination-review-modern-latent-variable-models/what-are-the-primary-advantages-and-limitations-of-using-generative-adversarial-networks-gans-compared-to-other-generative-models/
[79] https://www.nature.com/articles/s41598-025-92377-y
[80] https://www.lunit.io/company/blog/review-enhanced-deep-residual-networks-for-single-image-super-resolution-winner-of-ntire-2017-sisr-challenge
[81] https://www.lyzr.ai/glossaries/generative-adversarial-network/
[82] https://arxiv.org/html/2409.06590v2
[83] https://www.sciencedirect.com/science/article/pii/S0957417422007084
[84] https://www.sciencedirect.com/science/article/abs/pii/S0030402622009032
[85] https://ieeexplore.ieee.org/document/9126804/
[86] https://ieeexplore.ieee.org/document/10539326/
[87] https://www.frontiersin.org/articles/10.3389/fnbot.2024.1436052/full
[88] https://arxiv.org/html/2310.04705v1
[89] https://www.mdpi.com/1099-4300/23/6/767/pdf
[90] https://pmc.ncbi.nlm.nih.gov/articles/PMC8233773/
[91] http://arxiv.org/pdf/1607.07680.pdf
[92] https://arxiv.org/pdf/1904.07523.pdf
[93] http://arxiv.org/pdf/1511.07122.pdf
[94] https://arxiv.org/pdf/2111.09957.pdf
[95] https://arxiv.org/abs/2305.03387v1
[96] https://arxiv.org/abs/2505.10589
[97] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART94976781
[98] https://jeongwooyeol0106.tistory.com/32
[99] https://arxiv.org/abs/2211.16678
[100] https://www.ije.ir/article_150976.html
[101] https://arxiv.org/pdf/2107.12679.pdf
[102] https://arxiv.org/pdf/1712.05927.pdf
[103] https://downloads.hindawi.com/journals/mpe/2020/5217429.pdf
[104] https://arxiv.org/pdf/2101.10165.pdf
[105] https://arxiv.org/pdf/1908.06382.pdf
[106] https://arxiv.org/pdf/2204.13620.pdf
[107] https://www.mdpi.com/2076-3417/10/1/375/pdf?version=1578480996
[108] https://pmc.ncbi.nlm.nih.gov/articles/PMC10221380/
[109] https://www.mdpi.com/2072-4292/11/21/2578/pdf
[110] http://arxiv.org/pdf/2010.04634.pdf
[111] https://www.sciencedirect.com/topics/computer-science/the-super-resolution-generative-adversarial-network
[112] https://wikidocs.net/146367
[113] https://www.mdpi.com/2313-433X/11/5/163
[114] https://www.hindawi.com/journals/ijis/2024/3255233/
[115] https://journal.esrgroups.org/jes/article/view/8531
[116] https://arxiv.org/abs/2211.12845
[117] https://downloads.hindawi.com/journals/wcmc/2021/5579090.pdf
[118] https://arxiv.org/pdf/2401.05633.pdf
[119] https://arxiv.org/abs/1808.03344v3
[120] http://arxiv.org/pdf/1606.01299v3.pdf
[121] https://arxiv.org/pdf/2104.14951.pdf
[122] https://www.mdpi.com/2072-4292/14/12/2895/pdf?version=1655453787
[123] https://downloads.hindawi.com/journals/sp/2020/8877851.pdf
[124] https://www.mdpi.com/1424-8220/23/19/8213/pdf?version=1696141992
[125] https://paperswithcode.com/task/image-super-resolution
[126] https://www.themoonlight.io/ko/review/low-res-leads-the-way-improving-generalization-for-super-resolution-by-self-supervised-learning
[127] https://www.sciencedirect.com/science/article/pii/S0939388922001003
