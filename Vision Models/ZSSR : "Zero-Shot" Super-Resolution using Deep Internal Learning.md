# ZSSR : "Zero-Shot" Super-Resolution using Deep Internal Learning | Image generation, zero shot learning

### 핵심 주장과 주요 기여

**ZSSR(Zero-Shot Super-Resolution)**은 외부 훈련 데이터나 사전 훈련 없이 딥러닝의 힘을 활용하는 혁신적인 초해상도 기법입니다[1][2]. 논문의 핵심 주장은 기존 지도학습 기반 초해상도 방법들이 특정한 열화 조건(예: bicubic downscaling)에 제한되어 실제 이미지에서 성능이 저하되는 문제를 해결한다는 것입니다[1][2].

**주요 기여**는 다음과 같습니다:
- 최초의 비지도 CNN 기반 초해상도 방법 제안[1][2]
- 테스트 시점에서 입력 이미지 자체에서 추출한 내부 정보만을 활용한 image-specific CNN 훈련[1][2]
- 이상적이지 않은 촬영 조건에서도 적응 가능한 유연성 확보[1][2]
- 사전 훈련 불필요 및 적은 계산 자원으로 운영 가능[1][2]

### 해결하고자 하는 문제

기존 지도학습 기반 초해상도 방법들은 ** 제한적이라는 근본적인 문제**를 가지고 있습니다[1]. 구체적으로:

1. **사전 정의된 열화 모델 의존성**: 주로 bicubic downscaling으로 생성된 LR 이미지에만 최적화됨[1][2]
2. **실제 환경에서의 성능 저하**: 센서 노이즈, 이미지 압축, 비이상적 PSF 등이 포함된 실제 이미지에서 poor 성능[1][2]
3. **일반화 능력 부족**: 훈련 조건과 다른 환경에서 적응 불가[1][2]

### 제안하는 방법과 모델 구조

**ZSSR의 핵심 방법론**은 단일 이미지 내의 **internal recurrence of information**을 활용하는 것입니다[1][2]. 

**수학적 정의**:
- 입력 이미지 I를 downscaling하여 I↓s 생성 (s는 원하는 SR scale factor)
- 네트워크는 I↓s로부터 I를 복원하도록 훈련
- 훈련된 네트워크를 I에 적용하여 I↑s (고해상도 출력) 생성[1]

**모델 구조**:
- **간단한 완전 합성곱 네트워크**: 8개 hidden layers, 각 64 channels[1]
- **ReLU 활성화 함수** 사용[1]
- **Residual learning**: 보간된 LR과 HR parent 간의 잔차만 학습[1]
- **L1 loss + ADAM optimizer** 사용[1]

**데이터 증강 전략**:
- 이미지를 여러 해상도로 downscaling하여 "HR fathers" 생성[1]
- 각 HR father를 다시 downscaling하여 "LR sons" 생성[1]
- 4가지 회전 (0°, 90°, 180°, 270°)과 수직/수평 미러링으로 8배 증강[1]

### 성능 향상 결과

**이상적 조건에서의 성능**:
- Set5 데이터셋에서 PSNR/SSIM: ×2 (37.37/0.9570), ×3 (33.42/0.9188), ×4 (31.13/0.8796)[1]
- 기존 SRCNN보다 우수하고 VDSR과 유사한 성능 달성[1]
- 비지도 방법 중에서는 SelfExSR 대비 큰 성능 향상[1]

**비이상적 조건에서의 뛰어난 성능**:
- 비이상적 downscaling kernel 환경에서 EDSR+ 대비 **+1dB PSNR 향상**[1]
- 실제 kernel 정보 제공 시 **+2dB 향상**[1]
- 노이즈/압축 아티팩트가 있는 이미지에서 지도학습 방법들보다 현저히 우수[1]

### 모델의 한계점

**주요 한계점들**:

1. **긴 추론 시간**: 테스트 시점에서 훈련이 필요하여 단일 scale factor 증가에 평균 54초 소요[1][3][4]
2. **반복적 패턴 의존성**: 이미지 내부에 반복되는 구조가 부족한 경우 성능 저하[5][6]
3. **노이즈 증폭**: 내부 정보 활용 과정에서 노이즈가 함께 확대되는 문제[7]
4. **단일 이미지 제약**: 한 번에 하나의 이미지에만 특화되어 다른 이미지에 재사용 불가[4]

### 일반화 성능 향상 가능성

**내부 vs 외부 학습의 상호보완성**:
ZSSR은 이미지별 적응형 학습을 통해 기존 방법들과는 다른 **일반화 접근법**을 제시합니다[1]. 특히:

- **Image-specific adaptation**: 각 이미지의 고유한 특성에 맞춰 네트워크가 적응[1]
- **Unknown degradation handling**: 다양한 열화 조건에 대한 robust한 대응 가능[1]
- **Internal-External learning combination**: 내부 학습과 외부 학습의 결합으로 더 나은 성능 가능성 시사[1]

연구에서는 일부 픽셀은 내부 학습(ZSSR)을, 다른 픽셀은 외부 학습(EDSR+)을 선호한다는 흥미로운 발견을 제시하며, 이는 **내부-외부 학습을 결합한 하이브리드 접근법**의 가능성을 제안합니다[1].

### 미래 연구에 미치는 영향과 고려사항

**긍정적 영향**:

1. **Zero-shot learning paradigm 확산**: ZSSR 이후 다양한 zero-shot 기반 방법들이 제안됨[8][9][10][11][12]
2. **Internal statistics 활용**: 이미지 내부 통계 정보의 중요성 재조명[1][13]
3. **Real-world application 확대**: 의료 영상, 생물학적 데이터, 역사적 이미지 등 다양한 분야 적용[1][14][15]

**미래 연구시 고려사항**:

1. **계산 효율성 개선**: 테스트 시점 훈련 시간 단축을 위한 meta-learning 접근법 (MZSR 등) 필요[16][4]
2. **하이브리드 모델 개발**: 내부-외부 학습을 효과적으로 결합하는 방법론 연구[1]
3. **강건성 향상**: 반복 패턴이 부족한 이미지에서도 효과적인 방법 개발[5][6]
4. **Multi-modal integration**: 다양한 센서 데이터와의 융합 가능성 탐색[8][9]

**결론적으로**, ZSSR은 초해상도 분야에서 **패러다임 전환**을 이끌어낸 중요한 연구로, 실세계 적용 가능성을 크게 향상시켰으며, 향후 zero-shot learning과 internal statistics 기반 방법론 발전의 기반을 마련했습니다[1][2].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/72717d06-9108-417c-ae47-8032891b4f88/1712.06087v1.pdf
[2] https://ieeexplore.ieee.org/document/8578427/
[3] https://dlgari33.tistory.com/2
[4] https://airsbigdata.tistory.com/209
[5] https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Low-Res_Leads_the_Way_Improving_Generalization_for_Super-Resolution_by_Self-Supervised_CVPR_2024_paper.pdf
[6] https://arxiv.org/html/2403.02601v1
[7] https://jkimst.org/upload/pdf/KIMST-2023-26-3-234.pdf
[8] https://ieeexplore.ieee.org/document/10023452/
[9] https://ieeexplore.ieee.org/document/9992006/
[10] https://arxiv.org/abs/2401.06144
[11] https://www.mdpi.com/2078-2489/14/1/33
[12] https://ieeexplore.ieee.org/document/10983879/
[13] https://www.nature.com/articles/s41598-023-28462-x
[14] https://www.mdpi.com/1424-8220/24/21/7083
[15] https://www.nature.com/articles/s41467-024-48575-9
[16] https://openaccess.thecvf.com/content_CVPR_2020/papers/Soh_Meta-Transfer_Learning_for_Zero-Shot_Super-Resolution_CVPR_2020_paper.pdf
[17] https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf
[18] https://faculty.cc.gatech.edu/~hays/papers/SR_ICCP.pdf
[19] https://arxiv.org/abs/1501.00092
[20] https://arxiv.org/abs/1712.06087
[21] https://arxiv.org/abs/2011.11020
[22] https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
[23] http://www.wisdom.weizmann.ac.il/~vision/zssr/
[24] https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Yoon_Simple_and_Efficient_Unpaired_Real-World_Super-Resolution_Using_Image_Statistics_ICCVW_2021_paper.pdf
[25] https://www.mdpi.com/2076-3417/11/3/1092
[26] https://deepmal.tistory.com/29
[27] https://onlinelibrary.wiley.com/doi/10.1155/2023/8860842
[28] https://bovit.tistory.com/20
[29] https://github.com/assafshocher/ZSSR
[30] https://arxiv.org/html/2405.18872v1
[31] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART106035588
[32] https://www.semanticscholar.org/paper/b5654ec2442d5a2034aac8e58623877e882c67ba
[33] https://www.semanticscholar.org/paper/eb0486bfbfd8c055ab0a461726c20b58944ba829
[34] https://www.semanticscholar.org/paper/c917661ae4fb1fe407664ac6a6f615015acb8924
[35] https://www.semanticscholar.org/paper/bfb8759ed84f27dbadf0eee7a307618bea0cf491
[36] https://www.semanticscholar.org/paper/2e3e47ddf6403a49b7b324ca82475a8066fedac7
[37] https://www.semanticscholar.org/paper/e690cd4bfdaea09cfeb0954fc0958867ccd45717
[38] https://www.semanticscholar.org/paper/982c856788b4e21db6906a497971db36c33da191
[39] https://www.semanticscholar.org/paper/d3237bdf2a896dc9a87414cbe86e4d6a64824061
[40] https://ai-woods.tistory.com/49
[41] https://xoft.tistory.com/3
[42] https://white-joy.tistory.com/9
[43] https://dbstndi6316.tistory.com/376
[44] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ddnm/
[45] https://dlgari33.tistory.com/3
[46] https://hi-guten-tag.tistory.com/203
[47] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11554465
[48] https://deepmal.tistory.com/31
[49] https://mole-starseeker.tistory.com/82
[50] https://www.koreascience.or.kr/article/CFKO202130759709602.pdf
[51] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002709332
[52] https://deepmal.tistory.com/28
[53] https://www.semanticscholar.org/paper/9268cec27bbbfdcf497595319b6a61eea027cabf
[54] https://arxiv.org/abs/2206.10012
[55] https://arxiv.org/abs/2212.13556
[56] https://arxiv.org/abs/2504.01928
[57] https://www.semanticscholar.org/paper/692aabc07a61bf8fcb5195fda7733abc9bc516c9
[58] https://arxiv.org/abs/2502.01458
[59] https://ieeexplore.ieee.org/document/9965850/
[60] https://arxiv.org/abs/2205.06898
[61] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf
[62] https://pmc.ncbi.nlm.nih.gov/articles/PMC5487374/
[63] https://dl.acm.org/doi/10.1145/3617733.3617743
[64] https://arxiv.org/html/2402.18929v1
[65] https://www.mdpi.com/2227-7390/11/7/1653
[66] https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Low-Res_Leads_the_CVPR_2024_supplemental.pdf
[67] https://arxiv.org/abs/2205.07019
[68] https://neurips.cc/virtual/2023/poster/72113
[69] https://www.sciencedirect.com/science/article/abs/pii/S0045793024002275
[70] https://www.sciencedirect.com/science/article/abs/pii/S0923596522000595
[71] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13646/136460N/Image-super-resolution-by-exploring-spatial-and-frequency-information/10.1117/12.3056210.full
[72] https://www.sciencedirect.com/science/article/pii/S2666521220300053
[73] https://arxiv.org/html/2109.14335v2
[74] https://ieeexplore.ieee.org/document/10540131/
[75] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12926/3007304/Pushing-the-limits-of-zero-shot-self-supervised-super-resolution/10.1117/12.3007304.full
[76] https://arxiv.org/html/2312.12122v1
[77] https://arxiv.org/abs/2208.11313
[78] https://arxiv.org/abs/2002.12213
[79] http://arxiv.org/pdf/1712.06087.pdf
[80] https://arxiv.org/pdf/2006.01339.pdf
[81] http://arxiv.org/pdf/2404.09640.pdf
[82] http://arxiv.org/pdf/2405.02171.pdf
[83] https://arxiv.org/pdf/2312.14551.pdf
[84] https://pmc.ncbi.nlm.nih.gov/articles/PMC7610492/
[85] http://arxiv.org/pdf/2307.03416.pdf
[86] https://pure.kaist.ac.kr/en/publications/single-image-super-resolution-using-lightweight-cnn-with-maxout-u
[87] https://paperswithcode.com/task/image-super-resolution
[88] https://www.semanticscholar.org/paper/f96c90ec057ca5ea8ff13215ed9a71993846fdb7
[89] https://www.semanticscholar.org/paper/4a531ec13254da683b5d0bb21e97b5165a6ef9ea
[90] https://dl.acm.org/doi/pdf/10.1145/3617232.3624852
[91] http://arxiv.org/pdf/2310.05208.pdf
[92] http://arxiv.org/pdf/2304.13029.pdf
[93] https://drops.dagstuhl.de/storage/00lipics/lipics-vol275-approx-random2023/LIPIcs.APPROX-RANDOM.2023.56/LIPIcs.APPROX-RANDOM.2023.56.pdf
[94] https://www.spiedigitallibrary.org/journals/journal-of-applied-remote-sensing/volume-17/issue-3/032405/Self-supervised-embedding-for-generalized-zero-shot-learning-in-remote/10.1117/1.JRS.17.032405.pdf
[95] https://arxiv.org/abs/1504.07959
[96] https://arxiv.org/pdf/2111.09794.pdf
[97] https://arxiv.org/pdf/2011.08641.pdf
[98] http://arxiv.org/pdf/2402.02196.pdf
[99] https://velog.io/@qsdcfd/Meta-Transfer-Learning-for-Zero-Shot-Super-Resolution
[100] https://techblog.lycorp.co.jp/ko/how-to-evaluate-ai-generated-images-1
[101] https://koreascience.kr/article/JAKO202231363940914.page
[102] https://aclanthology.org/2021.emnlp-main.702
[103] https://arxiv.org/abs/2407.12404
[104] https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F7A4648992580A5DD7A05DD685CC8324/S0010417523000233a.pdf/div-class-title-exceptions-to-socialism-gender-ethnicity-and-the-transformation-of-soviet-development-in-comparative-perspective-div.pdf
[105] https://www.cambridge.org/core/services/aop-cambridge-core/content/view/B926F12F398B0ADCD17F3676A554E916/S0147547924000176a.pdf/div-class-title-soviet-inflection-points-a-play-in-three-acts-div.pdf
[106] https://www.shs-conferences.org/articles/shsconf/pdf/2019/10/shsconf_cildiah2019_00104.pdf
[107] http://arxiv.org/pdf/2407.14315.pdf
[108] https://journals.umcs.pl/bc/article/download/13221/pdf
[109] https://pmc.ncbi.nlm.nih.gov/articles/PMC1361106/
[110] https://www.mdpi.com/2071-1050/13/20/11389/pdf
[111] https://pmc.ncbi.nlm.nih.gov/articles/PMC5637657/
[112] http://www.esciencecentral.org/journals/impact-of-political-systems-on-european-population-health-2332-0761.1000123.php?aid=32188
[113] https://www.mdpi.com/2071-1050/13/10/5536/pdf
