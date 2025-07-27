# HO-CPConv : Factorized Higher-Order CNNs with an Application to Spatio-Temporal Emotion Estimation | Emotion estimation

## 1. 핵심 주장과 주요 기여

### 핵심 주장
이 논문의 핵심 주장은 **텐서 분해(tensor decomposition) 기법을 활용한 고차원 합성곱 신경망의 효율적인 설계와 higher-order transduction을 통한 차원 확장**이다[1]. 저자들은 기존의 3D 합성곱 연산이 매우 높은 계산 비용을 요구한다는 문제를 해결하기 위해, CP 분해(Canonical Polyadic decomposition)를 활용한 새로운 접근법을 제안했다[1][2].

### 주요 기여

**1. 통합 이론적 프레임워크**
논문은 기존에 별개로 다뤄지던 두 접근법을 통합했다[1]:
- 텐서 분해를 통한 네트워크 압축
- MobileNet과 같은 효율적인 아키텍처 설계

저자들은 ResNet의 Bottleneck 블록, MobileNet, ResNext 등이 모두 텐서 분해의 특수한 경우임을 수학적으로 증명했다[1].

**2. CPHO-CPConv)**
핵심 기술인 HO-CPConv는 다음 수식으로 표현된다[1]:

$$ F = \rho\left(\Psi\left(X \times_0 U^{(T)}\right)\right) \times_0 \left(\text{diag}(\lambda)U^{(C)}\right) $$

여기서:
- $$\rho$$: 1D 공간 합성곱들의 연쇄
- $$\Psi$$: 비선형성(배치 정규화 + ReLU)
- $$U^{(T)}, U^{(C)}$$: 출력 채널과 입력 채널 팩터
- $$\lambda$$: 가중치 벡터

**3. Higher-Order Transduction**
가장 혁신적인 기여는 N차원에서 학습된 모델을 (N+K)차원으로 확장하는 transduction 메커니즘이다[1]:

$$ F = \hat{\rho}\left(\Psi\left(X \times_0 U^{(T)}\right)\right) \times_0 \left(\text{diag}(\lambda)U^{(C)}\right) $$

여기서 $$\hat{\rho}(X) = \rho(X) \star_{N+1} U^{(K_{N+1})}$$로, 새로운 차원에 대해서는 $$K_{N+1} \times R$$ 개의 매개변수만 추가로 학습하면 된다[1].

## 2. 해결하고자 하는 문제

### 주요 문제점들

**1. 3D 합성곱의 계산 복잡성**
일반적인 3D 합성곱은 $$C \times T \times K^3$$ 개의 매개변수를 가지는 반면, 제안된 HO-CP 합성곱은 $$R(C + T + 3K)$$ 개의 매개변수만 필요하다[1]. 예를 들어, $$32 \times 32 \times 16$$ 입력에 대해 기존 3D 합성곱 대비 최대 50배 이상의 FLOP 감소를 달성했다[1].

**2. 비디오 데이터의 부족**
감정 인식 분야에서 전문가가 주석을 단 대규모 비디오 데이터셋의 부족 문제를 해결하기 위해, 정적 이미지에서 먼저 학습한 후 시간적 차원으로 확장하는 전략을 제안했다[1].

**3. 기존 효율적 아키텍처의 이론적 통합 부족**
MobileNet, ResNext 등의 효율적 아키텍처들이 각각 독립적으로 개발되었으나, 이들이 모두 텐서 분해의 특수한 경우임을 이론적으로 증명하여 통합된 관점을 제공했다[1].

## 3. 모델 구조

### 아키텍처 설계

**1. 기본 구조**
제안된 모델은 ResNet-18을 백본으로 사용하되, 모든 합성곱 레이어를 HO-CPConv로 대체한다[1]. 

**2. Transduction 과정**
1. **Transduction Process 단계**: AffectNet과 같은 정적 이미지 데이터셋에서 spatial factors (공간적 요소)를 학습
2. **3D 확장 단계**: 학습된 spatial factors를 고정하고 temporal factor (시간적 요소)만 새로 학습
3. **Fine-tuning 단계**: 모든 매개변수를 함께 최적화

**3. 자동 랭크 선택**

$$ L_{reg} = L + \gamma \sum_{l=0}^{L-1} |\lambda_l| $$

여기서 $$\gamma$$는 희소성을 제어하는 매개변수로, L1 정규화를 통해 각 레이어의 최적 랭크를 자동으로 결정한다[1]. 실험에서 전체 매개변수의 8-15%를 0으로 설정할 수 있음을 확인했다[1].

## 4. 성능 향상 및 한계

### 성능 향상

**1. 매개변수 효율성**
- 기존 3D ResNet-18: 33M 매개변수
- 제안 모델: 11M 매개변수 (66% 감소)
- ResNet-(2+1)D: 31M 매개변수[1]

**2. 데이터셋별 성능**

| 데이터셋 | 메트릭 | 기존 최고 성능 | 제안 모델 | 개선율 |
|---------|-------|--------------|-----------|--------|
| AffectNet | Valence CCC | 0.66 | 0.71 | +7.6% |
| AffectNet | Arousal CCC | 0.54 | 0.63 | +16.7% |
| SEWA (Temporal) | Valence CCC | 0.59 | 0.84 | +42.4% |
| AFEW-VA (Temporal) | Valence CCC | 0.51 | 0.64 | +25.5% |

**3. 일반화 성능**
CIFAR-10에서 MobileNet-v2와 유사한 성능(94%)을 달성하면서도 더 적은 매개변수(2.29M vs 2.30M)를 사용했다[1].

### 한계점

**1. 랭크 선택의 민감성**
CP 분해는 랭크 선택에 민감하며, 부적절한 랭크는 성능 저하를 야기할 수 있다[3][4]. 

**2. 학습 안정성**
텐서 분해 기반 합성곱은 처음부터 학습하기 어려우며, 적절한 초기화와 정규화가 필요하다[1].

**3. 특정 도메인 제한**
현재 연구는 주로 감정 인식에 집중되어 있어, 다른 컴퓨터 비전 태스크에서의 일반화 성능이 충분히 검증되지 않았다.

## 5. 일반화 성능 향상 가능성

### 이론적 기반

**1. 차원 무관한 확장성**
Higher-order transduction은 임의의 N차원 데이터를 (N+K)차원으로 확장할 수 있는 일반적인 프레임워크를 제공한다[1]. 이는 다음과 같은 응용이 가능하다:
- 2D 이미지 → 3D 비디오
- 3D 볼륨 → 4D 시공간 데이터
- 일반적인 N-D → (N+K)-D 확장

**2. 크로스 도메인 전이**
정적 이미지에서 학습된 spatial representation이 비디오 도메인에서도 효과적으로 작동함을 실험적으로 증명했다[1]. 이는 다음을 시사한다:
- 제한된 고차원 데이터로도 효과적인 학습 가능
- 사전 학습된 모델의 효율적인 재사용
- 도메인 적응 비용 최소화

**3. 매개변수 효율적 확장**
새로운 차원 추가 시 필요한 추가 매개변수가 $$K_{N+1} \times R$$로 매우 제한적이어서, 오버피팅 위험을 크게 줄일 수 있다[1].

### 실제 일반화 증거

**1. 다양한 데이터셋에서의 일관된 성능**
논문에서 제시된 결과는 세 가지 서로 다른 특성을 가진 데이터셋에서 일관되게 우수한 성능을 보여준다[1]:
- AffectNet (대규모, 정적, in-the-wild)
- SEWA (다문화, 동적, 자연스러운 상호작용)  
- AFEW-VA (영화 클립, 도전적인 조건)

**2. 아키텍처 독립성**
제안된 방법은 특정 네트워크 구조에 의존하지 않고 다양한 CNN 아키텍처에 적용 가능하다는 점에서 높은 일반화 가능성을 보인다[1].

## 6. 향후 연구에 미치는 영향과 고려사항

### 긍정적 영향

**1. 효율적인 고차원 학습의 새로운 패러다임**
- 의료 영상 (3D/4D 데이터)에서의 활용 가능성[5]
- 로보틱스에서의 시공간 인식[6]
- 자율주행에서의 다차원 센서 융합

**2. 이론적 통합의 촉진**
기존의 분리된 연구 영역들을 통합하는 이론적 프레임워크를 제공하여 향후 연구의 체계화에 기여할 것으로 예상된다[1].

**3. 실용적 응용 확장**
모바일 및 엣지 디바이스에서의 실시간 비디오 분석이 가능해져 다양한 실제 응용 분야로의 확장이 기대된다[7].

### 향후 연구 고려사항

**1. 최적화 알고리즘 개선**
현재 CP 분해의 비볼록성으로 인한 학습 불안정성 문제를 해결하기 위한 새로운 최적화 기법 연구가 필요하다[4][8].

**2. 다양한 텐서 분해 기법 탐색**
CP 분해 외에도 Tucker 분해, Tensor Ring 등 다른 분해 기법들과의 조합 연구가 필요하다[9][10].

**3. 자동 아키텍처 탐색**
Neural Architecture Search (NAS)와 결합하여 태스크별 최적의 텐서 분해 구조를 자동으로 찾는 연구가 유망하다.

**4. 해석가능성 향상**
텐서 분해를 통해 학습된 각 요소의 의미를 해석하고 설명할 수 있는 방법론 개발이 중요하다.

**5. 대규모 데이터셋에서의 검증**
현재 실험은 상대적으로 작은 규모의 데이터셋에서 수행되었으므로, ImageNet 규모의 대용량 데이터셋에서의 성능 검증이 필요하다.

이 연구는 효율적인 고차원 신경망 설계에 있어 중요한 이론적 기반과 실용적 방법론을 제공하여, 향후 컴퓨터 비전과 딥러닝 분야의 발전에 상당한 영향을 미칠 것으로 예상된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/aa12629a-4c0a-4f50-8ff6-e3695a4dc74a/1906.06196v2.pdf
[2] https://ieeexplore.ieee.org/document/9157354/
[3] https://www.semanticscholar.org/paper/62e348e26976c3ef77909b9af9788ebc2509009a
[4] https://www.mdpi.com/2076-3417/14/4/1491
[5] https://link.springer.com/10.1007/s11548-020-02178-z
[6] https://ieeexplore.ieee.org/document/8715379/
[7] https://ieeexplore.ieee.org/document/10477992/
[8] https://arxiv.org/abs/2205.15307
[9] https://dl.acm.org/doi/10.1145/3409073.3409094
[10] https://dl.acm.org/doi/10.1145/3702641
[11] http://ieeexplore.ieee.org/document/7881725/
[12] https://www.jsr.org/hs/index.php/path/article/view/4916
[13] https://arxiv.org/abs/2401.03384
[14] https://arxiv.org/abs/1412.6553
[15] http://proceedings.mlr.press/v44/huang15convolutional.pdf
[16] https://proceedings.neurips.cc/paper_files/paper/2023/file/e9b8a3362a6d9a7f9f842bd2d919e1a0-Paper-Conference.pdf
[17] https://arxiv.org/abs/2005.13746
[18] https://arxiv.org/abs/2311.05908
[19] https://openreview.net/forum?id=go4zzXBWVs
[20] https://github.com/ruihangdu/Decompose-CNN
[21] https://openreview.net/forum?id=gPKTTAfYBp
[22] https://proceedings.mlr.press/v28/sutskever13.pdf
[23] http://www.navisphere.net/6064/speeding-up-convolutional-neural-networks-using-fine-tuned-cp-decomposition/
[24] https://www.bmvc2021-virtualconference.com/conference/papers/paper_1631.html
[25] https://arxiv.org/abs/1212.1936
[26] https://velog.io/@godhj/Neural-Network-Compression-Tensor-Decomposition
[27] https://www.themoonlight.io/en/review/convolution-tensor-decomposition-for-efficient-high-resolution-solutions-to-the-allen-cahn-equation
[28] https://www.sciencedirect.com/science/article/abs/pii/S0893608023006767
[29] https://www.koreascience.kr/article/JAKO202113259287312.page
[30] https://ieeexplore.ieee.org/document/10328680/
[31] https://ieeexplore.ieee.org/document/10933146/
[32] https://ieeexplore.ieee.org/document/10141618/
[33] https://linkinghub.elsevier.com/retrieve/pii/S0045790623004871
[34] https://www.mdpi.com/2076-3425/13/4/685
[35] https://scispace.com/pdf/deep-spatio-temporal-features-for-multimodal-emotion-53l9icadik.pdf
[36] https://pmc.ncbi.nlm.nih.gov/articles/PMC10028317/
[37] https://thesai.org/Publications/ViewPaper?Volume=9&Issue=8&Code=IJACSA&SerialNo=43
[38] https://arxiv.org/abs/2501.06787
[39] https://www.nature.com/articles/s41598-024-65276-x
[40] https://pubmed.ncbi.nlm.nih.gov/36046470/
[41] https://openaccess.thecvf.com/content_CVPR_2020/papers/Kossaifi_Factorized_Higher-Order_CNNs_With_an_Application_to_Spatio-Temporal_Emotion_Estimation_CVPR_2020_paper.pdf
[42] https://onlinelibrary.wiley.com/doi/10.1155/2022/7450637
[43] https://paperswithcode.com/paper/3d-cnn-for-facial-emotion-recognition-in
[44] https://www.sciencedirect.com/science/article/abs/pii/S0010482525006286
[45] https://www.sciencedirect.com/science/article/pii/S1877050924008731
[46] https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.872311/full
[47] https://www.sciencedirect.com/science/article/abs/pii/S0010482523001543
[48] https://dl.acm.org/doi/10.1007/978-3-030-64559-5_23
[49] https://koreascience.kr/article/JAKO201900937437570.pdf
[50] https://www.jask.or.kr/articles/article/K47X/
[51] https://ieeexplore.ieee.org/document/10894781/
[52] https://www.nationaleducationservices.org/facial-emotion-recognition-with-crossdataset-generalization-capabilities/pid-2230847558
[53] https://ieeexplore.ieee.org/document/10674828/
[54] https://www.mdpi.com/1424-8220/23/3/1080
[55] https://ieeexplore.ieee.org/document/10435159/
[56] https://ieeexplore.ieee.org/document/10229334/
[57] https://arxiv.org/abs/2301.10906
[58] https://www.mohammadmahoor.com/pages/databases/affectnet/
[59] http://jeankossaifi.com/pdfs/sewa.pdf
[60] http://jeankossaifi.com/pdfs/afewva.pdf
[61] https://arxiv.org/html/2410.22506v1
[62] https://www.nature.com/articles/s41597-019-0209-0
[63] https://dl.acm.org/doi/10.5555/3143567.3143655
[64] https://www.themoonlight.io/en/review/affectnet-a-database-for-enhancing-facial-expression-recognition-with-soft-labels
[65] https://openaccess.thecvf.com/content/ICCV2021W/ABAW/papers/Zhang_Continuous_Emotion_Recognition_With_Audio-Visual_Leader-Follower_Attentive_Fusion_ICCVW_2021_paper.pdf
[66] https://ibug.doc.ic.ac.uk/resources/afew-va-database/
[67] https://arxiv.org/abs/2410.22506
[68] https://pmc.ncbi.nlm.nih.gov/articles/PMC9205566/
[69] https://www.sciencedirect.com/science/article/pii/S0262885617300379
[70] https://pmc.ncbi.nlm.nih.gov/articles/PMC7579283/
[71] https://www.isca-archive.org/interspeech_2019/schmitt19_interspeech.html
[72] https://paperswithcode.com/dataset/afew-va
[73] https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet
[74] https://arxiv.org/abs/2212.05581
[75] https://www.semanticscholar.org/paper/e3b41424637d716f63ade41390ff4b969853d049
[76] https://arxiv.org/pdf/1701.07148.pdf
[77] http://arxiv.org/pdf/1908.04471.pdf
[78] https://arxiv.org/pdf/2308.04595.pdf
[79] https://arxiv.org/pdf/1801.05243.pdf
[80] https://arxiv.org/pdf/1905.10145.pdf
[81] http://arxiv.org/pdf/2005.13746.pdf
[82] https://dx.plos.org/10.1371/journal.pone.0267091
[83] https://arxiv.org/pdf/2210.10184.pdf
[84] https://pmc.ncbi.nlm.nih.gov/articles/PMC9009670/
[85] https://arxiv.org/pdf/1810.08612.pdf
[86] https://www.sciencedirect.com/science/article/pii/S0045782524007618
[87] https://arxiv.org/html/2204.07756v3
[88] https://arxiv.org/abs/2410.15519
[89] https://linkinghub.elsevier.com/retrieve/pii/S1361841520300943
[90] http://link.springer.com/10.1007/s11548-019-02006-z
[91] https://arxiv.org/pdf/1705.04515.pdf
[92] https://pmc.ncbi.nlm.nih.gov/articles/PMC11666491/
[93] https://arxiv.org/abs/2011.09280
[94] https://arxiv.org/pdf/2303.06632.pdf
[95] https://arxiv.org/html/2307.03068
[96] https://www.mdpi.com/2076-3417/11/24/11738/pdf
[97] https://arxiv.org/abs/1910.01254
[98] http://arxiv.org/pdf/2404.18327.pdf
[99] https://arxiv.org/pdf/2305.19379.pdf
[100] https://www.mdpi.com/1424-8220/23/10/4777
[101] https://sejong.elsevierpure.com/en/publications/facial-expression-recognition-in-videos-an-cnn-lstm-based-model-f
[102] https://www.sciencedirect.com/science/article/pii/S1110866520301389
[103] https://www.mdpi.com/2073-431X/13/4/101
[104] https://www.sciencedirect.com/science/article/pii/S1746809424006669
[105] https://ieeexplore.ieee.org/document/10511123/
[106] https://ieeexplore.ieee.org/document/10651071/
[107] https://arxiv.org/pdf/1708.03985.pdf
[108] https://arxiv.org/pdf/2412.01860.pdf
[109] https://arxiv.org/pdf/2303.09162.pdf
[110] https://www.mdpi.com/1424-8220/23/3/1080/pdf?version=1673956722
[111] https://arxiv.org/pdf/2203.13436.pdf
[112] http://arxiv.org/pdf/2404.14975.pdf
[113] https://arxiv.org/pdf/2201.12705.pdf
[114] https://pmc.ncbi.nlm.nih.gov/articles/PMC9921901/
[115] https://www.mdpi.com/1424-8220/21/9/3046/pdf
[116] https://paperswithcode.com/dataset/sewa-db
[117] https://www.sciencedirect.com/science/article/abs/pii/S0262885617300379
[118] https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format
[119] https://arxiv.org/abs/1901.02839
