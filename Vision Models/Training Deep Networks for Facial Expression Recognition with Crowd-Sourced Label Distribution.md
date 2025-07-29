# Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution | Image classification, Facial emotion recognition

## 핵심 주장 및 주요 기여

이 논문은 크라우드소싱으로 수집된 노이즈가 많은 라벨로부터 효과적으로 딥 컨볼루션 신경망(DCNN)을 훈련하는 방법을 제시한다[1]. **주요 기여는 전통적인 다수결 투표(majority voting) 방식보다 라벨 분포를 완전히 활용하는 확률적 라벨 추출(Probabilistic Label Drawing, PLD)과 교차 엔트로피 손실(Cross-Entropy Loss, CEL) 방식이 더 우수한 성능을 보인다는 것을 실증적으로 증명한 것이다**[1].

연구진은 FER 데이터셋의 각 이미지에 대해 10명의 태거가 라벨링한 새로운 FER+ 데이터셋을 구축하고, 네 가지 서로 다른 훈련 방식을 비교 분석했다[1][2]. 특히 **기존의 65±5% 정확도를 보이던 크라우드소싱 라벨의 품질 문제를 해결하기 위해 다중 라벨 분포를 효과적으로 활용하는 새로운 접근법을 제안했다**[1].

## 해결하고자 하는 문제

### 핵심 문제
**크라우드소싱을 통해 수집된 라벨의 노이즈 문제**가 주요 해결 과제다[1]. 감정은 매우 주관적이며, 동일한 얼굴 이미지에 대해 두 사람이 정반대의 의견을 가지는 것이 일반적이다[1]. 또한 크라우드소싱 플랫폼의 작업자들은 낮은 보수로 인해 품질 보장보다는 작업량 증가에 더 관심을 갖는다[1].

### 데이터 품질 분석
연구진은 태거 수와 품질의 관계를 분석한 결과, 3명의 태거 사용 시 일치도가 46%에 불과했지만, 5명 사용 시 67%, 7명 사용 시 80% 이상으로 향상되었다[1]. 이는 **태거 수가 최종 라벨 품질에 미치는 영향이 크다는 것을 의미한다**[1].

## 제안하는 방법론 및 수식

### 네트워크 아키텍처
연구진은 커스텀 VGG13 모델을 사용했다[1]. 입력은 64×64 해상도의 그레이스케일 이미지이며, 출력은 8개의 감정 클래스(neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)다[1].

### 네 가지 훈련 방식

**1. 다수결 투표(Majority Voting, MV)**
라벨 분포의 다수를 단일 태그로 사용하는 방식이다[1]:

$$\hat{p}^i_k = \begin{cases} 1 & \text{if } k = \arg\max_j p^i_j \\ 0 & \text{otherwise} \end{cases}$$

손실 함수는 표준 교차 엔트로피 손실을 사용한다[1]:

$$L = -\sum_{i=1}^N \sum_{k=1}^8 \hat{p}^i_k \log q^i_k$$

**2. 다중 라벨 학습(Multi-Label Learning, ML)**
여러 감정을 허용하는 방식으로, 임계값 θ 이상의 감정들을 모두 수용한다[1]:

$$L = -\sum_{i=1}^N \arg\max_k I_θ(p^i_k) \log q^i_k$$

여기서 $$I_θ(p^i_k)$$는 임계값 함수다[1]:

$$I_θ(p^i_k) = \begin{cases} 1 & \text{if } p^i_k > θ \\ 0 & \text{otherwise} \end{cases}$$

**3. 확률적 라벨 추출(Probabilistic Label Drawing, PLD)**
각 훈련 에포크에서 라벨 분포로부터 무작위로 감정 태그를 추출하는 방식이다[1]:

$$\tilde{p}^i_k(t) = \begin{cases} 1 & \text{if } k = \text{choice}(p^i_j) \\ 0 & \text{otherwise} \end{cases}$$

손실 함수는[1]:

$$L(t) = -\sum_{i=1}^N \arg\max_k \tilde{p}^i_k(t) \log q^i_k$$

**4. 교차 엔트로피 손실(Cross-Entropy Loss, CEL)**
라벨 분포를 직접 목표로 하는 방식이다[1]:

$$L = -\sum_{i=1}^N \sum_{k=1}^8 p^i_k \log q^i_k$$

### 확률 분포 정규화
모든 방식에서 다음 조건이 만족된다[1]:

$$\sum_{k=1}^8 q^i_k = 1; \quad \sum_{k=1}^8 p^i_k = 1$$

## 모델 구조

커스텀 VGG13 네트워크는 다음과 같이 구성된다[1]:
- **입력층**: 64×64 그레이스케일 이미지
- **컨볼루션 레이어**: 10개의 컨볼루션 레이어, 최대 풀링과 드롭아웃 레이어가 교차 배치
- **첫 번째 블록**: 3×3 커널을 가진 64개의 컨볼루션 레이어 2개, 이후 맥스 풀링과 25% 드롭아웃
- **완전 연결층**: 각각 1024개의 숨겨진 노드를 가진 2개의 덴스 레이어, 50% 드롭아웃
- **출력층**: 소프트맥스 레이어

## 성능 향상 결과

### 정확도 비교
5회 실험 평균 결과는 다음과 같다[1]:

| 방식 | 정확도 | 표준편차 |
|------|--------|----------|
| **MV** | 83.852% | ±0.631% |
| **ML** | 83.966% | ±0.362% |
| **PLD** | **84.986%** | ±0.366% |
| **CEL** | **84.716%** | ±0.239% |

**PLD와 CEL 방식이 전통적인 MV 방식보다 1% 이상 높은 성능을 보였으며, t-값이 약 3.1로 99%-99.5% 확률로 통계적으로 유의미한 결과를 나타냈다**[1].

### 혼동 행렬 분석
최고 성능 네트워크의 혼동 행렬 분석 결과, 대부분의 감정에서 좋은 성능을 보였지만 **disgust와 contempt에서는 상대적으로 낮은 성능을 보였다. 이는 FER+ 훈련 세트에서 해당 감정으로 라벨링된 예제가 매우 적기 때문이다**[1].

## 한계점

### 데이터 불균형 문제
**Disgust와 contempt 감정의 경우 훈련 데이터가 부족하여 인식 성능이 떨어진다**[1]. 이는 크라우드소싱 데이터의 고질적인 문제인 클래스 불균형을 보여준다.

### ML 방식의 예상보다 낮은 성능
다중 라벨 학습 방식이 예상보다 좋지 않은 성능을 보였는데, 연구진은 **훈련 시에는 여러 감정을 허용하지만 테스트 시에는 다수 감정만 사용하는 훈련-테스트 불일치가 원인**이라고 분석했다[1].

### 제한된 아키텍처 실험
논문에서는 VGG13 모델만을 사용했으며, 다양한 DCNN 모델 비교는 연구 목적이 아니라고 명시했다[1]. 이는 제안된 방법론의 일반화 가능성을 평가하는 데 한계가 있다.

## 일반화 성능 향상 가능성

### 라벨 분포 활용의 장점
**PLD와 CEL 방식은 라벨 분포의 불확실성을 모델 훈련에 직접 반영함으로써 더 로버스트한 특성 학습이 가능하다**[2][3]. 이는 노이즈가 많은 실제 환경에서 모델의 일반화 성능을 향상시킬 수 있다.

### 드롭아웃의 효과
**35k개의 이미지만을 가진 FER+ 훈련 세트에서도 드롭아웃 레이어가 모델 과적합을 효과적으로 방지했다**[1]. 이는 제한된 데이터에서도 일반화 성능을 유지할 수 있음을 시사한다.

### 데이터 증강의 기여
**아핀 변환을 통한 실시간 데이터 증강이 번역, 회전, 스케일링에 대한 모델의 로버스트성을 향상시켰다**[1]. 이러한 접근법은 다양한 환경에서의 일반화 성능 향상에 기여한다.

### 확률적 접근의 우수성
PLD 방식의 확률적 특성은 **DisturbLabel 접근법과의 유사성을 보이며, 이는 라벨 노이즈에 대한 로버스트성을 제공한다**[1][4]. 여러 에포크에 걸쳐 다양한 라벨을 경험함으로써 모델이 더 일반화된 표현을 학습할 수 있다.

## 미래 연구에 미치는 영향

### 노이즈 라벨 학습 분야의 발전
이 연구는 **크라우드소싱 노이즈 라벨 학습 분야에서 중요한 기준점을 제시했다**[3][5]. 특히 라벨 분포를 직접 활용하는 접근법이 단순한 다수결 방식보다 우수함을 입증했다.

### 다중 모달 학습으로의 확장
**감정 인식에서 여러 모달리티(시각, 음성, 텍스트)를 결합하는 연구에서 노이즈 처리 방법론으로 활용 가능하다**[6]. 각 모달리티에서 수집된 노이즈 라벨을 효과적으로 통합하는 데 응용할 수 있다.

### FER+ 데이터셋의 영향
**Microsoft에서 공개한 FER+ 데이터셋은 얼굴 표정 인식 연구의 표준 벤치마크가 되었다**[7][8]. 다중 라벨 분포를 제공하는 최초의 대규모 데이터셋으로서 후속 연구의 기반을 마련했다.

## 앞으로 연구 시 고려할 점

### 클래스 불균형 해결
**Disgust와 contempt 같은 희소 클래스에 대한 성능 개선을 위해 클래스 불균형 처리 기법의 통합이 필요하다**[1]. 오버샘플링, 언더샘플링, 또는 가중치 조정 등의 방법을 고려해야 한다.

### 다양한 아키텍처 검증
**제안된 방법론이 다양한 딥러닝 아키텍처(ResNet, EfficientNet 등)에서도 일관되게 작동하는지 검증이 필요하다**[9]. 이는 방법론의 일반화 가능성을 확인하는 데 중요하다.

### 실시간 처리 고려사항
**실제 응용에서는 실시간 처리가 중요하므로, 제안된 방법들의 계산 복잡도와 추론 속도를 고려한 최적화가 필요하다**[10]. 특히 모바일 환경에서의 효율성 검증이 중요하다.

### 크로스 도메인 적용성
**다른 감정 인식 도메인(음성, 텍스트)이나 다른 분류 문제에서의 적용 가능성을 탐구해야 한다**[11][12]. 이를 통해 방법론의 범용성을 확인할 수 있다.

### 노이즈 레벨에 따른 적응성
**다양한 노이즈 레벨에서의 성능 변화를 분석하고, 노이즈 정도에 따라 적응적으로 방법을 선택하는 메커니즘 연구가 필요하다**[13][14]. 이는 실제 환경의 다양한 조건에 대응하기 위함이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/46a35e8a-a00a-43b7-9cc4-1016a2f107a3/1608.01041v2.pdf
[2] https://dl.acm.org/doi/10.1145/2993148.2993165
[3] https://arxiv.org/pdf/2007.08199.pdf
[4] https://epubs.siam.org/doi/10.1137/1.9781611978032.47
[5] https://pure.kaist.ac.kr/en/publications/learning-from-noisy-labels-with-deep-neural-networks-a-survey
[6] https://arxiv.org/html/2407.17716v1
[7] https://github.com/microsoft/FERPlus
[8] https://paperswithcode.com/dataset/fer
[9] https://ieeexplore.ieee.org/document/10426615/
[10] https://www.mdpi.com/2076-3417/14/3/1156
[11] https://www.aclweb.org/anthology/P19-1112
[12] https://arxiv.org/html/2407.06902v1
[13] https://ieeexplore.ieee.org/document/10656694/
[14] https://ieeexplore.ieee.org/document/10526454/
[15] https://ieeexplore.ieee.org/document/10743540/
[16] https://ieeexplore.ieee.org/document/9676449/
[17] https://link.springer.com/10.1007/s11042-023-17510-3
[18] https://ijai.iaescore.com/index.php/IJAI/article/view/21643
[19] https://peerj.com/articles/cs-2866
[20] https://arxiv.org/abs/1608.01041
[21] https://arxiv.org/pdf/1608.01041.pdf
[22] https://arxiv.org/abs/2503.20428
[23] https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
[24] https://www.kaggle.com/datasets/msambare/fer2013
[25] https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf
[26] https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer
[27] https://arxiv.org/abs/2007.08199
[28] https://paperswithcode.com/task/facial-expression-recognition
[29] https://github.com/vicksam/fer-model
[30] http://dmqa.korea.ac.kr/activity/seminar/377
[31] https://paperswithcode.com/task/facial-expression-recognition-1
[32] https://www.semanticscholar.org/paper/121ef6b3783c6f3d83d07cd2e07a1d2fbea3e097
[33] https://linkinghub.elsevier.com/retrieve/pii/S0888613X22001207
[34] https://ieeexplore.ieee.org/document/9444560/
[35] https://www.ijcai.org/proceedings/2020/356
[36] https://www.machinelearningmastery.com/cross-entropy-for-machine-learning/
[37] https://www.ijcai.org/proceedings/2019/0204.pdf
[38] https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning
[39] https://openaccess.thecvf.com/content/CVPR2021/papers/Ortego_Multi-Objective_Interpolation_Training_for_Robustness_To_Label_Noise_CVPR_2021_paper.pdf
[40] https://openreview.net/forum?id=nFEQNYsjQO
[41] https://arxiv.org/abs/2011.05231
[42] https://openaccess.thecvf.com/content/ICCV2021W/ABAW/papers/Gera_Noisy_Annotations_Robust_Consensual_Collaborative_Affect_Expression_Recognition_ICCVW_2021_paper.pdf
[43] https://openreview.net/forum?id=2NKumsITFw
[44] https://neptune.ai/blog/cross-entropy-loss-and-its-applications-in-deep-learning
[45] https://www.sciencedirect.com/science/article/abs/pii/S0957417425025990
[46] https://neurips.cc/virtual/2023/poster/70516
[47] https://kr.mathworks.com/matlabcentral/answers/467318-using-a-cross-entropy-loss-function-in-the-deep-learning-toolbox-for-a-probability-distribution-targ
[48] https://arxiv.org/abs/2404.17113
[49] https://www.themoonlight.io/ko/review/learning-from-crowdsourced-noisy-labels-a-signal-processing-perspective
[50] https://ieeexplore.ieee.org/document/10746065/
[51] https://ieeexplore.ieee.org/document/10746197/
[52] http://arxiv.org/pdf/1608.01041.pdf
[53] https://pmc.ncbi.nlm.nih.gov/articles/PMC9165031/
[54] https://arxiv.org/pdf/2209.10448.pdf
[55] http://arxiv.org/pdf/2101.03477.pdf
[56] https://arxiv.org/html/2410.22506v1
[57] https://arxiv.org/pdf/2503.20428.pdf
[58] http://arxiv.org/pdf/2303.00180.pdf
[59] https://downloads.hindawi.com/journals/wcmc/2022/7094539.pdf
[60] http://arxiv.org/pdf/2209.15402.pdf
[61] https://downloads.hindawi.com/journals/cin/2023/7850140.pdf
[62] https://paperswithcode.com/task/learning-with-noisy-labels
[63] https://www.sciencedirect.com/science/article/abs/pii/S0167865522003105
[64] https://paperswithcode.com/sota/facial-expression-recognition-on-fer-1
[65] https://ieeexplore.ieee.org/document/10203735/
[66] https://arxiv.org/pdf/2306.03116v1.pdf
[67] https://arxiv.org/pdf/2103.10869.pdf
[68] https://arxiv.org/pdf/2007.05836.pdf
[69] https://arxiv.org/pdf/2202.02200.pdf
[70] http://arxiv.org/abs/1712.04577
[71] https://arxiv.org/pdf/2208.03207.pdf
[72] https://arxiv.org/pdf/2203.04199.pdf
[73] https://arxiv.org/pdf/2312.06221.pdf
[74] https://arxiv.org/html/2305.19518
[75] http://arxiv.org/pdf/2306.11650.pdf
[76] https://www.sciencedirect.com/science/article/abs/pii/S1746809424007742
[77] https://arxiv.org/abs/2411.17113
[78] https://www.v7labs.com/blog/cross-entropy-loss-guide
[79] https://dl.acm.org/doi/10.1145/3736426.3736510
[80] https://towardsdatascience.com/how-neural-networks-learn-a-probabilistic-viewpoint-0f6a78dc58e2/
