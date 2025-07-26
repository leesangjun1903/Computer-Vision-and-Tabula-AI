# Siamese Neural Networks for One-shot Image Recognition | Meta-learning

### 핵심 주장과 주요 기여

이 논문은 Siamese Neural Networks를 활용한 one-shot image recognition에 대한 선구적 연구로, 다음과 같은 핵심 주장을 제시합니다[1]:

**핵심 주장**: 기존 머신러닝 모델들이 많은 데이터를 필요로 하는 한계를 극복하기 위해, Siamese Neural Networks를 이용하여 단 하나의 예시만으로도 새로운 클래스를 인식할 수 있는 방법론을 제안합니다[1][2].

**주요 기여**:
1. **구조적 혁신**: 두 개의 동일한 네트워크가 가중치를 공유하면서 입력 쌍의 유사도를 측정하는 독특한 구조 제안[1][3]
2. **일반화 능력**: 재훈련 없이 완전히 새로운 클래스들에 대해서도 강력한 판별 능력을 보여줌[1][2]
3. **실용적 성능**: Omniglot 데이터셋에서 92%의 정확도를 달성하여 기존 딥러닝 모델들을 크게 상회하는 성과[1][2]

### 해결하고자 하는 문제

**1. 데이터 부족 문제 (Data Scarcity)**
전통적인 딥러닝 모델들은 좋은 성능을 위해 대량의 레이블된 데이터를. 하지만 실제 상황에서는 클래스당 하나 또는 소수의 예시만 가능한 경우가 많습니다[1][4][5].

**2. 일반화 문제 (Generalization Challenge)**
기존 모델들은 새로운 클래스가 나타날 때마다 전체 모델을 재훈련해야 하는 비효율성을 가지고 있습니다[1][6][5].

**3. 도메인 특화 지식 의존성**
많은 기존 방법들이 특정 도메인에 특화된 지식에 의존하여 범용성이 떨어지는 문제가 있었습니다[1].

### 제안하는 방법론과 수식

**핵심 아키텍처**:
논문에서 제안하는 Siamese Network는 두 개의 동일한 CNN 구조가 가중치를 공유하는 구조입니다[1][7]:

**거리 메트릭**:
최종 예측값은 다음과 같이 계산됩니다[1]:

$$ p = \sigma\left(\sum_j \alpha_j |h^{(j)}\_{1,L-1} - h^{(j)}_{2,L-1}|\right) $$

여기서:
- $$\sigma$$는 sigmoid 활성화 함수
- $$\alpha_j$$는 학습 가능한 가중치 매개변수
- $$h^{(j)}\_{1,L-1}$$, $$h^{(j)}_{2,L-1}$$는 각각 두 네트워크의 $$(L-1)$$번째 은닉층의 $$j$$번째 특징벡터

**손실 함수**:
정규화된 교차 엔트로피 손실을 사용합니다[1]:

$$ L(x^{(i)}_1, x^{(i)}_2) = y(x^{(i)}_1, x^{(i)}_2) \log p(x^{(i)}_1, x^{(i)}_2) + (1-y(x^{(i)}_1, x^{(i)}_2)) \log(1-p(x^{(i)}_1, x^{(i)}_2)) + \lambda^T |w|^2 $$

### 모델 구조

**1. 컨볼루션 레이어 구조**[1]:
- 4개의 컨볼루션 레이어 (각각 64, 128, 128, 256개의 필터)
- 필터 크기: 10×10, 7×7, 4×4, 4×4
- 최대 풀링과 ReLU 활성화 함수 사용

**2. 완전연결 레이어**[1]:
- 4096개의 뉴런을 가진 완전연결 레이어
- L1 거리 계산을 통한 유사도 측정
- Sigmoid 함수를 통한 최종 예측

**3. 가중치 공유 메커니즘**[1][7]:
- 두 네트워크가 동일한 가중치를 공유하여 대칭성 보장
- 유사한 이미지들이 특징 공간에서 가까운 위치에 매핑되도록 보장

### 성능 향상 및 결과

**1. Omniglot 데이터셋 결과**[1][2]:
- One-shot 분류 정확도: 92.0%
- 기존 베이스라인 대비 현저한 성능 향상:
  - 1-Nearest Neighbor: 21.7%
  - Deep Boltzmann Machine: 62.0%
  - Hierarchical Deep: 65.2%

**2. 검증 작업 성능**[1]:
- 30k 훈련 데이터: 90.61% → 91.90% (데이터 증강 후)
- 150k 훈련 데이터: 91.63% → 93.42% (데이터 증강 후)

**3. 크로스 도메인 일반화**[1]:
- Omniglot에서 훈련된 모델을 MNIST에 적용: 70.3% (1-NN: 26.5%)

### 모델의 일반화 성능 향상

**1. 메트릭 학습 기반 접근**[1][8]:
Siamese Network는 분류가 아닌 유사도 학습에 초점을 맞춤으로써 새로운 클래스에 대한 일반화 능력을 획득합니다. 이는 인간이 새로운 개념을 빠르게 학습하는 방식과 유사합니다[1].

**2. 전이 학습 효과**[1][9]:
학습된 특징 표현이 도메인 간 전이가 가능함을 MNIST 실험을 통해 입증했습니다. 이는 모델이 문자 인식에 특화된 저수준 특징이 아닌 범용적인 시각적 특징을 학습했음을 시사합니다[1].

**3. 데이터 증강의 효과**[1]:
- 아핀 변환을 통한 데이터 증강이 성능 향상에 크게 기여
- 8가지 변환 적용으로 1-3% 성능 향상 달성

### 한계점

**1. 계산 복잡도**[1]:
컨볼루션 연산은 완전연결 레이어보다 계산 비용이 높아 훈련 시간이 오래 걸립니다.

**2. 하이퍼파라미터 민감성**[1]:
베이지안 최적화를 통한 광범위한 하이퍼파라미터 튜닝이 필요하며, 학습률, 모멘텀, 정규화 등 다양한 파라미터에 민감합니다.

**3. 도메인 간 성능 격차**[1]:
Omniglot에서 MNIST로의 전이 시 성능이 상당히 감소 (92% → 70.3%)하여 완전한 도메인 불변성은 달성하지 못했습니다.

**4. 확장성 한계**[8]:
매우 많은 클래스가 있는 경우, 모든 쌍별 비교가 필요하여 계산 복잡도가 급격히 증가할 수 있습니다.

### 미래 연구에 미치는 영향

**1. Few-shot Learning 분야 개척**[10][11]:
이 논문은 few-shot learning 분야의 기초를 마련했으며, 이후 Prototypical Networks, Matching Networks, MAML 등 다양한 메타러닝 방법론들의 발전에 영감을 제공했습니다[12][13].

**2. 메트릭 러닝의 대중화**[14][15]:
Siamese Network 구조는 메트릭 러닝 접근법의 효과를 입증하여 이후 연구들이 거리 기반 학습에 더 많은 관심을 갖게 했습니다.

**3. 실용적 응용 확산**[16][11][17]:
얼굴 인식, 서명 검증, 제품 식별 등 다양한 실제 응용 분야에서 Siamese Network가 활용되고 있습니다.

### 향후 연구 시 고려사항

**1. 아키텍처 개선**:
- 어텐션 메커니즘 통합으로 중요한 특징에 더 집중[18]
- Residual Connection 등을 통한 더 깊은 네트워크 구조 탐색

**2. 손실 함수 개선**:
- Contrastive Loss, Triplet Loss 등 다양한 메트릭 러닝 손실 함수 비교 연구[9]
- 하드 네거티브 마이닝 등 고급 샘플링 전략 적용

**3. 메타러닝과의 결합**[19][20]:
- MAML, Prototypical Networks 등과의 결합을 통한 성능 향상
- 그래프 신경망 등 새로운 아키텍처와의 융합

**4. 실제 환경 적용**:
- 노이즈가 있는 실제 데이터에 대한 강건성 개선
- 실시간 처리를 위한 경량화 연구

이 논문은 one-shot learning과 few-shot learning 분야의 패러다임을 바꾼 중요한 연구로, 현재까지도 많은 후속 연구들의 기반이 되고 있습니다[10][8][21]. 특히 데이터가 부족한 상황에서의 머신러닝 응용에 새로운 가능성을 제시했다는 점에서 그 의의가 큽니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ebfb8202-ba78-4877-8923-b6e8994093d5/oneshot1.pdf
[2] https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
[3] https://tyami.github.io/deep%20learning/Siamese-neural-networks/
[4] https://en.wikipedia.org/wiki/One-shot_learning_(computer_vision)
[5] https://encord.com/blog/one-shot-learning-guide/
[6] https://serokell.io/blog/nn-and-one-shot-learning
[7] https://builtin.com/machine-learning/siamese-network
[8] https://www.mdpi.com/2076-3417/11/17/7839
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC7148474/
[10] https://www.semanticscholar.org/paper/f216444d4f2959b4520c61d20003fa30a199670a
[11] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12969/3014396/One-shot-deformed-face-recognition-via-Siamese-neural-network/10.1117/12.3014396.full
[12] https://scholar.google.com.hk/citations?user=iBeDoRAAAAAJ&hl=th
[13] https://arxiv.org/pdf/1902.03477.pdf
[14] https://openaccess.thecvf.com/content/ACCV2022/papers/Jung_Few-Shot_Metric_Learning_Online_Adaptation_of_Embedding_for_Retrieval_ACCV_2022_paper.pdf
[15] https://openreview.net/forum?id=q7t7q8kJQa
[16] https://prosiding-senada.upnjatim.ac.id/index.php/senada/article/view/125
[17] https://lib.jucs.org/article/70484/
[18] https://openreview.net/pdf?id=BJjn-Yixl
[19] https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-023-01063-6
[20] https://royalsocietypublishing.org/doi/10.1098/rsos.230706
[21] https://arxiv.org/abs/2105.08149
[22] https://link.springer.com/10.1007/s00034-022-02062-y
[23] https://rhcsky.tistory.com/6
[24] https://ai-information.blogspot.com/2019/07/meta-002-siamese-neural-networks-for.html
[25] https://belvederef.github.io/cv-notebook/computer-vision-theory/one-shot-learning.html
[26] https://jxnjxn.tistory.com/84
[27] https://www.youtube.com/watch?v=SthmLerAeis
[28] https://www.geeksforgeeks.org/machine-learning/one-shot-learning-in-machine-learning-1/
[29] https://ygseo.tistory.com/179
[30] https://simonezz.tistory.com/100
[31] https://arxiv.org/pdf/2502.04580.pdf
[32] https://arxiv.org/html/2407.00100v1
[33] http://arxiv.org/pdf/2502.05414.pdf
[34] https://arxiv.org/pdf/1411.4257.pdf
[35] http://arxiv.org/abs/1710.05871
[36] http://arxiv.org/pdf/2305.17040.pdf
[37] https://arxiv.org/pdf/2306.04891.pdf
[38] https://arxiv.org/html/2405.17234
[39] https://www.geeksforgeeks.org/machine-learning/omniglot-classification-task/
[40] https://paperswithcode.com/author/gregory-koch
[41] https://www.baeldung.com/cs/siamese-networks
[42] https://cims.nyu.edu/~brenden/papers/LakeEtAlOmniglotProgress.pdf
[43] https://www.tutorialspoint.com/understanding-omniglot-classification-task-in-machine-learning
[44] https://www.semanticscholar.org/paper/Siamese-Neural-Networks-for-One-Shot-Image-Koch/f216444d4f2959b4520c61d20003fa30a199670a
[45] https://www.mdpi.com/2071-1050/14/18/11484
[46] https://paperswithcode.com/paper/siamese-neural-networks-for-one-shot-image
[47] https://en.wikipedia.org/wiki/Siamese_neural_network
[48] https://www.bibsonomy.org/bibtex/26f83b8c4cf316e77e6f6ce1e97411b30/bsc
[49] https://arxiv.org/abs/1607.08378
[50] https://ieeexplore.ieee.org/document/10182466/
[51] https://ieeexplore.ieee.org/document/9431773/
[52] https://arxiv.org/abs/1910.09798v1
[53] https://arxiv.org/pdf/2006.15343.pdf
[54] https://arxiv.org/pdf/1801.03329.pdf
[55] https://arxiv.org/pdf/2201.08815.pdf
[56] https://www.mdpi.com/2076-3417/11/17/7839/pdf
[57] https://arxiv.org/pdf/1811.11507.pdf
[58] https://arxiv.org/html/2501.07008v1
[59] https://arxiv.org/pdf/2012.07403.pdf
[60] https://arxiv.org/abs/2301.06957
[61] https://journal.umy.ac.id/index.php/jrc/article/download/20135/8919
[62] https://pareto.ai/blog/one-shot-learning
[63] https://www.sciencedirect.com/science/article/abs/pii/S0031320323000821
[64] https://kalelpark.tistory.com/53
[65] https://www.baeldung.com/cs/image-recognition-one-shot-learning
[66] https://arxiv.org/pdf/2402.14951.pdf
[67] https://arxiv.org/html/2410.20482v1
[68] http://arxiv.org/pdf/2406.02745.pdf
[69] https://arxiv.org/pdf/2207.06196.pdf
[70] http://arxiv.org/pdf/2310.10616.pdf
[71] https://arxiv.org/html/2405.19307v1
[72] https://arxiv.org/html/2311.08360v3
[73] https://arxiv.org/html/2502.14795v2
[74] https://arxiv.org/html/2409.15963
[75] https://arxiv.org/html/2502.02869
[76] https://dl.acm.org/doi/pdf/10.1145/3546036
[77] https://arxiv.org/pdf/2103.06254.pdf
[78] https://github.com/brendenlake/omniglot
[79] https://github.com/matbambbang/siamese-networks-omniglot-pytorch
[80] https://scholar.google.com/citations?user=iBeDoRAAAAAJ&hl=ko
[81] https://www.sciencedirect.com/science/article/abs/pii/S2352154619300051
