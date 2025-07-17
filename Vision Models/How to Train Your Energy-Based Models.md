# How to Train Your Energy-Based Models

## 핵심 주장과 주요 기여

이 논문은 Energy-Based Models (EBMs) 훈련의 **세 가지 주요 접근법**을 체계적으로 분석하고 비교하여, EBM 훈련의 이론적 기반과 실용적 방법론을 제공합니다[1][2].

**주요 기여:**
- **통합된 관점 제시**: MCMC 기반 최대우도 추정, Score Matching, Noise Contrastive Estimation 간의 이론적 연결성 규명[2][3]
- **실용적 가이드라인**: 각 방법의 장단점과 적용 상황에 대한 명확한 지침 제공[4][5]
- **최신 연구 동향 종합**: 2021년 시점의 EBM 훈련 기법들을 포괄적으로 정리[2][3]

## 해결하고자 하는 문제

### 1. 정규화 상수의 비계산성 문제

EBM의 핵심 문제는 **분배 함수(partition function) $$Z_θ$$의 계산 불가능성**입니다[1][2]:

$$p_θ(x) = \frac{\exp(-E_θ(x))}{Z_θ}$$

여기서 $$Z_θ = \int \exp(-E_θ(x)) dx$$는 일반적으로 계산이 불가능합니다[4][6].

### 2. 기존 생성 모델의 제약

**기존 모델의 한계:**
- 자기회귀 모델: 조건부 분포의 곱으로 인한 구조적 제약[1]
- 흐름 기반 모델: 가역 변환의 요구사항[1]
- VAE: 방향성 잠재변수 모델의 제약[1]

EBM은 이러한 제약 없이 **임의의 비선형 회귀 함수**로 에너지 함수를 매개변수화할 수 있어 더 표현력 있는 확률 분포를 모델링할 수 있습니다[1][2].

## 제안하는 세 가지 훈련 방법

### 1. MCMC 기반 최대우도 추정 (MLE with MCMC)

**수식적 정의:**
로그 확률의 그래디언트는 다음과 같이 분해됩니다[1]:

$$\nabla_θ \log p_θ(x) = -\nabla_θ E_θ(x) - \nabla_θ \log Z_θ$$

여기서 두 번째 항은 다음과 같이 근사할 수 있습니다[1]:

$$\nabla_θ \log Z_θ = \mathbb{E}_{x \sim p_θ(x)} [-\nabla_θ E_θ(x)]$$

**Langevin MCMC 샘플링:**
$$x_{k+1} \leftarrow x_k + \frac{ε^2}{2} \nabla_x \log p_θ(x_k) + ε z_k$$

여기서 $$\nabla_x \log p_θ(x) = -\nabla_x E_θ(x)$$입니다[1][7].

**한계:** 
- MCMC 수렴까지 계산 비용 높음[7][8]
- Contrastive Divergence의 편향된 그래디언트 추정[9][10]

### 2. Score Matching (SM)

**기본 아이디어:**
두 분포의 **score function**(로그 확률 밀도의 그래디언트)이 같으면 분포가 같다는 원리 활용[1][11]:

$$\nabla_x \log p_θ(x) = -\nabla_x E_θ(x)$$

**Fisher Divergence 최소화:**

$$D_F(p_{data}(x) \| p_θ(x)) = \mathbb{E}\_{p_{data}(x)} \left[\frac{1}{2} \|\nabla_x \log p_{data}(x) - \nabla_x \log p_θ(x)\|^2\right]$$

**실용적 형태 (Integration by Parts):**

$$D_F(p_{data}(x) \| p_θ(x)) = \mathbb{E}\_{p_{data}(x)} \left[\frac{1}{2} \sum_{i=1}^d \left(\frac{\partial E_θ(x)}{\partial x_i}\right)^2 + \frac{\partial^2 E_θ(x)}{(\partial x_i)^2}\right]$$

#### 2.1 Denoising Score Matching (DSM)

**노이즈 추가 접근법:**

$$\tilde{x} = x + ε$$

**DSM 목적함수:**

$$D_F(q(\tilde{x}) \| p_θ(\tilde{x})) = \mathbb{E}_{q(x,\tilde{x})} \left[\frac{1}{2} \|\nabla_x \log q(\tilde{x}|x) - \nabla_x \log p_θ(\tilde{x})\|^2\right]$$

**장점:** 2차 도함수 계산 불필요[12][13]
**단점:** 노이즈 분포와 원래 분포의 불일치[1][11]

#### 2.2 Sliced Score Matching (SSM)

**투영 기반 접근법:**

$$D_{SF}(p_{data}(x) \| p_θ(x)) = \mathbb{E}\_{p_{data}(x)} \mathbb{E}\_{p(v)} \left[\frac{1}{2}(v^T \nabla_x \log p_{data}(x) - v^T \nabla_x \log p_θ(x))^2\right]$$

**효율적 계산:**

$$\sum_{i=1}^d \sum_{j=1}^d \frac{\partial^2 E_θ(x)}{\partial x_i \partial x_j} v_i v_j = \sum_{i=1}^d \frac{\partial}{\partial x_i} \left(\sum_{j=1}^d \frac{\partial E_θ(x)}{\partial x_j} v_j\right) v_i$$

이는 $$O(d^2)$$ 대신 $$O(d)$$ 계산으로 가능합니다[1].

### 3. Noise Contrastive Estimation (NCE)

**이진 분류 문제로 변환:**
데이터와 노이즈를 구별하는 분류 문제로 밀도 추정을 근사[1][14]:

$$p_{n,θ}(y=0|x) = \frac{p_n(x)}{p_n(x) + νp_θ(x)}$$

**NCE 목적함수:**

$$θ^* = \arg\max_θ \mathbb{E}\_{p_{n,data}(x,y)}[\log p_{n,θ}(y|x)]$$

**장점:** 
- 정규화 상수를 학습 가능한 매개변수로 처리[1][15]
- 적절한 노이즈 분포 선택 시 효과적[15][16]

## 모델 구조와 아키텍처

### 에너지 함수 설계

**일반적 형태:**
$$E_θ(x) = \text{CNN}(x; θ)$$

**핵심 특징:**
- **아키텍처 자유도**: 임의의 신경망 구조 사용 가능[1][17]
- **도메인 특화**: 그래프 신경망(분자), 구면 CNN(구면 이미지) 등 적용 가능[1]
- **정규화 기법**: Spectral Normalization으로 Lipschitz 상수 제한[18]

### Score-Based 생성 모델

**다중 노이즈 스케일 접근법:**
서로 다른 노이즈 레벨에서 score 함수 학습[1][19]:

$$p_{σ_i}(x) = \int p(x_0) \mathcal{N}(x; x_0, σ_i^2 I) dx_0$$

**Noise-Conditional Score Network:**
$$s_θ(x, σ) ≈ \nabla_x \log p_{σ}(x)$$

## 성능 향상 메커니즘

### 1. 일반화 성능 개선

**표현력 향상:**
- 정규화 상수 제약 없는 모델링[1][2]
- 복잡한 다중 모달 분포 학습 가능[20][19]

**모드 붕괴 방지:**
- Score Matching의 다중 노이즈 스케일 기법[1][19]
- 큰 노이즈에서 작은 노이즈로의 점진적 샘플링[19]

### 2. 계산 효율성

**MCMC 개선:**
- Persistent Contrastive Divergence[1][21]
- 리플레이 버퍼 활용[18]
- 향상된 그래디언트 추정[10][22]

**Score Matching 최적화:**
- SSM의 선형 복잡도 계산[1]
- 분산 감소 기법 (Control Variates)[1]

## 한계 및 도전 과제

### 1. 계산상의 한계

**MCMC 기반 방법:**
- 느린 수렴 속도[7][8]
- 높은 차원에서의 혼합 문제[23][24]

**Score Matching:**
- 분리된 모드 간 가중치 학습 어려움[1][19]
- 고차원에서의 분산 증가[1]

### 2. 이론적 한계

**일관성 문제:**
- DSM의 노이즈 분포 의존성[1][11]
- Contrastive Divergence의 편향[9][10]

**최적화 안정성:**
- 에너지 함수의 급격한 변화[18]
- 훈련 불안정성[10][22]

## 향후 연구 방향과 영향

### 1. 연구에 미치는 영향

**확산 모델 연구:**
- Score-based 방법이 확산 모델 발전에 직접적 영향[25][19]
- Denoising Score Matching의 확산 모델 적용[25][11]

**하이브리드 접근법:**
- EBM과 다른 생성 모델의 결합[26][27]
- 대조 학습과의 통합[28]

### 2. 향후 고려사항

**효율적 샘플링:**
- 비MCMC 기반 훈련 방법 개발[29][27]
- 신경망 기반 적응적 샘플러[24]

**이론적 발전:**
- 수렴성 보장 알고리즘[23]
- 편향 제거 기법[7][10]

**실용적 응용:**
- 대규모 데이터에서의 확장성[20][30]
- 다양한 도메인 적용[31][32]

이 논문은 EBM 훈련의 이론적 기반을 확립하고 실용적 지침을 제공함으로써, 현대 생성 모델링 연구의 중요한 이정표가 되었습니다. 특히 Score-based 방법론은 이후 확산 모델 발전의 핵심 기반이 되어 생성 AI 발전에 지대한 영향을 미쳤습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d1b99e03-7a70-4ae7-bdc4-f0ba7c1f756e/2101.03288v2.pdf
[2] https://www.semanticscholar.org/paper/5d556dd3afff529de8cb694f88916b2d95fbdd3a
[3] https://arxiv.org/abs/2101.03288
[4] https://arxiv.org/pdf/2101.03288.pdf
[5] https://letter-night.tistory.com/218
[6] https://deep-learning-study-note.readthedocs.io/en/latest/Part%203%20(Deep%20Learning%20Research)/18%20Confronting%20the%20Partition%20Function/
[7] https://arxiv.org/abs/2312.02469
[8] https://arxiv.org/abs/2307.01668
[9] https://www.cs.toronto.edu/~hinton/absps/cdmiguel.pdf
[10] https://arxiv.org/abs/2012.01316
[11] https://johfischer.com/2022/09/18/denoising-score-matching/
[12] https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf
[13] https://www.activeloop.ai/resources/glossary/denoising-score-matching/
[14] https://deepai.org/machine-learning-glossary-and-terms/noise-contrastive-estimation
[15] https://deep-learning-study-note.readthedocs.io/en/latest/Part%203%20(Deep%20Learning%20Research)/18%20Confronting%20the%20Partition%20Function/18.6%20Noise-Contrastive%20Estimation.html
[16] https://arxiv.org/abs/1806.03664
[17] https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1114651/full
[18] https://ostin.tistory.com/354
[19] https://yang-song.net/blog/2021/score/
[20] https://www.semanticscholar.org/paper/a4727b807ae052c9dcddcf424a5d233bcc3c5a9e
[21] https://www.activeloop.ai/resources/glossary/persistent-contrastive-divergence/
[22] https://openreview.net/forum?id=daLIpc7vQ2q
[23] https://arxiv.org/abs/2310.12667
[24] https://pmc.ncbi.nlm.nih.gov/articles/PMC7996279/
[25] https://milvus.io/ai-quick-reference/how-does-denoising-score-matching-fit-into-diffusion-modeling
[26] https://arxiv.org/abs/2311.04071
[27] https://openreview.net/forum?id=46tjvA75h6
[28] https://openreview.net/forum?id=CZmHHj9MgkP
[29] https://arxiv.org/abs/2503.07021
[30] https://ieeexplore.ieee.org/document/10204299/
[31] https://www.semanticscholar.org/paper/efe9c1bf2fcd13da5e1ed1d7b03d034b2eadd49d
[32] https://aclanthology.org/2021.eacl-main.151
[33] https://ieeexplore.ieee.org/document/9596559/
[34] https://www.semanticscholar.org/paper/de579b0d090ac40dd638742b3a89260150495721
[35] https://www.semanticscholar.org/paper/6ddc1645883af317dbab87569df6e72822e0fc15
[36] https://www.semanticscholar.org/paper/dee64637647ec9c158bd0cb2eb5f2e57b188dbbc
[37] http://proceedings.mlr.press/v139/du21b/du21b.pdf
[38] https://sites.google.com/view/ebm-workshop-iclr2021
[39] https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss
[40] https://openreview.net/forum?id=hUGfSKeh1y
[41] http://www.cedar.buffalo.edu/~srihari/CSE676/18.6%20Noise%20Contrastive%20Estimation.pdf
[42] https://linkinghub.elsevier.com/retrieve/pii/S0196890422007889
[43] https://linkinghub.elsevier.com/retrieve/pii/S0360544221002073
[44] https://linkinghub.elsevier.com/retrieve/pii/S0360544218305577
[45] https://ieeexplore.ieee.org/document/10104114/
[46] https://www.frontiersin.org/articles/10.3389/fenrg.2023.1101049/full
[47] https://ieeexplore.ieee.org/document/9779444/
[48] https://ieeexplore.ieee.org/document/10693505/
[49] https://www.frontiersin.org/articles/10.3389/fenrg.2023.1338195/full
[50] https://andrewcharlesjones.github.io/journal/21-score-matching.html
[51] https://www.linkedin.com/pulse/limitation-generative-ai-models-non-probabilistic-abhay-gupta-ph-d--pdw2c
[52] https://pmc.ncbi.nlm.nih.gov/articles/PMC10020340/
[53] https://stats.stackexchange.com/questions/626021/normalizing-constant-calculation-of-strauss-process
[54] https://proceedings.neurips.cc/paper/2021/file/eae15aabaa768ae4a5993a8a4f4fa6e4-Paper.pdf
[55] https://proceedings.neurips.cc/paper_files/paper/2023/file/8e176ef071f00f1b233461c5ad5e1b24-Paper-Conference.pdf
[56] https://arxiv.org/html/2402.00759v3
[57] https://sambaiga.github.io/talk/DNN-EBM2020.pdf
[58] https://arxiv.org/abs/2501.18528
[59] https://betanalpha.github.io/assets/case_studies/generative_modeling.html
[60] https://proceedings.neurips.cc/paper/8619-implicit-generation-and-modeling-with-energy-based-models.pdf
[61] https://www.sciencedirect.com/science/article/pii/S0004370219301948
[62] https://arxiv.org/pdf/2501.18528.pdf
[63] https://www.semanticscholar.org/paper/6d9a3d849928c1e569a5fc10ff72241edfe42b15
[64] https://www.semanticscholar.org/paper/d966edcc545f2b6a8ee2403da237eafc2330e048
[65] https://ocw.snu.ac.kr/sites/default/files/NOTE/week10(MCMC,%20Bolzman%20Machine).pdf
[66] https://papers.neurips.cc/paper_files/paper/2022/file/3e25d1aff47964c8409fd5c8dc0438d7-Paper-Conference.pdf
[67] https://courses.cs.washington.edu/courses/cse599i/20au/resources/L16_ebm.pdf
[68] http://proceedings.mlr.press/v9/sutskever10a/sutskever10a.pdf
[69] https://mathoverflow.net/questions/436603/training-an-energy-based-model-ebm-using-mcmc
[70] https://openreview.net/forum?id=7962B4nXX7
[71] https://bsi-ni.brain.riken.jp/database/file/345/345.pdf
[72] https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
[73] https://uvadl2c.github.io/lectures/Advanced%20Generative%20&%20Energy-based%20Models/modern-based-models/lecture%204.2.pdf
[74] https://www.semanticscholar.org/paper/7c2de353f3151a0f88736af9be354ab5e773bff0
[75] https://direct.mit.edu/books/book/3826/chapter/125508/Energy-Based-Models
[76] https://aclanthology.org/2021.eacl-main.151.pdf
[77] https://arxiv.org/html/2310.12667
[78] https://arxiv.org/html/2407.15238v1
[79] http://arxiv.org/pdf/2304.10707.pdf
[80] https://arxiv.org/pdf/2303.04343.pdf
[81] https://arxiv.org/pdf/2309.05803.pdf
[82] https://arxiv.org/pdf/2406.13661.pdf
[83] https://arxiv.org/pdf/1903.08689.pdf
[84] http://arxiv.org/pdf/2111.13772.pdf
[85] https://arxiv.org/html/2501.12667v1
[86] https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-Python-NCE%EB%9E%80Noise-Contrastive-Estimation
[87] https://www.sciencedirect.com/science/article/abs/pii/S0893608025001790
[88] https://arxiv.org/html/2502.00336v1
[89] https://arxiv.org/abs/2406.04043
[90] https://ieeexplore.ieee.org/document/10472171/
[91] http://arxiv.org/pdf/1711.03130.pdf
[92] https://arxiv.org/pdf/2104.07531.pdf
[93] https://arxiv.org/pdf/1910.02720.pdf
[94] http://arxiv.org/pdf/2304.06094.pdf
[95] https://arxiv.org/pdf/1909.06878.pdf
[96] https://arxiv.org/pdf/2006.06897.pdf
[97] https://arxiv.org/pdf/1901.08508.pdf
[98] https://arxiv.org/pdf/2109.03237.pdf
[99] https://arxiv.org/pdf/2011.12216.pdf
[100] https://en.wikipedia.org/wiki/Normalizing_constant
[101] https://arxiv.org/html/2402.01103v3
[102] https://janghan-kor.tistory.com/1859
[103] https://www.ijcai.org/proceedings/2021/0587.pdf
[104] https://ojs.aaai.org/index.php/AAAI/article/view/17250
[105] https://academic.oup.com/gji/article/232/3/1957/6783161
[106] https://arxiv.org/html/2405.11179v1
[107] http://arxiv.org/pdf/2312.02469.pdf
[108] https://arxiv.org/html/2307.01668
[109] http://arxiv.org/pdf/1908.03491.pdf
[110] http://arxiv.org/pdf/2406.02490.pdf
[111] https://arxiv.org/pdf/1808.09095.pdf
[112] https://arxiv.org/pdf/2310.11232.pdf
[113] https://paperswithcode.com/paper/training-energy-based-models-with-diffusion
