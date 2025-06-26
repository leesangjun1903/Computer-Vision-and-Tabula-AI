# EfficientFormer: Vision Transformers at MobileNet Speed - 모바일 기기에서의 고속 Vision Transformer 연구

## 개요

본 보고서는 2022년 Snap Inc.와 Northeastern University에서 발표한 "EfficientFormer: Vision Transformers at MobileNet Speed" 논문에 대한 종합적인 분석을 제공합니다[1][2][3]. 이 연구는 Vision Transformer(ViT) 모델이 MobileNet과 같은 경량 CNN과 동등한 속도로 모바일 기기에서 동작할 수 있는지에 대한 근본적인 질문에서 출발하여, 효율적인 transformer 아키텍처 설계를 통해 이를 실현했습니다.

## 해결하고자 하는 문제

### 핵심 문제 정의

Vision Transformer는 컴퓨터 비전 분야에서 뛰어난 성능을 보여주었지만, 몇 가지 심각한 제약사항이 있었습니다[1][4][5]. 첫째, ViT 기반 모델들은 막대한 수의 매개변수와 attention mechanism으로 인해 경량 합성곱 신경망보다 일반적으로 몇 배 느린 추론 속도를 보였습니다. 둘째, 이러한 높은 계산 복잡도는 모바일 기기와 같은 자원 제약적 하드웨어에서의 실시간 응용 프로그램 배포를 특히 어렵게 만들었습니다.

### 기존 접근법의 한계

기존 연구들은 network architecture search나 MobileNet 블록과의 hybrid 설계를 통해 ViT의 계산 복잡도를 줄이려고 시도했지만, 여전히 만족스럽지 못한 추론 속도를 보였습니다[1][2][6]. 특히 MobileViT와 같은 hybrid 모델들도 lightweight CNN들보다 현저히 느린 성능을 보였으며, 단순히 MHSA와 MobileNet 블록을 교환하는 것만으로는 파레토 곡선을 개선하기 어려웠습니다.

## 제안하는 방법

### 체계적인 지연시간 분석

연구진은 iPhone 12와 CoreML 컴파일러를 활용하여 ViT 기반 모델들의 on-device 지연시간에 대한 포괄적인 분석을 수행했습니다[1][7]. 이 분석을 통해 네 가지 핵심 관찰사항을 도출했습니다.

**관찰 1: 패치 임베딩의 비효율성**
큰 커널과 stride를 갖는 patch embedding이 모바일 기기에서 속도 병목현상을 일으킨다는 것을 발견했습니다. 대부분의 컴파일러가 large kernel convolution을 지원하지 않고 Winograd 같은 기존 가속화 알고리즘으로부터 가속화되지 않기 때문입니다.

**관찰 2: 차원 일관성의 중요성**
token mixer 선택에 있어 일관된 특성 차원이 중요하며, Multi-Head Self Attention(MHSA)이 반드시 속도 병목현상은 아니라는 것을 확인했습니다. 특히 4D tensor와 3D tensor 간의 빈번한 reshape 연산이 주요 지연요인임을 발견했습니다.

**관찰 3: 정규화 기법의 영향**
CONV-BN이 LN(Layer Normalization)-Linear보다 지연시간 측면에서 유리하며, 정확도 하락은 일반적으로 수용 가능한 수준임을 확인했습니다.

**관찰 4: 활성화 함수의 하드웨어 의존성**
비선형성의 지연시간은 하드웨어와 컴파일러에 따라 달라지며, 특정 환경에서는 GeLU가 ReLU와 유사한 성능을 보일 수 있음을 발견했습니다.

### Dimension-Consistent Design 패러다임

이러한 분석을 바탕으로, 연구진은 네트워크를 4D partition과 3D partition으로 나누는 dimension-consistent design을 제안했습니다[1][8][7].

**4D Partition (MB4D)**
초기 단계에서는 CONV-net 스타일로 구현된 연산자들을 사용하여 저수준 특성을 추출합니다. 여기서 token mixer로는 pooling을 사용하며, 다음과 같은 수식으로 표현됩니다:

$$I_i = \text{Pool}(X_i^{B,C_j,\frac{H}{2^{j+1}},\frac{W}{2^{j+1}}}) + X_i^{B,C_j,\frac{H}{2^{j+1}},\frac{W}{2^{j+1}}}$$

$$X_{i+1}^{B,C_j,\frac{H}{2^{j+1}},\frac{W}{2^{j+1}}} = \text{Conv}_{B,G}(\text{Conv}_{B,G}(I_i)) + I_i$$

**3D Partition (MB3D)**
마지막 단계에서는 3D tensor에 대해 linear projection과 attention을 수행하여 MHSA의 전역적 모델링 능력을 활용합니다:

$$I_i = \text{Linear}(\text{MHSA}(\text{Linear}(\text{LN}(X_i^{B,\frac{HW}{4^{j+1}},C_j})))) + X_i^{B,\frac{HW}{4^{j+1}},C_j}$$

$$X_{i+1}^{B,\frac{HW}{4^{j+1}},C_j} = \text{Linear}(\text{Linear}_G(\text{LN}(I_i))) + I_i$$

여기서 MHSA는 다음과 같이 정의됩니다:

$$\text{MHSA}(Q,K,V) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{C_j}} + b\right) \cdot V$$

### Latency-Driven Slimming 알고리즘

효율적인 모델을 얻기 위해 새로운 지연시간 기반 슬리밍 방법을 제안했습니다[1][7]. 이 방법은 세 가지 단계로 구성됩니다:

**1단계: Supernet 사전 훈련**
Gumbel Softmax sampling을 사용하여 각 블록의 중요도 점수를 획득합니다:

$$X_{i+1} = \sum_n \frac{e^{(\alpha_i^n + \epsilon_i^n)/\tau}}{\sum_n e^{(\alpha_i^n + \epsilon_i^n)/\tau}} \cdot \text{MP}_{i,j}(X_i)$$

**2단계: 지연시간 룩업 테이블 구축**
다양한 너비(16의 배수)를 가진 MB4D와 MB3D의 on-device 지연시간을 수집합니다.

**3단계: 네트워크 슬리밍**
중요도 점수를 기반으로 다음 세 가지 동작 중에서 선택합니다:
- Depth Reduction (DR): 가장 중요하지 않은 MetaPath에 대해 Identity 선택
- Width Reduction (WR): 가장 중요하지 않은 Stage의 너비 감소
- MB3D Reduction (MR): 첫 번째 MB3D 제거

각 동작은 정확도 하락 대비 지연시간 개선 비율(-Δ%/Δms)을 기준으로 선택됩니다.

## 모델 구조

### 전체 아키텍처

EfficientFormer는 patch embedding과 여러 meta transformer block(MB)의 스택으로 구성됩니다[1]:

$$Y = \prod_{i}^{m} \text{MB}_i(\text{PatchEmbed}(X_0^{B,3,H,W}))$$

여기서 $$X_0$$은 배치 크기 B와 공간 크기 [H,W]를 가진 입력 이미지이고, Y는 원하는 출력, m은 총 블록 수(깊이)입니다.

각 MetaBlock은 다음과 같이 표현됩니다:

$$X_{i+1} = \text{MB}_i(X_i) = \text{MLP}(\text{TokenMixer}(X_i))$$

### Stage별 구성

네트워크는 4개의 Stage로 구성되며, 각 Stage는 동일한 공간 크기를 처리하는 여러 MetaBlock으로 구성됩니다:

**Stage 1 & 2**: MB4D 블록만 사용하여 저수준 특성 추출
**Stage 3 & 4**: MB4D와 MB3D 블록을 선택적으로 사용하여 고수준 특성 추출

### 구체적인 모델 변형

**EfficientFormer-L1**: 12.3M 매개변수, 1.3 GMACs
**EfficientFormer-L3**: 31.3M 매개변수, 3.9 GMACs  
**EfficientFormer-L7**: 82.1M 매개변수, 10.2 GMACs

## 성능 향상

### ImageNet-1K 분류 성능

EfficientFormer는 기존 모델들을 크게 앞서는 성능을 보였습니다[1][3][9]:

**CNN과의 비교**
- EfficientFormer-L1: iPhone Neural Engine에서 MobileNetV2×1.4와 동일한 1.6ms 지연시간으로 79.2% top-1 정확도 달성 (MobileNetV2×1.4: 74.7%)
- EfficientFormer-L3: EfficientNet-B0와 유사한 속도로 82.4% 정확도 달성 (EfficientNet-B0: 77.1%)

**ViT와의 비교**
- EfficientFormer-L3가 DeiT-Small보다 높은 정확도(82.4% vs 81.2%)를 4배 빠른 속도로 달성
- PoolFormer-S36보다 1% 높은 정확도를 3배 빠른 속도로 달성

**Hybrid 모델과의 비교**
- EfficientFormer-L1이 MobileViT-XS보다 4.4% 높은 정확도를 다양한 하드웨어에서 빠른 속도로 달성
- 유사한 추론 시간에서 EfficientFormer-L7이 MobileViT-XS보다 8.5% 높은 정확도 달성

### 다운스트림 태스크 성능

**객체 탐지 및 인스턴스 분할**
COCO 2017에서 Mask-RCNN을 사용한 실험 결과[1]:
- EfficientFormer-L3가 ResNet50 백본보다 3.4 box AP, 3.7 mask AP 향상
- PoolFormer-S24 백본보다 1.3 box AP, 1.1 mask AP 향상

**의미적 분할**
ADE20K 데이터셋에서 Semantic FPN을 사용한 실험 결과[1]:
- EfficientFormer-L3가 PoolFormer-S24보다 3.2 mIoU 향상
- 전역적 attention을 통해 장거리 의존성을 더 잘 학습하여 고해상도 밀집 예측 태스크에서 유리

## 일반화 성능 향상 가능성

### 아키텍처 설계의 일반성

EfficientFormer의 dimension-consistent design과 4D 블록의 CONV-BN fusion 같은 대부분의 설계는 일반적인 목적으로 사용 가능합니다[1][7]. 이러한 설계 원칙들은 다른 플랫폼에서도 적용 가능하며, hardware-friendly한 특성을 가지고 있어 다양한 환경에서의 일반화를 지원합니다.

### 다양한 응용 분야에서의 검증

논문에서 제시된 실험 결과들은 EfficientFormer가 단순히 ImageNet 분류에서만 우수한 것이 아니라, 객체 탐지, 인스턴스 분할, 의미적 분할 등 다양한 컴퓨터 비전 태스크에서 일관되게 좋은 성능을 보임을 증명했습니다[1]. 이는 모델의 일반화 능력이 뛰어남을 시사합니다.

### 전이 학습 잠재력

후속 연구들에서 EfficientFormer가 다양한 의료 영상 분야에서 활용되고 있는 것을 확인할 수 있습니다[10][11]. 특히 뇌종양 검출에서 99.61%의 높은 정확도를 달성한 사례나, 알츠하이머병 진단에서의 활용 사례는 EfficientFormer의 전이 학습 능력과 일반화 성능을 보여줍니다.

### 효율성과 성능의 균형

EfficientFormer는 모바일 기기에서의 효율성을 추구하면서도 성능 저하를 최소화했습니다[1][12]. 이러한 균형은 다양한 하드웨어 환경과 응용 분야에서 모델을 적용할 때 중요한 요소이며, 실제 배포 환경에서의 일반화 성능을 보장하는 핵심 요소입니다.

## 한계

### 플랫폼별 최적화의 필요성

EfficientFormer의 실제 속도는 다른 플랫폼에서 달라질 수 있습니다[1][7]. 예를 들어, 특정 하드웨어에서 GeLU가 잘 지원되지 않고 HardSwish가 효율적으로 구현되는 경우, 활성화 함수를 그에 맞게 수정해야 할 필요가 있습니다.

### 탐색 알고리즘의 단순성

제안된 latency-driven slimming은 단순하고 빠르지만, 탐색 비용을 고려하지 않는다면 열거 기반의 무차별 탐색을 통해 더 나은 결과를 얻을 수 있을 것입니다[1][7].

### 메모리 효율성 고려사항

장문맥 transformer 배포에서 나타나는 것처럼, KV cache의 크기가 메모리 효율성에 미치는 영향을 고려할 때[13], EfficientFormer 역시 더 긴 시퀀스나 더 큰 입력에 대해서는 메모리 최적화가 필요할 수 있습니다.

## 미래 연구에 대한 영향

### 효율적인 ViT 설계의 새로운 패러다임

EfficientFormer는 효율적인 Vision Transformer 설계에 대한 새로운 패러다임을 제시했습니다[1][2][14]. dimension-consistent design과 latency-driven optimization은 후속 연구들의 기반이 되고 있으며, EfficientFormerV2[12][15]와 같은 발전된 모델들의 토대를 마련했습니다.

### 모바일 AI의 발전 가속화

이 연구는 모바일 기기에서 transformer 모델의 실용적 배포 가능성을 증명함으로써[1][16][17], 모바일 AI 응용 프로그램의 발전을 크게 가속화했습니다. 특히 실시간 이미지 분석, 증강현실, 모바일 컴퓨터 비전 응용 분야에서 새로운 가능성을 열었습니다.

### Neural Architecture Search의 진화

EfficientFormer의 latency-driven slimming 방법론은 하드웨어 인식 NAS(Neural Architecture Search) 분야에 새로운 방향을 제시했습니다[18][7]. 실제 지연시간을 직접 최적화하는 접근법은 기존의 FLOPs나 매개변수 수 기반 최적화보다 실용적인 결과를 제공합니다.

### 엣지 컴퓨팅과 분산 추론

ED-ViT[19]와 같은 후속 연구들이 보여주듯이, EfficientFormer의 설계 원칙은 엣지 기기에서의 분산 추론으로 확장되고 있습니다. 이는 IoT 환경과 엣지 컴퓨팅에서 transformer 모델의 활용 가능성을 크게 확장시켰습니다.

## 앞으로 연구 시 고려할 점

### 하드웨어별 최적화 전략

미래 연구에서는 다양한 하드웨어 플랫폼(GPU, TPU, FPGA 등)에 특화된 최적화 전략을 개발해야 합니다[14][4]. 각 플랫폼의 특성을 고려한 아키텍처 수정과 컴파일러 최적화가 필요합니다.

### 메모리 효율성과 확장성

장문맥 처리나 고해상도 이미지 처리를 위한 메모리 효율적인 attention mechanism 개발이 중요합니다[13]. KV cache 최적화, gradient checkpointing, 그리고 mixed precision 훈련 등의 기법을 통합한 종합적인 접근이 필요합니다.

### 자동화된 설계 공간 탐색

더 정교한 자동화된 아키텍처 탐색 방법론 개발이 필요합니다[18][20]. 특히 다중 목적 최적화(정확도, 지연시간, 에너지 효율성, 메모리 사용량)를 동시에 고려하는 탐색 알고리즘이 중요합니다.

### 일반화와 전이 학습 능력 향상

도메인 적응과 few-shot 학습 능력을 강화하는 방향으로 연구가 진행되어야 합니다[21]. 특히 의료, 자율주행, 산업 검사 등 특수 도메인에서의 효율적인 전이 학습 방법론 개발이 중요합니다.

### 지속 가능한 AI 개발

에너지 효율성과 탄소 발자국을 고려한 지속 가능한 AI 모델 설계가 중요해지고 있습니다[22][23]. 모델의 전체 라이프사이클(훈련, 배포, 추론, 업데이트)에서의 환경 영향을 고려한 설계 방법론이 필요합니다.

EfficientFormer는 Vision Transformer의 실용적 배포 가능성을 보여준 혁신적인 연구로, 모바일 AI의 새로운 시대를 열었습니다. 이 연구가 제시한 설계 원칙과 최적화 방법론은 앞으로도 효율적인 transformer 모델 개발의 핵심 기반이 될 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3f6a6571-b098-4b70-ad42-86a282e67bb1/2206.01191v5.pdf
[2] https://arxiv.org/abs/2206.01191
[3] https://papers.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf
[4] https://ojs.aaai.org/index.php/AAAI/article/view/33759
[5] https://openreview.net/pdf?id=NXHXoYMLIG
[6] https://ieeexplore.ieee.org/document/10959660/
[7] https://proceedings.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Supplemental-Conference.pdf
[8] https://arxiv.org/pdf/2206.01191.pdf
[9] https://proceedings.neurips.cc/paper_files/paper/2022/file/5452ad8ee6ea6e7dc41db1cbd31ba0b8-Paper-Conference.pdf
[10] https://ieeexplore.ieee.org/document/10691343/
[11] https://www.mdpi.com/2227-7390/13/12/1927
[12] https://ieeexplore.ieee.org/document/10377927/
[13] https://arxiv.org/html/2405.08944v1
[14] https://arxiv.org/html/2503.02891v1
[15] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Rethinking_Vision_Transformers_for_MobileNet_Size_and_Speed_ICCV_2023_paper.pdf
[16] https://github.com/snap-research/EfficientFormer
[17] https://arxiv.org/abs/2305.19365
[18] https://ieeexplore.ieee.org/document/10376989/
[19] https://arxiv.org/html/2410.11650v1
[20] https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf
[21] https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/68695/1/Domain-Adaptive%20Vision%20Transformers%20for%20Generalizing%20Across%20Visual%20Domains.pdf
[22] https://arxiv.org/html/2502.16627v2
[23] https://www.themoonlight.io/en/review/optimization-strategies-for-enhancing-resource-efficiency-in-transformers-large-language-models
[24] https://www.ewadirect.com/proceedings/ace/article/view/9964
[25] https://journals.sagepub.com/doi/full/10.3233/JIFS-231440
[26] https://www.codenary.co.kr/discoveries/156
[27] https://da2so.tistory.com/53
[28] https://arxiv.org/html/2502.05800v1
[29] https://arxiv.org/abs/2411.06119
[30] https://www.frontiersin.org/articles/10.3389/fpls.2023.1256773/full
[31] https://www.mdpi.com/1424-8220/24/4/1331
[32] https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710294.pdf
[33] https://proceedings.mlr.press/v201/duman-keles23a/duman-keles23a.pdf
[34] https://stackoverflow.com/questions/65703260/computational-complexity-of-self-attention-in-the-transformer-model
[35] https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-498
[36] https://www.aclweb.org/anthology/2020.sustainlp-1.20.pdf
[37] https://arxiv.org/pdf/2212.08059.pdf
[38] https://arxiv.org/pdf/2210.06659.pdf
[39] https://www.youtube.com/watch?v=AipN0tbJj4o
[40] https://cvpr.thecvf.com/virtual/2023/poster/22623
[41] https://arxiv.org/pdf/2210.13452.pdf
[42] https://ieeexplore.ieee.org/document/9927347/
[43] https://aclanthology.org/2021.eacl-main.113
[44] https://www.mdpi.com/2076-3417/10/21/7817
[45] https://www.jstage.jst.go.jp/article/mrms/20/2/20_mp.2019-0199/_article
[46] https://ajmc.aut.ac.ir/article_5213.html
[47] https://arxiv.org/abs/2408.04593
[48] https://openreview.net/forum?id=NXHXoYMLIG
[49] https://openreview.net/forum?id=U49N5V51rU
[50] https://webisoft.com/articles/vision-transformer-model/
[51] https://proceedings.mlr.press/v202/li23l/li23l.pdf
[52] https://birjournal.com/index.php/bir/article/view/344
[53] https://account.jpr.winchesteruniversitypress.org/index.php/wu-j-jpr/article/view/130
[54] https://journalajarr.com/index.php/AJARR/article/view/1056
[55] https://iptek.its.ac.id/index.php/ijmeir/article/view/21475
[56] https://jurnal.usk.ac.id/JAROE/article/view/36135
[57] https://www.emerald.com/insight/content/doi/10.1108/MEDAR-11-2023-2229/full/html
[58] https://www.tdworld.com/transmission-reliability/article/55238823/mobile-transformer-brings-stability-to-instability
[59] https://arxiv.org/abs/2502.07417
[60] https://arxiv.org/abs/2207.05501
[61] https://www.tandfonline.com/doi/full/10.1080/01431161.2023.2283904
[62] https://arxiv.org/abs/2404.19066
[63] https://velog.io/@minkyu4506/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-EfficientFormer-Vision-Transformers-at-MobileNet-Speed-%EB%A6%AC%EB%B7%B0
[64] https://ostin.tistory.com/78
[65] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12527/2663761/Real-time-crowd-counting-via-mobile-friendly-Vision-Transformer-network/10.1117/12.2663761.full
[66] https://arxiv.org/abs/2403.20041
[67] https://ieeexplore.ieee.org/document/10690344/
[68] https://ieeexplore.ieee.org/document/10376776/
[69] https://www.sciencedirect.com/science/article/abs/pii/S0925231225010896
[70] https://tryolabs.com/blog/2022/11/24/transformer-based-model-for-faster-inference
[71] https://www.semanticscholar.org/paper/4f8bd045f1f40f7061d75bd024950b1011eaaddd
[72] https://www.semanticscholar.org/paper/b4958a003ba79cd4966dd66e71f69703ceec72ac
[73] http://link.springer.com/10.1007/s00354-019-00083-x
[74] https://www.semanticscholar.org/paper/4afd99f5d36a6687bf43724168a834d683c2ef67
[75] https://arxiv.org/abs/2402.15938
[76] https://linkinghub.elsevier.com/retrieve/pii/S2643651524002346
[77] https://huggingface.co/docs/transformers/en/model_doc/efficientformer
[78] https://www.koreascience.kr/article/JAKO202325643250869.page
[79] https://www.sciencedirect.com/science/article/pii/S0031320324008434
[80] https://2021-01-06getstarted.tistory.com/59
[81] https://link.springer.com/10.1007/s11356-024-34535-9
[82] https://link.springer.com/10.1007/s10479-023-05251-3
[83] https://gspjournals.com/ijrebs/index.php/ijrebs/article/view/67/78
[84] https://www.emerald.com/insight/content/doi/10.1108/MEDAR-01-2024-2317/full/html
[85] https://arxiv.org/html/2308.09372
[86] https://www.mdpi.com/2073-431X/14/5/171
[87] https://dl.acm.org/doi/fullHtml/10.1145/3530811
