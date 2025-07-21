# MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer | Image classification, Object detection

## 핵심 주장 및 주요 기여

**MobileViT**는 모바일 기기를 위한 경량화된 비전 트랜스포머 아키텍처로, CNN과 ViT의 장점을 결합하여 **"Transformers as Convolutions"**라는 독창적인 관점을 제시한다[1]. 이 논문은 기존 ViT의 무거운 연산량과 CNN의 공간적 지역성 한계를 동시에 해결하는 혁신적인 해결책을 제안한다[2][3].

### 주요 기여점

**1. 혁신적인 하이브리드 아키텍처**: MobileViT는 표준 convolution의 3단계 연산(unfolding, local processing, folding)에서 local processing을 transformer의 global processing으로 대체하여 CNN과 ViT의 특성을 모두 갖는 새로운 블록을 설계했다[1][3].

**2. 뛰어난 성능**: ImageNet-1k에서 약 600만 개의 파라미터로 78.4%의 top-1 정확도를 달성하여 MobileNetv3보다 3.2%, DeIT보다 6.2% 높은 성능을 보였다[2][3].

**3. 일반화 성능**: 기존 ViT 계열과 달리 기본적인 데이터 증강만으로도 우수한 성능을 달성하며, L2 정규화에 덜 민감한 특성을 보인다[1][3].

## 해결하고자 하는 문제

### 기존 방법들의 한계

**CNN의 한계**: 공간적 지역성(spatial locality)으로 인해 전역적 표현 학습이 제한적이다[1][2].

**ViT의 한계**: 
- 높은 연산 복잡도: 자기주의 메커니즘의 $$O(k^2)$$ 복잡도로 모바일 기기에 부적합[4][5]
- 과적합 문제: 광범위한 데이터 증강과 L2 정규화 필요[1]
- 이미지별 귀납적 편향(image-specific inductive bias) 부족[3]

## 제안하는 방법 및 모델 구조

### MobileViT Block 구조

MobileViT 블록은 세 가지 주요 구성요소로 이루어져 있다[1][3]:

**1. Local Representations**: 
입력 텐서 $$X \in \mathbb{R}^{H \times W \times C}$$에 $$n \times n$$ 표준 컨볼루션과 1×1 point-wise 컨볼루션을 적용하여 $$X_L \in \mathbb{R}^{H \times W \times d}$$를 생성한다[3].

**2. Transformers as Convolutions**:
$$X_L$$을 $$N$$개의 비중첩 패치 $$X_U \in \mathbb{R}^{P \times N \times d}$$로 전개하고, 각 패치 위치 $$p$$에 대해 다음과 같이 트랜스포머를 적용한다[1][3]:

$$X_G(p) = \text{Transformer}(X_U(p)), \quad 1 \leq p \leq P$$

여기서 $$P = wh$$, $$N = \frac{HW}{P}$$이다.

**3. Fusion**: 
전역 정보가 인코딩된 $$X_G$$를 다시 $$X_F \in \mathbb{R}^{H \times W \times d}$$로 재구성하고, point-wise 컨볼루션으로 차원을 조정한 후 원본 입력과 연결하여 최종적으로 $$n \times n$$ 컨볼루션으로 융합한다[1][3].

### 다중 스케일 샘플러

효율적인 학습을 위해 가변 배치 크기를 사용하는 다중 스케일 샘플러를 도입했다[1]. 공간 해상도 집합 $$S = \{(H_1, W_1), \ldots, (H_n, W_n)\}$$에서 $$t$$번째 반복의 배치 크기는 다음과 같이 계산된다:

$$b_t = \frac{H_n W_n b}{H_t W_t}$$

이를 통해 더 작은 해상도에서는 더 큰 배치 크기를 사용하여 GPU 활용도를 높이고 훈련 효율성을 개선했다[1].

## 성능 향상 및 실험 결과

### 분류 성능
- **ImageNet-1k**: MobileViT-S (5.6M 파라미터)로 78.4% top-1 정확도 달성[1][2]
- **경량 CNN 대비**: MobileNetv2 대비 5%, ShuffleNetv2 대비 5.4%, MobileNetv3 대비 7.4% 향상[3]

### 객체 탐지 성능
- **MS-COCO**: SSDLite와 결합하여 MobileNetv3 대비 5.7% mAP 향상[1][2]
- 파라미터 수는 1.8배 적으면서도 더 높은 성능 달성[3]

### 의미 분할 성능
- **PASCAL VOC 2012**: DeepLabv3와 결합하여 MobileNetv2 대비 1.4% mIOU 향상[1]
- ResNet-101 대비 9배 적은 파라미터로 경쟁력 있는 성능[1]

## 일반화 성능 향상

### 뛰어난 일반화 능력
MobileViT는 기존 ViT 계열 모델들과 달리 **훈련과 검증 오차 간의 격차가 매우 작아** 우수한 일반화 성능을 보인다[1][3]. Figure 3에서 보여지듯이 MobileViT-S는 CNN 수준의 일반화 능력을 달성했다[3].

### 강건성
- **데이터 증강 민감성**: 기본적인 증강(random resized cropping, horizontal flipping)만으로 우수한 성능 달성[1][3]
- **L2 정규화 강건성**: Weight decay 값에 덜 민감하여 하이퍼파라미터 튜닝 부담 감소[1]
- **다중 스케일 학습**: 다양한 입력 해상도에서 일관된 성능 유지[1]

## 모델의 한계점

### 연산 복잡도
MobileViT의 자기주의 메커니즘 복잡도는 $$O(N^2Pd)$$로 이론적으로는 표준 ViT의 $$O(N^2d)$$보다 높다[1]. 하지만 실제로는 더 효율적인 성능을 보인다.

### 모바일 기기 추론 속도
모바일 기기에서 MobileNetv2보다 느린 추론 속도를 보인다[1]. 이는 다음 두 가지 이유 때문이다:
1. 트랜스포머를 위한 최적화된 CUDA 커널의 부재[1]
2. CNN에 특화된 하드웨어 최적화(배치 정규화 융합 등)의 부재[1]

### FLOPs vs 실제 성능 괴리
FLOPs 지표와 실제 모바일 기기에서의 추론 속도 간에 차이가 존재한다[1]. 이는 메모리 접근, 병렬성 정도, 플랫폼 특성 등을 고려하지 않기 때문이다.

## 향후 연구에 미치는 영향

### 하이브리드 아키텍처의 새로운 패러다임
MobileViT는 "Transformers as Convolutions" 개념을 통해 하이브리드 아키텍처 설계에 새로운 관점을 제시했다[1][3]. 이후 연구들에서 이 개념을 발전시킨 다양한 변형 모델들이 등장했다:

- **MobileViTv2**: 분리 가능한 자기주의 메커니즘으로 $$O(k)$$ 복잡도 달성[4][5]
- **SwiftFormer**: 효율적인 덧셈 주의 메커니즘으로 2배 빠른 추론 속도[6]
- **MicroViT**: ESHA 메커니즘으로 40% 높은 효율성 달성[7]

### 모바일 컴퓨터 비전의 방향성
MobileViT는 모바일 기기에서 트랜스포머의 실용적 적용 가능성을 입증했다[8][9]. 의료 영상 진단[9][8], 농업 응용[10], 품질 검사[11] 등 다양한 실제 응용 분야에서 활용되고 있다.

## 향후 연구 시 고려사항

### 하드웨어 최적화
모바일 기기에서의 트랜스포머 추론 가속을 위한 전용 하드웨어 설계와 최적화 연구가 필요하다[12]. 현재 CNN에 특화된 최적화가 트랜스포머에도 적용되어야 한다[1].

### 효율성과 정확도의 균형
FLOPs와 실제 추론 속도 간의 괴리를 줄이기 위한 새로운 효율성 지표 개발이 필요하다[1]. 또한 양자화[13], 프루닝, 신경망 구조 탐색 등을 통한 추가 최적화가 중요하다[14].

### 확장성과 적응성
다양한 비전 태스크와 도메인에 적응할 수 있는 더 유연한 하이브리드 아키텍처 설계가 필요하다[15]. 특히 의료[16], 농업[10], 제조업[11] 등 특화된 응용 분야에서의 성능 최적화가 중요하다.

### 이론적 기반 강화
하이브리드 아키텍처의 표현 능력과 일반화 성능에 대한 이론적 이해를 깊화하는 연구가 필요하다[1]. 이를 통해 더 원리적이고 체계적인 설계 방법론을 개발할 수 있을 것이다.

MobileViT는 모바일 비전 트랜스포머의 실용적 구현을 위한 중요한 이정표를 제시했으며, 향후 경량화된 AI 모델 개발에 지속적인 영향을 미칠 것으로 예상된다[2][3].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/ab2d6bc1-0ad2-406e-b4e7-edda634bd810/2110.02178v2.pdf
[2] https://www.semanticscholar.org/paper/da74a10824193be9d3889ce0d6ed4c6f8ee48b9e
[3] https://openreview.net/pdf?id=qUcX0Zn5ROG
[4] https://arxiv.org/abs/2206.02680
[5] https://openreview.net/forum?id=tBl4yBEjKi
[6] https://ieeexplore.ieee.org/document/10376776/
[7] https://ieeexplore.ieee.org/document/11043206/
[8] https://www.mdpi.com/2076-3417/14/18/8115
[9] https://www.medrxiv.org/content/10.1101/2024.10.24.24316057v1.full-text
[10] https://www.frontiersin.org/articles/10.3389/fpls.2023.1256773/full
[11] https://ieeexplore.ieee.org/document/10873172/
[12] https://ieeexplore.ieee.org/document/10558190/
[13] https://ieeexplore.ieee.org/document/10457283/
[14] https://ieeexplore.ieee.org/document/10376989/
[15] https://pseudo-lab.github.io/All-About-ViT/docs/ch12/MobileViT_v3.html
[16] https://ieeexplore.ieee.org/document/10981108/
[17] https://ieeexplore.ieee.org/document/10651021/
[18] https://link.springer.com/10.1007/s00530-024-01312-0
[19] https://wikidocs.net/236118
[20] https://da2so.tistory.com/46
[21] https://arxiv.org/abs/2110.02178
[22] https://arxiv.org/html/2307.09283v7
[23] https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002998479
[24] https://huggingface.co/docs/transformers/model_doc/mobilevit
[25] https://arxiv.org/html/2504.08481v1
[26] http://arxiv.org/pdf/2110.02178.pdf
[27] https://paperswithcode.com/method/mobilevit
[28] https://www.sciencedirect.com/science/article/abs/pii/S0952197625000570
[29] https://leechanhyuk.github.io/paper_review/Mobile-ViT-review/
[30] https://www.nature.com/articles/s41598-024-75901-4
[31] https://learnopencv.com/mobilevit-keras-3/
[32] https://keras.io/examples/vision/mobilevit/
[33] https://ieeexplore.ieee.org/document/10538361/
[34] https://library.seg.org/doi/10.1190/geo2022-0757.1
[35] https://www.mdpi.com/2079-9292/13/24/5009
[36] https://iopscience.iop.org/article/10.1088/1742-6596/2562/1/012012
[37] https://www.worldscientific.com/doi/10.1142/S0129065725500157
[38] https://pmc.ncbi.nlm.nih.gov/articles/PMC10892637/
[39] https://journals.sagepub.com/doi/full/10.1177/15589250241233758
[40] https://ai-scholar.tech/en/articles/image-recognition%2Fmobilevit
[41] https://kalelpark.tistory.com/67
[42] https://pseudo-lab.github.io/All-About-ViT/docs/ch1/01_code.html
[43] https://arxiv.org/html/2403.08368v1
[44] https://openreview.net/forum?id=vh-0sUt8HlG
[45] https://vds.sogang.ac.kr/wp-content/uploads/2023/02/2023-%EB%8F%99%EA%B3%84%EC%84%B8%EB%AF%B8%EB%82%98_%EC%9C%A0%ED%98%84%EC%9A%B0.pdf
[46] https://arxiv.org/pdf/2111.01353.pdf
[47] https://dataloop.ai/library/model/apple_mobilevit-xx-small/
[48] https://openreview.net/forum?id=W2gO9bYYG5P
[49] https://www.mdpi.com/2077-0472/15/6/571
[50] https://arxiv.org/html/2410.22709v2
[51] https://worldscientific.com/doi/10.1142/S0218001424540089
[52] http://anale.steconomiceuoradea.ro/en/2024/02/15/practicing-of-renewable-energy-auction-scheme-expected-societal-economic-gains-for-the-developing-countries/
[53] https://egarp.lt/index.php/aghel/article/view/117
[54] https://iopscience.iop.org/article/10.1088/1755-1315/1337/1/012072
[55] https://pressto.amu.edu.pl/index.php/psn/article/view/47388
[56] https://ieeexplore.ieee.org/document/10307550/
[57] https://journals.lww.com/10.1097/SAP.0000000000003608
[58] https://www.emerald.com/insight/content/doi/10.1108/IJOTB-05-2024-0093/full/html
[59] https://ebooks.iospress.nl/doi/10.3233/FAIA231515
[60] https://openreview.net/pdf?id=vh-0sUt8HlG
[61] https://arxiv.org/html/2502.05800v1
[62] https://pubmed.ncbi.nlm.nih.gov/38400488/
[63] https://arxiv.org/html/2406.04820v1
[64] https://pajamacoder.tistory.com/38
[65] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Rethinking_Vision_Transformers_for_MobileNet_Size_and_Speed_ICCV_2023_paper.pdf
[66] https://wepub.org/index.php/IJCSIT/article/view/4108
[67] https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1498025/full
[68] https://www.sciencedirect.com/science/article/pii/S0378377425000794
[69] https://dl.acm.org/doi/10.1145/3508396.3512869
[70] https://ieeexplore.ieee.org/document/10971255/
[71] https://arxiv.org/abs/2110.02178v1
[72] https://arxiv.org/pdf/2309.01310.pdf
[73] https://pmc.ncbi.nlm.nih.gov/articles/PMC10562605/
[74] https://arxiv.org/pdf/2212.08059.pdf
[75] https://arxiv.org/pdf/2206.02680.pdf
[76] https://www.frontiersin.org/articles/10.3389/fpls.2023.1256773/pdf?isPublishedV2=False
[77] https://www.mdpi.com/1424-8220/24/4/1331
[78] https://arxiv.org/pdf/2204.05525.pdf
[79] https://arxiv.org/pdf/2309.05829.pdf
[80] https://dl.acm.org/doi/10.1145/3703187.3703208
[81] https://oa.upm.es/81020/2/paper.pdf
[82] https://github.com/matteo-stat/mobilevit-a-mobile-friendly-vision-transformer
[83] https://ieeexplore.ieee.org/document/10491267/
[84] https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-06/063006/VA-YOLO--a-foreign-object-debris-detection-method-on/10.1117/1.JEI.33.6.063006.full
[85] https://arxiv.org/html/2406.04820
[86] http://arxiv.org/pdf/2211.10526.pdf
[87] http://arxiv.org/pdf/2311.15157v1.pdf
[88] https://arxiv.org/pdf/2206.09959.pdf
[89] https://arxiv.org/pdf/2309.12424.pdf
[90] https://arxiv.org/pdf/2207.07268.pdf
[91] https://arxiv.org/pdf/2309.11523.pdf
[92] https://ejpe.org/journal/article/view/205
[93] https://arxiv.org/pdf/2206.01191.pdf
[94] http://arxiv.org/pdf/2307.09283.pdf
[95] https://arxiv.org/pdf/2305.19365.pdf
[96] https://arxiv.org/pdf/2503.02891.pdf
[97] https://arxiv.org/pdf/2303.09730.pdf
[98] https://pmc.ncbi.nlm.nih.gov/articles/PMC10576942/
