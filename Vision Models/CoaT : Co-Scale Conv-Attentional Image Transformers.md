# CoaT : Co-Scale Conv-Attentional Image Transformers | Object detection, Semantic segmentation

## 핵심 주장 및 주요 기여

**CoaT (Co-Scale Conv-Attentional Image Transformers)**는 비전 트랜스포머의 두 가지 핵심 한계를 해결하여 이미지 분류 성능을 향상시키는 혁신적인 아키텍처를 제안합니다. 논문의 핵심 주장은 **다중 스케일 정보 통합**과 **계산 효율적인 어텐션 메커니즘**이 비전 트랜스포머의 성능 개선에 필수적이라는 것입니다.[1][2]

### 주요 기여점

1. **Co-Scale 메커니즘**: 서로 다른 스케일의 인코더 브랜치 무결성을 유지하면서 스케일 간 효과적인 소통을 가능하게 하는 메커니즘으로, fine-to-coarse, coarse-to-fine, cross-scale 모델링을 실현합니다.[1]

2. **Conv-Attentional 모듈**: 팩터화된 어텐션 모듈에서 상대적 위치 임베딩을 효율적으로 구현하는 conv-like 연산을 통해 기존 self-attention 계층 대비 현저한 계산 효율성 향상을 달성합니다.[1]

## 해결하고자 하는 문제

### 기존 비전 트랜스포머의 한계

**1. 계산 복잡도 문제**
- 표준 scaled dot-product attention의 $$O(N^2)$$ 공간 복잡도와 $$O(N^2C)$$ 시간 복잡도[3]
- 고해상도 이미지에서 N ≫ C일 때 감당할 수 없는 계산량

**2. 다중 스케일 정보 부족**  
- ViT와 DeiT는 고정된 패치 크기의 단일 이미지 그리드 기반으로 제한됨[4][1]
- 16×16 패치 분할로 인한 세부 정보 모델링 능력 제한

## 제안하는 방법 및 수식

### 1. Factorized Attention Mechanism

기존의 scaled dot-product attention:

$$\text{Att}(X) = \text{softmax}\Big(\frac{QK^T}{\sqrt{C}}\Big)V$$[3]

CoaT의 factorized attention:

$$\text{FactorAtt}(X) = \frac{Q}{\sqrt{C}}\Big(\text{softmax}(K)^T V\Big)$$[3]

이 방식으로 $$O(NC + C^2)$$ 공간 복잡도와 $$O(NC^2)$$ 시간 복잡도를 달성하여 N에 대해 선형적 계산 복잡도를 구현합니다.[3]

### 2. Convolutional Relative Position Encoding

**Convolutional Relative Position Encoding (CRPE)**:

$$\hat{EV}^{(l)} = Q^{(l)} \circ \text{Conv1D}(P^{(l)}, V^{(l)})$$[3]

$$\hat{EV} = Q \circ \text{DepthwiseConv1D}(P, V)$$[3]

2D 이미지의 경우:

$$\hat{EV}^{\text{img}} = Q^{\text{img}} \circ \text{DepthwiseConv2D}(P, V^{\text{img}})$$[3]

최종 conv-attention:

$$\text{ConvAtt}(X) = \frac{Q}{\sqrt{C}}\Big(\text{softmax}(K)^T V\Big) + \hat{EV}$$[3]

### 3. Co-Scale Architecture

**Serial Blocks**: 순차적으로 해상도를 줄이며 채널 용량을 확장
**Parallel Blocks**: Feature interpolation을 통한 cross-scale attention으로 다중 스케일 정보 융합[1]

## 모델 구조

### CoaT-Lite
- 4개 직렬 블록으로 구성된 피라미드 구조
- $$H/4 \times W/4 \times C_1$$부터 $$H/32 \times W/32 \times C_4$$까지 계층적 다운샘플링

### CoaT  
- 직렬 블록과 병렬 블록 조합
- 병렬 그룹에서 F2, F3, F4 특징 맵들의 cross-scale interaction
- 3개 스케일의 CLS 토큰 집계를 통한 분류 수행[3]

## 성능 향상

### ImageNet-1K 결과
- **CoaT-Lite Tiny**: 77.5% (5.7M 파라미터)
- **CoaT Tiny**: 78.3% (5.5M 파라미터)  
- **CoaT-Lite Small**: 81.9% (20M 파라미터)
- **CoaT Small**: 82.1% (22M 파라미터)

동일 크기 CNN 및 ViT 모델들을 상당한 마진으로 상회하는 성능을 보여줍니다.[3]

### 다운스트림 태스크
- **객체 검출 (Mask R-CNN)**: ResNet, PVT, Swin Transformer 대비 우수한 성능
- **인스턴스 분할**: CoaT Small이 49.0% APb, 43.7% APm 달성[3]

## 한계점

### 계산 비용 문제
- CoaT 모델들이 Swin Transformer 대비 높은 정확도를 달성하지만 **더 큰 지연시간/FLOPs**를 가짐
- 병렬 그룹이 계산적으로 부담스러우며, 연산들이 병렬로 실행되지 않아 지연시간 오버헤드 발생[3]

### 메모리 사용량
- 고해상도 병렬 블록으로 인한 높은 메모리 요구량
- CoaT Small: 371M 메모리 vs Swin-T: 222M 메모리[3]

## 일반화 성능 향상 가능성

### 다중 스케일 특성의 범용성

**1. 아키텍처 독립성**  
CoaT의 co-scale 메커니즘은 다양한 ViT 아키텍처에 적용 가능하며, 특히 계층적 구조가 없는 vanilla ViT/DeiT에서 가장 큰 개선을 보여줍니다. 이는 attention saturation 문제 완화를 통한 깊은 네트워크의 효과적 학습을 가능하게 합니다.[5]

**2. 도메인 적응성**
Conv-attentional 메커니즘의 상대적 위치 인코딩은 **해상도 변화에 동적으로 적응**할 수 있어 다양한 입력 크기와 도메인에서 강건한 성능을 제공합니다.[6]

**3. 특징 표현의 풍부함**
다중 스케일 특징 학습은 **로컬과 글로벌 정보의 균형잡힌 통합**을 가능하게 하여 의료 영상, 객체 검출, 시계열 분석 등 다양한 응용 분야에서 활용 가능성을 보여줍니다.[7][8][9]

### 효율성과 정확성의 균형
팩터화된 어텐션과 convolutional position encoding의 조합은 **계산 효율성을 유지하면서도 표현력을 보존**하여 리소스 제약 환경에서의 배포 가능성을 높입니다.[10][11]

## 향후 연구에 미치는 영향

### 1. 아키텍처 설계 패러다임 전환

**하이브리드 접근법의 정착**  
CoaT는 CNN의 로컬 특징 추출과 Transformer의 글로벌 모델링 능력을 효과적으로 결합한 선구적 사례로, 이후 **하이브리드 아키텍처 연구의 기준점**이 되었습니다. 특히 의료 이미지 분할, UAV 추적 등에서 CNN-Transformer 융합 모델들이 CoaT의 설계 철학을 계승하고 있습니다.[12][13][14][10]

**다중 스케일 어텐션의 표준화**  
Co-scale 메커니즘은 **스케일 간 정보 교환**의 중요성을 입증하여, 이후 연구들에서 다중 스케일 특징 학습이 핵심 설계 원칙으로 자리잡게 했습니다.[9][15]

### 2. 효율적 어텐션 메커니즘 발전

**팩터화 기법의 확산**  
CoaT의 factorized attention은 **선형 복잡도 달성**의 실용적 방법론을 제시하여, 이후 FAST, Higher-Order Transformers 등에서 텐서 분해와 커널화 기법으로 발전되었습니다.[16][17][18]

**위치 인코딩 혁신**  
Convolutional position encoding은 **동적 해상도 적응**의 가능성을 보여주어, 시계열과 의료 영상 분야에서 도메인별 위치 인코딩 연구를 촉진했습니다.[8][19][9]

### 3. 성능-효율성 트레이드오프 최적화

**리소스 제약 환경 대응**  
CoaT의 계산량 증가 한계는 이후 **경량화 연구**의 중요성을 강조하여, MobileUNETR, EViT-UNet 등 모바일/엣지 디바이스용 효율적 모델 개발을 가속화했습니다.[11][10]

## 향후 연구 고려사항

### 1. 계산 효율성 개선 방향

**병렬 처리 최적화**  
현재 CoaT의 병렬 그룹에서 발생하는 지연시간 오버헤드를 해결하기 위해 **GPU 병렬화에 최적화된 구현**과 **high-resolution 병렬 블록 축소** 방안이 필요합니다.[3]

**메모리 효율적 설계**  
고해상도 특징 맵 저장으로 인한 메모리 부담을 완화하기 위한 **점진적 특징 재사용**과 **압축 기법** 도입이 요구됩니다.[5]

### 2. 일반화 능력 강화

**도메인 적응 메커니즘**  
다양한 응용 분야별 특성을 고려한 **적응적 co-scale 가중치**와 **도메인별 위치 인코딩** 전략 개발이 필요합니다.[6]

**스케일 동적 조정**  
입력 데이터 특성에 따른 **적응적 스케일 선택**과 **동적 병렬 블록 구성** 방법론 연구가 중요합니다.

### 3. 새로운 응용 영역 탐색

**3D 데이터 처리**  
CoaT의 설계 원리를 **3D 의료 영상**, **비디오 분석** 등 고차원 데이터로 확장하는 연구가 필요합니다.[20][18]

**멀티모달 학습**  
텍스트, 이미지, 음성 등 **이종 데이터 융합**을 위한 co-scale 메커니즘 활용 방안 모색이 중요합니다.[21]

CoaT는 비전 트랜스포머의 패러다임 전환점이 되어 효율적이면서도 강력한 다중 스케일 모델링의 가능성을 제시했으며, 이후 하이브리드 아키텍처와 효율적 어텐션 연구의 토대가 되었습니다. 향후에는 계산 효율성과 일반화 능력의 균형점을 찾는 연구가 핵심 과제가 될 것입니다.

[1] https://ieeexplore.ieee.org/document/9710209/
[2] https://arxiv.org/abs/2104.06399
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2d21d284-59c5-4a01-9b32-3f6dfa09ccc4/2104.06399v2.pdf
[4] https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Co-Scale_Conv-Attentional_Image_Transformers_ICCV_2021_paper.pdf
[5] https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_You_Only_Need_Less_Attention_at_Each_Stage_in_Vision_CVPR_2024_paper.pdf
[6] https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CPVTCONDITIONAL-POSITIONAL-ENCODINGS-FOR-VISION-TRANSFORMERS
[7] https://www.tandfonline.com/doi/full/10.1080/01691864.2024.2381812
[8] https://link.springer.com/article/10.1007/s10618-023-00948-2
[9] https://www.nature.com/articles/s41598-025-95361-8
[10] https://arxiv.org/abs/2409.03062
[11] https://ieeexplore.ieee.org/document/10981108/
[12] https://ieeexplore.ieee.org/document/9974664/
[13] https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.14297
[14] https://ieeexplore.ieee.org/document/10161487/
[15] https://arxiv.org/abs/2303.16892
[16] https://arxiv.org/abs/2402.07901
[17] https://openreview.net/forum?id=MxGGdhDmv5
[18] https://arxiv.org/abs/2412.02919
[19] https://arxiv.org/abs/2305.16642
[20] https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf
[21] https://ieeexplore.ieee.org/document/10635763/
[22] https://www.frontiersin.org/articles/10.3389/fmars.2023.1328436/full
[23] https://ieeexplore.ieee.org/document/10446332/
[24] https://ieeexplore.ieee.org/document/11085698/
[25] https://www.tandfonline.com/doi/full/10.1080/17538947.2023.2261770
[26] https://ieeexplore.ieee.org/document/9879679/
[27] https://arxiv.org/pdf/2104.06399.pdf
[28] https://arxiv.org/pdf/2106.04803.pdf
[29] https://arxiv.org/html/2407.06673v1
[30] https://arxiv.org/pdf/2107.06263.pdf
[31] https://arxiv.org/pdf/2010.11929.pdf
[32] https://arxiv.org/pdf/2106.05786.pdf
[33] https://arxiv.org/pdf/2103.14899.pdf
[34] http://arxiv.org/pdf/2309.05674.pdf
[35] https://arxiv.org/pdf/2210.01820.pdf
[36] https://arxiv.org/abs/2106.09681
[37] https://arxiv.org/abs/2505.14719
[38] https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Rethinking_and_Improving_Relative_Position_Encoding_for_Vision_Transformer_ICCV_2021_paper.pdf
[39] https://github.com/mlpc-ucsd/CoaT
[40] https://aclanthology.org/2024.lrec-main.1478.pdf
[41] https://lcyking.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-CoaT-Co-Scale-Conv-Attentional-Image-Transformers
[42] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/mvit/
[43] https://koreascience.kr/article/CFKO202220859207197.pdf
[44] https://www.ki-it.com/xml/37318/37318.pdf
[45] https://arxiv.org/html/2502.12370v1
[46] https://github.com/rishikksh20/CoaT-pytorch
[47] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/mvitv2/
[48] https://www.mdpi.com/2075-4418/13/2/178
[49] https://ieeexplore.ieee.org/document/10632054/
[50] https://ieeexplore.ieee.org/document/10581855/
[51] https://ieeexplore.ieee.org/document/10960840/
[52] https://ieeexplore.ieee.org/document/10717736/
[53] https://ieeexplore.ieee.org/document/10590387/
[54] https://ieeexplore.ieee.org/document/10410409/
[55] https://arxiv.org/pdf/2503.02891.pdf
[56] https://arxiv.org/pdf/2502.05800.pdf
[57] https://arxiv.org/pdf/2206.01191.pdf
[58] http://arxiv.org/pdf/2405.00314.pdf
[59] https://arxiv.org/pdf/2302.08374.pdf
[60] https://arxiv.org/pdf/2309.02031.pdf
[61] http://arxiv.org/pdf/2406.07488.pdf
[62] https://arxiv.org/pdf/2402.00033.pdf
[63] http://arxiv.org/pdf/2405.03882.pdf
[64] https://arxiv.org/pdf/2310.04134.pdf
[65] https://pmc.ncbi.nlm.nih.gov/articles/PMC10830169/
[66] https://arxiv.org/abs/2104.11227
[67] https://www.themoonlight.io/en/review/higher-order-transformers-efficient-attention-mechanism-for-tensor-structured-data
[68] https://openreview.net/forum?id=uf9EEOkcYR
[69] https://arxiv.org/html/2502.05800v1
[70] https://arxiv.org/html/2404.13434v1
[71] https://bitsofchris.com/p/how-to-implement-factorized-attention
[72] https://arxiv.org/html/2308.09372v2
[73] https://www.geeksforgeeks.org/machine-learning/sparse-transformer-stride-and-fixed-factorized-attention/
[74] https://helda.helsinki.fi/bitstreams/a6100bca-5d8c-449b-ae6f-a5fca821e608/download
[75] https://link.springer.com/article/10.1007/s40747-025-01904-x
