# DeiT : Training data-efficient image transformers & distillation through attention

## 핵심 주장

DeiT (Data-efficient Image Transformers) 논문은 **대규모 데이터셋 없이도 ImageNet-1K만으로 경쟁력 있는 Vision Transformer를 훈련할 수 있다**는 혁신적인 주장을 제시했습니다[1][2]. 기존 ViT가 JFT-300M과 같은 수억 장의 이미지를 요구했던 것과 달리, DeiT는 단일 컴퓨터에서 3일 이내에 훈련 가능한 효율적인 모델을 제안했습니다[1][3].

## 해결하고자 하는 문제

### 1. 데이터 의존성 문제
기존 Vision Transformer는 "충분하지 않은 데이터로 훈련할 때 일반화 성능이 떨어진다"는 한계가 있었습니다[1]. 이는 CNN과 달리 transformer가 locality나 translation equinductive bias**가 부족하기 때문입니다[4].

### 2. 계산 자원 요구량
ViT 훈련에는 방대한 컴퓨팅 인프라가 필요했고, 이는 많은 연구자들의 접근을 제한했습니다[1][5].

## 제안하는 방법

### 1. Distillation Token을 통한 Knowledge Distillation

DeiT의 핵심 혁신은 **distillation token**의 도입입니다[1]. 이는 기존의 class token과 함께 사용되어 teacher 모델로부터 지식을 전달받습니다.

수식적으로 표현하면:

**Hard-label distillation loss:**

$$
L_{\text{hardDistill}}^{\text{global}} = \frac{1}{2}L_{CE}(\psi(Z_s), y) + \frac{1}{2}L_{CE}(\psi(Z_s), y_t)
$$

여기서 $$y_t = \text{argmax}_c Z_t(c)$$ 는 teacher의 hard decision입니다[1].

**전체 distillation objective:**

$$
L_{\text{global}} = (1-\lambda)L_{CE}(\psi(Z_s), y) + \lambda\tau^2 KL(\psi(Z_s/\tau), \psi(Z_t/\tau))
$$

### 2. Attention 메커니즘을 통한 지식 전달

Multi-head Self Attention은 다음과 같이 계산됩니다:

$$
\text{Attention}(Q, K, V) = \text{Softmax}(QK^T/\sqrt{d})V
$$

distillation token은 self-attention을 통해 patch token들과 상호작용하며, teacher의 지식을 학습합니다[1].

## 모델 구조

### DeiT 변형 모델들[1]

| 모델 | 임베딩 차원 | 헤드 수 | 레이어 수 | 파라미터 수 | 처리량 (이미지/초) |
|------|-------------|---------|-----------|-------------|-------------------|
| DeiT-Ti | 192 | 3 | 12 | 5M | 2536 |
| DeiT-S | 384 | 6 | 12 | 22M | 940 |
| DeiT-B | 768 | 12 | 12 | 86M | 292 |

### Teacher 모델 선택
**CNN을 teacher로 사용하는 것이 transformer teacher보다 효과적**임을 실험적으로 증명했습니다. 이는 CNN의 inductive bias가 transformer에 전달되기 때문입니다[1][6].

## 성능 향상

### ImageNet-1K 결과[1]
- **DeiT-B**: 81.8% top-1 accuracy (distillation 없이)
- **DeiT-B⚗**: 83.4% top-1 accuracy (distillation 사용)
- **DeiT-B⚗↑384**: 85.2% top-1 accuracy (고해상도 fine-tuning)

이는 JFT-300M으로 pre-train된 ViT-B (84.15%)를 능가하는 성능입니다[1][7].

### 전이 학습 성능[1]
- CIFAR-10: 99.1%
- CIFAR-100: 91.3%
- Oxford-102 flowers: 98.8%
- Stanford Cars: 92.9%

## 일반화 성능 향상 메커니즘

### 1. Inductive Bias 전달
CNN teacher로부터의 distillation을 통해 **locality와 spatial hierarchy 같은 inductive bias**가 transformer에 전달됩니다[1][6]. 이는 특히 데이터가 제한적인 상황에서 일반화 성능을 크게 향상시킵니다.

### 2. 다양한 데이터 증강 기법
- **RandAugment**: AutoAugment보다 효과적
- **Mixup**: 0.8 확률
- **CutMix**: 1.0 확률  
- **Random Erasing**: 0.25 확률
- **Repeated Augmentation**: 3회 반복[1]

### 3. 정규화 기법
- **Stochastic Depth**: 0.1
- **Label Smoothing**: ε = 0.1
- **Weight Decay**: 0.05 (ViT의 0.3보다 크게 감소)[1]

## 한계점

### 1. Teacher 모델 의존성
여전히 강력한 CNN teacher 모델이 필요하며, 이는 추가적인 훈련 비용을 수반합니다[6].

### 2. 아키텍처 혁신의 부재
DeiT는 기존 ViT 구조를 그대로 사용하고 distillation token만 추가했을 뿐, 근본적인 아키텍처 개선은 제한적입니다[1].

### 3. 초기 훈련 단계의 불안정성
Transformer는 초기화에 민감하며, 적절한 하이퍼파라미터 튜닝이 필요합니다[1].

## 향후 연구에 미치는 영향

### 1. Efficient Vision Transformer 연구의 촉진
DeiT는 **데이터 효율성**이라는 새로운 연구 방향을 제시했습니다. 이후 T2T-ViT[4], LeViT[8], MobileViT 등 다양한 효율적인 transformer 모델들이 등장했습니다.

### 2. Knowledge Distillation 기법의 발전
Transformer 특화 distillation 기법들이 활발히 연구되고 있습니다:
- **DearKD**: 초기 단계 distillation[9]
- **ViTKD**: Feature-based distillation[10]
- **SpectralKD**: Spectral analysis 기반[11]

### 3. 의료 영상 분야로의 확산
작은 데이터셋에서도 효과적인 DeiT의 특성으로 인해 **의료 영상 분야**에서 활발히 활용되고 있습니다[12].

## 앞으로 연구 시 고려할 점

### 1. 더 나은 Teacher-Student 매칭
CNN과 Transformer 간의 **구조적 차이**를 더 효과적으로 bridge하는 방법이 필요합니다[13].

### 2. Self-supervised Learning과의 결합
Distillation과 self-supervised learning을 결합한 **하이브리드 접근법** 연구가 필요합니다[14].

### 3. Long-tail Distribution 대응
불균형 데이터셋에서의 성능 향상을 위한 **DeiT-LT** 같은 확장 연구가 중요합니다[15][16].

### 4. 계산 효율성 개선
실제 배포 환경에서의 **latency와 throughput 최적화**가 지속적으로 필요합니다[17][18].

DeiT는 Vision Transformer의 실용화에 결정적인 기여를 했으며, 특히 **제한된 자원으로도 높은 성능을 달성할 수 있는 길**을 제시했습니다. 이는 AI 연구의 민주화와 실제 산업 응용에 큰 영향을 미쳤습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3a166d22-e6c8-4057-93bd-603554dbf429/2012.12877v2.pdf
[2] https://arxiv.org/pdf/2012.12877.pdf
[3] https://proceedings.mlr.press/v139/touvron21a.html
[4] https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.pdf
[5] https://arxiv.org/abs/2012.12877
[6] https://stuartfeeser.com/blogs/ai-engineers/deit-vs-vit/index.html
[7] https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf
[8] https://vds.sogang.ac.kr/wp-content/uploads/2023/01/210114_%E1%84%83%E1%85%A9%E1%86%BC%E1%84%80%E1%85%A8%E1%84%89%E1%85%A6%E1%84%86%E1%85%B5%E1%84%82%E1%85%A1_%E1%84%89%E1%85%B5%E1%86%B7%E1%84%8C%E1%85%A2%E1%84%92%E1%85%A5%E1%86%AB.pdf
[9] https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_DearKD_Data-Efficient_Early_Knowledge_Distillation_for_Vision_Transformers_CVPR_2022_paper.pdf
[10] https://arxiv.org/abs/2209.02432
[11] https://www.semanticscholar.org/paper/ca2c8d0f9f7064228704226219cb8ca872bae594
[12] https://www.sciendo.com/article/10.2478/ijanmc-2025-0014
[13] https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Yang_ViTKD_Feature-based_Knowledge_Distillation_for_Vision_Transformers_CVPRW_2024_paper.pdf
[14] https://openreview.net/forum?id=SCN8UaetXx
[15] https://arxiv.org/abs/2404.02900
[16] https://openaccess.thecvf.com/content/CVPR2024/papers/Rangwani_DeiT-LT_Distillation_Strikes_Back_for_Vision_Transformer_Training_on_Long-Tailed_CVPR_2024_paper.pdf
[17] https://link.springer.com/10.1007/s11263-023-01861-3
[18] https://arxiv.org/abs/2506.11093
[19] https://books.openedition.org/aaccademia/7057
[20] https://aclanthology.org/2020.semeval-1.150
[21] https://aclanthology.org/2020.semeval-1.149
[22] https://ojs.aaai.org/index.php/AAAI/article/view/16476
[23] https://aclanthology.org/2020.wat-1.10
[24] https://www.semanticscholar.org/paper/821602f9f2a5d22cb2cb66ce4f40d02a8128dd9e
[25] https://books.openedition.org/aaccademia/7420
[26] https://www.semanticscholar.org/paper/6449f1512714a8ca5f309f2dd18edfe4ca22f6cc
[27] https://ai.meta.com/blog/significantly-faster-vision-transformer-training/
[28] https://187cm.tistory.com/84
[29] https://day-to-day.tistory.com/53
[30] https://hyoseok-personality.tistory.com/entry/Paper-Review-DeiT-Training-data-efficient-image-transformers-distillation-through-attention
[31] https://kalelpark.tistory.com/63
[32] https://paperswithcode.com/sota/image-classification-on-imagenet?p=deepvit-towards-deeper-vision-transformer
[33] https://deep-learning-study.tistory.com/806
[34] https://velog.io/@jsleeg98/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Training-data-efficient-image-transformers-distillation-through-attention
[35] https://arxiv.org/html/2308.09372v2
[36] https://velog.io/@hseop/Simple-Review-DeiT-Training-data-efficient-image-transformers-distillation-through-attention
[37] https://ieeexplore.ieee.org/document/10943339/
[38] https://ieeexplore.ieee.org/document/9710747/
[39] https://ieeexplore.ieee.org/document/10446293/
[40] https://arxiv.org/abs/2402.03317
[41] https://ieeexplore.ieee.org/document/10177144/
[42] https://www.semanticscholar.org/paper/0da3cefb4e15347760f64f5045a34b38515ef0f0
[43] https://arxiv.org/html/2202.13393v3
[44] http://arxiv.org/pdf/2302.02108.pdf
[45] https://developer.nvidia.com/blog/training-a-state-of-the-art-imagenet-1k-visual-transformer-model-using-nvidia-dgx-superpod/
[46] https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Cumulative_Spatial_Knowledge_Distillation_for_Vision_Transformers_ICCV_2023_paper.pdf
[47] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08449.pdf
[48] https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=3
[49] https://www.sciencedirect.com/science/article/abs/pii/S0950705124011651
[50] https://www.sciencedirect.com/science/article/pii/S0925231224017697
[51] https://koasas.kaist.ac.kr/handle/10203/321776
[52] https://doonby.tistory.com/53
[53] https://www.semanticscholar.org/paper/c723187a2230749b1e706df2217e928c8271a660
[54] https://www.semanticscholar.org/paper/c9d1f170f3d041791558b78101e8b291597d7f28
[55] https://ieeexplore.ieee.org/document/11078373/
[56] https://ieeexplore.ieee.org/document/10447955/
[57] https://ieeexplore.ieee.org/document/10864498/
[58] https://openaccess.thecvf.com/content/WACV2025/papers/Chowdhury_Bandit_Based_Attention_Mechanism_in_Vision_Transformers_WACV_2025_paper.pdf
[59] https://arxiv.org/html/2504.14366v2
[60] https://www.sciencedirect.com/science/article/pii/S1051200422004766
[61] https://yonghip.tistory.com/entry/Training-data-efficient-image-transformers-distillation-through-attention-%EB%A6%AC%EB%B7%B0DeiT-2021
[62] https://www.youtube.com/watch?v=oHnv_S9N1J8
[63] https://www.sciencedirect.com/science/article/pii/S0925231223010950
[64] https://hyunseo-fullstackdiary.tistory.com/422
[65] https://github.com/daekeun-ml/gitbook/blob/master/ml/computer-vision-transformer-based/deit-training-data-efficient-image-transformers-and-distillation-through-attention.md
[66] https://www.semanticscholar.org/paper/707710387b2ab2ccac3326eb478df6b2f4498584
[67] https://ieeexplore.ieee.org/document/9151078/
[68] https://arxiv.org/html/2405.02730v1
[69] https://arxiv.org/html/2411.03286
[70] http://arxiv.org/pdf/2212.09748v2.pdf
[71] https://arxiv.org/html/2503.12590v1
[72] https://arxiv.org/html/2411.08196
[73] https://arxiv.org/html/2503.10618
[74] https://arxiv.org/pdf/2211.11315.pdf
[75] https://arxiv.org/html/2502.17196v1
[76] https://arxiv.org/pdf/2207.09240.pdf
[77] https://arxiv.org/abs/2205.14100
[78] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003173418
[79] https://visionhong.tistory.com/29
[80] https://www.semanticscholar.org/paper/4d3bbc7a5f8dee882a084688a5cc57562215924c
[81] https://arxiv.org/abs/2405.18616
[82] https://arxiv.org/pdf/2205.01580.pdf
[83] https://arxiv.org/pdf/2106.01548.pdf
[84] https://arxiv.org/pdf/2101.11986v1.pdf
[85] http://arxiv.org/pdf/2106.03746.pdf
[86] https://arxiv.org/pdf/2107.06263.pdf
[87] https://arxiv.org/pdf/2204.00993.pdf
[88] https://arxiv.org/pdf/2303.01870.pdf
[89] https://arxiv.org/pdf/2106.10270.pdf
[90] https://arxiv.org/pdf/2105.02723.pdf
[91] https://dl.acm.org/doi/10.1145/3596286.3596292
[92] https://ieeexplore.ieee.org/document/10263583/
[93] https://arxiv.org/pdf/2107.01378.pdf
[94] https://arxiv.org/pdf/2503.01888.pdf
[95] https://arxiv.org/ftp/arxiv/papers/2310/2310.00369.pdf
[96] https://arxiv.org/html/2502.10683v1
[97] https://arxiv.org/pdf/2502.14226.pdf
[98] https://arxiv.org/pdf/2403.06213.pdf
[99] https://aclanthology.org/2021.acl-long.504.pdf
[100] https://arxiv.org/pdf/2104.02096.pdf
[101] http://arxiv.org/pdf/2204.12997.pdf

## DeiT : Training data-efficient image transformers & Distillation through attention | Image classification
DeiT는 이미지 분류를 위한 **트랜스포머 기반 모델**로, 적은 데이터와 컴퓨팅 자원으로 고성능을 달성하기 위해 설계되었습니다. 기존 Vision Transformer(ViT)는 대규모 데이터(수억 장)와 고사양 인프라가 필요했으나, DeiT는 **ImageNet 데이터만으로 3일 이내** 학습이 가능합니다[1][2][7]. 핵심 혁신은 데이터 증강 기술과 "증류 토큰(distillation token)"을 이용한 지식 증류(distillation)입니다.

### 핵심 메커니즘  
#### 1. **데이터 효율성 달성 기술**  
- **고도화된 데이터 증강**:  
  - Rand-Augment, Mixup, CutMix 등의 기법으로 데이터 다양성 확보[3][1].  
  - 예: 이미지 일부를 무작위로 제거하거나(CutMix), 여러 이미지 혼합(Mixup).  
- **반복 증강(Repeated Augmentation)**:  
  동일한 이미지를 변형해 배치에 여러 번 포함시켜 데이터 부족 문제 해결[6][8].  

#### 2. **증류 토큰(Distillation Token)**  
- **작동 원리**:  
  - 입력 시퀀스에 `[CLS]`(클래스 토큰) 외 `[DLS]`(증류 토큰) 추가[3][4].  
  - `[DLS]`는 **교사 모델(Teacher)**의 예측 결과를 학습 대상으로 삼음.  
  - 교사 모델은 일반적으로 CNN 기반(예: RegNet)[1][5].  
- **학습 과정**:  
  - **두 가지 손실 함수** 적용:  
    - `[CLS]`: 실제 레이블과의 오차 계산.  
    - `[DLS]`: 교사 모델의 예측과의 오차 계산[4][5].  
  - 주의 메커니즘(Self-Attention)으로 두 토큰이 상호작용하며 학습[5][6].  

#### 3. **트랜스포머 구조 최적화**  
- **계층적 설계**:  
  | 모델       | 임베딩 차원 | 헤드 수 | 파라미터 |  
  |------------|-------------|---------|----------|  
  | **DeiT-Ti**| 192         | 3       | 5M       |  
  | **DeiT-S** | 384         | 6       | 22M      |  
  | **DeiT-B** | 768         | 12      | 86M      |  
  - 작은 모델(DeiT-Ti/S)은 ResNet-18/50 수준의 경량화[6][8].  

### 성능 및 장점  
- **ImageNet 정확도**:  
  - DeiT-B: 83.1% (단일 크롭), 증류 적용 시 **85.2%** 달성[1][7].  
  - EfficientNet 대비 동급 정확도 + 빠른 추론 속도[8].  
- **자원 효율성**:  
  - 8-GPU 단일 노드에서 53시간 내 학습 완료[7][8].  
- **다운스트림 적용성**:  
  CIFAR-10/100, Oxford-102 등 다른 데이터셋에서도 우수한 전이 학습 성능[6][7].  

### 한계와 의의  
- **의존성**: 증류 성능은 교사 모델의 질에 영향받음[5][9].  
- **의의**:  
  - **최초로 트랜스포머를 소규모 데이터로 학습**한 모델[2][7].  
  - 하드웨어 제약 환경에서도 고성능 비전 트랜스포머 적용 가능성 확대[3][9].  

> "DeiT는 증류 토큰을 통해 교사 모델의 지식을 주의 메커니즘으로 흡수함으로써,  
> 데이터 효율성과 성능을 동시에 잡은 혁신적 모델입니다." – Hugo Touvron(DeiT 논문 저자)[1][6].

[1] https://arxiv.org/abs/2012.12877
[2] https://huggingface.co/docs/transformers/en/model_doc/deit
[3] https://demystifyml.co/deit-data-efficient-image-transformer
[4] https://zenn.dev/yuto_mo/articles/8bc30de613bb4f
[5] https://huggingface.co/docs/transformers/v4.30.0/model_doc/deit
[6] https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf
[7] https://proceedings.mlr.press/v139/touvron21a.html
[8] https://arxiv.org/pdf/2012.12877.pdf
[9] https://serp.ai/data-efficient-image-transformer/
[10] https://nithish96.github.io/Computer%20Vision/transformers/Data-efficient%20Image%20Transformer%20(DeiT)/
[11] https://paperswithcode.com/method/deit
[12] https://aitechtrend.com/transforming-computer-vision-with-data-efficient-image-transformers-deitsimage-transformer-architecture/
[13] https://sicorps.com/ai/deconstructing-deit-a-deep-dive-into-data-efficient-image-transformer-training/
[14] https://huggingface.co/docs/transformers/main/en/model_doc/deit
[15] https://clarifai.com/facebook/image-classification/models/general-image-recognition-deit-base
[16] https://www.reddit.com/r/MachineLearning/comments/1eog6j9/p_training_a_vision_transformer_on_a_small_dataset/
[17] https://www.betterhealth.vic.gov.au/health/healthyliving/healthy-eating
[18] https://www.healthline.com/nutrition/how-to-eat-healthy-guide
[19] https://www.mayoclinic.org/healthy-lifestyle/childrens-health/in-depth/nutrition-for-kids/art-20049335
[20] https://www.nhlbi.nih.gov/health/educational/lose_wt/eat/calories.htm
[21] https://www.youtube.com/watch?v=HobIo2oT0xY
[22] https://dataloop.ai/library/model/facebook_deit-base-patch16-384/
[23] https://arxiv.org/html/2502.18691v1
[24] http://arxiv.org/pdf/2404.02900.pdf
[25] https://hyper.ai/en/headlines/de2a84f6fb22caf1411137aeba1a501b
[26] https://www.datacamp.com/tutorial/complete-guide-data-augmentation
[27] https://github.com/FrancescoSaverioZuppichini/DeiT
[28] https://www.youtube.com/watch?v=s7MO3o7lD1w
[29] https://deepai.org/publication/detrdistill-a-universal-knowledge-distillation-framework-for-detr-families
[30] https://kids.britannica.com/students/article/food-and-nutrition/274373
[31] https://www.who.int/news-room/fact-sheets/detail/healthy-diet
[32] https://www.healthline.com/health/balanced-diet
[33] https://en.wikipedia.org/wiki/Diet_(nutrition)

# Reference
- https://187cm.tistory.com/84
- https://hyoseok-personality.tistory.com/entry/Paper-Review-DeiT-Training-data-efficient-image-transformers-distillation-through-attention
- https://velog.io/@heomollang/DeiT-%EA%B4%80%EB%A0%A8-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-04-Training-data-efficient-image-transformers-distillation-through-attentionDeiT
