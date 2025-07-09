# A Survey on Visual Transformer

Transformer 기반 모델이 컴퓨터 비전(CV) 전반에 확산되는 과정을 총괄적으로 정리한 논문 **“A Survey on Visual Transformer”**(Han et al., IEEE TPAMI 2023)는 자연어 처리(NLP)에서 검증된 Self-Attention 메커니즘을 시각 영역에 이식하려는 최근 연구를 체계화하였다. 아래에서는 논문의 핵심 주장, 주요 기여, 기술적 내용(수식 포함), 성능 향상 포인트·한계, 일반화 가능성, 향후 연구 지향점을 단계별로 자세히 설명한다.

## 개요: 핵심 주장과 주요 기여

- **모델 패러다임 전환**: CNN → Vision Transformer(ViT)로의 이동이 단순 아키텍처 교체가 아니라,  
  ① 긴 범위(전역) 의존성 학습, ② 비국소적 표현 능력, ③ 대규모 사전학습을 통한 범용성 확장을 이끈다고 주장한다[1]
- **네 가지 축으로 분류**  
  1. Backbone 설계  
  2. High/Mid-Level Vision (Detection·Segmentation·Pose 등)  
  3. Low-Level Vision (Super-Resolution·Inpainting 등)  
  4. Video Processing  
  각 영역마다 대표 모델·장단점을 체계적으로 비교했다.  
- **효율화·경량화 로드맵 제시**: pruning, distillation, quantization, NAS 등 **Efficient Transformer** 연구군을 별도 정리하여 실무 적용성을 강조했다.  
- **도전과제 & 연구 방향**:  
  -  **데이터 의존성·약한 귀납편향**에 따른 일반화 한계  
  -  **대규모 비지도 사전학습**과 **멀티모달 통합**의 잠재력  
  -  전역·지역 정보를 모두 활용하는 **Hybrid/CNN-Transformer** 설계 필요  

## 1. 문제 정의

기존 CNN은 지역(Receptive Field) 편향 덕에 소량 데이터에서도 강인하지만,  
전역 관계 학습·멀티모달 확장에 한계가 있었다.  
논문은 **“Self-Attention으로 전역 정보를 직접 모델링하면 시각 과제에도 같은 이득을 얻을 수 있는가?”**라는 큰 질문을 다룬다[1].

## 2. 제안 방법: Self-Attention 수식과 구조

논문은 새로운 모델을 제시하기보다 **표준 Transformer 블록을 시각 입력에 적용**하는 공통 공식을 재정의한다.

### 2.1 Self-Attention 재정식화

$$
\text{Attention}(Q,K,V)= \text{softmax}\Bigl(\frac{QK^{\top}}{\sqrt{d_k}}\Bigr)V 
$$

- $$Q=XW_Q,\;K=XW_K,\;V=XW_V$$  
- $$X$$: 패치 임베딩 행렬, $$d_k$$: Key 차원  
- 공간적 위치 정보를 위해 **고정(또는 학습) Positional Encoding**을 패치 토큰에 더한다.

### 2.2 모델 구조 전형
1. **패치 분할**: $$I\in\mathbb{R}^{H\times W\times C}$$ → $$N=(H\!W/p^2)$$개의 $$p\times p$$ 패치  
2. **선형 투영 & 위치 인코딩**  
3. **[CLS] 토큰** 추가 → 전역 표현  
4. **L개 Encoder 스택**  
5. 분류 헤드 또는 Task-특화 디코더 부착

## 3. 성능 향상 포인트

1. **대규모 사전학습**  
   - ViT-B가 ImageNet만으로는 ResNet-50을 근소 하회하지만, JFT-300 M 사전학습 뒤 **ImageNet Top-1 88.36%**로 역전[2][1].  
2. **토큰 설계 개선**  
   - TNT[3]·Swin은 **로컬 윈도우/서브패치** 도입으로 지역 정보 손실을 보완, FLOPs 대비 정확도 증가.  
3. **효율화 기법**  
   - Deformable DETR[4]은 Sparse Attention으로 학습 에폭 10× 감축, AP +4.2 상승.  
   - TinyBERT·Q8BERT 등 **지식 증류+양자화**로 모바일 환경 배치 가능.

## 4. 한계 및 일반화 관점

| 한계 | 원인 | 일반화 개선 방향 |
|------|------|------------------|
| **대규모 데이터 의존** | 귀납편향 약화 → 과적합 위험 | Masked Image Modeling(MAE)[5], Contrastive Pre-training(MoCo v3)[6] |
| **전역 계산량 O(N²)** | 모든 토큰 쌍 Attention | Sparse/Linear Attention(Katharopoulos et al.) |
| **로컬 세부 표현 부족** | 패치 평탄화로 위치 정보 손실 | Hybrid CNN-ViT(ConViT[3], CvT), Conditional Positional Encoding(CPVT) |
| **취약한 Robustness** | 구조적 편향 부족 | Adversarial Training·Strong Augmentation·Structured Initialization[7] |

> **일반화 성능 향상 핵심**:  
> (i) **데이터 규모 확대+Self-Supervised 학습**로 표현력 확보,  
> (ii) **Local Inductive Bias**(Convolution, Shifted Window 등) 삽입,  
> (iii) **Regularization/Distillation**으로 작은 데이터에도 견고한 성능 달성.

## 5. 향후 연구 영향 및 고려사항

- **범용 비전 백본**: ViT 계열은 NLP처럼 “One-for-All” 사전학습 백본으로 자리잡을 전망. 멀티태스크·멀티모달(예: GPT-4V) 확장이 자연스럽다.  
- **효율-성능 균형**: Edge/모바일 적용을 위한 경량 설계(Pruning, Low-bit Quantization, NAS) 연구가 가속될 것.  
- **데이터·라벨 의존성 완화**: MIM, CLIP류 대규모 비표주 학습이 필수. **Generalization & Robustness 벤치마크** 필요.  
- **해석 가능성**: Attention Map, Gating(ConViT) 분석을 통한 **모델 투명성** 연구 요구.  
- **친환경 AI**: 거대 모델 학습이 요구하는 에너지·탄소 비용을 고려한 **연산 효율화**도 중요 의제.

## 결론

본 Survey는 Vision Transformer 생태계를 **과제별·방법론별 로드맵**으로 정리하고,  
① **대규모 사전학습의 힘**, ② **로컬‧전역 혼합 설계의 필요성**, ③ **효율화 전략**을 명확히 했다.  
이는 추후 연구자들이 **일반화 강인·자원 효율**·**멀티모달 통합** 관점을 갖춘 차세대 비전 모델을 설계할 때 필수적 길잡이가 될 것이다.

[1] https://arxiv.org/abs/2012.12556
[2] http://thesai.org/Downloads/Volume14No8/Paper_30-An_Overview_of_Vision_Transformers_for_Image_Processing.pdf
[3] http://proceedings.mlr.press/v139/d-ascoli21a/d-ascoli21a.pdf
[4] https://arxiv.org/pdf/2305.09880.pdf
[5] https://dl.acm.org/doi/10.1109/TPAMI.2023.3268446
[6] https://jihyeonryu.github.io/2021-04-02-survey-paper1/
[7] https://arxiv.org/html/2505.19985v1
[8] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c5df94e8-c30b-488a-92bf-090ff178de1a/2012.12556v6.pdf
[9] https://www.sec.gov/Archives/edgar/data/1326205/000118518525000706/igc10k033125.htm
[10] https://www.sec.gov/Archives/edgar/data/1631282/000121390023079920/f10k2023_datasea.htm
[11] https://link.springer.com/10.1007/978-3-031-09282-4_25
[12] https://ieeexplore.ieee.org/document/10613466/
[13] https://ieeexplore.ieee.org/document/9716741/
[14] https://vciba.springeropen.com/articles/10.1186/s42492-023-00140-9
[15] https://ieeexplore.ieee.org/document/10767243/
[16] https://www.nature.com/articles/s41598-024-78774-9
[17] https://ieeexplore.ieee.org/document/10590648/
[18] https://ieeexplore.ieee.org/document/10065990/
[19] https://dl.acm.org/doi/10.1145/3729167
[20] https://ieeexplore.ieee.org/document/10088164/
[21] https://arxiv.org/html/2312.01232v1
[22] https://www.linkedin.com/pulse/advances-pretraining-scaling-large-scale-vision-models-jha-qgkbf
[23] https://openreview.net/pdf?id=_WnAQKse_uK
[24] https://arxiv.org/abs/2406.00237
[25] https://arxiv.org/abs/2111.06091
[26] https://www.semanticscholar.org/paper/A-Survey-on-Vision-Transformer-Han-Wang/d40c77c010c8dbef6142903a02f2a73a85012d5d
[27] https://coronasdk.tistory.com/1396
[28] https://arxiv.org/html/2406.00237v1
[29] https://jhtobigs.oopy.io/documents/vitsurvey
[30] https://openaccess.thecvf.com/content/ICCV2023W/NIVT/papers/Djenouri_A_Hybrid_Visual_Transformer_for_Efficient_Deep_Human_Activity_Recognition_ICCVW_2023_paper.pdf
[31] https://openreview.net/forum?id=NxoFmGgWC9
[32] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12610/1261021/Comparison-of-ResNet-50-and-vision-transformer-models-for-trash/10.1117/12.2671208.short
[33] https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html
[34] https://www.reddit.com/r/MachineLearning/comments/d0gnyp/d_what_is_the_inductive_bias_in_transformer/
[35] https://www.mdpi.com/2073-4395/15/1/77
[36] https://arxiv.org/pdf/2111.06091.pdf
[37] https://arxiv.org/pdf/2012.12556.pdf
[38] https://arxiv.org/pdf/2211.06004.pdf
[39] https://arxiv.org/abs/2101.01169
[40] https://arxiv.org/pdf/2304.09854.pdf
[41] http://arxiv.org/pdf/2305.09880.pdf
[42] https://www.mdpi.com/1424-8220/23/5/2385/pdf?version=1676983483
[43] https://pmc.ncbi.nlm.nih.gov/articles/PMC10006889/

# VISION TRANSFORMER 목록

Vision Transformer(이하 ViT)는 NLP용 Transformer를 순수 영상 입력(이미지 패치 시퀀스)에 적용한 것으로, Section 3에서는 ViT 및 그 파생 모델들이 CV 전 분야에 걸쳐 어떻게 활용되고 발전했는지를 네 가지 주요 응용 영역(백본, 고/중·저수준 비전, 동영상 처리)으로 나누어 심층 분석한다.

## 3.1 Backbone for Representation Learning  
– **순수 Transformer 백본**  
 -  ViT: 고해상도 이미지를 $$p\times p$$ 패치로 분할해 시퀀스로 변환한 뒤, [CLS] 토큰과 함께 Transformer Encoder에 입력.  
 -  DeiT: 데이터 효율화 기법(Data Augmentation, Token-based Distillation) 적용으로 ImageNet만으로도 83.1% 달성.  
 -  ViT 변형: Swin(Shifted Window), TNT(Transformer-in-Transformer), PVT(Pyramid ViT) 등 지역성(locality) 강화·계층 구조 도입.  
– **CNN+Transformer 하이브리드**  
 -  CvT, CeiT, CMT: FFN / Self-Attention에 Convolution 삽입해 지역-전역 정보 융합.  
 -  BoTNet: ResNet에 글로벌 Self-Attention 결합.  
– **Self-Supervised Pretraining**  
 -  iGPT: 픽셀 토큰 자가회귀 학습 후 분류 헤드 추가 fine-tune.  
 -  MoCo v3, MAE: contrastive / masked‐image‐modeling 기반 사전학습으로 레이블 없는 대규모 데이터 활용.  

【FLOPs vs. Accuracy 비교 차트】  
(create_chart 호출 생략)

## 3.2 High/Mid-level Vision  
– **Object Detection**  
 -  DETR: CNN+Transformer → Set Prediction, Bipartite Matching, NMS 불필요.  
 -  Deformable DETR: Sparse Attention으로 10× 빠른 수렴, AP+4.2↑.  
 -  UP-DETR, SMCA 등: 비지도 사전학습, Co-Attention 제약으로 수렴 가속, 작은 오버헤드.  
– **Segmentation**  
 -  Max-DeepLab: Panoptic Segmentation을 Transformer만으로 end-to-end 처리.  
 -  VisTR, ISTR: Instance Segmentation에 시계열과 3D CNN 결합.  
 -  SETR, Segmenter: Semantic Segmentation용 Transformer Encoder + 다단계 Aggregation.  
– **Pose Estimation**  
 -  METRO: Mesh Transformer로 3D 포즈·메시 복원, Query Masking.  
 -  TransPose, TokenPose: Keypoint별 Token 융합, Attention으로 의존 관계 학습.  

## 3.3 Low-level Vision  
– **Image Generation (Autoregressive)**  
 -  Image Transformer: 픽셀 시퀀스 생성, 국소 Self-Attention으로 해상도 극복.  
 -  Taming Transformer(VQGAN+Transformer): Codebook 기반 압축–생성 두 단계.  
 -  TransGAN, ViTGAN: GAN에 ViT 구조 적용해 고화질 합성.  
– **Image Processing**  
 -  IPT: Pretraining ImageNet → Super-resolution, Denoising, Deraining 멀티태스크.  
 -  TTSR: Reference-based SR에 Hard/Soft Attention 융합.  

## 3.4 Video Processing  
– **High-level**  
 -  Action Transformer: I3D+Transformer로 사람–환경 관계 모델링.  
 -  VideoBERT, Video Retrieval: 영상 특징–텍스트 통합 학습.  
– **Low-level**  
 -  ConvTransformer, Recurrent Transformer: Frame Synthesis 병렬/순차 예측.  
 -  STTN: Spatial-Temporal Attention으로 Video Inpainting.  

## 3.5 Multi-Modal Tasks  
– **CLIP**: 이미지·텍스트 Dual Encoder, Contrastive Pretraining으로 Zero-Shot 분류 성능 대폭 향상.  
– **DALL·E**, **CogView**: dVAE 토크나이저+Autoregressive Transformer로 Text→Image 생성.  
– **UniT**: 한 모델로 Detection, NLU, VQA 등 다중 모달·다중 태스크 통합.  

## 3.6 Efficient Transformer  
– **Pruning & Decomposition**: 헤드·레이어 제거, Low-rank 분해.  
– **Knowledge Distillation**: BERT→MiniLM/TinyBERT, Vision Transformer→Fine-grained Distillation.  
– **Quantization**: 8-bit, 4-bit 가중치·활성화 적용.  
– **Compact Design & NAS**: Hamburger Layer, Dynamic Conv, Self-Supervised NAS.  

“Transformer의 순수 전역처리 능력”과 “CNN의 지역성”을 어떻게 결합하고, 사전학습·자기지도학습·효율화로 확장하는지를 체계적으로 정리합니다. 이로써 CV 전 영역에서 **전역–지역 정보 융합**, **대규모 사전학습**, **경량화 전략** 등이 어떻게 구현되고 있는지 한눈에 보여 줍니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c5df94e8-c30b-488a-92bf-090ff178de1a/2012.12556v6.pdf
