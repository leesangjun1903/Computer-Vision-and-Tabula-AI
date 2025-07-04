# ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders | Image classification, Image detection, Image segmentation

ConvNet(CNN)은 수동 feature 엔지니어링에 의존하기보다는 다양한 시각적 인식 task에 대한 일반적인 feature 학습 방법을 사용할 수 있도록 함으로써 컴퓨터 비전 연구에 상당한 영향을 미쳤다.  
최근에는 원래 자연어 처리를 위해 개발된 transformer 아키텍처도 모델과 데이터셋 크기에 대한 강력한 확장 가능성으로 인해 인기를 얻었다.  
최근에는 ConvNeXt 아키텍처가 기존 ConvNet을 현대화했으며 순수 convolutional model도 확장 가능한 아키텍처가 될 수 있음을 보여주었다.  
그러나 신경망 아키텍처의 디자인을 탐색하는 가장 일반적인 방법은 여전히 ImageNet에서 supervised learning 성능을 벤치마킹하는 것이다.

본 논문은 ConvNeXt 모델에 마스크 기반 self-supervised learning을 효과적으로 만들고 transformer를 사용하여 얻은 결과와 유사한 결과를 달성하기 위해 동일한 프레임워크에서 네트워크 아키텍처와 MAE를 공동 설계할 것을 제안한다.

ConvNeXt V2는 기존 ConvNeXt 아키텍처와 마스크 기반 자기지도 학습(MAE)을 효과적으로 통합한 혁신적인 컨볼루션 신경망(ConvNet) 모델입니다. 단순한 결합으로는 성능 저하가 발생했으나, **완전 컨볼루션 마스크 오토인코더(FCMAE)** 프레임워크와 **글로벌 응답 정규화(GRN)** 계층을 도입해 문제를 해결했습니다. 이는 순수 컨볼루션 모델의 성능을 비전 트랜스포머 수준으로 끌어올린 핵심 연구입니다[1][3][7].

### 1. 문제 인식: 왜 ConvNeXt와 MAE의 단순 결합이 실패했나?
- ConvNeXt는 컨볼루션 기반, MAE는 트랜스포머 기반으로 **구조적 불일치** 발생[2][4].
- 마스킹된 입력 데이터로 학습할 때 **특징 붕괴(feature collapse)** 현상 발생[5][6].  
  → MLP 계층에서 채널 간 경쟁 부재로 표현력 저하.

### 2. 해결책: FCMAE + GRN의 협력 설계(Co-design)
#### 가. 완전 컨볼루션 마스크 오토인코더(FCMAE)
- **마스크 처리 최적화**: 가려진 영역을 희소 패치로 처리, **희소 컨볼루션**으로 가시 영역만 연산[6].
- **디코더 단순화**: 트랜스포머 디코더 대신 단일 ConvNeXt 블록 사용 → 전체 구조 완전 컨볼루션화[3][6].

#### 나. 글로벌 응답 정규화(GRN)
- **3단계 작동 원리**:
  1. **글로벌 특징 집계**: 공간적 특징 맵을 L2 노름 기반 풀링으로 벡터화[4].
  2. **특징 정규화**: 표준 편차 정규화 적용[4].
  3. **특징 보정**: 채널 간 경쟁 촉진 → 특징 붕괴 방지[5][7].
- **핵심 효과**: 채널 간 의존성 강화로 마스크 학습 시 안정성 향상[1][7].

### 3. 성능: 다양한 벤치마크에서 SOTA 달성
| 벤치마크         | ConvNeXt V2 성능                     | 주요 비교 대상 대비 향상 |
|------------------|---------------------------------------|--------------------------|
| ImageNet 분류    | 88.9% (Huge 모델)                    | 기존 ConvNet 대비 +2.1%  |
| COCO 객체 탐지   | 58.7% AP                             | 이전 최고치 대비 +1.5%  |
| ADE20K 세그먼테이션 | 58.0% mIoU                          | ConvNeXt V1 대비 +4.2%  |

**모델 규모별 성능**:
- **Atto** (3.7M 매개변수): ImageNet 76.7% 정확도[1]
- **Huge** (650M 매개변수): 공개 데이터만으로 88.9% 정확도[7]

### 4. 의의와 한계
- **의의**: 트랜스포머 의존 없이 컨볼루션 모델의 자기지도 학습 한계 극복[3][7].
- **한계**: GRN 추가로 인한 **계산 오버헤드** 발생[4].  
  → 경량화를 위한 Atto 모델 제공으로 일부 해결[1].

ConvNeXt V2는 아키텍처와 학습 프레임워크의 공동 설계를 통해 순수 컨볼루션 네트워크의 성능 한계를 재정의했으며, 특히 마스크 기반 사전 학습에서 트랜스포머 대비 효율성을 입증했습니다[1][7].

[1] https://arxiv.org/abs/2301.00808
[2] https://zenn.dev/nudibranch/articles/f4ee569465716d
[3] https://huggingface.co/docs/transformers/model_doc/convnextv2
[4] https://ai-scholar.tech/en/articles/image-recognition/convnext-v2
[5] https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf
[6] https://www.marktechpost.com/2023/01/25/meet-convnext-v2-an-ai-model-that-improves-the-performance-and-scaling-capability-of-convnets-using-masked-autoencoders/
[7] https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html
[8] https://dataloop.ai/library/model/facebook_convnextv2-large-22k-224/
[9] https://huggingface.co/docs/transformers/main/model_doc/convnextv2
[10] https://huggingface.co/timm/convnextv2_femto.fcmae
[11] https://huggingface.co/facebook/convnextv2-nano-22k-384
[12] https://github.com/facebookresearch/ConvNeXt-V2
[13] https://paperswithcode.com/paper/convnext-v2-co-designing-and-scaling-convnets
[14] https://mmpretrain.readthedocs.io/en/dev/papers/convnext_v2.html
[15] https://blog.csdn.net/weixin_50829873/article/details/130407270

# Reference
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/convnext-v2/
