## DeiT(Data-efficient Image Transformers) 개요  
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
