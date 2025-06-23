# Efficient Self-supervised Vision Transformers for Representation Learning (EsViT) 논문 설명

## 개요
“Efficient Self-supervised Vision Transformers for Representation Learning” 논문은 비지도 방식으로 비전 트랜스포머(ViT)의 효율성과 표현 학습 능력을 향상시키기 위해 제안된 EsViT(효율적 자체지도 학습 비전 트랜스포머) 방법을 다룹니다[1].  
- **목표**: 모델 규모와 연산 복잡도를 줄이면서도 ImageNet 선형 평가에서 기존 최첨단 모델을 능가하는 성능 달성  
- **핵심 기여**:  
  1. 멀티스테이지(Multi-stage) 트랜스포머 아키텍처 도입으로 연산량 대폭 절감  
  2. 비대조적(non-contrastive) 지역(region) 매칭 사전학습 과제 도입으로 세밀한 영역 의존성 학습  

---

## 1. 멀티스테이지 트랜스포머 아키텍처  
### 1.1 배경 및 필요성  
- 기존 Monolithic ViT는 전역적(Global) 자기-어텐션을 통해 우수한 표현을 학습하지만, 패치 수가 많아질수록 연산량이 $$O(N^2)$$로 증가하여 효율성 저하 발생[1].  
- 멀티스테이지 아키텍처는 여러 단계(Stage)로 토큰 수를 단계적으로 줄이면서 깊이(Feature 차원)는 확대해 계층적 표현을 생성한다[1].

### 1.2 구조  
1. **패치 머징(Patch Merging)**  
   - 초반부: 입력 이미지를 $$P\times P$$ 크기로 분할하여 토큰 생성  
   - 이후 단계: $$2\times2$$ 인접 토큰을 결합해 토큰 수 4배 축소, 차원은 2배 증가  
2. **희소(self-sparse) 자기-어텐션**  
   - 각 단계별 줄어든 토큰에서 제한된 영역 내 어텐션 수행으로 연산 절감  
3. **단계 반복**  
   - 위 두 모듈을 4단계 정도 반복해 최종 계층적 표현 생성[1].  

#### 장점과 단점  
- 장점: 연산량 및 메모리 사용 3~10배 절감  
- 단점: Monolithic ViT가 학습하던 지역별 세밀한 대응 관계(semantic correspondence) 학습 능력 일부 소실[1].  

---

## 2. 비대조적(non-contrastive) 지역(region) 매칭 과제  
### 2.1 동기  
- Monolithic ViT는 자체지도 학습만으로도 서로 다른 증강 뷰 사이에서 지역 간 대응을 자동 학습하는 성질을 보임(95% 정확도)[1].  
- 멀티스테이지 구조 전환 시 이 성질이 29%포인트 감소(66% 정확도)해 복원 필요성 대두[1].

### 2.2 과제 설계  
- **View-level 과제**(기존 DINO): 이미지 전체 뷰의 전역 표현 간 확률 분포 매칭
  
  $$L_V = -\frac{1}{|P|}\sum_{(s,t)}\,p_s \log p_t$$
  
  (여기서 $$p_s$$, $$p_t$$는 각각 학생·교사 네트워크의 전역 뷰 확률)[1].

- **Region-level 과제**(제안)  
  1. 상응하는 지역 쌍 찾기: 두 뷰의 지역 피처 벡터 간 코사인 유사도 최고인 인덱스 $$j^*$$ 선택  
  2. 비대조적 매칭 손실:
 
     $$L_R = -\frac{1}{|P|}\sum_{(s,t)}\frac{1}{T}\sum_{i=1}^{T} p_{j^*}\log p_i$$
     
     (지역별 MLP 확률 매칭, negative 샘플 불필요)[1].

- **최종 목적**:
   
  $$L = L_V + L_R$$
  
  학생 네트워크 파라미터는 크로스엔트로피 최소화로 업데이트하고, 교사 네트워크는 EMA 방식으로 갱신[1].

### 2.3 계산 비용  
- Monolithic ViT에서 지역 매칭 $$O(T^2)$$로 불가능 수준  
- 멀티스테이지 구조로 토큰 수 $$T=196\to49$$로 줄여 실용적 연산 비용 유지[1].

---

## 3. 성능 및 비교  
### 3.1 ImageNet 선형 평가  
| 모델                   | 파라미터(M) | 처리량(Img/s) | Top-1 (%) | k-NN (%) |
|------------------------|-------------|---------------|-----------|----------|
| MoCo-v3 (ViT-B/7)      | 85          | ~63           | 79.5      | –        |
| DINO (ViT-B/8)         | 85          | 63            | 80.1      | 77.4     |
| **EsViT (Swin-B, W=14)** | **87**      | **254**       | **81.3**  | **79.3** |

- EsViT(Swin-B, W=14) : MoCo-v3 대비 1.8%p 향상, 4× 처리량[1]  
- 대형 ResNet 대비 파라미터 16×, 처리량 8× 높은 효율성·성능 동시 달성  

### 3.2 전이 학습  
1. **18개 소규모 분류 데이터셋**  
   - 지도학습 Swin 대비 17/18 데이터에서 우위  
   - 평균 성능 +2.99%p 상승  

2. **COCO 물체 검출·분할**  
   - Swin-T 감독학습 대비 비슷한 AP 성능[1].  

3. **대규모 비지도 데이터 사전학습**  
   - ImageNet-1K, WebVision-v1, OpenImages-v4, ImageNet-22K  
   - 다양한 데이터 분포 학습 시 전이 성능 변화 관찰[1].

---

## 4. 질적 분석  
### 4.1 지역 대응 관계 시각화  
- DINO monolithic ViT: 10개 주요 대응 모두 정확 학습  
- 멀티스테이지 + View-level: 배경 영역 위주 오분류  
- EsViT + Region-level: 정확한 지역 간 매칭 복원[1].

### 4.2 어텐션 맵 시각화  
- DINO: 주요 객체 포oreground 강조 어텐션 학습  
- 멀티스테이지 + View-level: 어텐션 분포 단조로워짐  
- EsViT + Region-level: 헤드별 다양한 시맨틱 어텐션 재획득[1].

---

## 결론  
EsViT는 멀티스테이지 희소 자기-어텐션 아키텍처와 비대조적 지역 매칭 과제의 결합으로  
- 연산 효율성을 3–10× 향상  
- ImageNet 선형 평가 성능을 기존 최첨단보다 0.3–1.8%p 상회  
- 다양한 전이 학습 과제에서 지도학습 대비 동등 또는 우수한 성능 달성  

위 결과로 자체지도 학습 비전 시스템의 확장성과 효율성 향상을 제시한다[1].

[1] https://www.semanticscholar.org/paper/b70bb1855e217edffb5dfa0632e8216860821870
[2] https://arxiv.org/abs/2301.01431
[3] https://www.ssrn.com/abstract=4384165
[4] https://arxiv.org/abs/2204.07141
[5] https://ieeexplore.ieee.org/document/11014227/
[6] https://elibrary.asabe.org/abstract.asp?JID=5&AID=54149&CID=oma2023&T=1
[7] https://ieeexplore.ieee.org/document/9883983/
[8] https://ieeexplore.ieee.org/document/9878641/
[9] https://arxiv.org/abs/2106.09785
[10] https://openreview.net/forum?id=fVu3o-YUGQK
[11] https://deep-learning-study.tistory.com/845
[12] https://github.com/microsoft/esvit
[13] https://www.microsoft.com/en-us/research/publication/efficient-self-supervised-vision-transformers-for-representation-learning/
[14] https://openreview.net/pdf?id=He-Drd6101F
[15] https://iclr.cc/virtual/2022/poster/6312
[16] https://paperswithcode.com/method/esvit
[17] https://arxiv.org/abs/2305.00729
[18] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13517/3055190/Local-masking-meets-progressive-freezing--crafting-efficient-vision-transformers/10.1117/12.3055190.full
[19] https://paperswithcode.com/paper/sit-self-supervised-vision-transformer
[20] https://paperswithcode.com/paper/visual-representation-learning-with-self
