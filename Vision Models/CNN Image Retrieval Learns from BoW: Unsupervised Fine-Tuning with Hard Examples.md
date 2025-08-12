# CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples | Image retrieval

## 1. 핵심 주장 및 주요 기여
**핵심 주장**  
대규모의 수동 주석이 없는 이미지 콜렉션으로부터 3D 재구성(Structure-from-Motion)과 BoW 기반 검색을 활용해, *하드 포지티브* 및 *하드 네거티브* 예시를 자동 선별함으로써 CNN을 언슈퍼바이즈드 방식으로 효과적으로 파인튜닝할 수 있다. 이를 통해 소형 벡터(32∼512D)로도 최첨단 물체 검색 성능을 달성한다.

**주요 기여**  
1. **3D 재구성 기반 데이터 셋 선택**  
   – SfM에서 얻은 카메라 위치 및 포인트 가시성을 이용해 하드 포지티브(3가지 방식)와 하드 네거티브(2가지 방식) 샘플링을 제안.  
2. **언슈퍼바이즈드 하드 포지티브/네거티브 마이닝**  
   – 기존 GPS 또는 CNN 거리 기반보다 더 어려운 (harder) 예시 발굴로 학습 효율↑.  
3. **학습 기반 Whitening**  
   – 전통적인 PCA-whitening 대신, 매칭/비매칭 쌍의 공분산을 이용해 선형 판별(LDA) 형태의 프로젝션 학습, 성능 및 안정성 개선.  
4. **Compact Representation 최고 성능 갱신**  
   – 32D, 128D, 256D, 512D 전 범위에서 Oxford5k/Paris6k/Holidays 벤치마크 SOTA 경신.

***

## 2. 상세 설명

### 2.1 해결하고자 하는 문제  
- **문제**: 수동 라벨링 없이 CNN을 특정 객체(랜드마크) 검색에 맞춰 파인튜닝하는 방법  
- **한계**: 지도 학습 기반 파인튜닝은 대규모 어노테이션 필요, 기존 언슈퍼바이즈드 방식은 예시 난이도 한정

### 2.2 제안 방법

#### 2.2.1 이미지 표현: MAC  
마지막 합성곱층의 K 채널(feature map)에 대해 ReLU 후 전역 맥스풀링하여  

$$
f_k = \max_{x\in X_k} \{\,x \cdot 1(x>0)\,\},\quad \mathbf{f}=[f_1,\dots,f_K]^\top
$$  

ℓ₂ 정규화하여 두 이미지 유사도는 내적으로 계산.

#### 2.2.2 Siamese 네트워크와 Contrastive Loss  
두 브랜치가 파라미터를 공유하며 매칭(Y=1)／비매칭(Y=0) 쌍을 학습:  

$$
L_{i,j}=\tfrac12\bigl[Y_{ij}\|\bar f_i-\bar f_j\|^2+(1-Y_{ij})\max\{0,\tau-\|\bar f_i-\bar f_j\|\}^2\bigr]
$$  

여기서 $$\bar f$$는 ℓ₂ 정규화된 MAC 벡터, 마진 $$\tau=0.7$$.

#### 2.2.3 하드 포지티브／네거티브 샘플링  
- **포지티브**  
  - m₁(q): MAC 거리 최소  
    $$\arg\min_{i\in M(q)}\|\bar f(q)-\bar f(i)\|$$  
  - m₂(q): 공관측 포인트(inlier) 최대  
    $$\arg\max_{i\in M(q)}|P(q)\cap P(i)|$$  
  - m₃(q): 충분한 공관측 이상·스케일 변화 제한→무작위 선택  
    $$\{i:|P(q)\cap P(i)|/|P(q)|\ge t_i,\ \mathrm{scale}\le t_s\}$$  
- **네거티브**  
  - N₁(q): 모든 비매칭 중 유사도 기준 k-NN  
  - N₂(q): N₁과 유사하나 클러스터당 하나만  
  – 하드 네거티브는 매 에폭마다 재마이닝

#### 2.2.4 Whitening 및 차원 축소  
- 전통 PCA-whitening 대신, 매칭 쌍 공분산 $$C_S$$로 화이트닝, 비매칭 공분산 $$C_D$$로 회전:  
  
$$
  P = C_S^{-1/2}\,\mathrm{eig}(C_S^{-1/2}C_D C_S^{-1/2}),
  $$  
  
이후 중심화·프로젝션·ℓ₂정규화

### 2.3 모델 구조  
- AlexNet/VGG의 합성곱부만 사용, FC 레이어 제거  
- MAC+ℓ₂정규화 블록 삽입 후 Siamese 구조로 Contrastive Loss 학습

### 2.4 성능 향상 및 실험 결과  
- **벤치마크**: Oxford5k(+100k), Paris6k(+100k), Holidays  
- **결과**:  
  – 128D VGG-MAC: Oxford5k 76.8%→79.7%, Paris6k 70.8%→73.9% (mAP)  
  – 32D VGG-MAC: Oxford105k 47.6%, Paris106k 59.5% (SOTA)  
  – Learned whitening이 PCA-whitening 대비 일관성·성능 우수  
- **한계**:  
  – R-MAC 직접 학습 미적용(추후 가능)  
  – SfM 재구성 오류 시 예시 선정 품질 저하 우려

***

## 3. 일반화 성능 향상 가능성  
- **클러스터 제거 실험**: Oxford/Paris 클러스터 포함 여부 무관하게 유사 성능 유지→오버피팅 우려 낮음  
- **데이터 다양성**: 10→100→551 클러스터로 확장 시 검증 성능 점진 개선  
- **미래 제언**: 다양한 장면·조명·오브젝트에 특화된 비지도 클러스터링, 온라인 마이닝 기법 도입으로 일반화 추가 강화 가능

***

## 4. 향후 연구 영향 및 고려 사항  
- **영향**:  
  – 완전 비지도 환경에서 CNN 고성능 파인튜닝 패러다임 제시  
  – 하드 예시 샘플링과 학습 기반 프로젝션 활용 확대 기대  
- **연구 시 고려점**:  
  – 자동 재구성 실패 대비 강인한 예시 필터링  
  – R-MAC 등 지역 특징 학습 포함 확장  
  – 도메인 변화(실내·인물·자연) 대응을 위한 비지도 클러스터링 전략 접목

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9ccd84e6-b4dc-4a3e-9037-3051d738a99d/1604.02426v3.pdf
