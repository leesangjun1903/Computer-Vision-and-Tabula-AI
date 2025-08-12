# GeM : Fine-tuning CNN Image Retrieval with No Human Annotation | Image retrieval

## 1. 핵심 주장 및 주요 기여
이 논문은 **인간 주석 없이** 대규모 이미지 컬렉션에서 자동으로 추출된 3D 재구성 정보를 활용해 CNN을 파인튜닝함으로써, 기존에 수작업으로 라벨링된 데이터에 의존하던 이미지 검색 모델을 획기적으로 개선했다는 것을 주장한다.  
주요 기여는 다음과 같다.  
- 3D 구조-카메라 포즈 정보로부터 **hard-positive** 및 **hard-negative** 예시를 자동 선별하여 학습 데이터로 활용  
- 기존 PCA 기반 화이트닝을 대체하는 **판별적(Discriminative) 화이트닝** 기법 제안  
- 최대·평균 풀링을 일반화한 **Generalized-Mean(GeM) 풀링 레이어** 도입  
- 멀티스케일 표현 및 α-가중치 쿼리 확장(αQE) 기법으로 검색 성능 추가 향상  
- 라벨링 없는 완전 자율 학습으로 Oxford, Paris, Holidays 벤치마크에서 최첨단 성능 달성  

***

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제
- **인스턴스 이미지 검색**: 특정 객체를 묘사한 쿼리 이미지에 대해 대규모 미정렬 이미지 컬렉션에서 동일 객체를 찾아내는 과제  
- 기존 CNN 파인튜닝 방식은 수작업 라벨링(랜드마크 클래스 등)에 의존하며, 이는 비용·노동집약적이고 오류가 많음  

### 2.2 자동화된 학습 데이터 수집
1. 대규모 웹 이미지에 대해 BoW 기반 클러스터링 및 SfM(Structure-from-Motion)으로 3D 모델 구축  
2. 각 쿼이미지 q에 대해 동일 3D 클러스터 내의 카메라 위치·3D 포인트 관측 겹침 비율을 이용해  
   - hard-positive: 동일 객체 묘사를 보장하면서도 시점·스케일 변화가 큰 쌍 선택  
   - hard-negative: 다른 클러스터에서 유사도가 높되 클러스터당 하나만 선택  
3. 각 에폭마다 negative 예시 재탐색(mining)으로 “쉬운” 예시는 학습에서 제외  

### 2.3 네트워크 구조 및 손실 함수
- 입력: Fully-convolutional CNN 백본(VGG, ResNet 등)  
- GeM 풀링:  

$$ f_k = \bigl(\frac{1}{|X_k|}\sum_{x\in X_k} x^p\bigr)^{1/p} $$  
  
  – p→∞ → max pooling, p=1 → average pooling  
- ℓ₂ 정규화 후 유클리드 거리(내적)로 유사도 평가  
- **Contrastive loss** (positive/negative 쌍 모두 사용):  

$$
    L(i,j)=
    \begin{cases}
      \frac12\|\bar f(i)-\bar f(j)\|^2, & Y=1,\\
      \frac12 \max\{0,\tau-\|\bar f(i)-\bar f(j)\|\}^2, & Y=0.
    \end{cases}
  $$

### 2.4 Descriptor Whitening
- 전통적 PCA-whitening 대신, **intra-class covariance**와 **inter-class covariance**를 판별적으로 이용해  

$$
    P = C_S^{-1/2}\,\mathrm{eig}(C_S^{-1/2} C_D C_S^{-1/2})
  $$  
  
  형태의 선형 투영 학습  

### 2.5 멀티스케일 표현 및 α-QE
- 테스트 시 스케일 $$1,\frac1{\sqrt2},\frac12$$로 이미지 리사이즈 후 GeM으로 통합  
- AQE(단순 평균) 대신 랭킹별 유사도^α 가중 쿼리 확장:  

  $$\sum_{i=1}^{n_\mathrm{QE}} (\bar f(q)^\top \bar f(i))^\alpha \bar f(i)$$, α=3 권장  

***

## 3. 성능 향상 및 한계
- **Oxford5k/105k**, **Paris6k/106k**, **Holidays** 전 벤치마크에서 SOTA 달성  
- GeM 풀링, 판별적 화이트닝, 멀티스케일, α-QE 각 단계별로 5–10%p 이상 mAP 상승  
- 완전 무라벨 자율 학습에도 불구, 수작업 정제 데이터 기법 수준 달성  
- **한계**: SfM 재구성 가능한 ‘강건한’ 객체(랜드마크 등)에 의존하므로, 비정형·비강체(scene) 대상 일반화 한계  

***

## 4. 일반화 성능 향상 관점
- 3D 포즈·지오메트리 기반으로 선택된 hard 예시들은 **다양한 시점·배경·스케일** 사례를 포함  
- 네트워크가 객체의 본질적 형태 특징에 학습 집중 → 과적합 억제  
- 실제 실험에서 Oxford/Paris 클러스터 포함 여부에 따른 성능 차이 미미(±0.3%p) → **강한 일반화** 확인  

***

## 5. 향후 연구 영향 및 고려 사항
- **영향**: 라벨링 비용 없이 비지도·자동화로 최첨단 이미지 검색 달성 가능성을 제시  
- **고려점**:  
  - 비정형 장면·비강체 객체에 대한 3D 정보 확보 기법 개발  
  - SfM 실패 영역(반사, 투명체)에 대한 보완 (예: 합성 데이터, self-supervision)  
  - GeM p 파라미터·α 값 자동 최적화 및 다양한 백본 네트워크 확장 연구

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/74feee94-a0cb-4bd4-9f1d-5954134869af/1711.02512v2.pdf
