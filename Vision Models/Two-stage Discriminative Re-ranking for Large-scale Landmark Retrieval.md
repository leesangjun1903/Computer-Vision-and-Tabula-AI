# Two-stage Discriminative Re-ranking for Large-scale Landmark Retrieval | Image retrieval

## 1. 핵심 주장 및 주요 기여
**핵심 주장**  
큰 규모의 랜드마크 이미지 검색에서 단순한 비주얼 유사도 기반 검색은 시점, 조명, 실내·실외 등 시각적 변이가 큰 경우 정확도가 떨어진다. 본 논문은 라벨 정보를 활용한 두 단계의 판별적 재정렬(two-stage discriminative re-ranking) 기법을 도입하여 검색 결과를 대폭 개선한다고 주장한다.

**주요 기여**  
1. **Two-stage 재정렬 기법**  
   - Sort-step: k-NN 소프트 보팅을 통해 라벨 일치도가 높은 결과를 우선 순위로 재정렬  
   - Insert-step: 원 검색 결과에 누락된 동일 라벨의 이미지를 추가 삽입  
2. **Cosine softmax 임베딩 학습**  
   - ResNet-101 + GeM pooling을 사용하고 ArcFace(arc-angular margin) 손실로 학습  
3. **자동화된 데이터 클리닝**  
   - GLD-v2 학습셋에서 공간 검증(RANSAC+DELF)을 통한 노이즈 제거  
4. **최신 챌린지 상위권 성능**  
   - Google Landmark Retrieval 2019 챌린지 1위, Recognition 2019 3위 달성  

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결 과제  
- **시각적 변이성**: 동일 랜드마크라도 다양한 시점·실내외·조명 차이로 단순 CNN 임베딩 거리 기반 검색 한계  
- **노이즈 레이블**: 대규모 크롤링 데이터에 포함된 오탐·레이블 오류  

### 2.2 제안 방법  
1) **Embedding 학습**  
   - Backbone: ResNet-101 + GeM pooling → 2048→512차원 FC+BatchNorm  
   - 손실: ArcFace additive angular margin softmax  

$$
       L = -\frac1N \sum_{i=1}^N \log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}} + \beta\|W\|^2_2
     $$

2) **Offline 인덱스 예측**  
   - 모든 인덱스셋 샘플에 대해 학습셋을 대상으로 k-NN 소프트 보팅 수행  
   - 클래스별 유사도 점수 $$v(x,c)=\frac1k\sum_{x'\in N(x)}f(x')^\top f(x)\cdot\mathbf{1}(label(x')=c)$$  
3) **Online 검색 및 재정렬**  
   a) 기본 k-NN 검색 → 초기 순위 획득  
   b) Sort-step: 예측된 동일 클래스(“Positive”)를 상위로 재정렬  
   c) Insert-step: 검색에 포함되지 않은 예측 Positive를 순위 뒤편에 삽입, 점수 임계치 τscore 적용  

### 2.3 모델 구조  
- 입력 이미지 → ResNet-101 conv → GeM pooling(p=3.0) → FC(512) → BatchNorm → L2-normalized embedding  
- 재정렬 모듈은 embedding·라벨 정보·유사도 점수 기반 단순 k-NN 판별 모델  

### 2.4 성능 향상  
- GLD-v2 Private mAP@100:  
  - Baseline k-NN: 30.22 → Sort-step: 33.79 → +Insert-step: 36.85  
- αQE·EGT 등 기존 재정렬 대비 약 5~7%p 이상의 성능 우위  
- 데이터 클리닝 후 학습(v2-clean) 시 성능 추가 향상  

### 2.5 한계  
- **라벨 의존성**: 학습셋과 검색 대상 간 라벨 중복이 전제  
- **예측 불확실성**: 낮은 soft-voting 점수 시 부정확한 재정렬  
- **계산 비용**: offline k-NN soft-voting, spatial verification 클리닝의 높은 연산량  
- **일반화 제약**: 라벨 없는 신규 랜드마크 검색 시 재정렬 효과 제한  

***

## 3. 일반화 성능 향상 가능성  
- **라벨 활용 확대**: 사용자 메타데이터(캡션·태그)를 라벨로 삼아 실서비스 검색에도 적용 가능  
- **약지도 학습**: 소량의 라벨 정보만 있어도 re-ranking 적용 → 라벨 부족 도메인에 확장  
- **전달 학습 및 도메인 적응**: 다른 도시·문양·실내 디자인 데이터로 파인튜닝 후 재정렬로 지역 일반화  
- **효율적 구조 탐색**: 가벼운 k-NN 근사·지표 학습(FAISS, HNSW)으로 대규모와 실시간 적용  

***

## 4. 향후 연구 영향 및 고려사항  
- **메타데이터 융합 검색**: 시각+텍스트 라벨 동시 이용한 멀티모달 검색 연구 촉진  
- **재정렬 가치 재평가**: 전통적 QE·diffusion 기법에 라벨 정보 추가한 하이브리드 기법  
- **데이터 클리닝 자동화**: RANSAC 기반 외에도 학습기반 노이즈 필터링 알고리즘 개발  
- **저비용 재정렬**: online 처리량 감소 위한 threshold, 샘플링 전략 등 최적화 연구  
- **일반화 평가**: 라벨 없는 신규 인스턴스 검색 시 재정렬 성능 및 robustness 실험  

이 논문은 **라벨 정보의 재정렬 활용**이라는 직관적이면서도 효과적인 방법을 제시하여 이미지 검색 분야의 재정렬 연구에 새로운 방향을 제시한다. 앞으로는 다양한 도메인·메타데이터를 융합하고, 대규모 환경에서도 실시간 성능을 만족시키는 방법이 핵심 고려 사항이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a027d2b1-e067-478c-8ced-da7de2b1061a/2003.11211v1.pdf
