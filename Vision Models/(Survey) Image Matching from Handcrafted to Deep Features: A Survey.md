# Image Matching from Handcrafted to Deep Features: A Survey 

## 1. 핵심 주장 및 주요 기여의 간결한 요약

이 논문은 **이미지 매칭(image matching)** 분야에서 기존의 **수작업(handcrafted) 기반** 특징점 기법부터 최근의 **딥러닝(Deep Learning) 기반** 기법까지 전체적인 발전 흐름과 최신 동향을 포괄적으로 정리한 서베이 논문이다. 저자들은 특징점 검출, 기술, 매칭, 후처리 등 주요 단계별로 고전적인 방법과 데이터 기반 방법(머신/딥러닝 기반)이 어떻게 발전해왔는지 분석하며, 최근의 end-to-end 학습 방식과 실제 응용에서의 성능, 한계, 오픈 문제점까지 체계적으로 소개했다. 특히, 기존 설문이 하나의 하위 분야나 아키텍처에만 집중한 것과 달리, 본 논문은 전체 파이프라인을 연결해 **종합적인 비교와 실험적 평가**를 제공한다.

***

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

- 이미지 매칭 기술의 폭넓은 발전과 다양화로 인해, **어떤 기법이 어느 상황/목적에 적합한지 객관적으로 비교하거나 선택하기 어려운 현실**이 존재한다.
- 최근 딥러닝의 등장으로 전통적인 수작업 방식과 학습 기반 방식 사이에 **정확도, 강인성, 계산 효율성 측면의 구체적 비교와 종합적 가이드라인 결여**가 문제로 제기되고 있음.
- 아울러, 이미지 품질 변화, 비정형/비유클리드 데이터에서의 매칭 등 실제 환경에서의 일반화(Generalization) 역시 미해결된 핵심 과제다.

### 2-2. 제안 방법 및 구조 (주요 수식, 파이프라인)

#### **(1) 전통적 Feature-based Image Matching Pipeline**

이미지 매칭은 대체로 다음과 같은 파이프라인을 따른다:

$$
\text{(a) Feature Detection} \rightarrow \text{(b) Feature Description} \rightarrow \text{(c) Feature Matching} \rightarrow \text{(d) Geometric Model Estimation}
$$

#### **(2) 각 단계별 주요 기법 요약**

**a) Feature Detection (특징점 검출)**
- Corner detectors: Harris [M], FAST, Shi-Tomasi 등
- Blob detectors: SIFT(DoG), SURF(DoH), MSER 등
- 최근: CNN 기반 detector(예: LIFT, SuperPoint, LF-Net …)

**b) Feature Description (특징점 기술)**
- Handcrafted: SIFT, SURF, ORB, BRIEF 등(gradient, intensity 기반 기술자)
- Learning-based: PCA-SIFT, LDA-SIFT, DeepDesc, L2Net, HardNet 등
- Patch-based Descriptor Learning  (Siamese/Triplet 구조, metric/descriptor joint learning 등)

**c) Feature Matching & Mismatch Removal**
- 거리 기반 매칭(Nearest Neighbor, NNDR 등)
- Outlier 제거: RANSAC, LPM(지역 제약), GMS(격자 기반), Learning-to-match(Deep Matching, SuperGlue)

**d) Geometric Model Estimation**
- Rigid, Affine, Non-rigid 변환 모델
- Graph Matching, Point Set Registration, Area-based (Dense/Registration) 등

#### **(3) 수식 예시 - 그래프 매칭(Quadratic Assignment Problem)**

이미지 간 매칭을 '그래프 매칭' 형태로 공식화 시,

$$
\max_{X} \quad \text{vec}(X)^T K\,\text{vec}(X)
$$

```math
\text{subject to}: X \in \{0,1\}^{n_1 \times n_2} \;,\; X\mathbf{1}_{n_2} \leq \mathbf{1}_{n_1} \;,\; X^T\mathbf{1}_{n_1} = \mathbf{1}_{n_2}
```

- $$K $$: affinity matrix (점/엣지 유사도)
- $$X $$: correspondence(permutation) matrix

#### **(4) 딥러닝 기반 통합 구조**
- Detector, Descriptor, Matching, Outlier 제거까지 **End-to-End 학습** (ex: SuperPoint→SuperGlue, D2-Net, ContextDesc 등)
- Patch Matching: Siamese/triplet/contrastive loss(supervised), ranking loss(unsupervised/self-supervised), hard negative mining

### 2-3. 성능 향상 및 한계

#### 주요 성능 향상 요인:
- **딥러닝 도입**으로 이미징 조건 변화(채도, 노이즈, 뷰포인트 등)에 강인한 특징점 추출·표현 가능
- **End-to-end 파이프라인 학습**으로, 감지/기술/매칭 단계별 'sub-optimal' 이슈 감소, 전체 성능 향상
- **Multiscale, context-aware, attention 기반 네트워크** 활용으로 복잡한 변형(Scaling, Affine, Non-rigid)에서도 강인한 성능 발휘
- **Hard Negative Mining, Global/Local Loss, Data Augmentation 등 학습 트릭**으로 일반화 및 매칭 정확도 개선

#### 대표적 한계점:
- 훈련 데이터(특히 ground-truth correspondence)가 없거나 적은 영역에선 **일반화 성능 제한**
- 딥러닝 기반 모델의 경우 **도메인/디바이스/센서 간 도약(Generalization across domain/sensors)이 약점**
- **복잡한 대변형(Severe non-rigid) 및 텍스처 희박 상황**에서는 deep/local descriptor 모두 한계
- 전통 방식 대비 학습, 추론시 **연산 자원 및 효율성 문제**
- **End-to-end 방식 본연의 '블랙박스'성**: 해석력(descriptive power) 및 설명가능성 부재

***

## 3. 모델 일반화 성능 향상과 관련 내용 정리

- **Self-supervised/Unsupervised Learning** 기반 detector/descriptor(예: SuperPoint의 Homography Adaptation, LF-Net)는 별도의 ground-truth가 없는 환경에서도 반복성과 강인성을 높임
- **신규 Loss Function 설계(Triplet/Ranking/Contrastive/Global Loss 등)**는 서로 다른 디바이스, 환경, 모듈러리티에서의 매칭 일관성, 분리도를 개선함
- **Multi-domain/Domain-adaptive descriptor 개발**이 연구되고 있으나, 여전히 unseen scene/target/general-purpose matching에는 한계
- **조합적 파이프라인(전통+학습 기반 혼합)** 및 multitask/transfer learning 전략은 일반화 성능을 높이는 주요 수단으로 부각
- **Graph/point set 기반 matching에 deep learning 적용**이 '비정형/비유클리드/미등록 데이터' 일반화에서 중요한 역할을 시작

***

## 4. 앞으로 연구에 미치는 영향과 고려할 점

### **연구 영향**
- 본 논문은 이미지 매칭 분야에서 **핸드크래프트 방식과 딥러닝 기반의 ‘전체 스펙트럼’을 아우르는 표준 레퍼런스**를 제공하므로, 신규 기법/하이브리드 방식 설계, 응용 시스템 구축 및 성능 평가 측면에서 기초 자료로 널리 인용될 것으로 보임.
- 실험적으로 제시된 다양한 데이터셋 및 상세 벤치마크 결과는 후속 연구의 기준 지표 및 새 평가 방법 개발의 출발점이 됨.

### **앞으로 연구 시 고려할 점**
- **일반화와 도메인 적응**: Generative, meta-learning, self-supervised 방식 등 다양한 도메인 간 매칭 적용성을 높일 연구 필요
- **설명가능성, 해석력**: End-to-end 매칭 과정에서 intermediate feature의 의미 및 매칭 원리 해석 연구 (explainable matching)
- **메모리/연산 효율과 경량화**: 모바일/엣지 환경 및 대용량 데이터 대응 위한 경량 네트워크, efficient matching 알고리즘 개발
- **multi-modal, multi-view, non-rigid/scene level matching** 등 확장 응용에 관한 general purpose matching 방법론 연구
- **Open-set recognition/Matching**, **Unseen Category adaptation** 문제를 근본적으로 해결할 수 있는 통합적인 파이프라인 개발

***

## 결론 요약

이 논문은 **이미지 매칭 기술의 전반적 요소를 총망라**하였으며, 현재 있기 마련인 ‘single-task, single-domain, supervised learning’의 한계를 극복할 필요성을 강조하고 있다. 앞으로는 **일반화, 효율성, 해석력, 통합 파이프라인** 개발에 집중해야 하며, 본 논문이 이 방향성의 중요한 디딤돌이 될 것으로 평가된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a79afbb9-9e1b-4421-809e-9b1c5d3f780c/s11263-020-01359-2.pdf

# 2 Feature Detection 

## 2.1 Overview of Feature Detectors (특징 검출기 개요)

**특징점의 정의와 분류**
- **특징점(Feature Points)**은 이미지에서 특별한 의미 구조를 나타내며, 다음과 같이 분류됩니다:
  - **Corner Feature**: "L", "T", "X" 형태의 교차점이나 윤곽선의 고곡률점
  - **Blob Feature**: 원형이나 타원형의 지역적 닫힌 영역  
  - **Line/Edge**: 선분이나 가장자리
  - **Morphological Region**: 형태학적 영역

**좋은 특징점의 조건**
1. **Repeatability (반복성)**: 동일한 물리적 위치에서 일관되게 검출
2. **Invariance (불변성)**: 기하학적 변환에 강건함
3. **Robustness (강건성)**: 노이즈와 조명 변화에 견딤
4. **Efficiency (효율성)**: 빠른 계산과 낮은 저장 요구사항

## 2.2 Corner Features (코너 특징)

### 2.2.1 Gradient-Based Detectors (그래디언트 기반 검출기)

**Moravec Detector (1977)**
- 최초의 자동 코너 검출 방법
- 8방향으로 이동된 윈도우에서 최소 강도 변화를 계산
- 한계: 방향성에 민감하고 연속적이지 않음

**Harris Corner Detector (1988)**
- Moravec의 한계를 해결한 가장 유명한 코너 검출기
- 2차 모멘트 매트릭스(auto-correlation matrix) 사용
- 수식: 고유값을 이용한 코너 응답 함수
- 장점: 회전과 조명 변화에 불변, 높은 반복성

**Shi-Tomasi Detector (1993)**
- Harris 개선 버전
- 추적 성능 향상을 위해 특징점을 더 "분산"시켜 배치
- 더 정확한 위치 결정

### 2.2.2 Intensity-Based Detectors (강도 기반 검출기)

**SUSAN (1997)**
- Smallest Univalue Segment Assimilating Nucleus
- 중심점과 주변 픽셀의 밝기 유사성 기반
- 그래디언트 계산 불필요로 빠른 구현

**FAST (Features from Accelerated Segment Test)**
- 원형 패턴의 픽셀들과 중심 픽셀을 이진 비교
- 기계학습(ID3 tree) 기반 최적화된 판단 기준
- 매우 높은 효율성과 반복성

**개선 버전들**
- **FAST-ER**: 향상된 반복성
- **AGAST**: 더 일반적이고 적응적인 결정 트리
- **ORB**: FAST와 Harris 응답을 결합

### 2.2.3 Curvature-Based Detectors (곡률 기반 검출기)

**처리 과정**
1. **가장자리 검출**: Canny, Sobel 등의 방법 사용
2. **곡선 스무딩**: 노이즈 제거 (가우시안 스무딩 등)
3. **곡률 추정**: 직접/간접 방법으로 곡률 계산
4. **코너 결정**: 곡률 극값점을 임계값으로 선택

**특징**
- 텍스처가 적은 이미지나 이진 이미지에 유용
- 의료 영상, 적외선 영상 등에 적합
- 계산량이 많지만 정확도가 높음

## 2.3 Blob Features (블롭 특징)

### 2.3.1 Second-Order Partial Derivative-Based Detectors

**LoG (Laplacian of Gaussian)**
- 스케일 공간 이론 기반
- 가우시안 컨볼루션 필터링으로 노이즈 감소
- 다중 스케일에서 정규화된 응답의 극값 검출

**DoG (Difference of Gaussians)**
- LoG의 근사로 계산 속도 향상
- SIFT에서 핵심적으로 사용

**DoH (Determinant of Hessian)**
- 헤시안 행렬의 고유값과 고유벡터 활용
- 아핀 변환에 더 강건함
- SURF에서 활용

**주요 방법들**
- **SIFT**: DoG 피라미드에서 지역 극값 검출
- **SURF**: 헤시안 기반, Haar 웨이블릿으로 가속화
- **ASIFT**: 완전한 아핀 불변성
- **KAZE/AKAZE**: 비선형 확산 필터링 사용

### 2.3.2 Segmentation-Based Detectors (분할 기반 검출기)

**MSER (Maximally Stable Extremal Regions)**
- 임계값의 넓은 범위에서 안정한 영역 추출
- 스케일 추정이 불필요
- 큰 시점 변화에 강건함
- 연결 요소의 분수령(watershed) 기반

**기타 방법들**
- 주곡률 영상의 분수령 영역 기반
- 색상 정보를 고려한 개선된 방법들

## 2.4 Learnable Features (학습 가능한 특징)

### 2.4.1 Classical Learning-Based Detectors

**전통적 기계학습 방법**
- **Decision Tree**: FAST에서 최초 적용
- **SVM**: 분류 기반 특징점 선택
- **Boosting**: 반복적 특징점 품질 개선

**특징**
- 수작업 경험과 사전 지식에 의존
- 주로 신뢰할 수 있는 특징 선택에 사용
- 원시 이미지에서 직접적인 특징 추출은 제한적

### 2.4.2 Deep Learning-Based Detectors

**학습 방식별 분류**

**1. Supervised Methods (지도학습)**
- **TILDE**: 회귀 모델로 반복 가능한 키포인트 검출
- **TCDET**: "표준 패치"와 "정준 특징" 개념 도입
- **Key.Net**: 수작업과 학습된 CNN 필터 결합

**2. Self-Supervised Methods (자기지도학습)**
- **SuperPoint**: 합성 데이터로 사전훈련 후 호모그래피 적응
- **Zhang & Rusinkiewicz**: "ranking" 손실과 "peakedness" 손실 결합

**3. Unsupervised Methods (비지도학습)**
- **DetNet**: 공분산 제약조건을 통한 일반적 정식화
- **Quad-net**: 변환 불변 분위수 순위 기반

**End-to-End 통합 방법**
- **LIFT**: 검출기, 방향 추정기, 기술자 공동 훈련
- **LF-Net**: 완전 컨볼루션 모델로 단일 포워드 패스
- **RF-Net**: 다중스케일 수용 필드 맵 활용

**장점과 한계**
- 장점: 높은 수준의 단서 활용, 우수한 성능
- 한계: 초기 방법들의 높은 계산 비용, 데이터 의존성

## 2.5 3-D Feature Detectors (3차원 특징 검출기)

### 2.5.1 Fixed-Scale Detectors (고정 스케일 검출기)

**LSP (Local Surface Patch)**
- 주곡률에 의한 형태 지수(shape index) 사용

**ISS (Intrinsic Shape Signature)**
- 산란 행렬의 고유값 분해 기반
- 고유값 비율로 비구별적 점들 제거

**HKS (Heat Kernel Signature)**
- 열 확산 과정의 성질 기반
- 시간 영역으로 제한된 열 커널

### 2.5.2 Adaptive-Scale Detectors (적응 스케일 검출기)

**Laplace-Beltrami Scale Space**
- 지역 평균 곡률을 반영하는 연산자
- 증가하는 지지 영역에서 함수 계산

**MeshDoG**
- 2D DoG의 3D 확장
- 다양체에 정의된 스칼라 함수에서 연산

**자동 스케일 선택**
- 지지 크기 증가를 통한 스케일 공간 구축
- NMS를 통한 스케일별 자동 선택

## 2.6 Summary (요약)

**전통적 방법의 특징**
- 더 많은 이미지 단서 사용 → 더 나은 강건성과 반복성
- 계산 비용 증가의 트레이드오프
- 근사화와 사전 계산으로 속도 향상

**학습 기반 방법의 장점**
- CNN을 통한 고수준 단서 활용
- 강도, 그래디언트, 2차 미분을 넘어선 정보 활용
- 최근 방법들(SuperPoint, Key.Net)은 실시간 성능 달성

**공통 개선 전략**
1. **효율성**: 근사 알고리즘, 병렬 처리
2. **강건성**: 스케일/아핀 정보 추정, 다중스케일 샘플링
3. **정확성**: 서브픽셀 정확도, NMS 전략
4. **불변성**: 회전, 스케일, 조명 변화에 대한 강건함

이러한 특징 검출 기술들은 이미지 매칭의 첫 번째 단계로서, 후속 기술자 추출과 매칭 과정의 품질을 결정하는 중요한 역할을 합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a79afbb9-9e1b-4421-809e-9b1c5d3f780c/s11263-020-01359-2.pdf

# 3 Feature Description

특징 기술(Feature Description)은 검출된 관심점(Interest Point)을 이미지 매칭이나 분류, 검색 등 후속 작업에서 **안정적이고 구별 가능한 벡터 표현**으로 변환하는 단계입니다. 이 장에서는 크게 네 부분으로 나누어 설명합니다.  

## 3.1 개요(Overview of Feature Descriptors)

1) **목적**  
   - 관심점 주변의 픽셀 패치를 **고차원 벡터**로 변환하여, “서로 대응되는 점”은 벡터 간 거리가 작고, “비대응 점”은 거리가 크도록 만드는 것  

2) **기본 절차**  
   a. 저수준 피처 추출  
      - 그레이디언트(SIFT), 강도 패턴(LBP), 필터 응답(HOG) 등 원시 영상 정보를 뽑아냄  
   b. **공간적 풀링(Spatial pooling)**  
      - 관심점 주변 패치를 격자(4×4)나 극좌표, 가우시안 창 등으로 나누어 통계값(히스토그램, 평균 등)을 구함  
   c. **정규화(Feature normalization)**  
      - 단험거리·L2 노름 등으로 벡터 크기를 조정해 밝기 변화·잡음에 강하게 함  

3) **핵심 과제**  
   - 불변성(Invariance): 회전·크기·조명 변화에 강해야 함  
   - 구별력(Discriminability): 서로 다른 패치는 충분히 멀리 분리되어야 함  
   - 효율성(Efficiency): 실시간 매칭을 위해 차원·연산량이 적어야 함  

***

## 3.2 수작업 설계(Handcrafted) 기술자

### 3.2.1 그레이디언트 통계 기반(Gradient Statistic)  
- 대표: SIFT, SURF  
- 관심점 패치를 4×4 셀로 나눈 뒤, 각 셀에서 8방향 그레이디언트 히스토그램을 계산해 128차원 부동소수점 벡터로 묶음  
- 장점: 회전·스케일 불변성, 우수한 구별력  
- 단점: 계산량·메모리 크기 큼  

### 3.2.2 LBP 통계 기반(Local Binary Pattern)  
- 대표: CS-LBP, CS-LTP, RLBP  
- 관심점 주변 픽셀을 중심 강도와 비교해 0·1 바이너리 패턴을 만들고 그 통계를 히스토그램으로 표현  
- 장점: 조명 변화·노이즈에 강하고 연산량 적음  
- 단점: 평탄 영역 구별력 낮음, 히스토그램 길이 김  

### 3.2.3 강도 비교 기반(Binary Descriptor)  
- 대표: BRIEF, ORB, BRISK, FREAK  
- 패치 내 랜덤·원형 샘플링 점 쌍의 강도를 단순 비교해 비트로 코딩  
- 장점: 메모리·매칭 속도 우수  
- 단점: 회전·크기 변화에 약해, 보통 방향 보정 필요  

### 3.2.4 강도 순서 기반(Order Statistic)  
- 대표: OIOP/MIOP, LIOP  
- 픽셀 강도의 **순위(등수)** 정보를 통계화  
- 장점: 단조강도 변환·회전에 본질적 불변  
- 단점: 상대정렬·히스토그램 구축 비용  

***

## 3.3 기계 학습(Learning-Based) 기술자

### 3.3.1 고전 학습(Classical Learning)  
- PCA-SIFT: 그레이디언트 벡터를 PCA로 차원 축소  
- LDA 해싱, LSH 기반 방법: 해시 함수 학습으로 유사도 보존  
- AdaBoost 해싱, 최소 손실 해싱(minimal loss hashing) 등  

### 3.3.2 딥 러닝(Deep Learning)  
- **패치 매칭(Patch Matching)** 을 위해 Siamese·Triplet CNN 훈련  
  - DeepCompare, MatchNet: 두 패치 유사도 분류  
  - DeepDesc, TFeat, L2-Net, HardNet: 패치 ↦ descriptor 벡터  
- **엔드투엔드 통합**  
  - LIFT: 검출·회전·기술자 일체형  
  - SuperPoint: full-image 입력 → 관심점+기술자 동시 산출  
  - LF-Net, RF-Net, D2-Net: NMS·다중 스케일 포함, end-to-end  
- **학습 전략**  
  - *Hard negative mining*, *Triplet/contrastive/global loss*  
  - *Synthetic warping*, *Homography adaptation* 등 자율학습  

***

## 3.4 3차원(3-D) 기술자

### 3.4.1 수작업 설계  
- **공간 분포 히스토그램**: Spin Image, 3D Shape Context, PFH  
- **기하 속성 히스토그램**: SHOT, FPFH, MeshHOG  
- **스펙트럴**: HKS, WKS, Global Point Signature  

### 3.4.2 학습 기반  
- **스펙트럴 학습**: Spectral Descriptor 학습, Metric Learning  
- **비유클리드 CNN**: Geodesic CNN, Anisotropic CNN, PointNet  
- **뷰 퓨전**: 다중 투영/깊이 맵 입력, 3DConvNet, PointNet++  

***

이와 같이, 특징 기술 단계는 (1) 원시 영상 정보를 어떻게 뽑아내고, (2) 어떤 방식으로 통계·풀링하며, (3) 어떤 타입(부동소수점·이진)으로 정리하는지에 따라 매우 다양한 기법으로 발전해 왔습니다. 최근에는 기존 **수작업 설계 대신 데이터로부터 자동 최적화**하는 딥 러닝 기술자가 주류를 이루고 있어, 매칭 정확도·강인성·효율성 모두 크게 향상되고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a79afbb9-9e1b-4421-809e-9b1c5d3f780c/s11263-020-01359-2.pdf

# 4. Matching Methods

이미지 매칭은 크게 두 가지 범주로 나눌 수 있습니다.  
-  영역 기반 매칭(Area-Based Matching)  
-  특징 기반 매칭(Feature-Based Matching)  

아울러 최근 딥러닝을 활용한 기법들이 등장하며, 2D 이미지뿐 아니라 3D 포인트 셋에 대한 매칭도 활발히 연구되고 있습니다.  
다음에서는 목차 순서대로 각 기법의 원리와 특징을 정리합니다.

***

## 4.1 전체 개관

-  영역 기반 매칭  
  – 이미지 픽셀 강도를 직접 비교해 전체 영상 정합(Registration) 수행  
  – 상관관계, 위상 상관, 상호정보(MI) 등 유사도 척도 + 최적화  

-  특징 기반 매칭  
  – (1) 특징점 감지 → (2) 기술자(Descriptor) 생성 → (3) 기술자 간 유사도 매칭  
  – 매칭 → 변환 모델(예: 아핀, 투영 등) 추정 → 영상 재샘플링/변환  

  – 직접 매칭(Direct Feature Matching)  
    · 그래프 매칭(Graph Matching)  
    · 포인트셋 정합(Point Set Registration)  

  – 간접 매칭(Indirect Feature Matching)  
    · 기술자 유사도 기반 ‘잠정 매칭’(Putative Matches) → 기하학 검증(예: RANSAC)  

-  딥러닝 기반 매칭  
  – 영상 전체를 입력으로 변환 모델 회귀(Registration, Stereo, Pose)  
  – 포인트셋을 입력으로 변환 모델 또는 매칭 이진 분류기 학습  

-  3D 매칭  
  – 3D 스핀 이미지 등 로컬 디스크립터 → 포인트셋 간 매칭  
  – 펑셔널 맵, 텐서 매칭, 그래프 매칭 확장 등  

***

## 4.2 영역 기반 매칭(Area-Based Matching)

1) 상관(SAD/NCC)  
  – 영상 윈도우별 픽셀 간 상관관계 최대화  
  – 계산량 크고 기하 변형에 민감  

2) 위상 상관(Phase Correlation)  
  – 푸리에 변환 위상만 이용해 전단·회전·스케일 정합  
  – 잡음·조명 변화에 강함, 스펙트럼 차 존재 시 성능 저하  

3) 상호정보(Mutual Information, MI)  
  – 서로 다른 센서(다중 모달) 영상 정합에 효과적  
  – 전역 최적화 어려움  

4) 최적화·변환 모델  
  – 정합 후 rigid/affine/TPS 등 변환 추정  
  – 변환 매개변수 최적화(연속·이산·혼합 기법)  

– 딥러닝 확장  
  – 전통적 유사도 → CNN 기반 유사도 학습  
  – 강화학습 에이전트가 순차적 변환 예측  
  – 한 번에 변환 회귀하는 end-to-end 모델 등장  

***

## 4.3 그래프 매칭(Graph Matching)

– 두 개 그래프의 정점 대응을 찾는 문제 → Quadratic Assignment Problem(QAP)  
  -  Lawler’s QAP: maximize vec(X)ᵀK vec(X)  
  -  Koopmans–Beckmann QAP: maximize tr(KₚᵀX) + tr(A₁XA₂Xᵀ)  

– NP-hard → 근사 해법  
  1) 스펙트럴 완화(Spectral Relaxation)  
    – 벡터 크기 제약(&|vec(X)|=1) → 고유벡터 풀이  
  2) 볼록 완화(Convex Relaxation, SDP)  
    – 변수 리프팅 후 반정의 계획법  
  3) Convex-to-Concave 완화  
    – 점진적 경로 추적(Path Following)  
  4) 연속 완화(Continuous Relaxation)  
    – 이산 제약 완화 → DS 행렬→ ADMM, 更新법 등 최적화  
  5) 하이퍼그래프 매칭(Tensor Matching)  
    – 3⁺차 고차항 affinity 반영  
  6) 대체 패러다임  
    – 확률적 대수(Random Walk), 몬테카를로, 클러스터링 기반  
  7) 다중 그래프 매칭(Multi-graph Matching)  
    – 순환 일관성(cycle-consistency) 제약  

***

## 4.4 포인트셋 정합(Point Set Registration)

– 두 점군 간 변환(T) 추정 → 포인트 대응 없이 전체 정합  
  – Iterative Closest Point(ICP) 계열: 최근접 대응 → 변환 추정 반복  
  – EM-based: GMM → E-step posterior → M-step 변환 갱신  
  – Density-based: 밀도 분포(L2, kernel) 최소화  
  – 전역 최적화: BnB, 유전 알고리즘, SDP, BnB+최적해 보장  
  – 비강건/강건 기준, RBF/TPS/non-rigid 모델 확장  

***

## 4.5 기술자 매칭+오류 제거(Descriptor Matching + Mismatch Removal)

1) 잠정 매칭 후보(Putative Matches)  
  – FT, NN, Mutual NN, NN-Distance-Ratio 등 기준  

2) 오류 제거(Geometric Verification)  
  a) Resampling RANSAC 계열  
    – minimal 샘플링→모델 예측→inlier 집합 평가  
    – MLESAC, PROSAC, LO-RANSAC, USAC, MAGSAC 등  
  b) Non-parametric 모델  
    – 매칭 함수 추정(SVR), 베이지안 강건 회귀(VFC)  
  c) Relaxed 제약  
    – 지역 일관성(Local Coherence), 그래프 매칭, Map-based 등  

***

## 4.6 딥러닝을 활용한 매칭(Learning for Matching)

-  영상 기반(Learning from Images)  
  – Registration, Stereo, Homography, Fundamental, Pose 예측  
  – PWC-Net, FlowNet, DispNet 등 end-to-end 레지스트레이션/스테레오  
  – 강화학습 에이전트로 순차적 변환 예측  

-  포인트 기반(Learning from Points)  
  – Fundamental/Fundamental Matrix 회귀(DSAC)  
  – RANSAC 샘플링 가이드(NG-RANSAC)  
  – 매칭 분류기(LFGC, SuperGlue, OAN) → 순전파로 맞춤/틀림 예측  

***

## 4.7 3D 매칭

1) 로컬 3D 디스크립터  
  – Spin Image, Shape Context, PFH, FPFH, Mesh-HoG, Spectral(HKS/WKS) 등  
  – 학습-기반: Spectral Metric Learning, 3D CNN, PointNet 계열  

2) Functional Maps  
  – Laplace–Beltrami eigenbasis → 전역 지도(Functional Map)  

3) Hypergraph/Graph Matching 확장  
  – 3D affinity tensor, high-order GM  

4) Point Set Registration  
  – CPD, GMM, ICP, GoICP, BnB-기반 최적화, 딥러닝 기반(RPM-Net, DCP)  

***

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a79afbb9-9e1b-4421-809e-9b1c5d3f780c/s11263-020-01359-2.pdf
