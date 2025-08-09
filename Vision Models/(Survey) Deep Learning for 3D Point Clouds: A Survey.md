# Deep Learning for 3D Point Clouds: A Survey 

## 1. 핵심 주장과 주요 기여
이 논문은 **3D 포인트 클라우드**에 대한 **딥러닝** 기법을 통합적으로 정리·분류한 최초의 통합적 서베이로, 다음을 제시한다.  
- 3D 형태 분류, 객체 검출·추적, 포인트 클라우드 분할(시맨틱·인스턴스·파트) 영역별 대표 기법 분류 체계 제안  
- Multi-view, Volumetric, Point-based(MLP, Convolution, Graph, Hierarchical) 방법론의 **체계적 분류·비교**  
- 주요 공개 데이터셋(ModelNet, KITTI, S3DIS 등)에 대한 **성능 비교 표** 제공  
- 향후 연구 방향 및 한계점(데이터 규모·연산 비용·일반화 문제) 제시  

## 2. 문제 정의, 방법론, 구조, 성능 및 한계  
### 2.1 해결하고자 하는 문제  
- **불규칙·비정형** 3D 포인트 클라우드 처리: 순서 불변성, 밀도 변화, 고차원성  
- **응용 과제**  
  - 3D 형태 분류(Shape Classification)  
  - 3D 객체 검출·추적(Object Detection & Tracking)  
  - 포인트 클라우드 분할(Semantic/Instance/Part Segmentation)  

### 2.2 제안된 분류 체계 및 주요 방법  
1. **Multi-view**  
   - 3D→2D 투영 후 2D CNN 활용 (MVCNN 등)  
2. **Volumetric**  
   - Voxelization + 3D CNN (VoxNet, OctNet)  
3. **Point-based**  
   - Pointwise MLP (PointNet, PointNet++)  
     -  $$f(\mathbf{x}_i)=\mathrm{MLP}(\mathbf{x}_i)$$, $$\mathbf{g}=\max_i f(\mathbf{x}_i)$$  
   - Continuous Convolution (PointConv, KPConv)  
     -  $$\mathbf{y}\_i=\sum_{j\in\mathcal{N}(i)} w(\mathbf{x}_j-\mathbf{x}_i)\,f(\mathbf{x}_j)$$  
   - Graph-based (EdgeConv, DGCNN)  
     -  그래프 $$\mathcal{G}=(V,E)$$ 상의 필터링  
   - Hierarchical Structure (Kd-Net, SO-Net)  
     -  KD-트리·SOM 기반 계층적 기능 추출  

## 3장 3D 형태 분류(Shape Classification)

3D 형태 분류는 ‘포인트 클라우드’ 형태로 표현된 3차원 객체를 **어떤 클래스(예: 의자, 테이블, 자동차 등)에 속하는지**를 예측하는 과제입니다. 이 섹션에서는 크게 세 가지 접근 방식을 다루고, 각 방법이 어떻게 포인트 클라우드의 특징을 추출·통합하여 최종 분류 결과를 얻는지 설명합니다.

***

### 1. Multi-view 기반 방법  
- **아이디어**  
  - 3D 데이터를 여러 방향에서 2D 이미지로 투영한 뒤, 일반적인 2D CNN을 적용  
  - 각 뷰(view)별로 얻은 특징을 결합하여 최종 3D 객체 표현을 만듦  
- **장점**  
  - 2D 이미지용 CNN을 그대로 활용 가능  
- **단점**  
  - 뷰 선택에 민감 → 가려진(occluded) 부분 정보 손실  
  - 투영 과정에서 3D 세부 구조를 완전히 보존하지 못함  

***

### 2. Volumetric(체적) 기반 방법  
- **아이디어**  
  - 3D 공간을 정규격자(voxel)로 나누고, 각 셀(voxel)에 물체 유무나 법선 벡터 등 값을 할당  
  - 3D CNN(3차원 합성곱 신경망)으로 체적 데이터를 처리  
- **장점**  
  - 격자 형태로 규칙적인 데이터 → 3D CNN 그대로 적용  
- **단점**  
  - 해상도(격자 크기)가 높아질수록 계산·메모리 비용이 기하급수적으로 증가  
  - 세밀한 형태 정보는 작은 격자 밖에서 손실될 수 있음  

***

### 3. Point-based(포인트) 기반 방법  
#### 3.1 Pointwise MLP (PointNet 계열)  
- **구성**  
  1) 각 포인트 좌표 $$(x_i,y_i,z_i)$$를 동일한 MLP(다층 퍼셉트론)로 독립 처리 → 고차원 특징 벡터 생성  
  2) **최댓값 풀링**(max-pooling)으로 모든 포인트 특징을 하나의 전역 특징으로 집계  
  3) 전역 특징 → 분류 레이어 → 클래스 확률 예측  
- **PointNet 한계**  
  - 포인트 간 관계(이웃 정보, 국소 기하 구조) 반영 불가  
- **PointNet++ 개선**  
  - 포인트 샘플링(FPS) → 반경 기반 그룹화 → 각 그룹에 PointNet 적용 → 다단계 계층적 추출  
  - 국소 지역 구조와 전역 정보를 모두 학습  

#### 3.2 Convolution 기반 방법  
- **연속(con-) 컨볼루션**  
  - 3D 공간의 포인트 분포를 연속 함수로 모델링 → 위치 차이에 따라 가중치 결정  
  - 예: **PointConv**, **KPConv**  
- **불연속(discrete) 컨볼루션**  
  - 포인트 주변을 셀(cell)로 나누어, 같은 셀에 속하는 포인트에 동일 가중치 할당  
  - 예: **Ψ-CNN**, **GeoConv**  

#### 3.3 Graph 기반 방법  
- **구성**  
  - 각 포인트를 그래프의 정점(vertex)으로, 이웃 포인트를 간선(edge)으로 연결  
  - 간선 특성(거리·방향 등)과 정점 특징을 활용해 **EdgeConv** 연산 수행  
  - 그래프 풀링·다층화로 전역 특징 획득  
- **장점**  
  - 불규칙 포인트 클라우드에서 직접 국소 구조와 전역 문맥 모두 포착  

#### 3.4 Hierarchical Data Structure 기반 방법  
- **구성**  
  - **Octree**, **Kd-트리**, **SOM(Self-Organizing Map)** 등 트리 구조로 포인트 집계  
  - 저수준(leaf) 노드에서 고수준(root) 노드로 계층적 특징 집계  
- **장점**  
  - 계층적·효율적 공간 분할 → 대규모 데이터에도 확장성  

***

## 핵심 정리  
1. **PointNet++**: MLP + 계층적 그룹화로 첫 물꼬를 튼 기법  
2. **PointConv/KPConv 등**: 진정한 ‘포인트 컨볼루션’으로 1–2% 성능 향상  
3. **Graph CNN(DGCNN, ECC)**: 포인트 간 관계 학습으로 92% 이상의 정확도 달성  
4. **트리 기반(SO-Net, KD-Net)**: 복합 공간 구조화로 효율적 특징 추출  

각 방법은 **정확도–연산 효율–메모리 사용량** 사이에서 절충점을 다르게 설정하므로, 응용 환경(대규모 실내 스캔 vs. 자율주행 vs. AR/VR)과 사용 컴퓨팅 자원에 따라 적합한 기법을 선택해야 합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/de33ae6d-6a1f-4236-97e2-a7ea183f69b8/1912.12033v2.pdf

## 4장 3D 객체 검출 및 추적(3D Object Detection and Tracking)

3D 객체 검출은 **포인트 클라우드**(LiDAR 등)에서 자동차, 사람, 자전거 등 개별 물체의 **3차원 바운딩 박스**를 찾아내는 과제입니다. 추적은 검출된 객체를 연속되는 프레임에서 일관되게 식별하여 **시간축을 따라 움직임**을 추적합니다.

***

### 4.1 3D 객체 검출

#### 4.1.1 Region Proposal 기반 방법  
1) **Region Proposal**  
   - 3D 공간에서 물체가 있을법한 후보 영역(Proposal)을 먼저 뽑아둠  
2) **RoI Feature Extraction**  
   - 각 후보 영역 내부 포인트만 모아 특징을 추출  
3) **Classification & Regression**  
   - 물체 종류(자동차·사람 등) 분류  
   - 박스 중심 좌표, 크기, 회전(θ) 회귀  

**세부 분류**  
- **Multi-view**:  
  - BEV(위에서 본 Bird’s-Eye View), 정면 뷰, 컬러 이미지 등 서로 다른 투영 시점의 특징을 합침  
  - MV3D, AVOD, ContFuse 등  
  - ⇒ 융합 성능 높지만 계산량 ↑  
- **Segmentation-based**:  
  - 먼저 포인트 구분(segmentation)으로 배경 제거 → 남은 포인트로 Proposal 생성  
  - PointRCNN, PointRGCN, STD 등  
  - ⇒ 배경 없는 고품질 후보, 복잡 장면에 유리  
- **Frustum-based**:  
  - 이미지를 이용해 2D 박스 추출 → 2D 박스의 3D “Frustum”(사다리꼴 모양) 영역 잘라서 3D 박스 검출  
  - Frustum PointNets, PointFusion, F-ConvNet 등  
  - ⇒ 2D 검출기 성능에 크게 의존  

#### 4.1.2 Single-Shot(End-to-End) 방법  
- **Proposal 없이 한 번에 예측**  
- **종류**  
  - **BEV 기반**: 포인트를 2D 격자(위에서 본 뷰)로 변환해 2D CNN 적용 (PIXOR, HDNET, BirdNet)  
  - **Voxel(격자) 기반**: 포인트를 3D 격자(voxel)로 바꿔 3D CNN 적용 (VoxelNet, SECOND, PointPillars)  
  - **Point 기반**: 원시 포인트 그대로 사용해 3D 검출 (3DSSD)  
- **특징**:  
  - 속도 빠름(25–60fps)  
  - 정확도는 Region Proposal 방식에 일부 열세  

#### 4.1.3 핵심 기술 비교  
| 방식                 | 장점                                         | 단점                              |
|----------------------|----------------------------------------------|-----------------------------------|
| Region Proposal      | 높은 정확도(90%↑)                            | 다단계 연산, 느림(3–12fps)        |
| Single-Shot          | 실시간 처리(25–60fps)                        | 장거리·소형 물체 검출 한계        |
| Multi-Modal Fusion   | 이미지·LiDAR 융합으로 성능 향상              | 설계 복잡, 데이터 동기화 필요      |
| Point-Based          | 정밀한 포인트 단위 처리                       | neighbor search 연산 비용         |

***

### 4.2 3D 객체 추적(3D Tracking)

1) **초기화**: 첫 프레임에서 객체 위치(바운딩 박스) 제공  
2) **추적**: 다음 프레임의 포인트 클라우드에서 동일 객체 박스 예측  
3) **방법**  
   - **3D Siamese Network**: 기준(템플릿) 박스 포인트와 후보 박스 포인트 특징을 비교  
     - Shape Completion 정규화, Hough Voting 사용 (3D Siamese)  
   - **2D→3D 하이브리드**: BEV에서 대략 후보 → 3D 시아네스 네트워크로 세밀 조정  
   - **Kalman 필터**: 시간적 위치 예측으로 후보 박스 생성  
4) **성능**  
   - 2D 추적보다 **조명·가림(occlusion)·스케일 변화**에 강함  
   - 실시간(40fps) 대응 가능  

***

### 4.3 3D Scene Flow 추정

- **목표**: 연속 두 프레임 포인트 클라우드 사이에서 각 포인트의 **3D 이동 벡터(Flow)** 계산  
- **응용**: 움직임 추적, 행동 분석, SLAM 보조  
- **대표적 네트워크**  
  - **FlowNet3D**: 포인트 단위 특징+모션 특징 융합 → 직접 회귀  
  - **HPLFlowNet**: 대규모 포인트에 효율적, 구조 정보 보존  
  - **Self-Supervised**: 순방향‐역방향 일관성(Cycle Loss)으로 레이블 없는 학습  

***

## 장·단점 비교

| 과제                  | 대표 방법             | 장점                                         | 단점                                  |
|----------------------|----------------------|----------------------------------------------|---------------------------------------|
| 객체 검출            | PointRCNN, PV-RCNN   | 정확도(3D AP 80–90%)                         | 느린 속도(5–12fps), 장거리 검출 약함  |
| 실시간 검출          | PointPillars, 3DSSD  | 속도(25–60fps)                               | 근거리 위주, 소형 물체 놓침           |
| 객체 추적            | 3D Siamese, P2B      | 강력한 추적 안정성                           | 추적 대상 프레임별 초기화 필요         |
| Scene Flow 추정      | FlowNet3D, HPLFlowNet| 동작 벡터 직접 예측                          | 고정밀 레이블 데이터 필요             |

***

### 요약 및 실용 팁  
- **고정밀·다중단계**: Region Proposal 방식  
- **실시간**: Single-Shot PointPillars, 3DSSD  
- **추적**: Siamese + Hough Vote, Kalman 필터 결합  
- **동작 분석**: FlowNet3D 계열  

응용 분야와 요구 정확도·속도 균형에 따라 위 기법을 적절히 선택 및 결합하면 3D 환경에서 높은 성능의 검출·추적 시스템을 구현할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/de33ae6d-6a1f-4236-97e2-a7ea183f69b8/1912.12033v2.pdf

## 5장 3D 포인트 클라우드 분할 (3D Point Cloud Segmentation)

포인트 클라우드 분할은 각 포인트에 **의미 있는(label)** 를 부여하여 장면·물체·부품 수준의 다양한 분할을 수행하는 과제입니다.  
- **Semantic Segmentation**: 장면 전체를 의미 있는 카테고리(도로·건물·사람 등)로 구분  
- **Instance Segmentation**: 같은 카테고리 내에서 개별 객체(각 자동차·각 사람)를 분리  
- **Part Segmentation**: 개별 객체를 더욱 세분화된 부품(사람의 팔·다리, 자동차의 바퀴·문)으로 분리  

***

### 5.1 3D Semantic Segmentation

포인트 클라우드를 **점 단위로** 레이블링하는 작업으로, **4가지 주요 접근법**이 존재합니다.

#### 5.1.1 Projection-based Methods  
1) **Multi-view 투영**  
   - 3D → 여러 2D 시점(위·정면·측면)으로 찍은 사진처럼 변환  
   - 2D CNN으로 픽셀별 예측 후 3D 포인트에 다시 매핑  
   - 예: SnapNet, TangentConv  
   - *장점*: 2D segmentation 기법 활용  
   - *단점*: 투영 시 가림/중첩/정보 손실 발생  

2) **Spherical 투영**  
   - LiDAR의 회전 범위를 구면 좌표계로 펼쳐 2D 이미지로 변환  
   - 예: SqueezeSeg, RangeNet++  
   - *장점*: LiDAR 전 범위 반영, 실시간 처리 가능  
   - *단점*: 구면화 단계에서 해상도 왜곡과 블러링  

#### 5.1.2 Discretization-based Methods  
1) **Dense Voxelization**  
   - 포인트를 3D 격자(voxel)로 변환 → 3D CNN 적용  
   - 예: SEGCloud, ScanComplete  
   - *장점*: 3D 공간 구조를 그대로 보존  
   - *단점*: 고해상도시 메모리·연산 급증  

2) **Sparse Convolution**  
   - 비어 있는 voxel 무시 → occupied voxel에만 연산  
   - 예: SparseConvNet, MinkowskiNet  
   - *장점*: 대규모 스캔 실시간 처리 가능  
   - *단점*: GPU 메모리 관리 복잡  

3) **Permutohedral Lattice**  
   - 고차원 격자를 효율적 색인 → Bilateral Convolution  
   - 예: SPLATNet, LatticeNet  

#### 5.1.3 Hybrid Methods  
- 2D 이미지(컬러) 특징 + 3D 기하 정보 동시 학습  
- 예: 3DMV, MVPNet, UPB  
- *장점*: 색상·형상 정보 융합  
- *단점*: 데이터 동기화, 네트워크 복잡도 증가  

#### 5.1.4 Point-based Methods  
- **원시 포인트**에 직접 MLP·Convolution·Graph·RNN 연산  
- **PointNet 계열**: 공유 MLP + Symmetric Pooling → 전역/지역 특징 학습  
- **Point Convolution**: continuous/discrete kernel 정의 (KPConv, PointConv)  
- **Graph-based**: 포인트를 그래프 노드로 보고 EdgeConv, GACNet 등 적용  
- **RNN-based**: 포인트 순서를 만들어 시공간 의존성 학습  
- **효율화**: RandLA-Net은 랜덤 샘플링 + Local Aggregation으로 대규모 분할 가능  

##### 방법별 장·단점 비교  
| 분류        | 대표 기법                | 장점                             | 단점                         |
|-------------|-------------------------|----------------------------------|------------------------------|
| Projection  | SnapNet, SqueezeSeg     | 2D 기법 활용, 실시간 가능        | 정보 손실, 투영 왜곡         |
| Voxel CNN   | SEGCloud, SparseConvNet | 3D 구조 보존                    | 해상도↑ 연산·메모리↑        |
| PointNet    | PointNet++, KPConv      | 직접 학습, 유연한 표현           | 이웃 탐색·연산 비용          |
| Graph/RNN   | DGCNN, 3P-RNN           | 국소 구조·컨텍스트 표현 우수     | 복잡도·메모리 사용↑         |

***

### 5.2 3D Instance Segmentation

**같은 의미 카테고리**의 개별 객체를 구분하는 기법이며, **두 가지 전략**이 있습니다.

1) **Proposal-based**  
   - 3D Object Detection 네트워크로 후보 바운딩 박스 생성  
   - 각 박스 내부를 PointNet/RPN 등으로 세분화하여 마스크 예측  
   - 예: 3D-SIS, GSPN, 3D-BoNet  

2) **Proposal-free**  
   - Semantic Segmentation 후 **클러스터링**으로 인스턴스 분리  
   - 포인트 간 유사도 학습 → Mean-Shift, K-Means, graph Cut 등을 적용  
   - 예: SGPN, ASIS, JSIS3D, PointGroup  

| 방식             | 장점                       | 단점                          |
|------------------|----------------------------|-------------------------------|
| Proposal-based   | 객체 단위 분할 정확도 높음 | 다단계 학습·후처리 복잡       |
| Proposal-free    | 단일 네트워크, 빠름       | 객체 경계 모호, 분할 일관성↓ |

***

### 5.3 3D Part Segmentation

**객체 내부 부품**(사람의 팔·다리, 자동차의 바퀴·문)을 식별하는 과제  
- **Voxel-based**: VoxSegNet  
- **Mesh+CRF**: SyncSpecCNN, Kalogerakis et al.  
- **Autoencoder/Branch**:  BAE-NET(무감독), PartNet(재귀 세분화)  
- **Zero-/Few-shot**: co-segmentation, group policy 학습  

***

## 요약 및 실용 팁
1. **장면 분할**:  
   - 대규모 스캔 → SparseConvNet, RandLA-Net  
   - 실시간 LiDAR → SqueezeSegV2, RangeNet++  
2. **인스턴스 분할**:  
   - 높은 객체 정확도 → GSPN, 3D-SIS  
   - 경량화·실시간 → PointGroup, ASIS  
3. **부품 분할**:  
   - 대량 데이터 → BAE-NET(무감독), PartNet(재귀 세분)  
   - 세밀한 레이블 필요시 VoxSegNet + AFA 활용  

응용 시 **장·단점**을 고려해 적절한 방법을 선택하고, Projection·Voxel·Point 방식의 **하이브리드** 구성도 검토하세요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/de33ae6d-6a1f-4236-97e2-a7ea183f69b8/1912.12033v2.pdf

### 2.3 모델 구조 예시  
```
Input: N×3 point cloud
→ Sampling & Grouping (FPS or radius)
→ Local Feature Learning (MLP or Continuous Conv)
→ Feature Aggregation (max-pool or graph-pool)
→ Global Classification / Box Regression / Point-wise Labeling
```

### 2.4 성능 향상  
- PointConv, KPConv 등 **연속컨볼루션** 계열이 **PointNet++** 대비 1–2% p 성능 향상  
- DGCNN 등 **그래프 기반** 모델: 국소구조 포착 개선, ModelNet40 OA 92% 초과  
- PointPillars, SECOND 등 **Sparse 3D CNN** 활용 시 연산 효율 대폭 개선(>60 fps)  
- VoteNet: 투표 기반 제안으로 인도어 3D 검출에서 mAP 대폭 상승  

### 2.5 한계  
1. **연산·메모리 비용**  
   - 고정밀 voxelization·dense conv 어려움  
   - KNN·FPS 기반 구조화 단계 병목  
2. **데이터 규모 및 다양성**  
   - 레이블링 어려움, real-world occlusion과 노이즈 대응 부족  
3. **일반화(Generalization)**  
   - 센서 스펙·밀도·환경 변화에 민감  
   - 도메인 적응 및 self-supervised 학습 연구 초기 단계  

## 3. 일반화 성능 향상 가능성  
- **도메인 적응**: SqueezeSegV2 등 **Unsupervised Domain Adaptation** 기법 활용  
- **Self- and Weakly-Supervision**: 랜덤 자가 지도 학습(FlowNet3D 학습), 부분 레이블 학습  
- **랜덤 샘플링 + Sparse Convolution**(RandLA-Net): 큰 스케일에서도 메모리·연산 절감  
- **Multi-Task Learning**(MMF): 2D–3D 크로스모달 피처 공유로 robust features 획득  
- **데이터 증강(Augmentation)**: PointAugment 등 학습 가능한 증강 전략 도입  

## 4. 미래 영향 및 고려사항  
- 3D 딥러닝의 **표준 프레임워크** 토대 제공: 후속 연구의 분류 기준·비교 벤치마크로 활용  
- **효율적 구조화**(Point–Voxel, Submanifold Conv) 및 **자가 학습(self-supervised)** 융합 기대  
- 실제 로봇·자율주행 적용 시 **다양한 센서·환경 일반화** 문제 해결 필요  
- **대규모·고해상도** 포인트 클라우드 대응, **경량화 모델** 연구가 중요한 후속 과제

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/de33ae6d-6a1f-4236-97e2-a7ea183f69b8/1912.12033v2.pdf
