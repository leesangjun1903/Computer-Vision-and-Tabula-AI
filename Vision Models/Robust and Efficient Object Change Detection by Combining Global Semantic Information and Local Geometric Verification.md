# Robust and Efficient Object Change Detection by Combining Global Semantic Information and Local Geometric Verification | 3D object detection

## 1. 핵심 주장 및 주요 기여
이 논문은 로봇이 실내 환경에서 **새로 나타난 객체(new objects)** 와 **단순 이동된 객체(moved objects)** 를 구별해 검출하는 문제를 다룬다.  
- 전통적 전역 장면 차분(global scene differencing)은 센서 노이즈와 위치 추정 오차에 민감하고, 가구나 소도구의 단순 이동(permanent object readjustment)을 새 객체로 오인하는 한계가 있다.  
- 제안 방법은  
  1) **전역적(semantic) 검출**: 3D 재구성에 대해 CNN 기반 3D 시맨틱 분할을 수행한 뒤, 유의미한 수평면(탁자, 책상, 선반 등) 위의 돌출 영역을 객체 후보로 추출  
  2) **국부적(geometric) 검증**: 각 객체 후보 영역을 참조 맵(reference map)에서 로컬하게 정렬(ICP)한 뒤, 겹침 비율(overlap)이 낮으면 novel object로 판단  

이를 통해 작은 객체까지 효과적으로 검출하면서도, 이동된 가구 등으로 인한 오검출(false positive)을 크게 줄인다.

**주요 기여**  
- 시맨틱 정보와 로컬 기하 검증을 통합한 **오픈-셋(Open-Set) 3D 신규 객체 검출 프레임워크** 제안  
- 다양한 크기와 복잡도의 실내실험 환경(5종)에서 31회 관측, 260개 객체 주석으로 구성된 **새로운 로봇 객체 변동 검출 데이터셋** 공개  
- 기존 기법 대비 검출율(recall) 및 정밀도(precision)를 모두 크게 향상시켜, 평균 F1-score 0.69 달성  

***

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제
- **Novel Object Change Detection**: 참조 맵에 없던 신규 객체와, 참조 맵상의 객체가 단순히 위치만 바뀐(permanent) 객체를 구분해 검출  
- 조건: 사전 객체 모델 없이, 오픈-셋 환경(open-set)에서 동작해야 함

### 2.2 제안 방법 개요
1) **3D 시맨틱 분할**  
   - SparseConvNet 기반 네트워크로 ScanNet 데이터셋 학습  
   - 재구성된 포인트클라우드(voxblox) 각 정점(vertex)에 클래스 라벨 부여  

2) **수평면 기반 객체 후보 추출**  
   - 탁자(table), 책상(desk), 선반(shelf) 등 객체가 놓이는 수평면에 속한 정점만 선별  
   - RANSAC으로 평면 추정 후, 평면 위 돌출 영역을 Euclidean 클러스터링해 객체 후보로 수집  

3) **국부 기하 검증(Local Verification)**  
   (식 (1))  

$$ D = \{c \mid c\in C \wedge \nexists s\in S, \|c-s\| < d\} $$  
   
   - $$C$$: 관측된 후보 포인트, $$S$$: 참조 맵 포인트, $$d$$: 거리 임계값  
   - 두 단계 ICP:  
     a. 후보의 지지 평면(supporting plane)끼리 정렬(plane-plane ICP; x,y축 회전 고정)  
     b. 객체 클러스터만 z축 회전 및 x,y 이동만 허용해 정렬(object-map ICP)  
   - 정렬 후 겹침 비율이 낮으면(novel), 높으면(permanent)으로 분류  

4) **결과 후처리**  
   - 소규모 노이즈 제거, 클러스터링 오류 보완  

### 2.3 모델 구조
- 입력: Dense 3D 점군 + RGB 색상 (voxblox TSDF → 포인트클라우드 변환)  
- 백본: SparseConvNet (21 클래스)  
- 모듈: 시맨틱 분할 → 평면 검출+클러스터링 → 2단계 ICP 검증 → 최종 novel object 결정  

***

## 3. 성능 향상 및 한계

### 3.1 성능 평가
- **데이터셋**: 5개 환경(작은 방, 큰 방, 부엌, 거실, 사무실), 31회 관측, 260개 신규 객체  
- **비교 기법**: Octomap 기반 differencing, Meta-room  
- **평균 결과 (전체 31회)**  
  - Octomap: Precision 0.18, Recall 0.54, F1 0.23  
  - Meta-room: Precision 0.25, Recall 0.43, F1 0.22  
  - **제안 기법 (full)**: Precision 0.63, Recall 0.70, **F1 0.64**  

### 3.2 한계 및 실패 사례
- **인접 객체 분리 실패**: 소형 신규 객체가 큰 가구에 근접·접촉 시 클러스터링이 분리하지 못해, ICP 겹침 높아짐 → 오검출  
- **극단적 이동**: 참조 맵 대비 이동 거리가 더 크면(permanent object 재배치) novel로 오인하거나, 너무 작으면 miss  

***

## 4. 일반화 성능 향상 가능성

- **다양한 재구성 기법 적용**: Kintinuous, ElasticFusion, ScalableFusion 출력에도 동일 방법 적용 가능성 확인, F1 0.93~0.96  
- **온톨로지 기반 시맨틱 컨텍스트 확장**: 고수준 공간·객체 관계 지식을 통합해, semantic plane 검출 및 local verification 개선  
- **학습 기반 특징 기술자(feature descriptor) 결합**: 작은 객체 분리 및 정확 검증을 위해 3DMatch 등 로컬 디스크립터 활용  
- **적응적 클러스터링**: 객체 크기·밀집도에 따른 가변적 클러스터링 임계값 적용  

***

## 5. 향후 연구에 미치는 영향 및 고려 사항

- **장기 자율운영 로봇**: 사무실·가정·공장 등 반복 관측 환경에서, **실시간 신규 객체 감지** 및 **이상 상황(도난·배치 오류) 알림** 시스템 발전  
- **멀티센서·다중모달 통합**: RGB-D 외에 LiDAR, 초음파 센서 결합으로 노이즈 강건성 추가 확보  
- **온톨로지·상황인지(context-awareness)**: 객체 용도·관계 기반 검증으로 오검출 감소 및 고차원 추론 가능  
- **데이터셋 확장**: 야외·반개방 공간, 군중 환경, 조명 변화 등 다양한 조건 포함한 벤치마크 구축  

앞으로는 이러한 방향을 고려하여, **고도화된 시맨틱-기하 통합 검출** 기법이 스마트 로봇 및 자율 시스템 분야의 핵심 모듈로 자리잡을 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e4e973a4-d75b-4cb8-9f4b-7d3881a7a733/Robust_and_Efficient_Object_Change_Detection_by_Combining_Global_Semantic_Information_and_Local_Geometric_Verification.pdf
