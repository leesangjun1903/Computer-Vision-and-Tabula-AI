# Structure from Motion with Objects

이 연구는 **영상에서 점(point) 대신 객체(object)의 바운딩 박스만으로** 카메라 보정 3D 객체 위치·크기·자세를 동시에 복원하는 방법을 제안합니다. 전통적인 SfM은 이미지 속 여러 점을 추적하여 3D 구조와 카메라를 복원하지만, 이 논문은 객체 감지기의 결과(BB)를 활용해 같은 목표를 달성합니다.

## 1. 다루는 문제  
- 기존 SfM: 2D 점들의 좌표 궤적을 모아 measurement 행렬을 만들고, 이를 저차 분해(SVD)해 카메라와 3D 점을 복원.  
- 한계: 객체의 크기·형상·자세 정보는 전혀 반영되지 않음.  
- 이 논문: 이미지에서 “객체”로 인식된 바운딩 박스를 **타원(ellipse)** 으로 근사하여, 이 타원 정보를 통해 3D 객체(ellipsoid)와 카메라를 동시에 복원.

## 2. 전체 흐름  
1) **2D 바운딩 박스 → 타원**  
   각 프레임에서 객체 바운딩 박스 안에 꼭 들어맞는 타원을 추정  
2) **타원 → 듀얼 코닉(Dual Conic)**  
   타원을 표현하는 3×3 행렬을 adjoint 연산으로 변환(dual conic)  
3) **듀얼 쿼드릭(Dual Quadric)과 연결**  
   3D 객체는 ellipsoid(quadric)로 표현. orthographic(평행 투영) 카메라 모델 하에서  
   dual conic = P · (dual quadric) · Pᵀ 관계 성립  
4) **측정 행렬 구성 및 factorization**  
   - 모든 프레임·모든 객체의 dual conic을 세로로 쌓아 거대한 행렬 C 구성  
   - C는 rank ≤ 10 제약이 있으므로 SVD로 분해 → 초기 affine 해 획득  
5) **Metric 업그레이드 (Affine → Metric)**  
   - 먼저 프레임별로 타원 중심을 평균으로 정규화하여 translation 성분 제거  
   - 행렬 G를 재배열해 두 블록(rank-3, rank-6)으로 분리  
     -  rank-3 서브문제로 카메라 회전·병진 파라미터 선형 회복  
     -  rank-6 서브문제로 객체 형상(shape)·크기(size) 파라미터 복원  
6) **2D 점 추적 통합(옵션)**  
   - 2D 점을 “퇴화된 쿼드릭(degenerate quadric)”으로 간주해 같은 파이프라인에 포함  
   - 객체 개수가 적을 때 포인트 트랙이 도움

## 3. 주요 장점  
- **폐쇠형(Closed-form)** 선형 해: 비선형 최적화 없이 SVD와 선형 연산으로 빠르고 안정적  
- **동시 self-calibration**: 카메라 그리고 객체 3D 위치·형상·자세를 한 번에 처리  
- **점과 객체 통합**: 추가 정보로 2D 점 트랙을 쉽게 결합 가능  
- **견고성**: 합성 데이터와 Kinect 실제 데이터에서 바운딩 박스 오차에도 안정적인 복원 성능 확인

## 4. 한계 및 개선 방향  
- **원근 투영 미지원**: orthographic(평행)만 다룸 → 원근 모델로 확장 필요  
- **형상 단순화**: 객체를 ellipsoid로 근사 → 복잡·비대칭 물체에는 부정확  
- **최종 정교화 부족**: 선형 초기 해만 제공 → non-linear 최적화로 정밀도 향상 여지  
- **최소 객체 수**: 객체만 사용할 경우 3개 이상 필요 → 점 트랙 없으면 제한적

## 5. 일반화 성능 향상 방안  
- **Perspective SfM**: dual conic/quadric 관계를 원근 투영으로 일반화  
- **딥러닝 검출기 결합**: CNN 기반 특징으로 바운딩 박스 정확도 개선  
- **슈퍼쿼드릭 확장**: ellipsoid 대신 superquadric으로 복잡한 형상 모델링  
- **강건성 강화**: RANSAC, robust cost 함수 등으로 검출 잡음·결측 대응  
- **실시간 파이프라인**: 경량화된 factorization으로 실시간 응용

## 6. 향후 연구 고려 사항  
- 전체 파라미터(카메라+쿼드릭) 동시 최적화를 위한 비선형 refinement  
- 복합 객체(비구형, 비강체) 모델 통합  
- 실시간 SLAM/AR 애플리케이션에서의 응용 및 확장  

이 방법은 “객체”만으로 3D 구조를 복원한다는 **새로운 방향**을 제시하여, 기존 점 기반 SfM을 넘어 **3D 객체 인식, 포즈 추정 연구**에 큰 영향을 미칠 것입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/19f775dd-6e99-4b5d-a916-575d802dfc04/Crocco_Structure_From_Motion_CVPR_2016_paper.pdf

# Structure from Motion with Objects

**핵심 주장 및 주요 기여**  
이 논문은 전통적인 점(feature) 기반 구조-움직임fM) 기법을 객체 검출 결과만으로 확장하여, 2D 이미지 시퀀스 속 객체 바운딩 박스만을 이용해 카메라의 affine 보정과 객체의 3D 위치·크기·자세를 동시에 폐쇠형(closed-form)으로 복원할 수 있음을 보인다. 주요 기여는 다음과 같다:  
- 객체 검출(bounding box)으로부터 타원(ellipse)을 피팅하여 dual conic 행렬을 구성  
- Tomasi–Kanade factorization을 객체 쿼드릭(quadric)과 연결해, dual conic 행렬 $$C\in\mathbb R^{6F\times N}$$에 대해  
  $$C = G\,V,\quad \mathrm{rank}(C)\le10$$  
  인 affine factorization 성질 도출  
- 중심 정규화 후 rank-3/6 제약을 단계적으로 풀어 카메라 파라미터와 쿼드릭의 병진·형상·크기 매개변수를 선형 방식으로 추정  
- 2D 점 추적도 ‘퇴화된 쿼드릭(degenerate quadric)’으로 통합, 점과 객체를 동시에 factorization 가능  
- 합성·실험 데이터에서 검출 오차에 대해 견고함 확인 및 Kinect 데스크셋으로 실세계 검증  

## 1. 해결 과제  
전통적 SfM은 2D 포인트 매칭 궤적을 이용해 measurement 행렬의 저차(rank) 특성을 활용했다. 그러나 실제 장면에서는 객체 검출기가 제공하는 바운딩 박스 정보도 풍부하며, 이로부터 3D 객체 위치와 카메라 보정 정보를 동시에 얻고자 한다. 기존 점 기반 방법으로는 객체의 크기·자세 정보가 활용되지 못함.

## 2. 제안 방법  
### 2.1. 객체 검출 → Dual Conic  
- 각 프레임 $$f$$의 객체 $$i$$에 대해 바운딩 박스에 내접하는 타원 $$\mathbf D_{fi}\in\mathbb R^{3\times3}$$을 추정  
- Dual conic:

```math
\mathbf C_{fi}=\text{adj}(\mathbf D_{fi})
```

### 2.2. 쿼드릭과의 관계  
Affine 카메라 $$\mathbf P_f=[\mathbf R_f\;\mathbf t_f;\;0\;1]$$ 하에서 dual quadric $$\mathbf Q_i=\text{adj}(\mathbf E_i)$$와  

$$
\mathbf C_{fi} \propto \mathbf P_f\,\mathbf Q_i\,\mathbf P_f^\top
$$  

를 만족.

### 2.3. Dual Conic Matrix Factorization  
1. 벡터화: $$\mathbf c_{fi}=\text{vech}(\mathbf C_{fi})\in\mathbb R^6$$ , $$\mathbf v_i=\text{vech}(\mathbf Q_i)\in\mathbb R^{10}$$  

2. 각 프레임마다 $$\mathbf G_f\in\mathbb R^{6\times10}$$ 구축 →  

$$
   \mathbf c_{fi} = \mathbf G_f\,\mathbf v_i
   $$  

3. 전 프레임·객체 축합:  

$$
   C = \begin{bmatrix}c_{11}&\cdots&c_{1N}\\\vdots&&\vdots \\ c_{F1}&\cdots&c_{FN}\end{bmatrix}
   = GV,\quad G\in\mathbb R^{6F\times10},\;V\in\mathbb R^{10\times N}
   $$  

4. SVD로 rank-10 근사 ($$C\approx \tilde G\tilde V$$) 후 구조 제약으로 mixing 행렬 $$Z$$ 추정  

### 2.4. Metric Upgrade via Rank-3/6 분할  
- 객체 중심 정규화(translation 평균 제거) → $$\bar G_f$$  
- $$\bar G_f$$를 재배열해 상·중·하 블록 분리  
  - 중간 블록(2×3) → rank-3 factorization → orthographic 카메라 회전·병진 획득  
  - 상단 블록(3×6) → rank-6 factorization → 쿼드릭 형상·크기 추정  
- 2D 점은 퇴화된 쿼드릭으로 삽입해 동일한 pipeline 적용  

### 2.5. 수식 예시  
Metric upgrade의 핵심은 중심화된 dual conic relation:  

$$
\bar C_{fi} = \bar P_f\,\bar Q_i\,\bar P_f^\top
$$  

SVD 후 rank-3 mixing 행렬 $$Z_l$$은 orthogonality 조건  

$$
R_f R_f^\top = I_2
$$  

를 이용해 구하며, shape 추정은  

$$
V_s = G_s^+ C_s
$$  

로 Gs의 의사역원(pseudoinverse)으로 각 객체별 형상 파라미터를 선형 복원.

## 3. 모델 구조 및 성능 향상  
- **단계적 factorization**: rank-10 → rank-3 → rank-6 분할로 복잡도·야코비안 행렬 절감  
- **동시 self-calibration**: affine 카메라 보정, 객체 3D 위치·형상·자세 복원 일괄 처리  
- **점·객체 통합**: sparse point tracks 추가로 최소 객체 수(3개) 제약 극복  
- **성능 검증**:  
  - 합성데이터: 회전·크기 잡음 최대치(Rotation error 45°, Size error 50%)에서도 3D 부피중첩(O₃D) 0.5 이상, 자세 오차 40° 이하로 견고  
  - Kinect 실제데이터: 8–25프레임, 8–15개 객체에서 BB→타원으로 복원한 투영 일치도 높음, 깊이 배치 일관성 확인  

### 한계  
- **Orthographic 가정**: 원경 모델만 지원, 원근(perspective)으로 확장 필요  
- **형상 단순화**: 비대칭 객체에는 Ellipsoid 근사 한계  
- **최종 정교화 부족**: non-linear 최적화 없는 초기 선형 해만 제공  

## 4. 일반화 성능 향상 가능성  
- **Perspective SfM 확장**: dual conic factorization을 원근행렬로 일반화하면, 실세계 카메라에 바로 적용 가능  
- **Deep detector 결합**: 바운딩 박스의 정확도 향상을 위해 CNN 기반 특징 피드백 통합  
- **비구형 객체 모델링**: ellipsoid 대신 superquadric 등 형태 파라미터화로 복잡 형상 대응  
- **Non-rigid/다중 모달 통합**: articulated 객체 및 동적 형태 변화 추적에 dual factorization 확장  

## 5. 향후 연구 방향 및 고려 사항  
- **비선형 최적화 통합**: 전체 파라미터(카메라·쿼드릭) 동시 refinement로 국소 최소 극복  
- **Perspective 모델링**: dual conic–quadric 관계의 원근 일반화 연구  
- **복합 객체 표현**: Ellipsoid 한계를 넘는 복합 쿼드릭(superquadric, NURBS) 적용  
- **잡음·결측 견고화**: 실검출 강인성 향상을 위한 RANSAC, robust cost 함수 도입  
- **실시간 파이프라인**: 효율적 factorization·업그레이드로 실시간 구조화 애플리케이션 지원  

이 논문은 객체 검출 결과를 활용한 SfM의 새로운 방향을 제시함으로써, 전통적인 점 기반 차원을 넘어 3D 객체 인식·포즈 추정 연구에 지대한 영향을 미칠 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/19f775dd-6e99-4b5d-a916-575d802dfc04/Crocco_Structure_From_Motion_CVPR_2016_paper.pdf
