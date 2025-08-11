# LassoNet: Deep Lasso-Selection of 3D Point Clouds | 3D Point Cloud Preparation, Lasso-Selection

**핵심 주장 및 주요 기여**  
LassoNet은 사용자가 2D 화면에 그린 라쏘(lasso)와 카메라 뷰포인트(viewpoint)를 입력으로 3D 포인트 클라우드에서 원하는 영역을 정확·효율적으로 선택하는 새로운 딥러닝 기반 방법을 제안한다. 기존의 밀도 기반 휴리스틱에 의존한 기법들이 데이터 분포나 뷰포인트 변화에 취약했던 한계를 극복하고, 다양한 포인트 클라우드 분포와 라쏘 형태에 견고하게 작동함을 입증했다.

***

## 1. 해결하려는 문제  
- 3D 포인트 클라우드를 2D 화면에 투영하면 **은폐(occlusion)**와 **시각적 군집(clutter)**이 발생하고, 마우스·터치 입력은 2D에 제한되어 있어 사용자가 의도한 3D 영역을 선택하기 어려움.  
- 기존 Cylinder–Selection, CAST 등은 라쏘 내부에 있는 포인트를 단순히 집합으로 추출하거나, local density 기반 휴리스틱으로 의도 영역을 추정하나,  
  - 포인트 밀도 차이가 적은 실세계 스캔 데이터(S3DIS)나 CAD 모델(ShapeNet)에서는 오히려 잘못된 클러스터를 선택  
  - 다중 뷰에서 여러 차례 라쏘와 부울 연산이 필요하여 비효율적

***

## 2. 제안 방법  
### 2.1 Latent Mapping 함수  
- 사용자 의도 선택: f(P, V, L) → Ps  
  - P: 3D 포인트 클라우드  
  - V: 카메라 뷰포인트 (위치·방향)  
  - L: 2D 라쏘 곡선  
- 목적: Jaccard distance $$d_J(P_s, P_t) = 1 - \frac{|P_s \cap P_t|}{|P_s \cup P_t|} $$ 최소화  

### 2.2 파이프라인 구성  
1. **Interaction Encoding**  
   - 3D 좌표 변환: 객체 공간 → 카메라 공간  
   - Naive Selection: 라쏘를 프러스텀(frustum)으로 확장하여 각 포인트에 이진 라벨 $$w_i$$ 부여  
2. **Filtering & Sampling**  
   - Intention Filtering: 라쏘의 2D 바운딩 박스를 1.2배 확장하여 외부 포인트 제거  
   - Farthest Point Sampling: 최대 $$T$$개 포인트로 다운샘플링, 다수 파티션 처리  
3. **Hierarchical Neural Network**  
   - 기반: PointNet++ 구조  
   - Abstraction: 지역 그룹별 레벨별 특징 벡터 추출  
   - Propagation: 전역·지역 특징을 개별 포인트에 전파  
   - 출력: 각 포인트 선택 확률 $$\rho_i$$, 임계값 0.5로 이진 분류  

***

## 3. 모델 구조 및 수식  
- **Loss**: 클래스 불균형 보정 교차엔트로피  

$$
    L = -\frac{1}{n} \sum_{i=1}^n \bigl[\theta_0 s_i \log \rho_i + \theta_1 (1 - s_i)\log(1-\rho_i)\bigr]
  $$  
  
  - $$s_i$$: 정답 레이블(타겟=1, 방해점=0), $$\theta_0,\theta_1$$: 클래스 가중치  
- **하이퍼파라미터**  
  - FPS 임계값 $$T = 20{,}480$$  
  - 그룹 크기 $$g$$: ShapeNet→2,048, S3DIS→32  
- **최적화**: Adam, 학습률 $$1\times10^{-3}$$에서 50 epoch마다 절반 감소, 배치 정규화·드롭아웃 적용  

***

## 4. 성능 평가  
- **데이터셋**  
  - ShapeNet: 2,332개 모델·19,432개 라쏘 레코드  
  - S3DIS: 272개 스캔·12,944개 레코드  
- **비교 대상**: CylinderSelection, SpaceCast  
- **정량 평가**  
  - F1 ↑, Jaccard distance ↓ 모두 LassoNet 우수  
  - ShapeNet: $$d_J$$ 0.08 vs 0.28, F1 0.95 vs 0.84  
  - S3DIS: $$d_J$$ 0.17 vs 0.61, F1 0.90 vs 0.57  
  - 응답 시간: ShapeNet 20.5ms, S3DIS 69.5ms  
- **사용자 연구** (16명, 9 과제)  
  - 과제 완료 시간·정확도 모두 LassoNet 최고  
  - SpaceCast는 천체 시뮬레이션에만 유리, 일반 실세계 데이터에선 예측 불안정  
  - CylinderSelection은 결과 예측 가능하나 복잡한 장면에 비효율적  

***

## 5. 일반화 성능 향상 가능성  
- **현재 한계**: ShapeNet·S3DIS 모델별로 별도 학습. 단일 모델 훈련 시 데이터 분포 차이로 성능 저하 (Jaccard distance ↑).  
- **개선 방안**  
  1. **Multi-source Domain Adaptation**: 서로 다른 도메인 간 특성 정렬하여 단일 모델로 일반화  
  2. **데이터 증강**: 기존 라쏘 레코드 변형 생성으로 레코드 수 확대  
  3. **최신 아키텍처 통합**: MCCNN 등 비균일 샘플링에 강건한 백본 도입  
  4. **뷰포인트·FOV 파라미터 추가 학습**: VR/AR 등 다양한 렌더링 환경 확장  

***

## 6. 결론 및 향후 고려사항  
- LassoNet은 **뷰포인트와 라쏘** 정보를 효과적으로 결합하여 3D 포인트 클라우드 선택을 자동화하고, 기존 휴리스틱 기법의 한계를 극복함을 입증했다.  
- **향후 영향**: 3D 인터랙션 연구 전반에 딥러닝 도입을 촉진하며, 포인트 클라우드 편집·분할, 증강현실 셀렉션 등 응용으로 확산 가능.  
- **연구 시 고려점**: 다중 도메인 일반화, 적은 데이터 상황에서의 성능 보장, 사용자 정교 제어(근접 수정) 보완, 비선형 투영·스테레오 환경 대응 등이 필수적이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/16c04f08-b104-4eb6-9a64-c9a60a985071/1907.13538v4.pdf
