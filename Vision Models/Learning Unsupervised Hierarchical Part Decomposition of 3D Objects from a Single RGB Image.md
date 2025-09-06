# Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image | 3D Segmentation, 3D reconstruction

**핵심 주장 및 주요 기여**  
이 논문은 **단일 RGB 이미지**만으로 3D 객체를 **피라미드식 계층 구조**(binary tree) 형태의 기하학적 프리미티브(superquadrics)로 **비지도 학습**하여 분할 및 재구성하는 모델을 제안한다.  
- **비지도 구조 학습**: 부품(part) 경계나 계층(structure) 레이블 없이, 전체 메시(mesh)만으로 계층적 파트 분해를 학습  
- **동적 프리미티브 수 제어**: 각 노드별 **재구성 품질**(IoU 예측)을 통해 불필요한 분기를 중단, 단순 파트는 적은 프리미티브로 표현  
- **파트별 계층 표현**: 피쳐 인코더 → 파티션 네트워크 → 구조 네트워크 → 기하 네트워크로 이어지는 모듈러 구조

***

## 1. 해결 과제  
기존 단일 뷰 3D 재구성 연구는  
- 로컬 형상 복원에 집중하거나  
- 플랫(flat) 파트 집합만 예측  
→ **높은 수준의 부품 간 관계**와 **계층 구조**를 반영한 해석 가능 재구성이 부재  

본 연구는 *어떤 물체가 어떤 파트로 구성되며*, 파트들이 **어떻게 계층적으로** 연결되는지를 학습 없이 추론하고, 그 결과를 이용해 3D 형상을 재구성하는 것을 목표로 한다.

***

## 2. 제안 방법  
### 2.1 모델 개요  
입력 $$I$$ (RGB 이미지 또는 볼륨) → ResNet-18 인코더 → 루트 특징 $$c^0_0$$  
재귀적으로 깊이 $$d=0\dots D$$까지 파티션  
- Partition Net: $$c^d_k \to (c^{d+1}\_{2k},\,c^{d+1}_{2k+1})$$  
- Structure Net: $$c^d_k \to h^d_k \in \mathbb R^3$$ (파트 중심)  
- Geometry Net: $$c^d_k \to (\lambda^d_k,\,q^d_k)$$  
  -  $$\lambda^d_k$$: superquadric 크기·형상 파라미터 $$\alpha, \epsilon$$, 위치·방향 변환 $$t,q$$  
  -  $$q^d_k$$: 예측 IoU (재구성 품질)

### 2.2 주요 수식  
1. **파트 중심 정의**  
   $$\displaystyle H=\{h^d_k\}$$, 각 $$h^d_k$$로 점군 $$X^d_k$$을 최근접 기준 할당  
2. **재구성 손실**  

$$L_{\mathrm{rec}}=\sum_{d=0}^D\sum_{(x,o)\in X}\mathrm{BCE}(G^d(x),o)+\sum_{d,k}\sum_{(x,o)\in X^d_k}\mathrm{BCE}(g^d_k(x),o)$$  
   
   -  $$G^d$$: 전체 프리미티브 합집합 점유 함수, $$g^d_k$$: 개별 프리미티브 점유 함수  
3. **구조 손실** (유사 k-means)  

$$L_{\mathrm{str}}=\sum_{d,k}\sum_{(x,o)\in X^d_k}o\,\|x-h^d_k\|^2$$  

4. **호환성 손실** (IoU 예측)  

$$L_{\mathrm{comp}}=\sum_{d,k}(q^d_k-\mathrm{IoU}(p^d_k,X^d_k))^2$$  

5. **근접성 손실** (vanishing gradient 방지)  

$$L_{\mathrm{prox}}=\sum_{d,k}\|t(\lambda^d_k)-h^d_k\|^2$$

전체: $$L=L_{\mathrm{str}}+L_{\mathrm{rec}}+L_{\mathrm{comp}}+L_{\mathrm{prox}}$$

***

## 3. 모델 구조  
- **Feature Encoder**: ResNet-18 (single view) 또는 3D CNN (볼륨)  
- **Partition Network**: 2-layer MLP, RELU  
- **Structure Network**: 2-layer MLP, RELU → 파트 중심 좌표  
- **Geometry Network**: 개별 MLP로 superquadric 파라미터 추정 및 $$q$$ 예측  

***

## 4. 성능 개선 및 한계  
### 4.1 성능 향상  
- ShapeNet 단일 뷰: OccNet 대비 평균 IoU↑, Chamfer-L1 유사 수준  
- D-FAUST 인체: SQs(superquadrics flat) 대비 IoU 0.608→0.699, Chamfer-L1 0.189→0.098  
- **계층 예측**: 동일 파트에 동일 노드가 일관되게 매핑되어 의미론적 파트 분해 달성  

### 4.2 한계  
- **계산 복잡도**: 최대 노드 수 $$2^D$$ 규모 → 학습·추론 비용  
- **형상 다양성 제약**: superquadric 기하학으로 복잡 곡면 표현 한계  
- **강인성**: 실제 조명·텍스처 잡음에 대한 일반화 분석 미흡  

***

## 5. 일반화 성능 향상 가능성  
- **입력 다양성**: 다중 뷰 영상 또는 실세계 RGB-D 센서 데이터로 확장  
- **프리미티브 확장**: 슈퍼쿼드릭 외 일반 볼록형(Convex) 또는 학습형 기하 도입  
- **대상 다양화**: 복잡 계층 구조(예: 기계·가구)·장면(scene) 전체로 확대  
- **강화 학습 기반 탐색**: 분할 결정 학습 시 RL 활용으로 동적 깊이 제어 최적화  

***

## 6. 향후 연구 영향 및 고려 사항  
- **구조 인식 3D 재구성**: 비지도 계층 학습을 통해 해석 가능 모델 설계 가속  
- **로봇·자율주행**: 물체 부품별 추적·조작에 활용할 수 있는 3D 파트 분해  
- **비전–언어 연동**: “문 손잡이” 등 부품 단위 질의 응답에 계층 정보 활용  
- **고려점**: 실제 환경 노이즈 대응, 학습 효율성 및 대규모 데이터셋 적용 시 계산 비용 최적화  

이 논문은 **비지도**로 3D 객체의 **계층적 부품 구조**를 동시에 학습·재구성하는 새로운 방향을 제시하며, 향후 **해석 가능한 3D 표현** 연구의 토대를 마련한다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/21f96c49-729f-4660-958b-14d22e62f331/2004.01176v1.pdf)
