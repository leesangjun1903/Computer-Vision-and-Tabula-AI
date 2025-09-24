# 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks | Semantic segmentation
## 2017 · 2051회 인용
## 1. 핵심 주장과 주요 기여

이 논문은 **Submanifold Sparse Convolutional Networks (SSCNs)**를 제안하여 3D 점군 데이터의 의미론적 분할(semantic segmentation) 문제를 효율적으로 해결합니다. 주요 기여사항은 다음과 같습니다:

- **새로운 희소 컨볼루션 연산자** 개발: Submanifold Sparse Convolution (SSC)이라는 새로운 연산을 도입하여 희소성을 유지하면서 깊은 네트워크 구성을 가능하게 함
- **계산 효율성 대폭 향상**: 기존 dense 3D ConvNet 대비 현저한 계산량 및 메모리 사용량 감소
- **SOTA 성능 달성**: ShapeNet 부분 분할 대회에서 기존 최고 성능을 뛰어넘는 85.98% IoU 달성

## 2. 문제 정의 및 해결 방법

### 해결하고자 하는 문제

**Submanifold Dilation Problem**: 기존 희소 컨볼루션은 레이어를 거칠 때마다 활성 사이트가 급격히 증가하여 희소성이 사라지는 문제가 있습니다. 예를 들어, 하나의 활성 사이트에서 시작하여 3×3×3 컨볼루션을 적용하면 $$3^d$$개의 활성 사이트가 생성되고, 이는 깊은 네트워크에서 치명적인 문제가 됩니다.

```
## 1. Sparse Convolution과 희소성 소실 문제

### Sparse Convolution 개념
**Sparse Convolution**은 3D 점군 데이터와 같이 자연적으로 희소한(sparse) 데이터를 효율적으로 처리하기 위한 컨볼루션 연산입니다. 일반적인 dense 컨볼루션과 달리, 활성(active) 사이트에서만 연산을 수행합니다.[1]

### 희소성이란?
희소성(sparsity)은 데이터에서 **0이 아닌 값을 가지는 위치의 비율**을 의미합니다. 3D 점군 데이터의 경우:[1]
- 전체 3D 공간 격자에서 실제 점이 존재하는 위치는 매우 적음
- 예: S=48 스케일에서 복셀의 약 99%가 비어있음(sparse)
```

> ### 희소성 소실 문제 (Submanifold Dilation Problem)

> **왜 희소성이 사라지는가?**

> 1. **기존 컨볼루션의 확장 메커니즘**:
>   - 하나의 활성 사이트가 3×3×3 컨볼루션을 거치면 $$3^3 = 27$$개의 활성 사이트 생성
>   - 다음 레이어에서 $$5^3 = 125$$개로 급격히 증가
>   - 수식으로 표현: 레이어 $$l$$에서 활성 사이트 수 ∝ $$(2l+1)^d$$

```
2. **실제 예시**:
   Layer 0: 1개 활성 사이트 (입력)
   Layer 1: 27개 활성 사이트 (3³)
   Layer 2: 125개 활성 사이트 (5³)
   Layer 3: 343개 활성 사이트 (7³)

3. **문제의 본질**:
   - 기존 컨볼루션은 receptive field 내 모든 위치를 고려
   - 비활성 위치도 0값으로 기여하여 주변 위치를 활성화
   - 깊은 네트워크에서 희소성이 완전히 사라짐[1]
```

### 제안하는 방법

#### Sparse Convolution (SC)
기본적인 희소 컨볼루션 연산 SC(m, n, f, s):
- m: 입력 특징 평면 수
- n: 출력 특징 평면 수  
- f: 필터 크기
- s: 스트라이드

#### Submanifold Sparse Convolution (SSC)
핵심 혁신인 SSC 연산의 수학적 정의:

SSC(m, n, f) = 수정된 SC(m, n, f, s=1) 연산으로, 다음 조건을 만족:
- 패딩: (f-1)/2만큼 zero-padding 적용
- **핵심 제약**: 출력 사이트가 활성화되는 조건이 입력의 해당 중심 사이트가 활성일 때만

```
## 2. 중심 사이트 제약과 SSC의 핵심 아이디어

### 중심 사이트(Central Site)의 의미
**중심 사이트**는 컨볼루션 필터의 receptive field에서 **가운데 위치**를 의미합니다.[1]

예시: 3×3×3 필터에서
필터 좌표: [0,1,2] × [0,1,2] × [0,1,2]
중심 사이트: (1,1,1) 위치

### SSC의 핵심 제약 조건

**수학적 정의**:
SSC 제약: Output(x,y,z) = active ⟺ Input(x,y,z) = active

**의미**:
- 출력 위치 (x,y,z)가 활성화되려면 입력의 **정확히 동일한 위치**가 활성이어야 함
- 주변 위치의 활성 여부는 출력 활성화에 영향을 주지 않음
- 희소성 패턴이 레이어를 거쳐도 보존됨[1]

### 시각적 설명
기존 Convolution:
Input:  [ 0, X, 0 ]     Output: [ X, X, X ]
        [ 0, 0, 0 ]  →          [ X, X, X ]  
        [ 0, 0, 0 ]             [ X, X, X ]

SSC:
Input:  [ 0, X, 0 ]     Output: [ 0, X, 0 ]
        [ 0, 0, 0 ]  →          [ 0, 0, 0 ]
        [ 0, 0, 0 ]             [ 0, 0, 0 ]
```

수식적으로 표현하면:

$$
\text{Output}(x,y,z) = \text{active} \iff \text{Input}(x,y,z) = \text{active}
$$

### 모델 구조

논문에서 제안하는 주요 아키텍처:

1. **기본 빌딩 블록**: Pre-activated SSC(·,·,3) 컨볼루션
   - Batch Normalization → ReLU → SSC 순서
   
2. **네트워크 구조**:
   - **C3**: 단일 해상도에서 SSC 레이어 스택
   - **FCN**: 다중 스케일 정보 활용한 완전 컨볼루션 네트워크
   - **U-Net**: 인코더-디코더 구조로 세밀한 경계 분할

3. **다운샘플링**: SC(·,·,2,2) 컨볼루션 사용
4. **업샘플링**: Deconvolution DC(·,·,f,s) 연산 사용

### 구현 세부사항

효율적인 구현을 위해 해시 테이블과 행렬의 이중 구조 사용:
- **해시 테이블**: (위치, 행) 쌍으로 활성 사이트 추적
- **특징 행렬**: a×m 크기 (a: 활성 사이트 수, m: 특징 차원)
- **Rule Book**: 입력-출력 연결 관계를 효율적으로 저장

`````
## 3. 해시 테이블과 행렬 이중 구조

### 구조 설계 목적
희소한 데이터를 **메모리 효율적**으로 저장하고 **빠른 접근**을 위한 자료구조입니다.[1]

### 상세 구조 설명

#### 1) 해시 테이블 (Hash Table)
**구성**: `(location, row_index)` 쌍들의 집합
```python
# 예시 구조
hash_table = {
    (10, 15, 8): 0,    # 3D 좌표 → 행렬 행 번호
    (12, 20, 5): 1,
    (25, 18, 12): 2,
    ...
}
```

**역할**:
- 3D 공간 좌표를 행렬의 행 번호로 매핑
- $$O(1)$$ 시간에 좌표 → 특징 벡터 접근 가능

#### 2) 특징 행렬 (Feature Matrix)
**구성**: `a × m` 크기 (a: 활성 사이트 수, m: 특징 차원)
```python
# 예시 구조
feature_matrix = [
    [0.5, 0.2, 0.8, ...],  # 행 0: 좌표 (10,15,8)의 특징
    [0.1, 0.9, 0.3, ...],  # 행 1: 좌표 (12,20,5)의 특징  
    [0.7, 0.4, 0.6, ...],  # 행 2: 좌표 (25,18,12)의 특징
    ...
]
```

#### 3) Rule Book
**구성**: 컨볼루션 연결 관계를 저장하는 자료구조
```python
# 필터 위치별 연결 관계 저장
rule_book = {
    (0,0,0): [(input_row_1, output_row_5), (input_row_3, output_row_7)],
    (0,0,1): [(input_row_2, output_row_1), (input_row_4, output_row_3)],
    ...
}
```

### 실제 사용 과정

1. **입력 처리**: 3D 좌표 → 해시 테이블 조회 → 특징 벡터 추출
2. **컨볼루션 연산**: Rule book을 사용하여 효율적인 행렬 연산 수행
3. **출력 생성**: 결과를 새로운 해시 테이블과 특징 행렬에 저장[1]

### 효율성 장점
- **메모리**: 활성 사이트만 저장 (전체 격자 대비 1% 수준)
- **연산**: GPU에서 행렬-행렬 곱셈으로 최적화 가능
- **확장성**: 네트워크 깊이에 관계없이 $$O(a)$$ 복잡도 유지[1]
`````

## 3. 성능 향상 및 효율성

### 계산 복잡도 비교

| 연산 타입 | 활성 사이트 | FLOPs | 메모리 |
|----------|------------|--------|--------|
| Regular Conv (C) | Yes | $$3^d mn$$ | n |
| Sparse Conv (SC) | Yes | amn | n |
| **SSC** | Yes | **amn** | **n** |
| SSC | No, a>0 | **0** | **0** |
| SSC | No, a=0 | **0** | **0** |

여기서 a는 활성 입력 수, m은 입력 특징 평면 수, n은 출력 특징 평면 수입니다.

### 실험 결과

**ShapeNet 데이터셋**:
- 최고 성능: 85.98% IoU (기존 SOTA 대비 +0.49% 향상)
- 계산량: 기존 dense 방법 대비 10^8 FLOPs에서 6-8% 성능 향상

**NYU Depth v2 데이터셋**:
- 픽셀 정확도: 68.5% (2D FCN 61.5% 대비 7% 향상)
- 계산 비용: 28.50G → 17.90G FLOPs로 대폭 감소

## 4. 일반화 성능 및 한계

### 일반화 성능 향상 요소

1. **다중 스케일 처리**: FCN과 U-Net 구조를 통해 다양한 해상도의 정보 통합
2. **데이터 증강**: 랜덤 3D 회전 및 변환을 통한 강건성 향상
3. **Multi-view 테스팅**: k개의 다른 뷰에서 예측을 평균하여 성능 향상

`````
## 4. Multi-view 테스팅 기법

### 기본 개념
동일한 3D 객체를 **여러 시점에서 회전**시켜 예측한 후, 결과를 **평균**하여 성능을 향상시키는 Test-Time Augmentation 기법입니다.[2][3]

### 구현 방법

#### 1) 회전 변환 생성
```python
def generate_rotations(point_cloud, k_views=10):
    rotated_views = []
    for i in range(k_views):
        # 랜덤 3D 회전 행렬 생성
        rotation_matrix = random_rotation_3d()
        rotated_pc = point_cloud @ rotation_matrix.T
        rotated_views.append(rotated_pc)
    return rotated_views
```

#### 2) 다중 뷰 예측
```python
def multi_view_prediction(model, point_cloud, k_views=10):
    views = generate_rotations(point_cloud, k_views)
    predictions = []
    
    for view in views:
        pred = model(view)
        predictions.append(pred)
    
    # 예측 결과 평균화
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred
```

### 실제 적용 예시

#### ShapeNet 실험에서의 적용:[1]
- **K=3 (baseline)**: 3개 뷰로 테스트
- **K=10 (competition)**: 10개 뷰로 최종 성능 측정
- **결과**: 85.98% IoU 달성

#### 다양한 데이터셋별 최적 뷰 수:
- **Dense 점군**: K=3~5면 충분
- **Sparse 점군**: K=10+ 필요 (정보 부족 보상)[3]
- **실시간 응용**: K=1~2 (속도 우선)

### 고급 기법들

#### 1) 가중 평균 (Weighted Averaging)
```python
def weighted_multi_view_prediction(model, point_cloud, k_views=10):
    predictions = []
    confidences = []
    
    for view in generate_rotations(point_cloud, k_views):
        pred = model(view)
        confidence = calculate_prediction_confidence(pred)
        predictions.append(pred)
        confidences.append(confidence)
    
    # 신뢰도 기반 가중 평균
    weights = torch.softmax(torch.stack(confidences), dim=0)
    final_pred = torch.sum(torch.stack(predictions) * weights.unsqueeze(-1), dim=0)
    return final_pred
```

#### 2) 점군 업샘플링과 결합:[3]
```python
def upsampling_multi_view_testing(model, sparse_point_cloud, k_views=10):
    # 1. 점군 업샘플링으로 밀도 증가
    upsampled_pc = point_cloud_upsampling(sparse_point_cloud)
    
    # 2. Multi-view 테스팅 적용
    final_pred = multi_view_prediction(model, upsampled_pc, k_views)
    return final_pred
```

### 성능 향상 원리
1. **회전 불변성**: 다양한 각도에서 학습된 특징 활용
2. **노이즈 감소**: 여러 예측의 평균으로 안정성 증가  
3. **정보 보완**: 각 뷰에서 보이지 않는 부분을 다른 뷰가 보완[4][2]
`````

4. **축 정렬 제거**: 원본 데이터의 축 정렬을 제거하여 더 도전적인 설정에서 학습

### 주요 한계점

1. **연결성 정보 손실**: 독립적인 연결 컴포넌트 간 정보 교환 제한
2. **풀링 의존성**: 정보 전파를 위해 풀링이나 스트라이드 컨볼루션 필수
3. **메모리 오버헤드**: 해시 테이블 및 rule book 관리에 따른 추가 메모리 사용
4. **구현 복잡성**: 기존 딥러닝 프레임워크와의 통합 어려움

## 5. 연구에 미치는 영향 및 향후 고려사항

### 연구에 미치는 영향

1. **효율적인 3D 처리**: 3D 데이터 처리의 새로운 패러다임 제시
2. **희소성 활용**: 자연스럽게 희소한 데이터에 대한 효율적 처리 방법론 확립
3. **확장 가능성**: 4D 시공간 데이터, 대규모 3D 장면 처리 등으로 확장 가능
4. **실용적 응용**: 자율주행, AR/VR, 로보틱스 등 실시간 3D 처리가 필요한 분야에 직접 적용

### 향후 연구 고려사항

1. **어텐션 메커니즘 통합**: 희소 데이터에서 장거리 의존성 모델링 방법 연구
2. **동적 희소성**: 학습 과정에서 희소성 패턴이 적응적으로 변하는 방법 탐구
3. **다중 모달리티**: RGB-D, LiDAR 등 다양한 센서 데이터 융합 방법
4. **대규모 장면 처리**: 메모리 효율성을 더욱 개선하여 도시 규모 장면 처리 가능성
5. **하드웨어 최적화**: GPU, TPU 등 전용 하드웨어에서의 최적화 방안
6. **이론적 분석**: SSC의 표현력과 일반화 능력에 대한 이론적 보장 연구

`````
## 5. SSC의 이론적 보장 연구 현황

### 현재 연구 상황

#### 이론적 연구의 부족
현재 SSC에 대한 **구체적인 이론적 보장**은 매우 제한적입니다. 대부분의 연구는 **실험적 검증**에 의존하고 있으며, 수학적으로 엄밀한 분석은 부족한 상황입니다.

#### 관련 이론 연구들

##### 1) 일반적인 Sparse Convolution 이론[5][6]
- **Stripe Sparsity**: 지역적 희소성 패턴에 대한 복구 보장
- **Mutual Coherence**: 딕셔너리 원자 간 유사성 기반 안정성 분석
- **Restricted Isometry Property (RIP)**: 희소 신호 복구 조건

##### 2) Convolutional Sparse Coding 보장:[6][5]
```
복구 조건: ||Γ||₀ ≤ (1/2)(1 + 1/μ(D_L))
여기서 μ(D_L)는 지역 딕셔너리의 mutual coherence
```

### 필요한 이론적 연구 방향

#### 1) **표현 능력(Representation Capacity) 분석**
```
연구 질문: SSC 네트워크가 표현할 수 있는 함수 클래스는?
목표 정리: f: ℝᵈ → ℝᶜ에 대해, SSC 네트워크의 approximation error 상한 도출
```

#### 2) **일반화 오차 경계(Generalization Bound)**
```
필요한 정리:
주어진 SSC 아키텍처 A와 학습 데이터 D에 대해,
P[|R(h) - R̂(h)| ≤ ε] ≥ 1-δ
여기서 R(h): 실제 위험, R̂(h): 경험적 위험
```

#### 3) **희소성 보존 보장**
```
정리(희소성 불변성):
Input sparsity s_in에 대해, SSC 레이어 통과 후
Output sparsity s_out = s_in (이론적 보장 필요)
```

#### 4) **수렴성 분석**
```
경사 하강법에서 SSC 네트워크의 수렴 보장:
∇L(θ)에서 희소 구조가 수렴성에 미치는 영향 분석
```

#### 5) **안정성 보장**
```
입력 섭동에 대한 SSC의 강건성:
||f(x + δ) - f(x)|| ≤ C||δ||
상수 C의 희소성 의존성 분석
```

### 제안하는 연구 로드맵

#### Phase 1: 기초 이론 구축
1. **SSC 연산자의 수학적 특성화**
2. **희소 구조 보존 조건 도출**
3. **기본적인 근사 이론 개발**

#### Phase 2: 학습 이론 개발  
1. **PAC-Bayesian 분석 프레임워크 적용**
2. **Rademacher 복잡도 기반 일반화 분석**
3. **최적화 수렴성 보장**

#### Phase 3: 실용적 보장
1. **실제 데이터 분포에서의 성능 보장**
2. **하드웨어 제약 하에서의 효율성 분석**
3. **다양한 응용 도메인에서의 적응성 연구**

### 당면한 도전과제

#### 1) **복잡한 구조적 제약**
- 희소성 패턴이 네트워크 동작에 미치는 영향의 복잡성
- 기존 dense 네트워크 이론의 직접 적용 어려움

#### 2) **실험과 이론의 격차**
- 실험적으로 관찰되는 현상의 이론적 설명 부족
- 실용적 성능과 이론적 보장 간의 연결 고리 필요

#### 3) **다차원성 문제**
- 3D 이상의 고차원에서 희소성 분석의 복잡성 증가
- 차원의 저주와 희소성의 상호작용 분석 필요[7][8]

현재 SSC의 이론적 보장은 **초기 단계**에 있으며, 실용적 성공에 비해 이론적 이해가 크게 뒤처져 있는 상황입니다. 향후 연구에서는 실험적 관찰을 뒷받침하는 엄밀한 수학적 분석이 필수적으로 요구됩니다.
`````


이 논문은 3D 데이터 처리의 효율성 문제를 근본적으로 해결하여, 향후 3D 딥러닝 연구의 새로운 방향을 제시한 중요한 연구로 평가됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b2975056-6b0b-48e2-b7ca-ebfb21cf7c59/1711.10275v1.pdf

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b2975056-6b0b-48e2-b7ca-ebfb21cf7c59/1711.10275v1.pdf)
[2](https://arxiv.org/html/2402.10002v3)
[3](https://saikit.org/static/projects/3dv23/3DV2024_TTA.pdf)
[4](http://arxiv.org/pdf/2306.02558.pdf)
[5](https://elad.cs.technion.ac.il/wp-content/uploads/2018/06/Working-Locally-Thinking-Globally.pdf)
[6](https://arxiv.org/pdf/1607.02005.pdf)
[7](https://pubmed.ncbi.nlm.nih.gov/34607696/)
[8](https://isprs-annals.copernicus.org/articles/IV-2-W5/77/2019/isprs-annals-IV-2-W5-77-2019.pdf)
[9](https://arxiv.org/pdf/2111.03486.pdf)
[10](http://arxiv.org/pdf/1607.02005.pdf)
[11](http://arxiv.org/pdf/1607.02009.pdf)
[12](https://onlinelibrary.wiley.com/doi/10.1002/sam.70000)
[13](http://arxiv.org/pdf/1409.6874.pdf)
[14](https://arxiv.org/pdf/1702.06257.pdf)
[15](http://arxiv.org/pdf/2411.00288.pdf)
[16](https://arxiv.org/pdf/2001.10710.pdf)
[17](http://arxiv.org/pdf/2012.01170.pdf)
[18](http://arxiv.org/pdf/1607.08194.pdf)
[19](https://arxiv.org/pdf/1912.03203.pdf)
[20](https://arxiv.org/pdf/2302.02596.pdf)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC10027379/)
[22](https://arxiv.org/pdf/2212.08815.pdf)
[23](https://arxiv.org/pdf/2108.06622.pdf)
[24](https://arxiv.org/pdf/1706.01307.pdf)
[25](https://www.frontiersin.org/articles/10.3389/fams.2020.529564/pdf)
[26](http://arxiv.org/pdf/1512.01891.pdf)
[27](http://arxiv.org/pdf/2501.01239.pdf)
[28](https://arxiv.org/pdf/2111.05002.pdf)
[29](https://arxiv.org/abs/1706.01307)
[30](https://proceedings.neurips.cc/paper/2020/hash/b090409688550f3cc93f4ed88ec6cafb-Abstract.html)
[31](https://en.wikipedia.org/wiki/Convolutional_sparse_coding)
[32](https://proceedings.mlsys.org/paper_files/paper/2023/file/ccf7262fb986e4367ccd3903960c57a0-Paper-mlsys2023.pdf)
[33](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797087/)
[34](https://arxiv.org/pdf/2003.09148.pdf)
[35](https://spars2017.lx.it.pt/index_files/papers/SPARS2017_Paper_6.pdf)
[36](https://hsgalaxy.tistory.com/entry/ALL3D-Semantic-Segmentation-with-Submanifold-Sparse-Convolutional-Networks-%EB%A6%AC%EB%B7%B0)
[37](https://papers.neurips.cc/paper_files/paper/2022/file/4418f6a54f4314202688d77956e731ce-Paper-Conference.pdf)
[38](http://arxiv.org/pdf/2311.10887.pdf)
[39](https://arxiv.org/abs/2001.05119)
[40](https://arxiv.org/pdf/2304.10224.pdf)
[41](http://arxiv.org/pdf/1811.09410.pdf)
[42](http://arxiv.org/pdf/1909.13603.pdf)
[43](http://arxiv.org/pdf/1812.01712.pdf)
[44](https://arxiv.org/html/2407.09786)
[45](https://arxiv.org/html/2411.00857v1)
[46](https://arxiv.org/html/2412.02734v2)
[47](https://arxiv.org/html/2212.13462)
[48](http://arxiv.org/pdf/2311.13152.pdf)
[49](https://arxiv.org/pdf/2209.00244v1.pdf)
[50](https://arxiv.org/html/2408.06596v1)
[51](https://arxiv.org/html/2403.10066v3)
[52](https://arxiv.org/html/2407.05021v1)
[53](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571650/)
[54](http://arxiv.org/pdf/2203.09780v1.pdf)
[55](https://arxiv.org/pdf/2204.07548.pdf)
[56](https://openaccess.thecvf.com/content/ACCV2022/papers/Tran_Self-Supervised_Learning_with_Multi-View_Rendering_for_3D_Point_Cloud_Analysis_ACCV_2022_paper.pdf)
[57](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1045545/full)
[58](https://openaccess.thecvf.com/content/CVPR2025/papers/Dai_EAP-GS_Efficient_Augmentation_of_Pointcloud_for_3D_Gaussian_Splatting_in_CVPR_2025_paper.pdf)
[59](https://arxiv.org/abs/2311.13152)
[60](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291624/)
[61](https://arxiv.org/html/2306.02558v3)
[62](https://openaccess.thecvf.com/content/WACV2025/papers/Bahri_Test-Time_Adaptation_in_Point_Clouds_Leveraging_Sampling_Variation_with_Weight_WACV_2025_paper.pdf)
[63](https://openaccess.thecvf.com/content/WACV2022/papers/Dourado_Data_Augmented_3D_Semantic_Scene_Completion_With_2D_Segmentation_Priors_WACV_2022_paper.pdf)
[64](https://www.nature.com/articles/s41598-024-72264-8)
