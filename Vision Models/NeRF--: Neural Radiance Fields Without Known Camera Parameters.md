# NeRF--: Neural Radiance Fields Without Known Camera Parameters | 3D reconstruction, photometric reconstruction

## 1. 핵심 주장과 주요 기여

**NeRF--**는 기존 NeRF의 핵심 제약사항인 사전 계산된 카메라 파라미터 요구사항을 제거하여, **카메라 파라미터와 3D 장면 표현을 동시에 최적화**하는 혁신적인 접근법을 제시합니다.

```
## 1. 카메라 파라미터 요구사항

카메라 파라미터는 3D 장면을 2D 이미지로 투영하는 과정에서 필요한 수학적 정보입니다.

**구성 요소:**
1) **내재 파라미터 (Intrinsic Parameters):**
   - 초점 거리 (f): 렌즈 중심에서 이미지 센서까지의 거리
   - 주점 (cx, cy): 이미지 센서의 중심점 좌표
   - 왜곡 계수들

2) **외재 파라미터 (Extrinsic Parameters):**
   - 6DoF 포즈: 3D 공간에서 카메라의 위치와 방향
   - 회전 (Rotation): 카메라가 어느 방향을 보고 있는지
   - 변환 (Translation): 카메라가 어디에 위치하는지

**기존 NeRF의 문제점:**
- 이러한 파라미터들이 미리 계산되어 있어야 함
- 실제 촬영 상황(모바일 폰 등)에서는 이 정보를 얻기 어려움
- COLMAP 같은 전처리 도구에 의존해야 함
```

### 주요 기여 3가지:

1. **카메라 파라미터 동시 최적화**: 광측정 재구성(photometric reconstruction)을 통해 카메라 파라미터를 학습 가능한 매개변수로 설정하여 NeRF 훈련과 함께 최적화할 수 있음을 증명

```
## 2. 광측정 재구성 (Photometric Reconstruction)

광측정 재구성은 픽셀의 색상 정보를 직접 비교하여 3D 구조를 복원하는 방법입니다.

**작동 원리:**
1) **입력**: 여러 시점에서 촬영한 2D 이미지들
2) **과정**: 
   - 각 픽셀의 색상값을 직접 비교
   - 같은 3D 점이 다른 시점에서 어떻게 보이는지 분석
   - 색상 일치도를 기반으로 카메라 위치와 3D 구조를 동시에 추정
3) **출력**: 카메라 파라미터 + 3D 장면 표현

**장점:**
- 특징점 매칭이 필요 없음
- 텍스처가 부족한 영역에서도 작동 가능
- 전역적으로 일관된 결과 제공

**비유**: 여러 각도에서 찍은 사진들의 색상 패턴을 맞춰가며 퍼즐을 맞추는 것과 같음
```

2. **BLEFF 데이터셋 도입**: 카메라 파라미터 추정과 새로운 시점 렌더링 품질을 평가하기 위한 path-traced 합성 장면 데이터셋 제공

```
## 3. BLEFF 데이터셋의 Path-traced 합성 장면

BLEFF (Blender Forward-Facing Dataset)는 이 논문에서 새로 제안한 합성 데이터셋입니다.

**Path-traced 합성 장면의 특징:**
1) **Path Tracing:**
   - 실제 광선의 물리적 경로를 시뮬레이션
   - 반사, 굴절, 그림자 등을 정확히 계산
   - 매우 사실적인 렌더링 결과 생성

2) **합성 장면:**
   - Blender로 제작된 3D 모델들
   - 14개의 다양한 장면 (비행기, 공, 욕실, 침실 등)
   - 실제 물리 법칙을 따르는 조명과 재질

3) **데이터셋의 장점:**
   - 정확한 ground truth 카메라 파라미터 제공
   - 다양한 카메라 움직임 패턴으로 테스트 가능
   - 16가지 궤적 변형 (회전/병진 섭동 조합)

**목적**: 카메라 파라미터 추정 정확도와 새로운 시점 렌더링 품질을 동시에 평가
```

3. **포괄적 분석**: 다양한 카메라 움직임 패턴에서의 훈련 행동 분석을 통해 대부분의 시나리오에서 COLMAP 기반 방법과 비교 가능한 성능을 달성함을 입증

```
## 4. COLMAP

COLMAP은 Structure-from-Motion (SfM) 분야에서 가장 널리 사용되는 오픈소스 도구입니다.

**주요 기능:**
1) **카메라 파라미터 추정:**
   - 여러 이미지에서 특징점 추출 및 매칭
   - 특징점 대응 관계를 이용해 카메라 위치/방향 계산
   - 내재/외재 파라미터 모두 추정

2) **3D 복원:**
   - 희소한 3D 점군 생성
   - Bundle Adjustment로 전역 최적화

3) **작동 과정:**
   입력 이미지들 → 특징점 검출 → 매칭 → 카메라 파라미터 추정 → 3D 점군 생성

**한계:**
- 특징점이 부족한 영역에서 실패
- 반복적인 패턴이나 동질적 영역에서 오류
- 빠른 시점 변화나 모호한 카메라 궤적에서 문제 발생
- 전처리 단계로 추가 복잡성 도입

```

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의
기존 NeRF 방법들은 **사전 계산된 카메라 파라미터**(내재/외재 파라미터)를 필요로 하며, 이는 실제 시나리오(예: 모바일 폰으로 촬영한 이미지)에서 접근하기 어렵습니다. COLMAP과 같은 전처리 단계는 다음과 같은 문제를 야기합니다:

- 추가적인 복잡성 도입
- 동질적 영역, 빠른 시점 의존적 외관 변화로 인한 오류 발생
- 모호한 카메라 궤적으로 인한 퇴화된 해 발생

```
## 5. 동질적 영역과 퇴화된 해

**동질적 영역 (Homogeneous Regions):**
- **정의**: 색상이나 텍스처가 거의 동일한 영역 (예: 흰 벽, 하늘, 잔디밭)
- **문제점**: 특징점 검출이 어려워 COLMAP이 실패하기 쉬움

**시점 의존적 외관 변화 (View-dependent Appearance Changes):**
- **정의**: 보는 각도에 따라 물체의 모양이나 색이 달라지는 현상
- **예시**: 반사면, 투명한 유리, 금속 표면의 반사
- **문제점**: 특징점 매칭 시 같은 점을 다른 점으로 인식하여 오류 발생

**퇴화된 해 (Degenerate Solutions):**
- **정의**: 수학적으로는 맞지만 실제로는 틀린 해답
- **발생 원인:**
  1) 모든 카메라가 한 직선상에 위치 (Pure Translation)
  2) 모든 카메라가 한 점 중심으로 회전 (Pure Rotation)  
  3) 카메라 움직임이 특정 평면에 제한

- **왜 문제인가?**
  * 기본 행렬(Fundamental Matrix) 계산 시 여러 해가 존재
  * 잘못된 해를 선택하면 완전히 틀린 카메라 파라미터 추정
  * 특히 선형 방정식 시스템에서 해가 유일하지 않음

**예시**: 직선으로 움직이며 촬영한 경우, 무한히 많은 카메라 위치가 같은 이미지를 생성할 수 있음
```

### 제안 방법

**수식적 정형화:**

기존 NeRF: 

$$ \Theta^* = \arg\min_\Theta L(\hat{I}|I, \Pi) $$

NeRF--:

```math
\Theta^*, \Pi^* = \arg\min_{\Theta,\Pi} L(\hat{I}, \hat{\Pi}|I)
```

여기서 Π는 내재 및 6DoF 포즈를 포함한 카메라 파라미터입니다.

```
## 6. 내재 및 6DoF 포즈

**내재 파라미터 (Intrinsic Parameters):**
- 카메라 자체의 물리적 특성을 나타내는 파라미터들

**구성 요소:**
1) 초점 거리 (f): 렌즈에서 센서까지의 거리 (픽셀 단위로 표현)
2) 주점 (Principal Point, cx, cy): 
   - 광축이 이미지 평면과 만나는 점
   - 보통 이미지 중심에 위치 (cx ≈ W/2, cy ≈ H/2)
3) 비등방성 스케일링 인수
4) 렌즈 왜곡 계수들

**6DoF 포즈 (6 Degrees of Freedom Pose):**
- 3D 공간에서 카메라의 위치와 방향을 완전히 기술하는 6개의 자유도

**구성:**
1) 3DoF Translation (병진): 3D 공간에서의 위치 (x, y, z)
2) 3DoF Rotation (회전): 3D 공간에서의 방향 (roll, pitch, yaw)

**수학적 표현:**
- 4×4 변환 행렬 T = [R|t]로 표현
- R: 3×3 회전 행렬 (SO(3) 그룹)
- t: 3×1 변환 벡터 (R³ 공간)

**실생활 예시:**
- 당신이 방에서 사진을 찍을 때
- 위치: 방의 어느 지점에 서 있는가? (3DoF)
- 방향: 어느 쪽을 보고 있는가? (3DoF)
```

**카메라 파라미터 표현:**
- **내재 파라미터**: 초점 거리 f (주점을 센서 중심으로 가정: cx ≈ W/2, cy ≈ H/2)
- **외재 파라미터**: SE(3) 공간의 카메라-세계 변환 행렬 Twc = [R|t]

```
## 7. 초점 거리와 SE(3) 변환 행렬

**초점 거리 (Focal Length) f:**

**물리적 의미:**
- 렌즈 중심에서 이미지 센서까지의 거리
- 단위: 보통 픽셀 또는 mm

**역할:**
- 시야각(Field of View) 결정
- f가 클수록: 망원 효과, 좁은 시야각
- f가 작을수록: 광각 효과, 넓은 시야각

**수학적 역할:**
- 3D 점을 2D 이미지 평면으로 투영할 때 사용
- 투영 공식: u = f * X/Z, v = f * Y/Z

**SE(3) 공간의 카메라-세계 변환 행렬:**

**SE(3) (Special Euclidean Group):**
- 3D 공간에서의 모든 강체 변환을 나타내는 수학적 군
- 회전 + 병진을 동시에 표현

**변환 행렬 구조:**
T = [R | t]  (4×4 행렬)
    [0 | 1]

여기서:
- R ∈ SO(3): 3×3 회전 행렬 (직교행렬, det(R)=1)
- t ∈ R³: 3×1 병진 벡터

**의미:**
- 세계 좌표계의 점을 카메라 좌표계로 변환
- 또는 그 역변환 (카메라 → 세계)
```

- **회전 표현**: 축-각 표현 φ := αω, φ ∈ R³

```
## 8. 축-각 표현의 기호들

**축-각 표현 (Axis-Angle Representation):**
φ := αω, φ ∈ R³

**기호별 의미:**

1) **ω (오메가):**
   - 의미: 정규화된 회전축 벡터 (단위 벡터)
   - 크기: ||ω|| = 1
   - 방향: 회전축의 방향을 나타냄
   - 예시: ω = [1]ᵀ (z축 중심 회전)

2) **α (알파):**
   - 의미: 회전각 (라디안 단위)
   - 범위: 보통 [0, π] 또는 [-π, π]
   - 양수: 오른손 법칙에 따른 회전 방향

3) **φ (파이):**
   - 의미: 실제로 최적화되는 벡터
   - 계산: φ = α × ω
   - 크기: ||φ|| = α (회전각)
   - 방향: φ/||φ|| = ω (회전축)

**장점:**
- 3개의 스칼라 값으로 3D 회전을 완전히 표현
- 특이점(singularity) 없음
- 최적화에 적합한 형태

**직관적 이해:**
- φ 벡터의 방향 = 회전축
- φ 벡터의 크기 = 회전각
- 예: φ = [0, 0, π/2]ᵀ → z축 중심으로 90도 회전

**물리적 비유:**
나사를 돌리는 것과 같음:
- 나사축 방향 = ω
- 돌리는 각도 = α  
- 전체 회전 = φ = α × ω
```

- **로드리게스 공식**: $$R = I + \frac{\sin(\alpha)}{\alpha}\phi^∧ + \frac{1-\cos(\alpha)}{\alpha^2}(\phi^∧)^2 $$

```
## 9. 로드리게스 공식

**로드리게스 공식 (Rodrigues' Formula):**
$$ R = I + \frac{\sin(\alpha)}{\alpha}\phi^∧ + \frac{1-\cos(\alpha)}{\alpha^2}(\phi^∧)^2 $$

**각 기호의 의미:**

1) **R**: 
   - 의미: 최종 회전 행렬 (3×3)
   - 역할: 3D 벡터를 회전시키는 변환 행렬

2) **I**:
   - 의미: 3×3 항등 행렬 (단위 행렬)
   - 값: [[1], [1], [1]]
   - 역할: 회전이 없는 기본 상태

3) **α**:
   - 의미: 회전각 (라디안)
   - 계산: α = ||φ|| (φ 벡터의 크기)

4) **φ**:
   - 의미: 축-각 벡터 (φ = α × ω)
   - 크기: 회전각, 방향: 회전축

5) **φ∧ (φ hat)**:
   - 의미: φ의 반대칭 행렬 (skew-symmetric matrix)
   - 계산: φ = [x, y, z]일 때
     ```
     φ∧ = [[ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]]
     ```
   - 역할: 외적(cross product)을 행렬 곱셈으로 표현

6) **(φ∧)²**:
   - 의미: φ∧를 자기 자신과 곱한 결과
   - 계산: φ∧ × φ∧

**공식의 직관적 이해:**
- 첫 번째 항 (I): 원래 상태 (회전 없음)
- 두 번째 항 (sin(α)/α φ∧): 첫 번째 회전 효과
- 세 번째 항 ((1-cos(α))/α² (φ∧)²): 두 번째 회전 효과

**물리적 의미:**
- 축-각 표현을 회전 행렬로 변환하는 공식
- 3D 공간에서 임의 축 중심의 회전을 표현
```

**렌더링 과정:**

광선 방향: 

$$ \hat{d}_{i,p} = \hat{R}_i \begin{bmatrix} (u-W/2)/\hat{f} \\ -(v-H/2)/\hat{f} \\ -1 \end{bmatrix} $$

픽셀 색상:

$$\hat{I}\_i(p) = R(p, \pi_i|\Theta) = \int_{h_n}^{h_f} T(h)\sigma(r(h))c(r(h),d)dh $$

여기서 $$T(h) = \exp\left(-\int_{h_n}^h \sigma(r(s))ds\right) $$는 누적 투과율입니다.

```
## 10. 픽셀 색상 공식

**픽셀 색상 공식:**

$$ \hat{I}_i(p) = R(p, \pi_i|\Theta) = \int_{h_n}^{h_f} T(h)\sigma(r(h))c(r(h),d)dh $$

**각 기호와 의미:**

1) **Îᵢ(p)**:
   - 의미: 이미지 i의 픽셀 p에서 렌더링된 색상
   - 형태: RGB 값 (r, g, b)

2) **R(p, πᵢ|Θ)**:
   - 의미: 렌더링 함수
   - 입력: 픽셀 위치 p, 카메라 파라미터 πᵢ, NeRF 파라미터 Θ
   - 출력: 해당 픽셀의 색상

3) **적분 범위 [hₙ → hf]**:
   - 의미: 광선을 따라 샘플링하는 거리 범위
   - hₙ: near bound (가까운 경계)
   - hf: far bound (먼 경계)

4) **T(h)**:
   - 의미: 누적 투과율 (accumulated transmittance)
   - 계산: $$ T(h) = \exp\left(-\int_{h_n}^h \sigma(r(s))ds\right) $$
   - 물리적 의미: 광선이 h 지점까지 막히지 않고 도달할 확률

5) **σ(r(h))**:
   - 의미: 위치 r(h)에서의 밀도(density)
   - 역할: 그 지점에서 광선이 흡수/산란되는 정도
   - 높은 값: 불투명한 물질, 낮은 값: 투명한 영역

6) **c(r(h), d)**:
   - 의미: 위치 r(h)에서 방향 d로 방출되는 색상
   - 입력: 3D 위치와 시선 방향
   - 출력: RGB 색상값

7) **r(h) = o + hd**:
   - 의미: 광선 위의 점
   - o: 카메라 원점 (camera origin)
   - d: 광선 방향 (ray direction)
   - h: 카메라로부터의 거리

**물리적 의미:**

이 공식은 실제 광선이 3D 공간을 통과하면서 색상을 수집하는 과정을 수학적으로 모델링:

1) 카메라에서 픽셀 방향으로 광선을 발사
2) 광선을 따라 여러 지점에서 샘플링
3) 각 지점에서 색상과 밀도를 NeRF 모델로 예측
4) 앞쪽부터 뒤쪽까지 색상을 누적하여 최종 픽셀 색상 계산

**직관적 이해:**
- 안개 속을 지나가는 빛을 생각해보세요
- 빛이 안개를 통과하면서 점점 약해지고 (T(h))
- 각 지점에서 안개가 빛을 흡수하거나 산란시키며 (σ)
- 동시에 각 지점에서 새로운 색상을 방출합니다 (c)
```

## 3. 모델 구조

### 네트워크 아키텍처
- 원본 NeRF보다 **경량화된 구조** (은닉층 차원: 256 → 128)
- **계층적 샘플링 제거**로 계산 효율성 향상

```
## 11. 계층적 샘플링 제거

**원본 NeRF의 계층적 샘플링:**
1) **Coarse Network:**
   - 광선을 따라 균등하게 64개 점 샘플링
   - 각 점에서 색상과 밀도 예측
   - 전체적인 형태 파악

2) **Fine Network:**
   - Coarse 결과를 기반으로 중요한 영역에 추가 128개 점 샘플링
   - 물체 경계나 세부 사항이 많은 곳에 집중
   - 더 정확한 렌더링 수행

**NeRF--에서 제거한 이유:**
1) **계산 효율성 향상:**
   - 두 번의 forward pass → 한 번의 forward pass
   - 네트워크 파라미터 수 감소
   - 훈련 시간 단축

2) **메모리 사용량 감소:**
   - 두 개의 네트워크 → 하나의 네트워크
   - GPU 메모리 절약

3) **카메라 파라미터 최적화에 집중:**
   - 이미 복잡한 joint optimization 문제
   - 계층적 샘플링 추가 시 최적화가 더 어려워짐

**대신 사용한 방법:**
- 단일 네트워크로 128개 점 샘플링
- 더 작은 은닉층 차원 (256 → 128)
- 적은 픽셀 샘플링 (전체 → 1024픽셀)

**결과:**
- 약간의 품질 저하는 있지만 여전히 경쟁력 있는 성능
- 훈련 시간과 메모리 사용량 크게 감소
- 카메라 파라미터 최적화에 더 안정적

```

```
## 12. GPU 요구사항

**논문에서 명시된 GPU 요구사항:**

**하드웨어:**
- 사용된 GPU: NVIDIA GTX 1080Ti (11GB VRAM)
- 훈련 시간: BLEFF 장면(27개 훈련 이미지) 기준 5.5시간

**비교 분석:**
- 원본 NeRF (단순화 버전): 5시간
- NeRF--: 5.5시간 (추가 30분은 온라인 광선 구성 때문)

**현재 기준 GPU 요구사항 추정:**

**최소 요구사항:**
- VRAM: 8GB 이상 (GTX 1070, RTX 2070)
- 계산 능력: GTX 1060 이상

**권장 사양:**
- VRAM: 11GB 이상 (GTX 1080Ti, RTX 2080Ti, RTX 3080)
- 현대적 GPU: RTX 3070/4070 이상

**최적 환경:**
- VRAM: 24GB 이상 (RTX 3090/4090, A6000)
- 더 큰 이미지나 복잡한 장면 처리 가능

**최적화 팁:**
1) 배치 크기 조정: 메모리에 맞게 픽셀 샘플링 수 조정
2) 혼합 정밀도: Float16 사용으로 메모리 절약
3) 그래디언트 체크포인팅: 메모리-시간 트레이드오프
```

- 이미지당 1,024픽셀, 광선당 128포인트 샘플링
- **3개의 독립적인 Adam 옵티마이저**: NeRF, 포즈, 초점거리 각각 최적화

### 훈련 세부사항
- 카메라 파라미터를 항등 행렬에서 초기화
- 초점거리는 이미지 너비로 초기화
- 학습률: 0.001 (NeRF 모델은 10 에포크마다 0.9954배 감소)
- 총 10,000 에포크 훈련
- 완전 미분 가능한 파이프라인으로 역전파를 통한 동시 최적화

## 4. 성능 향상 및 일반화 능력

### 양적 성능
- **NVS 품질**: COLMAP-NeRF와 비교 가능 (ΔSSIM 및 ΔLPIPS = 0.05, ΔPSNR = 1.0)
- **회전 추정 오차**: ~5°
- **초점거리 오차**: ~25픽셀
- **병진 섭동**에서 COLMAP보다 우수한 성능
- **회전 섭동**에서는 COLMAP보다 경쟁력이 떨어짐

```
## 13. NVS 품질, 병진 섭동, 회전 섭동

**1) NVS 품질 (Novel View Synthesis Quality):**

**정의**: 새로운 시점에서 렌더링된 이미지의 품질

**평가 지표:**
- PSNR (Peak Signal-to-Noise Ratio): 높을수록 좋음
- SSIM (Structural Similarity Index): 1에 가까울수록 좋음  
- LPIPS (Learned Perceptual Image Patch Similarity): 낮을수록 좋음

**의미**: 실제 사진과 AI가 생성한 이미지가 얼마나 비슷한지 측정

**2) 병진 섭동 (Translation Perturbation):**

**정의**: 카메라 위치를 원래 위치에서 조금씩 이동시키는 것

**표현 방식:**
- t010: ±10% 병진 섭동
- t020: ±20% 병진 섭동

**물리적 의미:**
- 1m × 1m 영역에서 촬영할 때, ±10%는 약 ±28.3cm 범위
- 실제 촬영 시 손떨림이나 위치 변화를 시뮬레이션

**NeRF--의 성능:**
- ±20%까지 COLMAP보다 우수한 성능
- 순수 병진 움직임에서 COLMAP이 실패하는 퇴화 케이스를 잘 처리

**3) 회전 섭동 (Rotation Perturbation):**

**정의**: 카메라 방향을 원래 방향에서 조금씩 회전시키는 것

**표현 방식:**
- r010: ±10도 회전 섭동
- r020: ±20도 회전 섭동

**물리적 의미:**
- 카메라를 들고 촬영할 때 발생하는 자연스러운 흔들림
- 완벽하게 정면을 보지 않고 약간 위/아래, 좌/우로 기울어진 상태

**NeRF--의 성능:**
- ±20도 이상에서 COLMAP보다 성능 저하
- 특징점 기반 방법(COLMAP)이 외관 변화에 더 강건
```

### 일반화 성능 분석

**카메라 움직임 패턴별 강건성:**

1. **병진 섭동**: ±20%까지 COLMAP보다 우수
2. **회전 섭동**: ±20° 이상에서 COLMAP보다 취약
3. **전체 6DoF**: 두 방법 모두 t020r020에서 실패

**특정 카메라 움직임에 대한 성능:**
- **회전 움직임**: NeRF--가 COLMAP 우월
- **병진 움직임**: NeRF--가 COLMAP 우월
- **객체 추적 움직임**: COLMAP이 NeRF-- 우월

**퇴화 케이스 처리:**
- COLMAP은 순수한 병진/회전 움직임에서 실패 (퇴화된 기본 행렬)
- NeRF--는 광측정 최적화로 인해 이러한 케이스를 더 잘 처리

```
## 14. 광측정 최적화의 장점

**잘 처리하는 케이스:**
1) 순수 병진 움직임 (Pure Translational Motion)
2) 순수 회전 움직임 (Pure Rotational Motion)  
3) 동질적 영역이 많은 장면
4) 특징점이 부족한 장면

**왜 더 잘 처리하는가?**

**1) 퇴화 케이스 문제 해결:**

**COLMAP의 문제:**
- 기본 행렬(Fundamental Matrix) 계산 시 퇴화
- 모든 카메라가 직선상에 있으면 무수히 많은 해 존재
- 특징점 대응만으로는 구분 불가능

**NeRF--의 해결:**
- 픽셀 색상을 직접 비교하여 최적화
- 전체 이미지의 색상 패턴을 동시에 고려
- 기하학적 제약 없이 광측정 일치도로 판단

**2) 특징점 의존성 제거:**

**COLMAP의 한계:**
- 특징점 검출 → 매칭 → 파라미터 추정 순서
- 특징점이 없으면 아예 작동 불가
- 잘못된 매칭 시 전체 결과 오류

**NeRF--의 장점:**
- 모든 픽셀을 동등하게 활용
- 텍스처가 부족해도 색상 일치도로 판단
- 전역적 최적화로 국소적 오류 보상

**3) 전역적 일관성:**

**광측정 최적화의 특성:**
- 모든 이미지를 동시에 고려
- 카메라 파라미터와 3D 구조를 joint optimization
- Bundle Adjustment와 유사한 전역 최적화

**구체적 예시:**
- 흰 벽면을 따라 촬영 (동질적 영역)
- COLMAP: 특징점 부족으로 실패
- NeRF--: 벽면의 미세한 색상 변화까지 활용하여 성공

**물리적 직관:**
- 같은 3D 점은 다른 시점에서도 같은 색으로 보여야 함
- 이 원리를 모든 픽셀에 적용하여 최적화
- 특징점이 없어도 색상 일치도만으로 충분
```

- 객체 추적 케이스에서는 감독 신호 부족으로 NeRF--가 문제 발생

## 5. 한계점

1. **객체 추적 움직임**에서 의미 있는 카메라 파라미터 추정 불가
2. **전방향 장면으로 제한**됨
3. **360° 장면 처리 불가**
4. **±20° 이상의 회전 섭동**에서 실패 가능성

## 6. 미래 연구에 대한 영향 및 고려사항

### 긍정적 영향
- **COLMAP 전처리 의존성 제거**로 엔드투엔드 최적화 가능

```
## 15. COLMAP 의존성 제거

**COLMAP이 지금까지 많이 사용된 이유:**

1) **기술적 우수성:**
   - 가장 정확하고 안정적인 SfM 도구
   - 다양한 장면에서 검증된 성능
   - 강력한 Bundle Adjustment 구현

2) **생태계 구축:**
   - 오픈소스로 널리 보급
   - 많은 연구에서 표준으로 사용
   - 풍부한 문서화와 커뮤니티 지원

3) **NeRF의 구조적 필요성:**
   - 원본 NeRF는 카메라 파라미터가 필수 입력
   - 3D 좌표와 시선 방향 없이는 작동 불가
   - 당시 기술로는 joint optimization이 어려웠음

**NeRF--에서 의존성을 제거한 방법:**

**1) Joint Optimization Framework:**

기존 방식: COLMAP → 카메라 파라미터 → NeRF 훈련
새로운 방식: 이미지 → NeRF + 카메라 파라미터 동시 최적화

**2) 미분 가능한 렌더링:**
- 카메라 파라미터를 학습 가능한 매개변수로 설정
- 광선 구성부터 최종 렌더링까지 모든 과정을 미분 가능하게 구현
- 역전파를 통해 카메라 파라미터까지 그래디언트 전달

**3) 적절한 초기화와 파라미터화:**
- 카메라를 원점에서 -z 방향으로 초기화
- 축-각 표현으로 회전 파라미터화
- 초점거리를 이미지 너비로 초기화

**4) 안정적인 최적화 전략:**
- 3개의 독립적인 옵티마이저 사용
- 서로 다른 학습률과 감소 스케줄
- Forward-facing 가정으로 탐색 공간 제한

**핵심 기술적 혁신:**

1) **온라인 광선 구성:**
   - 매 iteration마다 현재 카메라 파라미터로 광선 계산
   - 실시간으로 카메라 파라미터 업데이트 반영

2) **광측정 감독 신호:**
   - 픽셀 색상 차이를 직접 손실 함수로 사용
   - 중간 단계 없이 end-to-end 학습

3) **전역적 일관성:**
   - 모든 카메라 파라미터를 동시에 최적화
   - Bundle Adjustment와 유사한 효과

**실용적 영향:**
- 모바일 폰으로 찍은 사진 직접 사용 가능
- 전처리 단계 제거로 사용자 편의성 증대
- COLMAP 실패 케이스에서도 작동 가능
- 완전 자동화된 파이프라인 구현

**한계와 트레이드오프:**
- Forward-facing 장면으로 제한
- 초기화에 민감할 수 있음
- 특정 카메라 움직임 패턴에서는 COLMAP보다 성능 저하

이러한 혁신들을 통해 NeRF--는 기존의 복잡한 전처리 과정을 제거하고, 사용자가 간단히 촬영한 이미지만으로도 고품질의 새로운 시점 합성을 가능하게 만든 획기적인 연구입니다.
```

- **퇴화된 카메라 움직임의 더 나은 처리**
- **보정되지 않은 이미지 기반 NVS**의 길 개척
- 모바일 기기에서 촬영한 일반적인 이미지에 직접 적용 가능

### 앞으로의 연구 방향
1. **360° 장면으로의 확장**: 상대적 포즈 추정 및 시간적 관계 활용
2. **동적 장면 및 객체 처리**: 현재는 정적 장면에만 제한
3. **개선된 초기화 전략**: 더 나은 수렴을 위한 방법 개발
4. **객체 추적 움직임 개선**: 충분한 감독 신호 제공 방법 연구
5. **다른 NVS 방법과의 통합**: NeRF 외의 다른 방법들과 결합

### 연구 시 고려사항
- **초기화 방법의 중요성**: 카메라 파라미터 초기화가 수렴에 미치는 영향
- **비디오 시퀀스의 시간적 관계 활용**: 연속된 프레임 간의 관계 이용
- **이미지 쌍 간 상대적 포즈 추정**: 더 강건한 최적화를 위한 방법
- **큰 회전 섭동 처리**: 현재 한계를 극복하기 위한 새로운 접근법
- **계산 효율성 개선**: 실시간 또는 빠른 처리를 위한 최적화

이 논문은 **NeRF의 실용성을 크게 향상**시키는 중요한 기여를 했으며, 향후 uncalibrated image-based novel view synthesis 분야의 발전에 **핵심적인 기반**을 제공할 것으로 예상됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f5f83fb0-ad7f-4656-89f9-c4bd3438dc9e/2102.07064v4.pdf

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f5f83fb0-ad7f-4656-89f9-c4bd3438dc9e/2102.07064v4.pdf
