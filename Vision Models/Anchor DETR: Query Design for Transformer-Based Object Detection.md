# Anchor DETR: Query Design for Transformer-Based Object Detection | Object detection

## 1. 핵심 주장 및 주요 기여  
**Anchor DETR**은 기존 DETR의 학습 효율성과 해석 가능성 한계를 해결하기 위해, 객체 쿼리를 “학습된 임베딩”이 아닌 “명시적 위치 정보(앵커 포인트)”로 설계하고, 각 앵커 포인트에 다중 예측 패턴을 부여함으로써  
- 객체 쿼리의 **위치 집중성(Positional Focus)**을 강화  
- 동일 영역에 다수 객체가 존재할 때의 **다중 예측(Multiple Predictions)** 문제를 해결  
- Transformer 디코더의 표준 어텐션을 **Row-Column Decoupled Attention (RCDA)**으로 대체하여 메모리 효율성을 향상  

이를 통해 DETR 대비 **10× 적은 학습 에포크**(50 vs. 500)로 더 높은 AP를 달성하고, 실시간 검출 속도(19 FPS) 및 메모리 사용량 감소 효과를 동시에 얻는다.

## 2. 해결 과제 및 제안 방법

### 2.1 해결 과제  
- **쿼리의 위치 불명확성(Positional Ambiguity):** DETR의 학습된 쿼리는 물리적 의미가 없어, 특정 영역을 집중하기 어렵고 최적화가 느리다.  
- **한 영역 다수 객체 처리(One Region, Multiple Objects):** 하나의 공간에 여러 객체가 있을 때 단일 쿼리로는 예측 불가.  
- **높은 메모리 비용:** 표준 Transformer 어텐션의 $$O(N_qHW)$$ 복잡도가 대규모 고해상도 피처에 부담.

### 2.2 제안 방법

#### 2.2.1 앵커 포인트 기반 쿼리 설계  
- **앵커 포인트 $$\mathrm{Pos}_q \in ^{N_A\times2} $$** 를 학습하거나 균일 그리드로 초기화[1]
- 위치 임베딩 함수 $$g$$ (sine-cosine 또는 소형 MLP)로 인코딩:  

$$
Q_p = g(\mathrm{Pos}_q),\quad K_p = g(\mathrm{Pos}_k)
$$

- 디코더 초기 쿼리 $$Q = Q_f^{\mathrm{init}} + Q_p$$

#### 2.2.2 다중 패턴 예측  
- 각 앵커 포인트당 $$N_p$$개의 패턴 임베딩 $$\{P_i\}_{i=1}^{N_p}$$ 공유  
- 총 쿼리 수 $$N_q = N_A \times N_p$$  
- 디코더 입력 쿼리:  

$$
Q = Q_f^{\mathrm{init}} + g(\mathrm{Pos}\_q)\otimes\mathbf{1}_{N_p} + P
$$

이를 통해 한 포인트에서 여러 객체 예측 가능

#### 2.2.3 Row-Column Decoupled Attention (RCDA)  
- 2D 키 피처 $$K_f\in\mathbb{R}^{H\times W\times C}$$를 행/열 1D로 분리:  
  
$$K_{f,x}\in\mathbb{R}^{W\times C}$$, $$K_{f,y}\in\mathbb{R}^{H\times C}$$ (글로벌 평균 풀링)  

- 순차적 어텐션:  
  1. 행 어텐션: $$A_x=\mathsf{softmax}(\frac{Q_xK_x^\top}{\sqrt{d_k}})$$  
  2. 가중합: $$Z = A_x V$$  
  3. 열 어텐션: $$A_y=\mathsf{softmax}(\frac{Q_yK_y^\top}{\sqrt{d_k}})$$  
  4. 최종 출력: $$\mathrm{Out} = A_y Z$$  
- 메모리 복잡도 비율:  

$$
\frac{N_q H W M}{N_q H C} = \frac{W\times M}{C}
$$
  
일반적 C5(32×32) 기준 **유사 메모리**, 고해상도(예: C4)에서 **2–4× 절감**

## 3. 모델 구조 및 성능  
- **백본:** ResNet-50-DC5  
- **디코더 레이어:** 6, 쿼리 300포인트×3패턴 = 900개  
- **학습:** 50 에포크, AdamW, LR 1e–4, 디케이 0.1@40  
- **주요 성능:**  
  - AP 44.2, FPS 19 (DC5 single-level)  
  - DETR(500E) 대비 AP +0.9, 학습 시간 1/10  
  - Deformable DETR(50E multi-level) 대비 AP +0.4, 속도 +4 FPS  

### 한계  
- **앵커 포인트 수 및 패턴 조정 필요:** COCO 최적화된 값(300×3)이며, 소규모/특수 데이터셋에서 재조정 필요  
- **고해상도 효율:** RCDA는 메모리 절감 기대보다 작은 feature map에 한정해 장점 발휘  
- **쿼리 간 상호작용 감소:** 앵커 집중성 강화가 복잡한 상호 의존성 모델링을 제한할 가능성

## 4. 일반화 성능 및 향후 연구 고려 사항  
- **다양한 도메인 적용:** 학습된 앵커 포인트 분포가 COCO 객체 밀도에 최적화되어 있어, 의료 영상·위성 영상처럼 분포가 다른 도메인에서 일반화 성능 평가 필수  
- **데이터 규모 의존성:** 앵커 분포가 데이터셋 객체 분포와 연동되므로, 소규모 라벨링 환경에서 효과 저하 우려  
- **어텐션 변형 확장:** RCDA를 다른 Transformer 기반 비전 태스크(세분화·분류)로 확장 및 비교 연구  
- **동적 패턴 수 조절:** 입력 이미지별로 앵커당 예측 패턴을 동적으로 조정하여 과잉/과소 예측 방지  

## 5. 미래 영향 및 제언  
Anchor DETR는 **Transformer 기반 검출기의 해석가능성**과 **효율성** 양축에서 혁신을 제시하였다.  
- **엔드투엔드 실시간 검출** 연구에 강력한 단일-레벨 기준을 제공  
- **어텐션 최적화** 연구에 RCDA 구조의 확장 가능성 시사  
- **쿼리 설계** 분야에 “명시적 위치 정보” 도입 패러다임을 확립  

향후 연구 시에는 앵커 분포 자동 학습, 멀티-스케일 통합 쿼리, 그리고 도메인 적응 기반 일반화 성능 검증을 중점 고려해야 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/da9e657f-caab-4c74-b04b-36b17107078d/2109.07107v2.pdf
