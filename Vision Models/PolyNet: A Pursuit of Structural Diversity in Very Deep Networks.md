# PolyNet: A Pursuit of Structural Diversity in Very Deep Networks | Image classification

## 핵심 주장 및 주요 기여  
PolyNet은 **깊이(depth)**·**폭(width)** 외에 **구조적 다양성(structural diversity)** 을 새로운 차원으로 제안하며, 이를 통해 매우 깊은 네트워크에서도 성능이 계속 향상될 수 있음을 입증한다. 주요 기여는 다음과 같다.[1]
- **PolyInception 모듈**: Inception 블록들을 다항식(polynomial) 형태로 결합하여 표현력을 높이는 새로운 빌딩 블록(I + F + F² 등)을 제안.[1]
- **구조적 다양성 연구**: 서로 다른 PolyInception 구성을 체계적으로 비교·분석하여, 깊이만 키우는 방식보다 구조적 다양성을 높이는 방식이 더 효율적임을 실험적으로 증명.[1]
- **Very Deep PolyNet**: PolyInception을 활용해 ILSVRC 2012 기준 top-5 오류율을 단일 크롭 4.25%, 다중 크롭 3.45%로 종전 최고치(4.9%, 3.7%)를 경신.[1]

## 문제 인식  
기존 연구는 네트워크 성능 향상을 위해 깊이를 늘리거나 폭을 넓히는 데 집중해 왔으나,  
- 깊이 증가 시 100단계 이상부터 성능 향상 폭이 급격히 감소하며 학습 난이도도 증가  
- 폭 확장 시 계산량과 메모리 비용이 $$O(k^2)$$로 급증하여 현실적 한계 존재  
따라서 **깊이·폭 이외의 제3의 설계 축**이 필요하다는 문제를 제기한다.[1]

## 제안 기법  
### PolyInception 모듈  
Residual 유닛 $$(I + F)x = x + F(x)$$을 다항식 조합으로 확장하여, 예를 들어  

$$
(I + F + F^2) \cdot x = x + F(x) + F\bigl(F(x)\bigr)
$$

와 같은 형태로 만들고, 연산 효율을 위해 $$F(x)$$를 공유하는 **축약형(cascaded) 설계**를 이용한다.[1]
- **poly-n**: 파라미터 공유, $$I + F + \cdots + F^n$$  
- **mpoly-n**: 파라미터 비공유, $$I + F + G F + \cdots$$  
- **k-way**: 서로 다른 $$k$$ 개 블록 $$F, G, …$$을 병렬 추가

### 네트워크 구조  
- **스탬(stem)** + **Stage A (35×35)**, **B (17×17)**, **C (8×8)** 3개 스테이지  
- Stage A: 10×2-way, Stage B: 20개 PolyInception(2-way와 poly-3 혼합), Stage C: 10개(혼합)  
- 최종 softmax 연결, 입력 해상도는 331×331[1]

### 학습 전략  
- **RMSProp** + 배치 정규화, 초기 학습률 0.45→0.045→0.0045  
- **초기화-삽입(initialization by insertion)**: 기존 모델 파라미터 보존 후 새 블록만 랜덤 초기화  
- **잔차 스케일링(residual scaling)**: 잔차 경로에 계수 $$\beta=0.3$$ 적용  
- **확률적 경로 제거(stochastic paths)**: 과적합 방지 위해 학습 후반부에 최대 드롭 확률 0.25로 경로 일부 랜덤 제거[1]

## 성능 향상  
- ILSVRC single-crop top-5 오류율 4.25%→기존 Inception-ResNet-v2(4.9%) 대비 0.65%p 개선  
- multi-crop top-5 오류율 3.45%→3.7% 대비 0.25%p 개선  
- 깊이만 늘린 ResNet-500 대비도 우수하며, 동일 연산 예산 대비 구조적 다양성이 더 큰 성능 향상을 보임.[1]

## 한계  
- 연산 비용 및 메모리 사용량이 여전히 매우 큼(32 GPU, 배치 512 필요)  
- 설계 공간이 큰 만큼 하이퍼파라미터(폴리 차수, 구성 혼합 비율) 탐색 비용이 높음  
- 구조적 복잡성 증대로 인해 추론 속도가 실시간 응용에는 부담될 수 있음

## 일반화 성능 향상 요소  
PolyInception의 **다양한 연산 경로**가 내재적 앙상블(enforced ensemble) 효과를 내며, 학습 중 **확률적 경로 제거**와 **잔차 스케일링**이 결합되어 **공동 적응(co-adaptation)을 억제**함으로써 일반화 성능을 크게 향상시킨다.[1]

## 향후 연구에의 영향 및 고려 사항  
앞으로는 깊이·폭 외에 **토폴로지(architecture topology)** 설계가 주요 탐색 축으로 부상할 것으로 보인다.  
- PolyInception과 유사한 다항식 조합을 다른 블록(Inception-v4, Transformer 등)에 적용  
- 경량화 위해 **지식 증류**, **네트워크 프루닝**과 결합  
- 구조 다양성·하이퍼파라미터 자동 탐색(AutoML) 기법 통합  
이들 방향이 향후 네트워크 설계의 핵심 고려 요소가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/806b71e9-536a-4fb5-abf4-a3d5d689483e/1611.05725v2.pdf
