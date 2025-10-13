# Face Alignment using Cascade Gaussian Process Regression Trees | 2015 · 152회 인용, Face Alignment

**주요 주장 및 기여**  
이 논문은 **Cascade Gaussian Process Regression Trees(cGPRT)**라는 새로운 얼굴 정렬 기법을 제안한다. cGPRT는 기존의 Cascade Regression Trees(CRT)에 비해  
- 예측 시간은 유지하면서  
- Gaussian Process의 우수한 일반화 성능을 활용해  
더 나은 정렬 정확도를 달성한다.[1]

또한, **shape-indexed Difference of Gaussian(DoG) 특징**을 도입하여 지오메트릭 변동과 형태 추정 오차에 강인성을 확보한다.[1]

***

## 1. 해결하고자 하는 문제
기존 CRT 기반의 얼굴 정렬은  
- **Gradient Boosting** 방식 학습 시 과적합(overfitting)이 발생하기 쉽고,  
- 학습과 예측 간의 피팅 속도 차이로 인해 테스트 시 형태 추정이 불안정해진다.  
또한, **shape-indexed 특징**은 추정 형태에 의존적이어서 작은 추정 오차가 연쇄적으로 큰 오류를 유발한다.[1]

***

## 2. 제안 방법

### 2.1 기존 CRT 개요  
CRT는 입력 이미지 I에서 shape 벡터 $$s=(x_1,y_1,\dots,x_p,y_p)^\top$$를 초기화하고,  
각 단계별 트리 $$f_t(x; \theta_t)$$로부터 형태 보정값을 누적하여 업데이트한다:  

$$
\hat{s}_t = \hat{s}_{t-1} + f_t(x_t;\theta_t)
$$  

학습 시에는 잔차 $$r = s_{\text{gt}} - \hat{s}_{t-1}$$를 최소화하도록 그리디하게 트리를 구성한다.[1]

### 2.2 Gaussian Process Regression Trees(GPRT)  
GPR의 예측 함수 $$f(x)\sim\mathcal{GP}(0,k(x,x'))$$와 노이즈 모델 $$y = f(x)+\epsilon$$를 결합해,  
트리 집합으로 정의된 커널  

$$
k(x,x') = \sigma_k^2\frac{1}{M}\sum_{m=1}^M\mathbb{I}[\tau_m(x)=\tau_m(x')]
$$  

를 사용한다. 이를 통해 입력 쌍이 동일한 리프에 속한 횟수를 유사도로 측정한다.[1]

예측 시, N개의 학습 샘플에 대한 커널 가중합  

```math
\bar{f}^* = \sum_{i = 1}^{N} \alpha_i k(x_i, x ^ *)
```

를 효율적으로 $$O(M\log B)$$ 시간에 계산한다.[1]

### 2.3 Cascade GPRT(cGPRT)  
cGPRT는 T개의 GPRT를 **Product of Experts** 형태로 결합하여 각 단계별 잔차를 모델링한다:  

```math
p(f^*|x^*,M) \propto \prod_{t=1}^T p(f^*|x_t^*,M_t)
``` 

각 GPRT의 예측 평균과 분산 

```math
\{\bar{f}_t^*,\ , (\sigma_t^*)^2\}
```

를 조합해 최종 예측을 수행한다.[1]

예측 평균 업데이트:  

$$
\hat{s}_t = \hat{s}_{t-1} + \sum_{m=1}^M \bar{\alpha}_{t,m,\tau_{t,m}(x_t)}
$$

### 2.4 Shape-indexed DoG 특징  
1. Gaussian 필터로 다중 스케일 이미지 스무딩  
2. **Local retinal sampling pattern**에 기반한 픽셀 추출  
3. 추출된 가우시안 반응의 차(difference) 연산  

스케일은 샘플링 점과 중심점 간 거리 비례로 설정해, 멀리 있는 점은 넓은 영역을, 가까이 있는 점은 높은 분별력을 갖도록 설계했다.[1]

***

## 3. 성능 평가 및 한계

### 3.1 성능 향상  
- **HELEN (194 landmarks)**: 평균 오차 4.63 (기존 최저 4.90)  
- **300-W (68 landmarks)**: 평균 오차 5.71 (기존 최저 6.32)  
cGPRT는 기존 CRT 기반 방법 대비 가장 큰 개선을 보였으며, 특히 **300-W**처럼 난이도 높은 데이터셋에서 일반화 우수성을 입증했다.[1]

### 3.2 계산 비용  
- cGPRT (M=10, T=500) 예측 속도 93fps (CRT 기반 대비 다소 낮음)  
- cGPRTfast (M=10, T=100)로 속도 최적화 가능(871fps)  
학습 단계에서 커널 역행렬 계산 및 하이퍼파라미터 최적화 비용이 있으며, 대규모 데이터에 확장 시 연산 부담이 있다.[1]

### 3.3 한계  
- **학습 복잡도**: GPRT 하이퍼파라미터 추정(O((BM)³))  
- **데이터 의존성**: 2D landmark만 고려, 다양한 조명·표정·포즈에 한계  
- **실시간 적용**: 고속 예측을 위한 파라미터 조정 필요

***

## 4. 일반화 성능 및 향후 연구 고려사항

- **Gaussian Process의 불확실성 모델링** 덕분에 과적합 감소 및 일반화 성능이 크게 향상됐다.  
- **스케일-적응 DoG 특징**은 추정 오차에 강인해, 다양한 얼굴 변형 상황에서 안정적이다.

향후 연구에서는  
- 3D landmark 정렬로 확장  
- 대규모 및 다양한 인종·연령 데이터로 일반화 검증  
- 효율적 커널 근사 기법 도입을 통한 학습 및 예측 속도 개선  
- End-to-end 딥러닝 계열 모델과의 융합을 통한 특성 자동화 추출  

등을 고려할 필요가 있다.  

 Lee et al., “Face Alignment using Cascade Gaussian Process Regression Trees,” CVPR 2015.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7fa54a6b-0af0-45b7-aeef-7c8848403b80/Lee_Face_Alignment_Using_2015_CVPR_paper.pdf)
