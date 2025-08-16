# YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

## 1. 핵심 주장과 주요 기여  
YOLOv7은 실시간 객체 검출 분야에서 속도와 정확도 양 측면에서 새로운 **최첨단** 성능을 달성했다.  
- GPU V100에서 30 FPS 이상 실시간 검출 모델 중 최고 56.8% AP 기록  
- Transformer·Convolution 기반 검출기 대비 최대 500% 이상 빠른 추론 속도와 동등하거나 높은 정확도  
- 단일 MS COCO 데이터셋 학습만으로 달성한 결과  

**주요 기여**  
1. **Trainable Bag-of-Freebies**: 추론 비용 증가 없이 훈련 과정에서 정확도 높이는 다양한 모듈·기법 제안  
2. **Planned Re-parameterization**: ResNet/DenseNet 등 다양한 구조에 최적화된 리파라미터화 기법 설계  
3. **Coarse-to-Fine Lead-Guided Label Assigner**: 보조 헤드(auxiliary)와 주 헤드(lead)의 소프트 레이블 할당 방식을 계층화하여 일반화 성능 개선  
4. **Concatenation-Based Compound Scaling**: E-ELAN 아키텍처에 맞춰 깊이·넓이 동시 확장하는 복합 스케일링 전략 설계  

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- **추론 비용 제약** 하에 실시간 검출기의 정확도 향상 한계  
- 모델 리파라미터화 시 다른 구조(Residual, Concatenation)와의 **호환성 문제**  
- 다중 출력 헤드 존재 시 **소프트 레이블 할당** 불일치로 인한 학습 비효율  
- 기존 스케일링 방법이 **Concatenation 기반 모델**에 최적화되지 않음  

### 2.2 제안 방법

1) **Planned Re-parameterization**  
   - 일반적인 RepConv(3×3,1×1,Identity 합산)에서 Identity 분리  
   - Residual 연결 시: RepConv → RepConvN (identity 제거)  
   - Dense 연결 시: 동일 원칙 적용하여 그래디언트 다양성 확보  

2) **Coarse-to-Fine Lead-Guided Label Assigner**  
   - 주 헤드 예측과 GT를 통합해 소프트 레이블 생성  
   - Auxiliary 헤드는 **coarse**(높은 recall) 레이블, Lead 헤드는 **fine**(높은 precision) 레이블 학습     

```math
       \mathcal{L} = \lambda_{\text{coarse}} \mathcal{L}_{\text{aux}}^{\text{coarse}} + \lambda_{\text{fine}} \mathcal{L}_{\text{lead}}^{\text{fine}}
``` 
   
   - 거리 기반 상한 제약으로 coarse 레이블 중요도 동적으로 제어  

3) **Extended ELAN (E-ELAN) + Compound Scaling**  
   - 그룹 컨볼루션 통한 채널·카디널리티 확장, Shuffle-Merge 전략으로 gradient path 유지  
   - Depth×Width 동시 스케일 공식:  

$$
       d' = \alpha d,\quad w' = \beta w,\quad \alpha>1,\ \beta>1
     $$  
   
   - transition 레이어는 $$w'$$만 스케일링해 구조 특성 보존  

4) **기타 Bag-of-Freebies**  
   - Conv-BN 통합 배치정규화, YOLOR의 implicit knowledge 융합, EMA 모델 활용  

### 2.3 모델 구조  
- **Backbone**: E-ELAN 블록 스택  
- **Neck**: CSP 기반 FPN 구조  
- **Heads**: Auxiliary + Lead 이중 헤드, Coarse-to-Fine 레이블 할당  
- **Scaling**: YOLOv7, YOLOv7-X, YOLOv7-E6/D6/E6E 등 다양한 크기 제공  

### 2.4 성능 향상  
- YOLOv4 대비 파라미터 75%↓, 연산량 36%↓, AP +1.5%↑  
- YOLOR-CSP 대비 파라미터 43%↓, 연산량 15%↓, AP +0.4%↑  
- GPU V100 30–160FPS 영역에서 **최고 AP** 및 **최고 속도** 동시 달성  

### 2.5 한계  
- 복합 스케일링 최적 파라미터($$\alpha,\beta$$) 탐색 비용  
- Coarse 레이블 가중치 설정 민감도  
- 대규모 외부 데이터나 사전학습 활용 미검증  

***

## 3. 일반화 성능 향상 관점

- **Lead-Guided Label Assigner**를 통한 다중 헤드 동시 학습으로 데이터 분포 학습 효율 개선  
- **Planned Re-parameterization**로 다양한 구조에 일관된 gradient 다양성 제공  
- **Partial Auxiliary Head** 연결로 멀티 스케일 피처 손실 최소화  
- **EMA 모델**로 안정적인 파라미터 영역 탐색 및 일반화 강화  

이들의 시너지로 과적합 억제, 작은 데이터셋에서도 **우수한 일반화** 가능성 제시  

***

## 4. 향후 연구 영향 및 고려 사항

- **Unified Label Assigner**: 다른 검출기·세분화(task)에도 적용 가능성  
- **스케일링 법칙 일반화**: NAS 없이 구조 특성 반영한 수학적 스케일링 연구 확대  
- **리파라미터화 전략**: Transformer 기반 모듈·경량화 모델에 재파라미터화 적용 탐색  
- **하이퍼파라미터 자동화**: $$\alpha,\beta$$, Label 가중치 최적화 자동화  
- **일반화 이론적 분석**: Bag-of-Freebies가 학습 경로·손실 지형에 미치는 영향 규명  

이 논문은 실시간 객체 검출 성능을 새로운 경지로 끌어올리며, 향후 일반화 중심의 **훈련 기법**과 **구조 최적화** 연구에 중요한 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9a24fbc9-2f7b-41e7-9668-f6386d734edb/2207.02696v1.pdf
