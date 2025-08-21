# CANet: A Context-Aware Network for Shadow Removal | Shadow removal

**핵심 주장 및 기여**  
CANet은 그림자 영역과 비그림자 영역 간의 잠재적 컨텍스트 매칭 정보를 학습하고 이를 특징 공간에서 전이하여 고품질의 그림자 제거를 실현하는 **두 단계(context-aware) 구조**를 제안한다. 주요 기여는 다음과 같다.  
1. 그림자·비그림자 패치 간의 컨텍스트 매칭을 학습하는 **CPM(Contextual Patch Matching) 모듈**  
2. 매칭된 비그림자 패치 특징을 가우시안 샘플링 기반으로 부드럽게 전이하는 **CFT(Contextual Feature Transfer) 메커니즘**  
3. 첫 단계에서 전이된 L·A/B 채널 복원을 정제하여 최종 결과를 생성하는 **엔코더–디코더(DenseUNet) 구조의 두 번째 단계**  

***

## 1. 해결 과제  
- 그림자 제거 시 기존 물리 기반·단일 수용 영역 학습 기법들은 그림자 경계 아티팩트나 색 왜곡이 발생  
- 복잡한 장면에서 그림자와 유사한 텍스처를 가진 영역 간 전역적 매칭을 고려하지 못함  

## 2. 제안 방법  
### 2.1 CPM 모듈  
- 입력: 원본 그림자 이미지와 “빛-제거(light-unaware)” 이미지  
- 출력:  
  1) 두 패치의 매칭 유사도 $$s \in $$ (회귀)[1]
  2) 순서 정보(type: –1/0/1) 분류  
- 학습 손실:  

```math
    L_{CPM} = \|s_{\text{out}} - s_{\text{gt}}\|_2^2 \;+\; \text{CrossEntropy}(t_{\text{out}}, t_{\text{gt}})
```

### 2.2 CFT 메커니즘  
- 매칭된 상위 $$k$$개 비그림자 패치의 특징을, 가우시안 가중치 $$\phi(\Delta x,\Delta y)$$로 샘플링하여 전이  
- 채널별 가중 합산:  

```math
    F = \frac{\sum_{i=1}^k w_i F'_i}{\sum_{i=1}^k w_i},
    \quad w_i = \text{유사도}_i
```

- L·A/B 채널을 분리 처리하여 L 채널의 민감도 제어  

### 2.3 두 단계 네트워크  
- **Stage I**: DenseNet 기반 특징 추출 → CPM → CFT → L·A/B 채널 복원  
- **Stage II**: 복원된 채널 및 원본 그림자 영상을 입력으로 DenseUNet 적용 → 최종 정제  
- 총 손실:  

$$
    L_{\text{CANet}} = \lambda_1 L_{\text{rem}} + \lambda_2 L_{\text{per}} + \lambda_3 L_{\text{grad}}
  $$  

```math
    L_{\text{rem}} = \|I_{\text{gt}} - I_{\text{out}}\|_2^2,\quad
    L_{\text{per}} = \|\phi_{\text{VGG}}(I_{\text{gt}}) - \phi_{\text{VGG}}(I_{\text{out}})\|_1,\quad
    L_{\text{grad}} = \|\nabla I_{\text{gt}} - \nabla I_{\text{out}}\|_1
```

## 3. 성능 향상 및 한계  
- **정량 평가**: ISTD·SRD 데이터셋에서 그림자 영역 RMSE 8.86→7.82, 전체 RMSE 6.15→5.98로 기존 기법 대비 우수  
- **정성 평가**: 경계 아티팩트 감소, 컬러 왜곡 방지  
- **한계**:  
  - 비그림자 영역에 대응하는 컨텍스트 매칭이 부족한 경우 복원 실패  
  - 촬영 환경·노출 차이로 인한 데이터 불일치 시 컬러 일관성 문제  
  - 프레임별 처리는 가능하나 동영상 연속성 보장 미흡  

## 4. 일반화 성능 향상 관점  
- CPM을 통한 전역 매칭 학습으로 복잡한 장면에서도 다양한 텍스처 대응 가능  
- 빛-제거 입력으로 그림자 민감도 학습, 채널 분리 처리로 과·과소 보정 최소화  
- 학습 데이터 일관성을 확보하고, 추가 도메인(다양한 조명·재질) 샘플링을 통해 일반화 제고 여지  

***

## 5. 향후 연구 영향 및 고려 사항  
향후 연구에서는 CPM/CFT 구조를 다른 도메인(하이라이트 제거, 반사 제거 등)으로 확장하거나, 시간적 연속성을 반영한 비디오 레벨 최적화 연구가 기대된다. 또한 데이터 불일치에 강인한 도메인 적응 및 노출 보정 기법을 통합하면 **더 나은 일반화 성능**을 확보할 수 있을 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0c69258d-2b46-4278-a7df-6af3cb26d2df/2108.09894v1.pdf)
