# DPT : Vision Transformers for Dense Prediction | Depth estimation, Semantic segmentation

## 1. 논문의 핵심 주장 및 주요 기여  
- **핵심 주장**: 전통적인 컨볼루션 기반 백본을 대체해, Vision Transformer(ViT)를 활용한 **Dense Prediction Transformer(DPT)** 구조가 전역 수용 영역(global receptive field)과 고해상도 표현을 유지해, 밀집 예측(dense prediction) 성능을 유의미하게 향상시킨다[1].  
- **주요 기여**:  
  - ViT 토큰을 다양한 해상도의 이미지 형태(feature map)로 재구성하는 **Reassemble** 연산 제안[1].  
  - Reassemble된 특징을 RefineNet 기반 퓨전 블록으로 결합해 점진적 예측을 수행하는 **컨볼루션 디코더** 설계[1].  
  - 대규모 메타데이터셋(MIX6, 약 140만 이미지)을 활용해 **모노큘러 깊이 추정** 및 **시맨틱 분할**에서 현존 최고 성능 달성[1].

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- **밀집 예측 과제의 한계**:  
  - 컨볼루션 네트워크는 점진적 다운샘플링 과정에서 공간적 해상도와 세부 정보가 소실됨[1].  
  - 전역 문맥(global context) 확보를 위해 깊은 네트워크가 필요하나 계산·메모리 비용이 급증함[1].

### 2.2 제안 방법  
- **Reassemble 연산**: Transformer의 토큰 집합 $$\{t_l^i\}_{i=0}^{N_p}$$을

$$
    \text{Reassemble}_s(t) = (\text{Resample}\_s \circ \text{Concatenate} \circ \text{Read})(t)
$$
   
  으로 재구성[1].  
  - **Read**: 읽기(readout) 토큰 처리(무시, 덧셈, 투영)[1].  
  - **Concatenate**: $$N_p$$ 토큰 → $$\frac{H}{p}\times\frac{W}{p}$$ 특징 맵 변환[1].
  
  - **Resample $$_s$$ **: $$1\times1$$ · $$3\times3$$  (Transpose) 컨볼루션으로 해상도 조정[1].
   
- **컨볼루션 디코더**:  
  - 4단계 Reassemble 후, **RefineNet** 기반 퓨전 블록이 특징을 병합·업샘플링해 최종 예측 생성[1].

### 2.3 모델 구조  
| 구성 요소           | 세부 사항                                   |
|------------------|-----------------------------------------|
| 백본 (Encoder)     | ViT-Base(12층), ViT-Large(24층), ViT-Hybrid(ResNet50+12층) 활용[1] |
| Reassemble 단계    | 레이어별 토큰 재구성(예: Base: layers 3,6,9,12)[1]  |
| 디코더 (Decoder)   | RefineNet 스타일 퓨전 블록 x4, 최종 해상도 $$H/2\times W/2$$[1] |
| 출력 헤드         | (1) 모노큘러 깊이: 3× 컨볼루션+선형 투영 (2) 분할: 1×1 컨볼루션+업샘플링[1] |

### 2.4 성능 향상  
- **모노큘러 깊이 추정**  
  - Zero-shot 크로스 데이터셋 전이: DPT-Large가 MiDaS 대비 평균 28% 오류 감소[1].  
  - NYUv2, KITTI 미세 조정: 기존 기법 대비 RMSE·δ 지표 전반적 개선[1].
- **시맨틱 분할**  
  - ADE20K: DPT-Hybrid가 기존 최강 ResNeSt-200 대비 mIoU 약 2.7%p 향상[1].  
  - Pascal Context: 동일 구조로 mIoU 약 1.5%p 상승[1].

### 2.5 한계  
- **연산 비용**: ViT-Large는 파라미터 수(343M)가 크고 메모리 수요가 높음[1].  
- **데이터 의존성**: 대형 트랜스포머 특성상, 소규모 데이터셋만으로는 과적합 위험이 존재[1].  
- **실시간 적용**: 실시간 애플리케이션에는 여전히 경량화 필요[1].

## 3. 일반화 성능 향상 가능성  
- **전역 주의력**: 모든 토큰 간의 전역 self-attention으로 다양한 장면 변화에도 **컨텍스트 일관성** 유지 가능[1].  
- **Reassemble 유연성**: 다중 해상도 특징 결합 구조가 다양한 입력 크기에 **탄력적 대응** 제공[1].  
- **대규모 예비 학습**: ViT의 사전 학습 및 MIX6와 같은 풍부한 메타데이터 활용이 **0-shot 전이 성능**을 크게 강화[1].

## 4. 향후 영향 및 연구 시 고려 사항  
- **영향**:  
  - Dense prediction 과제에서 **트랜스포머 기반 백본** 채택 가속화.  
  - 후속 연구에서 **토큰 재구성 및 결합 전략** 다양화, 경량화 아키텍처 개발 촉진.  
- **고려 사항**:  
  - **효율적 경량화**: 모바일·임베디드 적용을 위한 하이브리드·프루닝 기법 연구 필요.  
  - **데이터 효율성**: 소량 데이터에서도 일반화 가능한 **준지도 학습**·자기 지도 학습 기법 통합.  
  - **해석 가능성**: attention 맵 분석을 통한 모델 내부 작동 메커니즘 이해 강화.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f9222748-240c-4292-bf69-9fd43bcb12ee/2103.13413v1.pdf
