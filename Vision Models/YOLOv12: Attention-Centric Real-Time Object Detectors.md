# YOLOv12: Attention-Centric Real-Time Object Detectors

## 핵심 주장 및 주요 기여  
YOLOv12는 실시간 객체 탐지에서 전통적으로 우위에 있던 CNN 기반 설계를 벗어나, **효율적인 attention 메커니즘을 도입하여 속도와 정확도 양립**을 구현한 최초의 YOLO 프레임워크다.  
1. Area Attention: 특수한 윈도우 분할 없이 특징 맵을 단순 재구성만으로 attention 계산량을 절반으로 줄이는 모듈  
2. R-ELAN (Residual Efficient Layer Aggregation Networks): 블록 내 잔차 연결과 채널 집약 구조를 결합해 대규모 모델 최적화와 파라미터·FLOPs 절감  
3. FlashAttention, 위치 지각(perceiver) 등 기존 attention 오버헤드를 줄이는 설계 최적화  

## 해결하고자 하는 문제  
- **Attention의 계산 복잡도**: Self-attention은 입력 길이 $$L$$에 대해 $$O(L^2 d)$$ 연산을 필요로 해 고해상도 이미지에서 실시간 성능을 달성하기 어렵다.  
- **비효율적 메모리 접근**: attention map과 softmax map의 잦은 메모리 I/O로 인해 실제 지연(latency)이 크게 늘어남.  

## 제안 방법  
### 1. Area Attention  
- 특징 맵 $$H\times W$$를 수직 또는 수평으로 $$l$$ 분할해 크기가 $$\tfrac{H}{l}\times W$$ 또는 $$H\times \tfrac{W}{l}$$인 영역을 생성  
- 단순 reshape만으로 attention을 계산하여 연산량을  

$$
    O(2n^2hd) \;\to\; O\bigl(\tfrac12n^2hd\bigr)
  $$  
  
  으로 절반 수준으로 감소  

### 2. Residual Efficient Layer Aggregation Networks (R-ELAN)  
- ELAN 구조의 병목점을 해소하기 위해 블록 단위 잔차 연결 및 스케일링(기본 0.01) 적용  
- 채널 정렬(transition)→블록 반복→합치기 구조를 병목화된 단일 경로로 재설계해 FLOPs와 파라미터 절감  

### 3. 아키텍처 최적화  
- FlashAttention 도입으로 메모리 접근 오버헤드 최소화  
- MLP 비율(Feed-forward) 기본 1.2로 조정해 attention 연산 비중 확대  
- Conv2d+BN, 대형 separable convolution(7×7)으로 위치 정보 지각(perceiver)  
- 불필요한 positional encoding 제거, 뒤단 블록 수 축소로 경량화  

## 모델 구조  
- 입력 해상도 640×640 기준으로 N, S, M, L, X 5개 스케일 제공  
- 백본: YOLOv11의 초기 두 단계 유지, 이후 R-ELAN 블록으로 구성  
- 넥과 헤드 역시 attention-centric 모듈로 교체해 end-to-end 실시간 연산 흐름  

## 성능 향상  
- COCO 50:95 mAP 기준 YOLOv12-N 40.6% (1.64ms), YOLOv12-S 48.0% (2.61ms), YOLOv12-X 55.2% (11.79ms) 달성  
- 동일 FLOPs 대비 이전 YOLOv11 대비 평균 +1.2~1.7% mAP, RT-DETRv2 대비 최대 1.5%p 우위 및 두 배 이상 빠른 속도  
- Area Attention으로 RTX 3080에서 N 모델 FP32 기준 0.7ms, S 모델 1.6ms 가량 속도 개선  
- FlashAttention 적용 시 N/S 모델 추가 0.3~0.4ms 절감  

## 일반화 성능 향상 가능성  
- **넓은 수용 영역**: Area Attention이 전역 문맥을 반영하며 소규모 객체부터 대규모 객체까지 균일한 특성 학습  
- **경량화된 잔차 구조**: R-ELAN의 스케일링된 잔차 연결이 과적합 제어 및 안정적 수렴 도모  
- **학습 안정성**: MLP 비율과 hierarchical 디자인 최적화로 다양한 데이터셋 전이 시 과대적합 억제  
- 다중 스케일 실험에서 작은 변화에도 일관된 mAP 향상 관찰, **특화된 fine-tuning 없이도 새로운 도메인 전송학습 기대**

## 한계  
- FlashAttention 지원 GPU에 종속(현재 Turing, Ampere, Ada, Hopper 아키텍처)  
- 640×640 해상도 외 고해상도 입력에서 실제 성능 저하 및 메모리 병목 가능  
- 복합 모듈로 인한 코드 복잡도 상승 및 배포 환경 제약  

## 미래 연구 영향 및 고려 사항  
YOLOv12는 실시간 객체 탐지에서 **attention 메커니즘의 실용화 가능성**을 제시하며, 향후 연구는 아래를 중점 고려해야 한다.  
- **Attention 경량화**: 더 낮은 연산 예산에서 area attention과 linear attention 결합 연구  
- **하드웨어 최적화**: FPGA/모바일 GPU 환경에서 FlashAttention 대체 기법 개발  
- **다중 모달 일반화**: 비전+언어, 3D 포인트 클라우드 등 다양한 입력 유형에 대한 attention-centric 설계 확장  
- **학습 효율성 개선**: 프리트레이닝 활용 및 self-supervised 방식으로 일반화 성능 극대화  

이로써 YOLOv12는 실시간 딥러닝 비전 모델 설계 패러다임을 전환하고, **attention-first 객체 탐지** 분야의 새로운 표준을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/59bd7f9f-0c1f-4ee0-a7e2-243400d8ec17/2502.12524v1.pdf
