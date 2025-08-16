# Swin Transformer V2: Scaling Up Capacity and Resolution | Image classification, Object Detection, Semantic Segmentation

**핵심 주장 및 주요 기여**  
Swin Transformer V2는 비전 트랜스포머 모델을 최대 30억 파라미터로 확장하고, 고해상도 입력(최대 1536×1536)에서도 안정적으로 학습·전이할 수 있도록 세 가지 주요 기법을 제안한다.  
1. **Residual Post-Norm + Scaled Cosine Attention**: 잔차 블록 후(normalization) 정규화와 코사인 유사도 기반 스케일 조정(attention)으로 대형 모델 학습 불안정성을 해소.  
2. **Log-Spaced Continuous Position Bias (Log-CPB)**: 윈도우 크기 변화에 유연한 위치 편향 생성을 위해 로그 스케일 좌표를 사용하는 메타 네트워크 도입.  
3. **SimMIM 자가지도 학습**: 방대한 라벨링 데이터 없이 1/40 규모의 데이터로 30억 파라미터 모델 사전학습.  

이 세 기법을 결합해 ImageNet-V2 분류(84.0% top-1), COCO 물체 검출(63.1 box AP), ADE20K 분할(59.9 mIoU), Kinetics-400 행동 인식(86.8%) 등 4대 벤치마크에서 모두 SOTA를 달성했다.

***

## 1. 해결하고자 하는 문제  
- 대형 비전 트랜스포머 학습 시, 깊은 층으로 갈수록 활성화 분포가 폭주하며 불안정(unstable)해짐.  
- 사전학습 해상도(저해상도)와 미세조정 해상도(고해상도) 간 위치 편향(interpolation) 성능 저하.  
- 라벨된 대규모 데이터 의존으로 인한 데이터 허기(data hunger).

***

## 2. 제안 방법

### 2.1 Residual Post-Norm & Scaled Cosine Attention  
- **Residual Post-Norm**  
  - 기존 Pre-Norm 구성에서 잔차 합산 전후에 LayerNorm 위치를 이동시켜 심층 누적(amplitude accumulation) 완화.  
- **Scaled Cosine Attention**  
  - 기존 $$\mathrm{Softmax}\bigl(QK^T/\sqrt{d}\bigr) $$ 대신  

$$
      \mathrm{Sim}(q_i, k_j) = \frac{\cos(q_i, k_j)}{\tau} + B_{ij}
    $$
    
  로 입력 크기에 무관한 코사인 유사도를 사용($$\tau$$: 학습 가능한 스칼라, $$B_{ij}$$: 위치 편향).  

이로써 심층 학습 안정성이 대폭 개선되고, 대형 모델 훈련이 가능해졌다.  

### 2.2 Log-Spaced Continuous Position Bias  
- **Continuous Position Bias**  

$$
    B(\Delta x, \Delta y) = G(\Delta x, \Delta y)
  $$
  
  2-layer MLP $$G$$로 임의 상대좌표에 대한 편향 생성.  
- **Log-Spaced Coordinates**  

$$
    \widetilde{\Delta x} = \mathrm{sign}(\Delta x)\,\log(1 + |\Delta x|),\quad
    \widetilde{\Delta y} = \mathrm{sign}(\Delta y)\,\log(1 + |\Delta y|)
  $$
  
  로그 변환으로 큰 해상도 차이 시에도 메타넷워크의 외삽(extrapolation) 부담 감소.

### 2.3 SimMIM 자가지도 학습  
- 입력 이미지 일부분을 마스킹하고 복원하도록 학습해, 라벨 없이도 표현 학습 가능.  
- JFT-3B 대비 40× 적은(70M) 라벨 데이터로 30억 파라미터 모델 사전학습.

***

## 3. 모델 구조 및 성능

| 모델 크기       | 파라미터 수 | 해상도(훈련→테스트)    | ImageNet-V2 Top-1 | COCO box AP | ADE20K mIoU | Kinetics-400 Top-1 |
|----------------|-------------|-------------------------|-------------------|-------------|-------------|--------------------|
| SwinV2-B       | 88M         | 192²→384²               | 78.08%            | 52.1        | 55.9        | —                  |
| SwinV2-L       | 197M        | 192²→384²               | 78.31%            | 52.7        | 58.4        | 84.9%             |
| **SwinV2-G**   | **3.0B**    | 192²→640²               | **84.00%**        | **63.1**    | **59.9**    | **86.8%**         |

- 모든 벤치마크에서 이전 최고 대비 1–4% 포인트 성능 향상.  
- 테스트 시 윈도우 크기 늘리면 추가 이익 가능(예: COCO AP +0.8).

***

## 4. 일반화 성능 향상 관점  
- **Log-CPB** 덕분에 사전학습(저해상도) 모델을 학습 없이도 다양한 해상도에 즉시 적용 가능, 테스트 시 윈도우 조정만으로 일반화 성능 개선.  
- **Residual Post-Norm + Cosine Attention**으로 과대적합(overfitting) 완화 및 심층 표현 안정화, 대규모 모델이 과도한 파라미터에도 튼튼하게 학습.  
- **SimMIM** 사전학습으로 라벨되지 않은 방대한 이미지에서 일반적 시각 표현 학습, 소수 라벨 데이터만으로도 downstream 과제에 잘 전이.

***

## 5. 한계 및 향후 연구 고려 사항  
- 3B 규모 학습을 위해 ZeRO, 체크포인트, 순차적 self-attention 등 복잡한 구현 필요.  
- Log-CPB 메타 네트워크가 매우 큰 윈도우 폭차(예: 8→64)에서는 외삽 한계 존재 가능.  
- 자가지도 방식 SimMIM 외에 더 강력한 SSL 기법 도입 시 추가 성능 여지.

### 향후 영향 및 연구 방향  
- 비전·언어 통합 멀티모달 모델의 공통 백본으로 활용 가능.  
- 로그 스케일 position bias는 3D 포인트 클라우드, 그래프 구조 등으로 확대할 잠재력.  
- 대형 모델 학습 안정화 기법들은 자연어·오디오 트랜스포머에도 이식 검토 필요.  
- 라벨 효율성 극대화를 위한 자가지도·반지도 학습 연구 강화가 요구됨.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5eaae8f-cccb-443d-890c-e006dbe05581/2111.09883v2.pdf
