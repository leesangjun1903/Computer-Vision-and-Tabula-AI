# Dynamic Convolution: Attention over Convolution Kernels | Image classification

**핵심 주장 및 주요 기여**  
본 논문[1]은 경량화된 CNN의 표현력을 저해하는 낮은 연산 예산 문제를 해결하기 위해, **단일 정적 커널이 아닌 입력 의존적(attention-based)으로 가중치를 동적으로 조합하는 다중 병렬 컨볼루션 커널**을 제안한다. 이를 통해 네트워크 깊이나 너비는 그대로 유지하면서도, 모델 복잡도를 효과적으로 높여 ImageNet 분류에서 최대 +4.5% Top-1 정확도 향상, COCO 키포인트 검출에서 +2.9 AP 향상을 달성했다.

## 1. 해결하고자 하는 문제  
- 경량 CNN은 낮은 FLOPs 제약 하에서 깊이(depth)와 폭(width)을 증가시키지 못해 표현력이 제한됨[1].  
- MobileNetV3-Small (66M MAdds) → Top-1 67.4%, MobileNetV3-Small (219M MAdds) → 75.2%로 성능 저하 심각.

## 2. 제안하는 방법  
### 2.1. Dynamic Perceptron 수식  

$$
y = g\bigl(\tilde W(x)^\top x + \tilde b(x)\bigr),\quad
\tilde W(x) = \sum_{k=1}^K \pi_k(x)\,\tilde W_k,\quad
\tilde b(x) = \sum_{k=1}^K \pi_k(x)\,\tilde b_k
$$

$$
\text{s.t.}\quad \pi_k(x)\ge0,\quad \sum_k\pi_k(x)=1
$$  

- $$K$$개의 병렬 커널 $$\{\tilde W_k,\tilde b_k\}$$을 입력 의존적 attention $$\pi_k(x)$$으로 가중 합.  
- Softmax 기반으로 $$\sum\pi_k=1$$ 제약하여 학습 안정화[1].

### 2.2. Dynamic Convolution 구조  
- 각 컨볼루션 레이어에 $$K$$개의 동일 구조 커널 병렬 배치.  
- Squeeze-and-Excitation 방식으로 global average pooling → FC-ReLU-FC → softmax(온도 τ)로 $$\pi_k(x)$$ 생성.  
- $$\tau$$를 크게 설정(초기 τ=30, 초 10 epochs 동안 점진적 감소)하여 초기 학습에서 평탄한 attention 분포 유지[1].  
- 추가 FLOPs: attention 계산 및 커널 집계 연산이 전체 컨볼루션 대비 4% 내외로 경미.

## 3. 모델 구조 및 세부 설정  
- **네트워크**: MobileNetV2/V3, ResNet-10/18.  
- **Dynamic Layer**: 모든 컨볼루션(첫 레이어 제외)에 $$K=4$$ 적용.  
- **학습 하이퍼파라미터**:  
  - τ annealing: 30→1 (초기 10 epochs), Learning rate cosine 또는 step decay.  
  - Optimizer: SGD (momentum 0.9), weight decay ≈1e-4, batch size 256.  
  - Regularization: label smoothing, dropout, mixup 등[1].

## 4. 성능 향상  
| 백본             | FLOPs    | Top-1 (%)     | Top-1 개선 |
|------------------|----------|---------------|------------|
| MobileNetV2×1.0  | 300M     | 72.0 → **75.2** | +3.2       |
| MobileNetV2×0.5  | 97M      | 65.4 → **69.9** | +4.5       |
| MobileNetV3-Small| 66M      | 67.4 → **70.3** | +2.9       |
| ResNet-18        | 1.81G    | 70.4 → **72.7** | +2.3       |

- COCO 키포인트 검출에서도 AP +1.6∼4.9 향상[1].  

## 5. 한계 및 일반화 성능  
- **한계**  
  - 모델 크기·실제 지연 시간(CPU 기준 +10%) 증가: 작은 연산 증가는 실제 최적화되지 않은 환경에서 더 큰 오버헤드 발생[1].  
  - $$K$$가 커질수록 학습 난이도·과적합 위험 증가: 성능 포화는 $$K=4$$ 근방[1].  
- **일반화 성능 향상 관련**  
  - 입력별 커널 조합으로 다양한 국소 특징 캡처 → 작은 네트워크에서도 표현력 강화.  
  - Mixup 및 dropout 등과 결합 시 regularization 효과로 외부 도메인에서도 견고성 기대 가능.  
  - 초기 평탄화된 attention 분포는 모든 커널 동시 최적화, 앙상블 효과 유도 → 과적합 억제.

## 6. 향후 연구에 미치는 영향 및 고려사항  
- **영향**  
  - NAS(Search) 공간에 *동적 커널 모듈* 추가: 입력 의존적 경량 앙상블 구조 설계.  
  - 비전 이외의 영역(음성·자연어)에서도 dynamic operator 확장.  
- **고려사항**  
  - 실제 배포 환경의 커널 집계 연산 가속화: hardware-aware 최적화 필요.  
  - 최적 적정 $$K$$ 탐색 및 온도 스케줄링 연구: 과적합 방지와 성능 균형 유지.  

[1] Dynamic Convolution: Attention over Convolution Kernels, Y. Chen et al., arXiv:1912.03458v2.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/70e2d549-634a-4fdb-ba04-d86f3771faaa/1912.03458v2.pdf
