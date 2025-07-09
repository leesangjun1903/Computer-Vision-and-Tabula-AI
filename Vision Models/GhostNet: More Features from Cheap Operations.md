# GhostNet: More Features from Cheap Operations | Image classification

**핵심 주장 및 주요 기여**  
GhostNet은 기존의 고비용 컨볼루션 연산을 저렴한 선형 변환(linear transformation) 연산으로 대체하여, 본질적(intrinsic) 특징 맵을 생성한 뒤 이를 기반으로 다수의 “고스트(ghost)” 특징 맵을 효율적으로 생성함으로써, 파라미터 수와 FLOPs를 대폭 절감하면서도 성능을 유지하거나 향상시킨다. 주요 기여는 다음과 같다.  
1. **Ghost 모듈**: 전통적 컨볼루션 연산을 두 단계로 분리  
   - 1차: $$m$$개의 intrinsic 특징 맵 $$Y'=X * f'$$ 생성  
   - 2차: 저렴한 $$d\times d$$ depthwise 컨볼루션 $$\Phi_{i,j}$$로 ghost 특징 맵 $$y_{ij}=\Phi_{i,j}(y'_i)$$ 생성  
2. **Ghost bottleneck**: MobileNetV3의 inverted residual block을 Ghost 모듈로 대체  
3. **GhostNet 아키텍처**: 일련의 Ghost bottleneck을 쌓아 경량화 네트워크 설계  
4. **실험 결과**:  
   - ImageNet에서 MobileNetV3 대비 유사 FLOPs에서 최대 +0.5% Top-1 정확도 향상  
   - ResNet-50 대체 시 2× 연산량 절감에도 원본 수준 정확도 유지  

## 1. 해결하고자 하는 문제  
- **임베디드 기기 제약**: 대용량 CNN은 메모리와 연산량이 커 모바일·임베디드 환경에 부적합  
- **특징 맵의 중복성**: 잘 학습된 CNN의 깊은 레이어 특징 맵에는 고차원 구조에도 불구하고 상당한 중복 정보가 존재  

## 2. 제안 방법

### 2.1 Ghost 모듈  
전통적 컨볼루션 연산  

$$
Y = X * f + b,\quad f\in\mathbb{R}^{c\times k\times k\times n}
$$  

Ghost 모듈  
1) **Primary convolution**:

$$
Y' = X * f',\quad f'\in\mathbb{R}^{c\times k\times k\times m},\quad m = \tfrac{n}{s}
$$  

2) **Ghost feature 생성**:

$$
y_{ij} = \Phi_{i,j}(y'_i),\quad i=1,\dots,m,\;j=1,\dots,s,\quad n=m\times s
$$  

여기서 $$\Phi_{i,j}$$는 $$d\times d$$ depthwise 컨볼루션 또는 항등 매핑이다.  

**이론적 이득**  

$$
\text{Speed-up}\approx s,\quad \text{Compression}\approx s
$$  

(수식 4, 5 참조)

### 2.2 Ghost bottleneck  
- **Expansion Ghost 모듈 → Depthwise(conv, stride=2) → Projection Ghost 모듈**  
- MobileNetV2/V3의 inverted residual 구조에 Ghost 모듈 적용, SE 모듈 결합

### 2.3 GhostNet  
- 첫 3×3 conv → 일련의 Ghost bottleneck 스테이지 → 글로벌 풀링 → 1×1 conv → FC  
- width multiplier $$\alpha$$로 채널 폭 조정 가능

## 3. 모델 구조 요약  
| Stage | Operator                 | #exp   | #out  | SE  | Stride |
|-------|--------------------------|--------|-------|-----|--------|
| 1     | Conv 3×3                 | –      | 16    | –   | 2      |
| 2–16  | Ghost bottleneck blocks  | 16→960 | 16→160| 조건부 | 1→2 |
| 17    | Conv 1×1                 | –      | 960   | –   | 1      |
| 18    | AvgPool 7×7              | –      | –     | –   | –      |
| 19    | Conv 1×1                 | –      | 1280  | –   | 1      |
| 20    | FC (1000 classes)        | –      | –     | –   | –      |

## 4. 성능 향상 및 한계

### 4.1 성능 향상  
- **ImageNet Classification**  
  - GhostNet-1.3× (226M FLOPs) → 75.7% Top-1  
  - MobileNetV3 Large 1.0× (219M FLOPs) → 75.2%  
- **ResNet-50 압축**  
  - Ghost-ResNet-50 (s=2) → 13.0M params, 2.2B FLOPs, 75.0% Top-1  
- **Latency**  
  - ARM 모바일(단일 스레드, TFLite)에서 MobileNetV3 대비 동등 정확도 시 약 10% 빠른 추론
- **Object Detection (COCO)**  
  - RetinaNet/Faster R-CNN 백본으로 활용 시 MobileNetV2/V3 대비 유사 mAP에 30%↓ FLOPs

### 4.2 한계  
- **Linear 변환의 한계**: depthwise 컨볼루션 외 다른 저비용 변환 탐색 부족  
- **자동 최적화 부재**: Ghost 모듈 내 $$s,d$$ 하이퍼파라미터 수동 설정  
- **Non-visual 태스크 일반화**: NLP, 음성 등 비이미지 도메인 적용성 검증 미비  

## 5. 일반화 성능 향상 가능성  
- **중복 특징의 효율적 재사용**: intrinsic 특징을 다양한 저비용 변환으로 확장함으로써 오버피팅 억제 및 표현력 강화  
- **Width multiplier**를 통해 모델 규모와 성능 균형 제어 가능 → 전이 학습 시 과적합 방지  
- **Ghost bottleneck** 특유의 저비용 연산 구조는 새로운 도메인(세그멘테이션, 비전 변환)으로 확장 잠재력  

## 6. 향후 연구와 고려 사항  
- **Adaptive Ghost 모듈**: 변환 수 $$s$$와 커널 크기 $$d$$를 데이터 특성에 맞춰 학습하는 메타 프루닝/신경 구조 탐색  
- **다양한 저비용 변환**: affine, 파동변환, channel shuffle 등 Ghost 특화 변환 실험  
- **비이미지 도메인 적용**: NLP·음성 분야에서 GhostNet 구조 효능 검증  
- **하드웨어 최적화**: FPGA·ASIC 환경에서 Ghost 모듈 연산 병렬화 및 메모리 배치 최적화 연구  

이로써 GhostNet은 “가성비 높은 특징 맵 생성” 패러다임을 제시하며, 경량화 모델 설계와 신경 구조 탐색 분야에 새로운 방향성을 제안한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/30eee873-20d9-4bff-a740-fa973c2bdd76/1911.11907v2.pdf
