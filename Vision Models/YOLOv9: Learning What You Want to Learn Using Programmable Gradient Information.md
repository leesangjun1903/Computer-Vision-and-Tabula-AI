# YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

**주요 주장**  
YOLOv9는 딥러닝 네트워크 학습 시 정보 병목(information bottleneck)으로 인한 중요한 입력 정보 손실을 완화하고, 경량 모델에서도 신뢰할 수 있는 그래디언트(gradient)를 제공하여 수렴(convergence) 성능을 크게 향상시킨다. 이를 위해 reversible 구조를 활용한 **Programmable Gradient Information (PGI)** 메커니즘과, 임의의 계산 블록을 지원하는 **Generalized Efficient Layer Aggregation Network (GELAN)** 아키텍처를 제안한다.

**주요 기여**  
- **PGI**: 역전파 시 auxiliary reversible branch를 통해 완전한 입력 정보를 복원하고, multi-level auxiliary information으로 다양한 의미 수준의 그래디언트를 조합함으로써 정보 병목과 deep supervision의 단점을 동시에 해결.  
- **GELAN**: CSPNet과 ELAN을 일반화하여 사용자가 원하는 계산 블록과 깊이(depth)를 자유롭게 조합 가능토록 설계, 경량성, 연산 효율, 성능을 균형 있게 확보.  
- **YOLOv9**: PGI와 GELAN을 결합한 실시간 객체 탐지 모델로, MS COCO에서 파라미터 수와 FLOPs를 대폭 줄이면서도 기존 최첨단 모델 대비 0.6% 이상의 AP 개선을 달성.  

***

## 해결하고자 하는 문제

1. **정보 병목(Information Bottleneck)**  
   딥 네트워크가 깊어질수록 입력 X의 본질적 정보 I(Y; X)가 점차 손실되어, 최종 출력 Ŷ에 대한 그래디언트가 불완전해지고 수렴 성능이 저하됨.  

   $$ I(X;X) \ge I(X; f_\theta(X)) \ge I(X; g_\phi(f_\theta(X))) $$  

2. **Deep Supervision 한계**  
   중간층에 보조 예측(head)을 추가하는 전통적 deep supervision은 shallow features와 깊은 features 간 정보 충돌을 야기하여 오히려 얕은(또는 가벼운) 모델에서 과소 파라미터화 현상을 일으킴.

***

## 제안 방법

### 1. Programmable Gradient Information (PGI)

- **수식적 정의**  
  - Reversible 함수 $$r_\psi$$와 그 역함수 $$v_\zeta$$를 활용해 완전한 정보 복원을 보장:  

    $$X = v_\zeta\bigl(r_\psi(X)\bigr), \quad I(X;X) = I(X; r_\psi(X)) $$  

  - auxiliary reversible branch를 통해 메인 브랜치의 특성(feature)에 잃어버린 정보를 보강하고, multi-level auxiliary information 네트워크에서 여러 예측 머리(head)로부터 받은 그래디언트를 통합하여 메인 브랜치에 공급.

- **아키텍처 구성**  
  1. **Main Branch**: GELAN 기반, 추론 시 단독 사용  
  2. **Auxiliary Reversible Branch**: DHLC나 RevCol 유사 구조로 설계, 완전한 정보 공급  
  3. **Multi-level Auxiliary Information**: FPN/PAN 유사 집성기 또는 ICN(Integrated Composite Network)으로 그래디언트 종합  

### 2. Generalized Efficient Layer Aggregation Network (GELAN)

- CSPNet의 feature partitioning과 ELAN의 gradient path planning을 결합  
- Conv, Res, Dark, CSP 등의 연산 블록(block)을 plug-and-play 방식으로 자유 조합  
- 깊이 조절(ELAN depth, CSP depth)을 통해 모델 규모(S, M, C, E)별 유연한 설계 가능  

***

## 모델 구조

| 구성 요소                   | 세부 사항                                                        |
|-----------------------------|------------------------------------------------------------------|
| Backbone & Neck             | CSP-ELAN 블록 반복, DOWN 모듈(풀링+Conv)                         |
| SPP-ELAN                    | Spatial Pyramid Pooling + ELAN                                    |
| Feature Fusion              | Up/Down 샘플링 후 Concat → CSP-ELAN                              |
| Prediction Head             | Decoupled head: classification, objectness, box regression 분리  |
| Auxiliary Branch            | 입력 → reversible 연산 → multi-level fusion → loss 계산           |

***

## 성능 향상

- **MS COCO 2017 (train-from-scratch)**  
  - YOLOv9-E: 57.3M 파라미터, 189.0G FLOPs → **55.6% AP**, AP50 72.8%, AP75 60.6%  
  - 기존 YOLOv8-X 대비 파라미터 −16%, FLOPs −27%, AP +1.7%  

- **Ablation**  
  - GELAN: Conv→CSP 블록 교체 시 소형 모델 AP +0.7% 향상  
  - PGI: auxiliary reversible(ICN)+lead-head guided training 조합 시 대형 모델 AP +0.6%  

***

## 한계 및 일반화 성능

- **한계**  
  - Auxiliary branch 학습 비용: 학습 시 메모리·연산 증가  
  - 복잡한 하이퍼파라미터(ELAN/CSP 깊이, fusion 구조) 튜닝 필요  
- **일반화 성능**  
  - PGI는 경량 모델에서도 auxiliary supervision 적용 가능토록 설계되어, 다양한 데이터셋과 도메인 전이(transfer)에서 robust한 수렴을 기대  
  - GELAN의 블록 유연성 덕분에, 새로운 연산 블록(예: Transformer block) 통합으로 최신 비전 아키텍처와도 호환  

***

## 향후 연구 및 고려 사항

1. **경량화된 Auxiliary Structure**  
   - 학습 비용을 줄이기 위한 reversible branch 경량화  
2. **하이퍼파라미터 자동화**  
   - ELAN/CSP 깊이, fusion 네트워크 구조 탐색을 위한 NAS(Neural Architecture Search) 적용  
3. **다양한 비전 태스크로 확장**  
   - 분할(segmentation), 자세 추정(pose estimation) 등에 PGI & GELAN 적용 가능성 연구  
4. **도메인 일반화**  
   - 저조도, 왜곡된 영상 등 까다로운 환경에서 PGI가 제공하는 안정적 그래디언트의 효과 검증  

이상으로, YOLOv9는 **정보 병목 해소**와 **경량 모델의 보조 학습**을 혁신적으로 결합하여, 실시간 객체 탐지 분야의 새로운 기준을 제시한다. 앞으로 PGI와 GELAN 모듈은 다양한 비전 모델 설계와 학습 전략에 핵심적인 영감을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9bdba5b0-74be-4c45-a3b6-cd8d0b8089fc/2402.13616v2.pdf
