# HAT : Activating More Pixels in Image Super-Resolution Transformer | Super-resolution

**핵심 주장:**  
Hybrid Attention Transformer(HAT)는 이미지 초해상도(SR) 과정에서 더 넓은 범위의 입력 픽셀 정보를 활성화함으로써 기존 Transformer 기반 모델이 갖는 국소 정보 활용 한계를 극복하고, 채널 주의(channel attention)와 창(window) 기반 자기주의(self-attention)를 결합하여 성능을 크게 향상시킨다.

**주요 기여:**  
1. 채널 주의와 창 기반 자기주의를 병렬로 융합하는 **Hybrid Attention Block(HAB)** 설계  
2. 이웃 창 간 상호작용을 강화하는 **Overlapping Cross-Attention Block(OCAB)** 도입  
3. 같은 과제(same-task)로 대규모 데이터(ImageNet) 사전 학습(pre-training) 후 소규모 데이터(DF2K)로 미세 조정(fine-tuning)하는 전략 제안  
4. 스케일업된 대형 모델(HAT-L)로 SR 성능 상한을 대폭 확장  

***

## 1. 해결하고자 하는 문제  
- 기존 CNN 기반 SR 모델은 국소적 수렴(local fitting)에 강하지만 전역 정보(global statistics) 활용이 제한적.  
- Transformer 기반 SwinIR 등은 자기주의 메커니즘을 쓰지만, 실제로는 입력의 넓은 픽셀 범위를 활용하지 못해 제한된 정보만으로 복원하며 블록 아티팩트가 발생함.  

***

## 2. 제안 방법

### 2.1 모델 구조 개요  
입력 $$I_{LR}\in\mathbb{R}^{H\times W\times C_{in}}$$ →  
1) 얕은 특징 추출: 3×3 합성곱 → $$F_0$$  
2) 깊은 특징 추출: Residual Hybrid Attention Group(RHAG) 반복 ×6  
3) 전역 잔차 연결 및 픽셀 셔플 업샘플링 → $$I_{SR}$$  

### 2.2 Hybrid Attention Block (HAB)  
– 표준 Swin Transformer 블록 내 창 기반 MSA(Window-MSA)와 병렬로 채널 주의 블록(CAB)을 결합  
– 수식  

$$
X_N = \mathrm{LN}(X),\quad
X_M = \mathrm{W\!-\!MSA}(X_N) + \alpha\,\mathrm{CAB}(X_N) + X,\\
Y = \mathrm{MLP}(\mathrm{LN}(X_M)) + X_M
$$  

– $$\alpha$$는 CAB 출력 가중치(기본값 0.01)  

### 2.3 Overlapping Cross-Attention Block (OCAB)  
– 이웃 창 간 직접 정보 교환을 위해 쿼리(Q)는 비겹치는 $$M\times M$$ 창, 키(K)와 값(V)는 겹치는 $$M_o\times M_o$$ 창으로 분할  
– $$M_o = (1+\gamma)\,M$$ ($$\gamma$$=0.5)  
– 같은 창 내부가 아닌 인접 영역 픽셀까지 토큰 단위 cross-attention 적용  

### 2.4 Same-Task Pre-training 전략  
– ImageNet 전체로 ×4 SR 과제 사전 학습(800K iter) → DF2K로 미세 조정(250K iter)  
– 멀티태스크 사전 학습 대비 단일 과제 대용량 데이터 학습이 SR 성능에 더 효과적임을 실험적으로 증명  

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- Urban100, Manga109 등 벤치마크에서 SwinIR 대비 PSNR +0.3~1.2dB 개선  
- 같은-task 사전 학습 시 Urban100 ×2 SR에서 +1.0dB 이상 성능 향상  
- HAT-L로 모델 규모 확장 시 성능 상한 대폭 상향  

### 3.2 일반화 성능 향상 가능성  
- CAB를 통한 전역 채널 정보 활용과 OCAB로 인접 창 상호작용 강화가 반복 패턴과 구조적 패턴에 강한 Urban100에서 특히 두드러진 성능 향상을 보임  
- 사전 학습 단계에서 다양한 도메인 이미지를 학습함으로써 여러 유형의 저해상도 열화에 대한 **모델 일반화 능력**이 강화됨  
- 실험적으로 CNN 대비 Transformer 계열이 대용량 데이터 학습 시 일반화 이점을 더 크게 누리는 양상을 확인  

### 3.3 한계 및 고려 사항  
- CAB 도입 시 연산량과 파라미터 수가 급증하므로, 계산 자원 제약 환경에서 경량화가 필요  
- OCAB 파라미터 $$\gamma$$와 CAB 압축비 $$\beta$$ 튜닝이 성능에 민감  
- 사전 학습-미세 조정의 최적 학습률·스케줄링 설정이 전체 성능에 큰 영향을 줌  

***

## 4. 향후 연구 영향 및 고려점

- **융합주의 구조 연구:** 전역 채널주의와 지역 창주의 결합 방식 다양화로 SR 성능·효율 균형화 가능성  
- **모델 경량화:** CAB, OCAB의 핵심 연산을 저비용으로 대체하거나 프루닝 및 지식 증류 활용  
- **사전 학습 도메인 확장:** 자연 이미지 외 의료·위성·저조도 등 특수 도메인 SR 일반화 연구  
- **하이브리드 멀티태스크 학습:** 같은-task 전략에 더해 유사 과제(예: 잡음 제거) 사전 학습이 주는 보완 효과 탐색  

이 논문은 Transformer 기반 SR 모델이 입력으로부터 더 많은 정보를 활용하도록 구조와 학습 방식을 혁신함으로써, 향후 초해상도 및 저수준 비전 전반의 **정보 활용 범위 확장 연구**와 **사전 학습 전략 최적화** 분야에 중요한 기준점을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/81ced4c8-d366-4a8f-bd0e-0228ae226ec1/2205.04437v3.pdf
