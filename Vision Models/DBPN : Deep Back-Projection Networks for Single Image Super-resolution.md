# Deep Back-Projection Networks for Single Image Super-resolution | Super resolution

**핵심 주장 및 주요 기여**  
Deep Back-Projection Networks (DBPN)은 저해상도(LR) 이미지와 고해상도(HR) 이미지 간의 상호 의존성을 적극 활용하기 위해, 반복적인 상향(Up-sampling) 및 하향(Down-sampling) 투영 단위를 도입한 최초의 딥러닝 기반 단일 이미지 초해상도(SISR) 네트워크다. 주요 기여는 다음과 같다.  
1. **반복 투영 유닛 (Iterative Projection Units)**: LR→HR 투영(Up-projection)과 HR→LR 투영(Down-projection) 단위를 교차로 쌓아, 각 단계에서 투영 오차(error feedback)를 계산·보정.  
2. **오류 피드백 메커니즘 (Error Feedback)**: 각 투영 단계에서 재투영된 특징과 원본 특징의 차이를 이용해 잔차(residual)를 학습, HR 복원 품질을 크게 향상.  
3. **딥 연결(concatenation)**: 모든 Up-projection 단계의 HR 특징 맵을 연결하여 최종 재구성, 다양한 해상도 특징을 한 네트워크로 융합.  
4. **모델 경량화 및 확장성**: 파라미터 공유(recurrent DBPN), Dense 연결, residual 학습 등을 통해 성능–파라미터 수 트레이드오프 최적화.

## 1. 문제 정의  
단일 이미지 초해상도(SISR)는 정보가 부족한 LR 이미지를 HR로 복원하는 ill-posed inverse problem이다. 기존 DNN 기반 방법들은 대부분 순방향(feed-forward) 구조로 LR 특징을 HR로 단일 투영하거나, 단계적(progressive) 업샘플링만 수행하여 대규모 배율(×8)에서 잔상(ringing), 체스보드(chessboard) 아티팩트 문제를 해결하지 못함.

## 2. 제안 방법  

### 2.1 투영 유닛 수식  
각 단계 $$t$$에 대해  
– **Up-projection** (LR→HR):  

$$
\begin{aligned}
H_t^{(0)} &= (L_{t-1} * p_t)\uparrow_s,\\
L_t^{(0)} &= (H_t^{(0)} * g_t)\downarrow_s,\\
e_t^l &= L_{t-1} - L_t^{(0)},\\
H_t^{(1)} &= (e_t^l * q_t)\uparrow_s,\\
H_t &= H_t^{(0)} + H_t^{(1)}.
\end{aligned}
$$

– **Down-projection** (HR→LR):  

$$
\begin{aligned}
L_t^{(0)} &= (H_t * g'_t)\downarrow_s,\\
H_t^{(0)} &= (L_t^{(0)} * p'_t)\uparrow_s,\\
e_t^h &= H_t^{(0)} - H_t,\\
L_t^{(1)} &= (e_t^h * g'_t)\downarrow_s,\\
L_t &= L_t^{(0)} + L_t^{(1)}.
\end{aligned}
$$

여기서 $$*$$는 컨볼루션, $$\uparrow_s$$ · $$\downarrow_s$$는 각각 배율 $$s$$의 업·다운샘플링 연산. $$p_t, q_t, g_t$$ 등은 학습 가능한 (de)convolution 필터.

### 2.2 전체 아키텍처  
1. **초기 특징 추출**: $$L_0 = \mathrm{conv}(3,n_0)$$(LR 입력) → 차원 축소 $$\mathrm{conv}(1,n_R)$$.  
2. **반복 백-프로젝션 단계**: Up-projection ㆍ Down-projection을 $$T$$단계 반복.  
3. **재구성**: 모든 Up단계의 HR 특징 $$\{H_1,\dots,H_T\}$$을 깊이 연결(concatenate) 후 $$\mathrm{conv}(3,3)$$로 최종 SR 이미지 생성.

### 2.3 변형 모델  
- **Dense DBPN**: 각 투영 유닛 입력으로 이전 모든 유닛 출력의 1×1 컨볼루션 병합 → gradient 흐름 개선.  
- **Recurrent DBPN**: Up/Down 유닛 파라미터 공유(재귀) → 파라미터 수 대폭 감소.  
- **Residual DBPN**: 최종에 LR을 bicubic 보간 후 잔차 학습 → 수렴 가속 및 품질 개선.

## 3. 성능 향상 및 한계  

| 모델            | Set5 PSNR (4×) | 파라미터 수 |
|-----------------|---------------:|------------:|
| LapSRN          |       31.54 dB |       0.81M |
| EDSR            |       32.46 dB |      43.2M  |
| D-DBPN (본 연구) | **32.40 dB**   |      10.2M  |
| DBPN-RES-MR64-3 | **32.65 dB**   |      10.2M  |

- **대규모 배율(8×)**에서 PSNR·SSIM 모두 기존 기법 대비 평균 0.3–0.6 dB 이상 우수.  
- 경량 모델(DBPN-R64-10)은 1.6M 파라미터로 실시간 애플리케이션에도 적용 가능.  
- PIRM2018 챌린지: Perceptual Index 기준 Region 2 1위, Region 1 3위, Region 3 5위.

**한계**  
- 반복 투영 단계 수 $$T$$ 증가 시 학습·추론 비용 선형 증가.  
- 최적 필터 크기(예: 8×8 for 4×)는 경험적 설정, 일반화 중요 데이터셋에 대한 자동 최적화 미흡.  
- 매우 복잡한 환경(노이즈, 압축 아티팩트)에서 성능 검증 필요.

## 4. 일반화 성능 향상 가능성  
- **Domain Adaptation**: 투영 유닛의 오류 보정 구조는 다양한 저품질 입력(노이즈·모션 블러)에 대한 적응 학습에 유리.  
- **Self-Supervised 학습**: Down-projection 오류를 이용해 unlabeled 영상으로도 재구성 오류 신호 생성 가능 → 학습 데이터 다양화.  
- **파라미터 공유 모델**: 공유 유닛 기반 구조는 타깃 도메인별 소량 파인튜닝으로 일반화 성능 강화.

## 5. 향후 연구 과제 및 영향  
- **Adaptive 투영 필터 학습**: 입력 영상 특성에 따라 자동으로 투영 필터 크기·개수를 조정하는 메커니즘 필요.  
- **효율적 다중배율 학습**: 단일 모델로 2×, 4×, 8× 등의 멀티스케일 SR 지원 및 일반화.  
- **노이즈·압축 내성**: 실제 저품질 영상을 대상으로 한 투영 유닛 강화 및 페어링 전략.  
- **초해상도 + 다운스트림 태스크**: SR 후 객체 검출·분할 등 후속 컴퓨터 비전 과제 성능 개선 검증.

DBPN은 **투영 오차 피드백** 개념을 SISR에 성공적으로 적용하여, 대규모 배율에서의 성능 한계를 돌파했고, SR 분야 후속 연구에 **반복 보정 메커니즘**과 **파라미터 효율적 설계**라는 새로운 방향을 제시했다. 앞으로는 **적응적·자기지도 학습**과의 결합, **실제 저품질 영상**에 대한 평가 확장이 중요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8f2d003a-d01b-45e4-919d-9159432ad3d6/1904.05677v2.pdf
