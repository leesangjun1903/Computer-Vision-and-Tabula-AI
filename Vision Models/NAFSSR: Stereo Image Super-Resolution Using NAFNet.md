# NAFSSR: Stereo Image Super-Resolution Using NAFNet | Super resolution

## 핵심 주장 및 주요 기여
NAFSSR은 단일 이미지 복원에 강력한 성능을 보이는 NAFNet을 기반으로, 간단한 **Stereo Cross-Attention Module (SCAM)** 을 추가하여 스테레오 영상 간의 상호 정보를 효과적으로 융합함으로써  
1) 파라미터 효율성을 유지하면서  
2) 기존 최첨단 대비 우수한 SR(슈퍼해상도) 성능을 달성  
3) NTIRE 2022 챌린지에서 1위 성적을 획득  
한다는 점을 주요 기여로 제시한다.

***

## 1. 해결하고자 하는 문제
스테레오 SR(Stereo Super-Resolution)은 좌·우 두 개의 저해상도(LR) 이미지를 입력받아 고해상도(HR) 영상을 재구성하는 과제이다.  
- **기존 문제점**: 단일 뷰 정보만 활용하거나 복잡한 모듈·손실함수를 설계하여 교차 뷰 정보를 끌어오는 방식이 시스템 복잡도를 크게 증가시킴.  
- **목표**: 단순한 구조로 intra-view(단일 뷰)와 cross-view(양 뷰) 정보를 모두 활용해 높은 SR 성능을 얻되, 파라미터 수와 연산량을 최소화.

***

## 2. 제안 방법

### 2.1 전체 구조  
NAFSSR은 아래 세 단계로 구성된다.  
1. **Intra-view feature extraction**  
   - 3×3 Convolution → NAFBlocks 반복 → 3×3 Convolution + Pixel Shuffle  
   - NAFBlock: 비선형 활성화 없이 SimpleGate와 Channel Attention(SE) 적용한 MBConv + FFN 구조  
2. **Cross-view feature fusion (SCAM)**  
   - 각 NAFBlock 후에 삽입  
   - 스케일드 닷프로덕트 어텐션을 좌→우, 우→좌 이중으로 계산하되 에피폴라선(수평선)상에 한정[식(3)]  
     $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt{C})\,V $$  
   - 최종 융합:  
     $$F_L = X_L + \gamma_L\,\mathrm{Attn}(X_L,X_R),\quad F_R = X_R + \gamma_R\,\mathrm{Attn}(X_R,X_L)$$  
3. **Reconstruction**  
   - Upsampling 후 residual 예측: $$I^{SR} = \mathrm{PixelShuffle}(F) + \mathrm{Bilinear}(I^{LR})$$

### 2.2 수식 요약
- NAFBlock 내 SimpleGate:  
  $$\mathrm{SimpleGate}(X) = X_1 \odot X_2$$  
- Channel Attention:  
  $$\mathrm{CA}(X) = X \odot W\,\mathrm{pool}(X)$$  
- L1 손실:  
  $$\mathcal{L} = \|I_L^{SR}-I_L^{HR}\|_1 + \|I_R^{SR}-I_R^{HR}\|_1$$

***

## 3. 성능 향상 및 한계

### 3.1 성능 향상
- **파라미터 대비 PSNR**: NAFSSR-T는 기존 최첨단 SSRDE-FNet 대비 79% 파라미터 절감하면서 동등 이상의 성능 달성.  
- **Scale-up 효과**: NAFSSR-S/B/L 모델 크기 확장 시 PSNR 0.29–0.48 dB 추가 상승.  
- **실험 결과**: KITTI/Middlebury/Flickr1024 전반에 걸쳐 2×/4× 업샘플링 성능 모두 크게 개선.  
- **연산 효율**: NAFSSR-T는 ≈5.1×, NAFSSR-S는 ≈2.6× 속도 개선.

### 3.2 한계
- **데이터 의존성**: 중간 크기 이상 모델은 소규모 스테레오 데이터셋(flickr1024)에 과적합 경향.  
- **복잡한 장면**: 큰 시차(disparity)나 occlusion이 심한 장면에서 교차 어텐션의 제약(수평선 기반)이 정보 손실 유발 가능.

***

## 4. 일반화 성능 향상 전략
1. **Stochastic Depth**: NAFSSR-S/B/L에서 적용해 overfitting 방지 및 out-of-distribution 성능 +0.16 dB 개선.  
2. **TLSC(Test-time Local Statistics Conversion)**: 테스트 시 global→local pooling 전환으로 train-test 불일치 해소, PSNR +0.03 dB 이상 상승.  
3. **채널 셔플**: 입력 채널 랜덤 셔플로 다양성 보강, 단일 augmentation으로도 +0.19 dB 향상.

***

## 5. 향후 연구에 미치는 영향 및 고려 사항
- **모듈 단순화**: NAFSSR은 “최소한의 어텐션 모듈”로도 교차 뷰 정보를 효과적으로 활용할 수 있음을 보여, 향후 경량화된 스테레오 비전 시스템 설계에 영감.  
- **일반화 연구**: Stochastic Depth와 TLSC 조합의 train-test 일관성 해소 기법은 다른 영상 복원 과제에도 적용 가능.  
- **제약 해소**: Epipolar 기반 어텐션 한계 극복을 위해 비수평선(disparity-aware) 확장 및 occlusion 처리 강화가 필요.  
- **대규모 데이터**: 보다 다양한 스테레오 데이터셋 확보 및 크로스도메인 학습으로 일반화 안정성 추가 검증 권장.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5b2442ca-03d8-4104-92f3-482f45874c17/2204.08714v2.pdf
