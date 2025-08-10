# Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration | Super resolution, Image restoration

## 1. 핵심 주장 및 주요 기여  
**Swin2SR**는 Swin Transformer V2를 기반으로 **압축된 저해상도 이미지의 초해상도(SR) 및 복원을 동시에 수행**하는 모델로, 다음과 같은 기여를 제시한다.  
- SwinIR 대비 **훈련 안정성**, **수렴 속도**, **데이터 효율성**을 크게 개선  
- JPEG 압축·초해상도·경량 SR 등 다양한 복원 과제를 **단일 아키텍처**로 통합  
- AIM 2022 챌린지에서 압축 이미지 SR 분야 **Top-5** 성과 달성  

## 2. 문제 정의 및 제안 기법  
### 2.1 해결 과제  
- JPEG 압축으로 인한 블록·링잉 아티팩트 제거  
- 원본 정보 손실된 저해상도(LR)에서 고해상도(HR) 복원  
- 다양한 압축 품질(q) 및 배율(×2, ×4) 대응  

### 2.2 모델 구조  
1) **Shallow Feature Extraction**: 초기 컨볼루션으로 저주파 정보 보존  
2) **Deep Feature Extraction**:  
   - Residual SwinV2 Transformer Block(RSTB) ×6  
   - 각 RSTB 내에 SwinV2 Transformer Layer(S2TL) ×6  
   - **Shifted Window Self-Attention V2**:  

$$
       \mathrm{Attention}(Q,K,V) = \mathrm{Softmax}\!\bigl(\tfrac{\cos(Q,K)}{\tau} + S\bigr)\,V
     $$

  - $$Q,K,V\in \mathbb{R}^{M^2\times d}$$, $$\tau$$: 학습 가능한 스칼라, $$S$$: 연속 위치 바이어스  
3) **Image Reconstruction**: 픽셀 셔플 업샘플링 후 잔차 연결으로 HR 생성  

### 2.3 추가 손실함수  
- **Auxiliary Loss**: 다운샘플 일관성 유지  

$$
    \mathcal{L}_{\rm aux} = \|D(y)-D(\hat y)\|_1
  $$  
- **High-Frequency Loss**: 고주파 디테일 복원  

$$
    \mathcal{L}_{\rm hf} = \|HF(y)-HF(\hat y)\|_1
    \quad,\quad HF(\cdot)=\cdot - (\cdot * b)
  $$  

## 3. 성능 향상 및 한계  
### 3.1 성능 개선  
- **JPEG 아티팩트 제거**: Classic5, LIVE1 데이터셋에서 단일 모델(q=10–40)로 최상위 PSNR/SSIM 달성  
- **클래식 SR (×2, ×4)**: SwinIR에 준하는 성능, 훈련 iteration 절반 수준  
- **경량 SR**: 파라미터 1.0M, MACs 199G으로 CARN·IMDN 수준 성능  
- **압축 이미지 SR**: AIM 2022 챌린지 테스트 PSNR 23.40dB로 Top-5 등극  

### 3.2 한계  
- 고주파 디테일 복원 시 일부 **블러링** 현상 남음  
- 매우 극단적 저품질(q≤10)에서 세밀한 텍스처 복원은 미흡  

## 4. 일반화 성능 향상 관점  
- SwinV2 기반 **연속 위치 바이어스**와 **코사인 어텐션** 도입으로 다양한 해상도·압축 환경에서 학습 불안정성 완화  
- **단일 모델**로 다중 품질·배율 대응 가능해, 테스트 시점 다양한 도메인에도 **추가 튜닝 없이** 적용  
- Auxiliary/HF 손실로 저·고주파 모두 고려하여 **도메인 편차**에 강인  

## 5. 향후 연구 영향 및 고려 사항  
- **대규모 비전 트랜스포머** 안정적 확장을 위한 SwinV2 구성 요소 일반 SR·복원에도 적용 가능  
- 압축·노이즈·블러 등 다중 저해상도 왜곡을 **통합 복원**하는 연구 추구  
- 한계 보완을 위해 고주파 복원용 **정교한 주파수 기반 헤드**나 **지각 손실(perceptual loss)** 도입 검토  
- 실시간 스트리밍·AR/VR 적용 시 **추론 속도 최적화** 및 **경량화** 전략 병행 필요  

***
**결론**: Swin2SR은 SwinV2 트랜스포머의 안정성과 일반화 장점을 활용해 압축된 저해상도 이미지 복원 분야에서 실질적 발전을 이끌었다. 후속 연구에서는 더욱 정교한 주파수 보존 기법과 다중 왜곡 대응, 실시간 퍼포먼스 개선에 주목해야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eba6d56d-3a46-4802-a126-6c9880493645/2209.11345v1.pdf
