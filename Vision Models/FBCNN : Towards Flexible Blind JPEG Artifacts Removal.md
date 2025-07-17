# FBCNN : Towards Flexible Blind JPEG Artifacts Removal | Image compression, JPEG Artifact correlation, Image restoration

# 핵심 요약

**“Towards Flexible Blind JPEG Artifacts Removal”** 논문의 주요 주장은 다음과 같다.  
- 단일 모델로 다양한 JPEG 품질(QF)을 처리하면서도, 입력 이미지를 복원할 때 **예측된 품질 인자(QF)를 조절**해 “노이즈 제거⇄디테일 보존” 간의 **사용자 정의 트레이드오프**를 실현할 수 있는 유연한 블라인드 네트워크 FBCNN(Flexible Blind CNN)을 제안한다.  
- 비정렬(shifted) 이중 JPEG 압축 상황에서도 **“우세 QF(dominant QF)” 보정** 및 **이중 압축 데이터 증강** 전략을 통해 실제 인터넷 이미지 복원 문제를 효과적으로 해결한다.

# 상세 설명

## 1. 문제 정의  
기존 DNN 기반 JPEG 복원은  
- 각 QF별로 **전용 모델**을 학습하거나,  
- 블라인드 방식으로 복원하되 QF를 예측하지 않아 **출력 제어 불가능**,  
- 다중 압축(실제 인터넷 이미지) 시 **비정렬 이중 압축**에 취약  
했다.

## 2. 제안 기법

### 2.1 모델 구조  
FBCNN은 네 부분으로 구성된다:  
1) **Decoupler**  
   - 입력 JPEG 이미지에서 영상 특징과 QF 특징을 분리 추출  
   - 4개 스케일, 각 4개 잔차 블록(residual block), 채널 수 64→128→256→512  
2) **QF Predictor**  
   - 512-차원 QF 특징 → 3-layer MLP → **예측 QF**  
   - $$L_{QF} = \frac1N\sum_i |QF_{est}^i - QF_{gt}^i|$$  
3) **Flexible Controller**  
   - 예측 QF → 4-layer MLP → 스케일별 $$\gamma,\beta$$ (affine 파라미터) 생성  
4) **Image Reconstructor**  
   - Decoupler의 이미지 특징에 QF 기반 $$\gamma,\beta$$ 적용 (QF Attention)  
   - $$F_{out} = \gamma \odot F_{in} + \beta$$  
   - $$L_{rec} = \frac1N\sum_i \|I_{rec}^i - I_{gt}^i\|_1$$  
   - 전체 손실: $$L_{total}=L_{rec} + \lambda L_{QF}$$, $$\lambda=0.1$$

### 2.2 이중 JPEG 처리  
- **우세 QF 보정(FBCNN-D)**:  
  - 모든 QF로 재압축→MSE 곡선의 첫 번째 최소값 위치가 진짜 첫 압축 QF  
- **데이터 증강(FBCNN-A)**:  
  - 랜덤한 픽셀 시프트(0–7) 후 2회 JPEG 적용  
  - 증강 시 QF 예측 손실 무시($$\lambda=0$$)

## 3. 성능 향상  
| 조건                  | 단일 JPEG 복원 (PSNR↑) | 비정렬 이중 JPEG (PSNR↑) |
|-----------------------|------------------------|---------------------------|
| 기존 DnCNN/QGAC       | 29.51–34.16 dB         | 28.20–32.32 dB            |
| **FBCNN**             | 29.75–34.53 dB (+0.3)  | 28.29–32.61 dB            |
| **FBCNN-D**           | —                      | 28.94–32.65 dB (+0.6)     |
| **FBCNN-A**           | —                      | 29.38–32.69 dB (+1.0)     |

- 단일 압축 복원에서 블라인드 SOTA 대비 **0.3 dB** 수준의 안정적 개선  
- 복잡한 비정렬 이중 압축에서는 **FBCNN-A**가 최대 **1.0 dB** 개선  

## 4. 한계  
- QF 예측이 어려운 **작은 패치**에서는 불안정  
- 증강된 이중 압축 외 **실제 메타데이터 없는 복잡 압축** 사례 일반화 필요  
- 논문 실험은 **MATLAB JPEG** 기준; 실제 카메라·SNS 압축 다양성 검증 미흡  

# 일반화 성능 향상 가능성  
- **QF 제어 변수**를 활용해 다양한 도메인(노이즈, 블러, 압축)으로 확장 가능  
- **비지도 우세 QF 추정** 방식은 새로운 압축 알고리즘에도 적용 전망  
- **증강 모델**에 다양한 리얼 월드 압축 체인(카메라→SNS→웹) 시뮬레이션 추가 시 일반화 강화  

# 향후 연구 영향 및 고려 사항  
- **사용자 설정 가능 복원** 패러다임 제시: 인터랙티브 이미지 에디팅·리스토어 분야 확장  
- **비정렬 다중 압축** 복원 연구 활성화: 실제 인터넷·SNS 데이터셋 구축과 벤치마크 필요  
- QF 예측 불확실성 관리용 **신뢰도 추정** 메커니즘 및 **메타러닝** 적용 검토  
- **다중 손상 복합** 시나리오(압축+노이즈+블러)에 대한 통합된 유연 복원 프레임워크 연구 제안

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a678a3a7-a008-4a64-a475-1ad5b517f4c3/2109.14573v1.pdf
