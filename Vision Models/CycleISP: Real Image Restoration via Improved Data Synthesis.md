# CycleISP: Real Image Restoration via Improved Data Synthesis | Image denoising, Image restoration

## 핵심 주장 및 주요 기여  
CycleISP는 실제 카메라 이미지의 노이즈 특성을 정교하게 모사하기 위해, 카메라의 ISP(Image Signal Processing) 파이프라인을 양방향으로 학습하는 **CycleISP 프레임워크**를 제안한다. 이를 통해 RAW 및 sRGB 공간에서 수백만의 사실적인 노이즈-클린 이미지 페어를 합성할 수 있으며, 해당 합성 데이터를 이용해 학습된 단일 이미지 복원 네트워크가 실제 카메라 벤치마크(DND, SIDD)에서 종전 최첨단 수준을 능가하는 성능을 달성함을 입증한다. 주요 기여는 다음과 같다.
- **디바이스-중립적 ISP 역·정방향 모델 학습**: 별도 카메라 파라미터 없이 sRGB→RAW, RAW→sRGB 변환을 모두 수행하는 양방향 CNN 구조 제안.  
- **정확한 노이즈 합성 모듈**: RAW 공간에서 신호종속성(shot) 및 비신호종속성(read) 노이즈를 추가한 뒤 ISP를 통해 sRGB 노이즈까지 현실적으로 구현.  
- **경량화된 복원 네트워크**: Dual Attention 기반의 Recursive Residual Group 구조를 활용하여 파라미터 수를 종전 대비 5배 축소하면서도 성능 향상.  
- **범용성 검증**: 노이즈 제거뿐 아니라 3D 영화용 스테레오 쌍의 색 정합(color matching)에도 효과적으로 활용됨을 보임.  

## 1. 해결하고자 하는 문제  
일반적인 심층 학습 기반 단일 이미지 복원(denoising) 모델은 대규모 데이터 확보가 어려운 RAW 노이즈 환경 대신, AWGN(첨가성 백색 가우시안 노이즈) 합성을 통해 학습된다. 그러나 실제 카메라 센서 노이즈는:
1) **신호 종속적(shot noise)**  
2) **채널·공간 상관성**  
3) **ISP에 의한 비선형 왜곡**  
등 AWGN 가정과 현저히 다르기 때문에, 학습된 네트워크는 실제 이미지에서 성능 저하를 겪는다.

## 2. 제안 방법  
### 2.1. CycleISP 프레임워크  
CycleISP는 두 개의 주요 가지(branch)로 구성된다:  
- **RGB2RAW**: sRGB 이미지 $$I_{rgb}$$를 입력받아 RAW 센서 측정값 $$ \hat I_{raw} $$으로 변환  
- **RAW2RGB**: RAW 데이터 $$I_{raw}$$를 모니터용 sRGB 이미지 $$ \hat I_{rgb} $$로 복원  

두 가지 네트워크 모두 Residual Group 내부에 **Dual Attention Block**(채널+공간 주의 메커니즘)을 여러 겹 쌓아 특징을 정교히 추출하며, 학습 손실은 L₁ 및 로그 도메인 L₁을 조합하여 강조치·저광량 영역 모두 복원토록 한다.  
학습은  
1) RGB2RAW와 RAW2RGB 개별 사전 학습  
2) 두 네트워크를 연결해 **공동 파인튜닝**(joint fine-tuning)  
단계로 이루어진다.

### 2.2. 현실적 노이즈 합성  
학습된 RGB2RAW에 클린 sRGB 이미지를 투입해 클린 RAW를 생성한 뒤, shot/read 노이즈를 랜덤 샘플링하여 RAW 영역에 주입한다. 이 노이즈화된 RAW를 RAW2RGB를 통해 sRGB 노이즈 이미지로 변환함으로써, RAW·sRGB 두 공간에 걸친 **1:1 클린-노이즈 페어**를 합성한다. 또한 SIDD 실샘플 데이터의 실제 노이즈 잔차(residue)를 이용한 추가 파인튜닝으로 합성 노이즈의 현실감을 더욱 제고한다.

## 3. 모델 구조  
- **Dual Attention Block (DAB)**:  
  - 채널 주의(Channel Attention): 전역 평균풀링→소·대역 필터→시그모이드 스케일링  
  - 공간 주의(Spatial Attention): 채널별 평균·최댓값 맵 결합→시그모이드  
- **Recursive Residual Group (RRG)**:  
  - 여러 DAB을 잇고, 그룹 단위 번갈아 skip connection을 적용  
- **Color Correction Branch (RAW2RGB 전용)**:  
  - 강한 블러 처리된 sRGB를 입력으로 색 정보만 추출하여 메인 경로와 결합  

이 구조로 파라미터 수 2.6M의 경량 네트워크를 구현하였다.

## 4. 성능 향상  
### 벤치마크 성능 (PSNR / SSIM)  
| 데이터셋 | 공간  | 이전 최고 (UPI) | Ours         | 향상치      |
|----------|------|----------------|-------------|------------|
| DND      | RAW  | 48.89 dB /0.982 | **49.13 dB /0.983** | +0.24 dB  |
| DND      | sRGB | 40.17 dB /0.962 | **40.50 dB /0.966** | +0.33 dB  |
| SIDD     | RAW  | 45.52 dB /0.980 | **52.41 dB /0.993** | +6.89 dB  |
| SIDD     | sRGB | 30.95 dB /0.863 | **39.47 dB /0.918** | +8.52 dB  |

특히 SIDD에서 대폭적인 PSNR 향상을 보여, 실제 노이즈 제거에 강력한 성능을 입증하였다.

## 5. 모델의 일반화 성능  
- **Cross-Dataset 일반화**: DND 전용으로 학습된 U-Net 모델을 SIDD에 입력 시, UPI 대비 +1.0 dB 성능 향상  
- **Color Matching 응용**: 스테레오 3D 영화 프레임 간 색 차이를 목표 뷰 색 정보로 보정하는 데 CycleISP를 활용, 기존 전통 기법 대비 더 높은 PSNR 및 시각적 일관성 확보  

이처럼 ISP 전 과정을 학습함으로써, 특정 카메라·데이터에 특화되지 않는 **디바이스-불변적** 복원·합성 능력을 갖춘다.

## 6. 한계 및 고려사항  
- **ISP 블랙박스**: Proprietary ISP 알고리즘 중 일부 단계가 완벽히 역추정되지 않을 경우, 합성 노이즈 분포가 실제와 미세 차이를 보일 수 있음.  
- **계산 비용**: dual attention과 다중 residual group 사용으로 연산량이 크며, 실시간 처리에는 추가 경량화가 요구됨.  
- **다중 모달리티**: 저조도, 압축 아티팩트 등 다른 노이즈 양상까지 합성하려면 CycleISP 확장이 필요.

## 7. 향후 연구에의 영향 및 고려점  
- **Low-Level Vision 통합 프레임워크**: 초해상도·탈블러링·색보정 등 다양한 복원 과제에 CycleISP 기반 현실적 합성 데이터 활용 확대  
- **ISP 단계별 해석성**: 블랙박스 대신 물리 기반 모듈을 도입하여 중간 표현의 해석성을 높이고, 카메라별 최적화를 가능케 하는 방향성  
- **경량화 및 실시간화**: 모바일·임베디드 환경 적용을 위한 Attention 모듈 단순화, 네트워크 프루닝(pruning)·양자화 연구 병행  

CycleISP는 “데이터 합성이 곧 모델의 실제 성능 직결”임을 재확인시킨 사례로, 향후 저수준 비전 연구의 표준 툴킷에 통합될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f53b3991-dc47-4e8c-bdfd-81591f5becf8/2003.07761v1.pdf
