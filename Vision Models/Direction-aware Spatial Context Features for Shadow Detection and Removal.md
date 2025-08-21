# Direction-aware Spatial Context Features for Shadow Detection and Removal | Shadow detection, Shadow removal

## 1. 핵심 주장 및 주요 기여
본 논문은 **그림자(shadow)** 를 단일 이미지에서 검출하고 제거하기 위해, **공간 맥락(spatial context)을 방향별로 선별 학습**하는 새로운 딥러닝 모듈인 DSC(Direction-aware Spatial Context) 모듈을 제안한다.  
- 글로벌 이미지 문맥 정보를 네 방향(상·하·좌·우)으로 순환 전파하는 **Spatial RNN**과  
- 각 방향별 중요도를 학습하는 **direction-aware attention** 메커니즘  
을 결합하여, 그림자 검출 및 제거 성능을 크게 향상시켰다.

## 2. 문제 정의 및 제안 방법
### 2.1 해결하고자 하는 문제
- 기존 그림자 검출/제거 기법은 로컬 색·조명 모델이나 단순 CNN 기반 학습으로 인해  
  -  **검출 정확도**(특히 경계나 미약한 그림자) 부족  
  -  **검은 물체**(shadow–like object)와 혼동  
  -  제거 후 **비일관적 색 보정**  
  등의 한계를 가짐.

### 2.2 제안 모듈: DSC
1. **Spatial RNN**  
   - 입력 특징 맵 $$X$$에 대해 1×1 conv → ReLU → 4방향 순환전파(translation):  

$$
       h_{i,j}^{(d)} = \max\bigl(\alpha_d\,h_{i+\Delta_i,j+\Delta_j}^{(d)} + h_{i,j},\,0\bigr),
     $$  
     
$$\alpha_d$$는 각 방향 $$d\in\{\text{up,down,left,right}\}$$의 가중치.

2. **Direction-aware Attention**  
   - 두 개의 3×3 conv + ReLU, 1×1 conv 층을 통해 $$W\in\mathbb{R}^{H\times W\times4}$$를 생성.  
   - 분리된 채널 $$W_d$$로 각 방향별 RNN 출력 $$C_d$$를 element-wise 곱하여 강조:  

$$
       \widetilde{C}_d = W_d \odot C_d.
     $$

3. **DSC 모듈 전체 흐름**  
   - 1라운드 RNN 전파 → attention 적용 → concat → 1×1 conv →  
   - 2라운드 동일 RNN 전파 (weight 공유) → attention → concat → 1×1 conv → ReLU → **출력 DSC 특징**.  

### 2.3 네트워크 구조
- 백본: VGG16 (또는 ResNet-101)  
- 각 백본 레이어(feature map 해상도별)마다 DSC 삽입  
- 각 레이어 출력 upsampling → multi-level integrated features(MLIF) → deep supervision으로 각 레이어와 최종 fusion 예측  
- 검출: weighted cross-entropy loss (shadow/non-shadow imbalance 보정)  
- 제거: shadow-free 이미지를 target으로 Euclidean loss  
- **색 보상**: ground-truth shadow-free 이미지 $$I_n$$를 non-shadow 영역에서 input $$I_s$$와 최소제곱으로 맞추는 선형 변환 $$T_f$$ 적용 후 학습

## 3. 성능 향상 및 한계
- **검출**: SBU(0.97 acc, BER 5.59)·UCF(0.95/10.38) 최고치  
  -  Attention 전파 라운드 2회, weight 공유가 핵심  
- **제거**: SRD RMSE 6.21, ISTD 6.67로 기존 최고치 경신  
- **일반화 성능**  
  -  SBU 훈련→UCF 테스트에서도 성능 유지 → 방향별 글로벌 문맥 학습이 다양한 장면 대비 내성 제공  
- **한계**  
  -  GPU 메모리 제약으로 shallow·deep 레이어 균형 필요  
  -  **소프트·미세 그림자**나 **복합 배경 내 조각 그림자**에서 여전히 오류  
  -  color compensation 선형 가정의 한계

## 4. 모델 일반화 및 확장 가능성
- **방향별 공간 문맥**은 그림자 외에도 *saliency detection*, *semantic segmentation* 등 다양한 픽셀 수준 과제에 적용 가능  
- RNN 기반 전역 문맥 처리 모듈을 다른 백본·경량화 구조(Transformer, MobileNet)에도 삽입하여 **저연산 디바이스** 대응  
- **비선형 색 보정** 또는 **adversarial color consistency**로 제거 품질 추가 향상 여지  

## 5. 향후 연구 및 고려사항
- **비디오 시퀀스**: 시공간적 그림자 변화 모델링  
- **Non-linear color transfer**: 복잡한 조명 변화 대응  
- **희소 데이터 상황**: self-supervised 또는 domain adaptation으로 라벨 부족 문제 완화  
- **경량화·실시간**: AR/VR, 모바일 카메라 적용을 위한 실시간 디텍션/제거 모듈 최적화  
- **다중 모달**(Depth, IR) 융합 탐구로 극한 조명 환경 대응  

위와 같은 방향으로 연구를 확장하면, DSC 모듈의 강점을 살려 다양한 비전 과제에서 **글로벌 문맥 인지력**을 크게 향상시킬 수 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/049679e7-e2d8-4902-8a08-7f6bd0765d44/1805.04635v2.pdf)
