# Video Super-Resolution Based on Deep Learning: A Comprehensive Survey

## **핵심 주장과 주요 기여**
이 논문은 딥러닝 기반 비디오 초해상도(Video Super-Resolution) 분야의 **첫 번째 종합적 서베이 논문**으로, 37개의 최신 방법론을 체계적으로 분석하고 분류한 연구입니다.

**주요 기여점:**
1. **새로운 분류 체계(Taxonomy) 제안**: Inter-frame 정보 활용 방식에 따른 7개 하위 카테고리 분류
2. **포괄적 분석**: 37개 최신 VSR 방법들의 구조, 구현 세부사항, 성능 비교
3. **벤치마크 성능 정리**: 주요 데이터셋에서의 정량적 성능 비교 분석
4. **응용 분야 정리**: 의료영상, 감시시스템, VR 등 다양한 활용 분야 제시
5. **향후 연구 방향 제시**: 8가지 주요 도전과제 및 연구 방향 제안
## **해결하고자 하는 문제와 제안 방법**
### **문제 정의**
비디오 초해상도는 저해상도(LR) 비디오로부터 고해상도(HR) 비디오를 복원하는 문제입니다. 단일 이미지 SR과 달리 **연속된 프레임 간의 시간적 정보를 효과적으로 활용**하는 것이 핵심입니다.

### **수식적 정의**
**비디오 열화 과정:**

$$I_i = \phi(\hat{I}\_i, \{\hat{I}\_j\}\_{j=i-N}^{i+N}; \theta_\alpha)$$

구체적으로는:

$$I_j = DBE_{i\rightarrow j}\hat{I}_i + n_j$$

여기서:
- $$I_i$$: i번째 저해상도 프레임
- $$\hat{I}_i$$: i번째 고해상도 프레임  
- $$D$$: 다운샘플링 연산, $$B$$: 블러 연산
- $$E_{i\rightarrow j}$$: 프레임 i에서 j로의 워핑 연산
- $$n_j$$: 노이즈

**복원 과정:**

$$\tilde{I}\_i = \phi^{-1}(I_i, \{I_j\}\_{j=i-N}^{i+N}; \theta_\beta)$$

# Video Super-Resolution Methods (Section 3)

비디오 초해상도(Video Super-Resolution, VSR)는 연속된 저해상도(LR) 프레임들로부터 고해상도(HR) 비디오를 복원하는 기술입니다. 이미지 초해상도(SISR)와 달리, 비디오는 **프레임 간의 시간적 정보**(motion, temporal consistency)를 활용해야 하므로 훨씬 더 복잡합니다.  

Section 3에서는 “**프레임 정렬(Alignment) 유무**”와 “**초해상도 과정에서 핵심적으로 쓰이는 기술**”에 따라 VSR 기법을 **두 가지 큰 범주**로 나누고, 그 아래 **7개 소분류**로 정리합니다.  

## 1. 두 가지 큰 범주

1. **Methods with Alignment (정렬 기반 방법)**  
   - 이 방법들은 먼저 인접 프레임을 타깃 프레임과 **정렬**(Alignment)한 뒤 초해상도를 수행합니다.  
   - 프레임 정렬을 위해 주로 **Motion Estimation & Motion Compensation (MEMC)** 혹은 **Deformable Convolution** 기법을 사용합니다.

2. **Methods without Alignment (비정렬 기반 방법)**  
   - 프레임을 별도로 정렬하지 않고, 딥러닝 네트워크가 **자체적으로** 공간(spatial) 및 시공간(spatio-temporal) 정보를 학습하도록 합니다.  
   - Alignment 단계가 없으므로 구조가 단순하지만, 네트워크 설계가 더욱 중요해집니다.

## 2. 소분류: 총 7가지

소분류는 “Alignment 유무” 및 “핵심 모듈”에 따라 나뉩니다.

1. **Alignment 기반 방법 (Methods with Alignment)**  
   a. Motion Estimation & Compensation (MEMC)  
   b. Deformable Convolution Alignment  

2. **Alignment 없이 학습하는 방법 (Methods without Alignment)**  
   a. 2D Convolution (순수 공간적 처리)  
   b. 3D Convolution (시공간 동시 처리)  
   c. Recurrent CNN (RCNN)  
   d. Non-Local Network (전역적 상관관계)  
   e. 기타 복합 접근 (Other)

각 소분류의 특징을 간단히 정리하면 다음과 같습니다.

| 소분류 이름                  | 특징                                                                                 |
|---------------------------|------------------------------------------------------------------------------------|
| 1a. MEMC                  | ● 광학 흐름(Optical Flow)로 프레임 간 움직임 추정● 추정된 흐름으로 이웃 프레임 워핑 |
| 1b. Deformable Convolution | ● CNN 레이어에 학습 가능한 오프셋 적용● 복잡한 움직임에 유연하게 대응                   |
| 2a. 2D Convolution         | ● 각 프레임을 2D 필터로 처리● 순수 공간 공간적 정보만 사용                            |
| 2b. 3D Convolution         | ● 3D 필터로 공간+시간 정보 동시 학습● 일종의 짧은 시퀀스 처리                          |
| 2c. Recurrent CNN (RCNN)   | ● 순환 구조(RNN/LSTM)로 장기 시간 의존성 모델링● 앞뒤 프레임 정보 순환적으로 활용          |
| 2d. Non-Local Network      | ● 전역적 위치 간 상관관계를 계산하는 블록 삽입● 국소 필터 한계를 벗어나 장거리 의존성 포착      |
| 2e. Other                  | ● 위 분류에 딱 들어맞지 않는 하이브리드·특수 기법들                                  |

## 3. 왜 이런 분류인가?

- **정렬(Alignment)** 단계를 명시적으로 두면, 네트워크가 프레임 간 움직임을 정확히 보정해 주어야 하며, 잘못된 흐름 추정(Flow) 시 화질 저하나 아티팩트가 생길 수 있습니다.  
- 반면 **비정렬** 방식은 네트워크가 **스스로** 특징을 학습하기 때문에, 정렬 오류에 덜 민감하지만, 더 강력한 네트워크 구조와 학습이 필요합니다.  

정리하자면, VSR은  
1) **이웃 프레임을 얼라인**한 뒤 복원할지,  
2) **프레임 얼라인 없이** 시공간 정보를 망막처럼 처리할지  
라는 두 가지 큰 갈림길이 있고, 이후에 쓰는 핵심 모듈(Optical Flow, Deformable Conv, 3D Conv, RCNN, Non-Local 등)에 따라 다양한 세부 방법이 탄생했습니다.  

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daeb3764-3a5c-4d45-98f2-1fbe93663e67/2007.12928v3.pdf

### **제안하는 분류 체계**
논문은 inter-frame 정보 활용 방식에 따라 VSR 방법들을 두 가지 주요 범주로 분류합니다:

**1. Methods with Alignment (정렬 기반 방법)**
- **MEMC (Motion Estimation & Motion Compensation)**: 17개 방법
  - 광학 흐름 추정으로 움직임 정보 추출
  - 움직임 보상을 통한 프레임 정렬
  - 예시: Deep-DE, VSRnet, VESPCN, FRVSR, BasicVSR

- **Deformable Convolution**: 5개 방법  
  - 학습 가능한 오프셋으로 유연한 수용장 구성
  - 복잡한 움직임과 조명 변화에 강인
  - 예시: EDVR, DNLN, TDAN, D3Dnet

# 정렬 기반 비디오 초해상도(Video Super-Resolution with Alignment)

정렬 기반 방법(Methods with Alignment)은 **인접 프레임(frame)을 목표 프레임(target frame)에 맞춰 정렬(alignment)**한 뒤, 이 정렬된 정보를 바탕으로 고해상도 영상을 복원하는 기법입니다. 대표적으로 두 가지 핵심 기술이 사용됩니다.

## 1. Motion Estimation & Motion Compensation (MEMC) 기반 방법  

### 1.1 개념  
1) **Motion Estimation (ME)**  
- 인접 프레임 간 픽셀 단위 움직임(광학 흐름, optical flow)을 추정  
- 전통 기법(예: Lucas–Kanade), 혹은 FlowNet, PWC-Net 같은 딥러닝 모델 사용  

2) **Motion Compensation (MC)**  
- 추정된 흐름 정보를 이용해 인접 프레임을 목표 프레임에 워핑(warping)  
- Bilinear interpolation, Spatial Transformer Network 등으로 구현  

### 1.2 분류  
- **전통 MEMC**: ME·MC 모두 전통 기법  
  - 예: Deep-DE, VSRnet, RRCN  
- **딥러닝 MEMC**: ME 또는 MC 단계에 CNN 모듈 도입  
  - 예: VESPCN, FRVSR, TOFlow, MEMC-Net, BasicVSR  

### 1.3 대표 모델  

| 모델    | ME/MC 구조                                    | 특징                                                                                          |
|--------|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| VESPCN | Coarse-to-fine CNN 기반 MCT 모듈               | 두 단계로 흐름 예측 → 워핑 → 재예측 후 재워핑; Sub-pixel convolution으로 업샘플링           |
| FRVSR  | 이전 HR 결과 순환 입력 + FlowNet 기반 흐름 예측 | 높은 시간 일관성; 이전 프레임 HR 정보를 활용해 연산량 ↓                                      |
| TOFlow | SpyNet 흐름 예측 + STN 기반 워핑               | 과제(task)-특화(optimal) 흐름 학습, SR/Interpolation/Deblur 등 다중 태스크에 적용 가능      |
| MEMC-Net| FlowNet+U-Net 기반 Kernel estimation + adaptive warping | 흐름·커널 동시 추정→프레임 별 가변 커널 워핑; ResNet18 특징 맥락(context) 활용하여 강화 |

## 2. Deformable Convolution 기반 방법  

### 2.1 개념  
- **Deformable Convolution (DConv)**  
  - 기존 고정격자(convolution grid)에 학습 가능한 오프셋(offset)을 더해  
  - 움직임·조명 변화에 유연하게 대응하며 정렬 수행  

### 2.2 핵심 모듈 흐름  
1) 타깃 프레임과 인접 프레임의 특징 맵(feature map) 연결  
2) 작은 CNN으로 **오프셋(offset)** 추정  
3) 이 오프셋을 기존 컨볼루션 필터에 적용→**가변적 수용 영역**으로 정렬  
4) 후속 레이어로 복원·업샘플링  

### 2.3 대표 모델  

| 모델    | Alignment 모듈                    | 특징                                                          |
|--------|----------------------------------|-------------------------------------------------------------|
| EDVR   | Pyramidal Cascading Deform. (PCD) | 피라미드 구조로 다중 스케일 특징 정렬 → TSA(Temporal-Spatial Attention)로 프레임 융합 |
| DNLN   | Cascaded Deform. + Non-Local Att.  | 계층적(offset) 예측 + 글로벌 비국소(non-local) 상관관계 반영             |
| TDAN   | Temporally Deformable Alignment   | 단일 레벨에서 타깃·이웃 특징 매칭→Offset 학습→워핑                          |
| D3Dnet | 3D Deformable Convolution         | 시공간(3D) DConv로 프레임 정렬 + 공간 특징 통합                          |

## 3. 정렬 기반 방법의 장·단점 및 선택 가이드  

### 장점  
- 인접 프레임 간 움직임을 명시적으로 보정 → 시간적 일관성 확보  
- 광학 흐름·가변 커널 등을 활용해 복잡한 모션에도 대응  

### 단점  
- 흐름 추정 오류 시 부정확한 정렬 → 아티팩트 발생  
- DConv나 Non-Local 연산의 계산 부담  
- 실시간·경량화·하드웨어 구현 시 제약  

### 선택 가이드  
- **정밀한 정렬이 관건** → MEMC-Net, EDVR (DConv) 추천  
- **실시간/경량화** → FRVSR (순환구조), TDAN (단일 스케일 DConv)  
- **장기 의존성 활용** → BasicVSR (양방향 RNN)  
- **글로벌 상관관계 강화** → DNLN, EDVR(TSA)  

정렬 기반 VSR은 **프레임 정렬** 단계가 핵심이며, 이 정렬의 정확도와 효율성을 높이는 MEMC·Deformable Conv 기법이 각각 발전해 왔습니다. 각 방법의 특징과 계산량을 고려해, 사용 환경(정확도·속도·리소스 제약 등)에 맞춰 최적의 모델을 선택하실 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daeb3764-3a5c-4d45-98f2-1fbe93663e67/2007.12928v3.pdf

**2. Methods without Alignment (비정렬 기반 방법)**
- **2D Convolution**: 공간적 정보만 활용 (2개)
- **3D Convolution**: 시공간 정보 동시 처리 (4개)
- **Recurrent CNN**: 장기 시간 의존성 모델링 (5개)
- **Non-Local**: 전역 상관관계 활용 (1개)
- **Other**: 기타 복합적 접근법 (3개)

# Methods without Alignment

Methods without Alignment(비정렬 기반 방법)은 **인접 프레임 간에 별도의 정렬(Alignment) 과정을 거치지 않고**, 네트워크가 스스로 시공간(spatio-temporal) 정보를 학습하도록 설계된 VSR 기법들입니다. 정렬 모듈이 없으므로 구조는 단순하지만, **공간·시간 정보를 효과적으로 추출·융합하는 네트워크 설계**가 핵심입니다. 이들은 크게 다섯 가지로 분류할 수 있습니다.  

## 1. 2D Convolution 방식  
- **아이디어**: 연속된 프레임을 그대로 2D 컨볼루션 네트워크에 입력 → 순수 공간적 특징 추출·융합  
- **장점**: 구조가 단순, 메모리·연산량 비교적 작음  
- **단점**: 시간 정보 활용이 간접적(네트워크 깊이·수용 영역에 의존)  
- **대표 모델**  
  - *VSRResFeatGAN* : GAN을 도입해 공간적 특징 강조, 시공간 의존성은 후방 네트워크가 학습  
  - *FFCVSR* : 프레임 간 유사도(컨텍스트)와 로컬 정보를 분리해 병렬 처리  

## 2. 3D Convolution 방식  
- **아이디어**: 3D 필터(k×k×k)로 공간+시간 정보를 동시에 학습  
- **장점**: 프레임 간 단기 의존성(움직임) 직접 처리  
- **단점**: 계산·메모리 비용 증가, 장기 의존성 포착은 한계  
- **대표 모델**  
  - *DUF* : 입력마다 동적 업샘플링 필터 생성 + 3D 컨볼루션으로 잔차 맵 학습  
  - *FSTRN* : k×k×k 컨볼루션을 1×k×k + k×1×1 분해 → 연산량 절감  
  - *3DSRnet* : 장면 전환 인식용 서브넷 도입 → 씬 변경 시 프레임 교체 후 3D SR  

## 3. Recurrent CNN(RCNN) 방식  
- **아이디어**: RNN/LSTM 구조로 순환 처리 → 장·단기 시간 의존성 모델링  
- **장점**: 긴 프레임 간 문맥 포착, 파라미터 효율적  
- **단점**: 훈련 난이도↑(그래디언트 소실), 매우 긴 시퀀스 처리 시 성능 한계  
- **대표 모델**  
  - *BRCN* : 순·역방향 양방향 순환망으로 과거·미래 정보 동시 활용  
  - *STCN* : ConvLSTM으로 시공간 특징 학습  
  - *RISTN* : 가역 블록(RIB) + 잔차-밀집 ConvLSTM → 공간·시간 특징 융합  
  - *RLSP* : 은닉 상태에 이전 HR 결과 전가 → 순환적 정보 전파  
  - *RSDN* : 구조(Structure) vs 세부(Detail) 분리 → 두 흐름별 처리 후 정보 교환  

## 4. Non-Local 방식  
- **아이디어**: 전역적 위치 간 상관관계 계산(Non-Local 블록) → 국소 CNN 한계 극복  
- **장점**: 장거리 시공간 의존성 → 복잡한 모션 포착  
- **단점**: O(N²) 연산 비용 발생 → 비용 절감 연구 필요  
- **대표 모델**  
  - *PFNL* : Non-Local ResBlock로 특징 추출 → Progressive Fusion 블록으로 시공간 융합  

## 5. 그 외(Other)  
- **아이디어**: 위 분류 외 하이브리드·특수 기법  
- **대표 모델**  
  - *RBPN* : 다중 이미지 SR(MISR) + 단일 이미지 SR(SISR)을 반복적(back-projection)으로 결합  
  - *STARnet* : SR + 보간(Interpolation) 동시 수행 → 공간·시간 다중 해상도 네트워크  
  - *DNSTNet* : 짧은/긴 시퀀스용 3D Conv + ConvLSTM + Region-level Non-Local → 정보 집약  

――――――  
**핵심 요약**  
Methods without Alignment는 **정렬 모듈 부재 대신 네트워크 설계로 시공간 정보를 학습**합니다.  
● 2D Conv: 구조 단순·경량  
● 3D Conv: 단기 모션 직접 포착  
● RCNN: 장기 의존성 학습  
● Non-Local: 전역 상관관계 활용  
● Other: 복합·특수 하이브리드  

사용 환경(계산 리소스, 모션 크기, 시퀀스 길이 등)에 따라 위 다섯 축에서 적합한 모델을 선택하면 됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daeb3764-3a5c-4d45-98f2-1fbe93663e67/2007.12928v3.pdf

## **성능 향상 및 한계점**
### **성능 분석 결과**
**Top 5 방법론 (×4 배율, Vimeo-90K-T):**
- IconVSR: 37.84 dB (BD), 37.47 dB (BI)
- EDVR: 37.61 dB (BI)  
- MuCAN: 37.32 dB (BI)
- RSDN: 37.23 dB (BD)

**모델 효율성 분석:**
- EDVR: 20.60 MB (고성능이지만 큰 모델)
- IconVSR: 8.70 MB (효율성과 성능의 균형)
- 경량 모델들: 3-6 MB 범위에서도 우수한 성능

### **주요 한계점**
**1. 방법론별 한계:**
- **MEMC 방법들**: 큰 움직임이나 조명 변화 시 optical flow 추정 부정확
- **Deformable Convolution**: 높은 계산 복잡도와 까다로운 수렴 조건
- **3D Convolution**: 높은 계산 비용으로 실시간 처리에 제약
- **Recurrent Networks**: 그래디언트 소실 문제와 장기 의존성 포착의 어려움

**2. 전반적 한계:**
- 대부분 bicubic degradation에 최적화 (실제와 괴리)
- 고정 배율(×2, ×4)에 특화되어 범용성 부족
- 장면 변화가 있는 비디오 처리 미흡

## **일반화 성능 향상 관련 내용**
### **현재 일반화 성능의 문제점**
**1. 합성 데이터 의존성:**
- 대부분 bicubic/Gaussian blur로 생성된 LR-HR 쌍으로 훈련
- 실제 비디오의 복잡한 열화 과정과 괴리
- Real-world 성능 저하의 주요 원인

**2. 도메인 특화 문제:**
- 특정 데이터셋/도메인에 과적합
- 다양한 비디오 콘텐츠에 대한 강인성 부족
- 고정 배율에만 최적화되어 실용성 제한

### **일반화 성능 향상 방안**
**1. 현실적 열화 모델링:**
- 복잡한 실제 열화 과정의 정확한 모델링 필요
- 다양한 노이즈, 블러, 압축 아티팩트 동시 고려

**2. 비지도 학습 접근:**
- 페어 데이터 없이도 학습 가능한 방법론 개발
- Self-supervised learning의 적극적 활용

**3. 도메인 적응 기법:**
- Transfer learning 및 meta-learning을 통한 다양한 비디오 도메인 적응
- 범용적 특성 학습을 위한 구조적 개선

## **향후 연구에 미치는 영향과 고려사항**
### **연구에 미치는 영향**
**1. 학술적 영향:**
- VSR 연구의 현황과 동향을 체계적으로 정리하여 연구 방향 제시
- 새로운 연구자들의 진입 장벽 완화 및 방법론 선택 가이드 제공
- Inter-frame 정보 활용의 중요성을 부각시켜 관련 연구 촉진

**2. 기술적 영향:**
- 각 방법론의 장단점을 명확히 제시하여 하이브리드 접근법 연구 촉진
- 성능 벤치마킹을 통한 객관적 평가 기준 제공

### **향후 연구 시 고려할 점**
**1. 우선순위 연구 과제:**
- **현실적 열화 모델링**: 실제 비디오 열화의 복잡성을 반영한 훈련 데이터 구축
- **임의 배율 지원**: 고정 배율의 한계를 극복한 범용 모델 개발
- **경량화**: 모바일 기기에서 실행 가능한 효율적 구조 설계

**2. 새로운 연구 방향:**
- **비지도 학습**: 페어 데이터셋 구축의 어려움을 해결할 수 있는 방법론
- **인간 지각 기반 평가**: PSNR/SSIM의 한계를 극복한 새로운 평가 지표
- **대배율 초해상도**: ×8, ×16 등 대배율 처리 능력

**3. 실용적 고려사항:**
- 실시간 처리를 위한 계산 효율성과 성능의 균형점 탐색
- 다양한 하드웨어 플랫폼(GPU, 모바일, 임베디드)에서의 최적화
- 장면 변화가 빈번한 실제 비디오에 대한 강인성 확보

이 논문은 VSR 분야의 **종합적 로드맵**을 제공함으로써, 향후 연구가 더욱 체계적이고 목표 지향적으로 진행될 수 있는 기반을 마련했습니다. 특히 일반화 성능 향상과 실용성 확보가 차세대 VSR 연구의 핵심 과제임을 명확히 제시했다는 점에서 큰 의미가 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/daeb3764-3a5c-4d45-98f2-1fbe93663e67/2007.12928v3.pdf
