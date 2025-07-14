# Deepfake Generation and Detection: A Benchmark and Survey

**핵심 주장 및 주요 기여**  
이 논문은 **딥페이크 생성과 탐지 분야의 통합적 현황**을 처음으로 광범위하게 정리하고,  
1) 주요 **태스크 정의(얼굴 교환, 재연, 대화 얼굴 생성, 속성 편집, 위변조 탐지)**의 통일화  
2) 대표 **데이터셋·평가 지표** 비교  
3) **GAN·VAE·확산(diffusion) 모델** 발전사 및 신기술(확산 기반 방법) 포함  
4) 분야별 **벤치마크**(최신 기법 대조 평가)  
를 수행했다[1].  

## 1. 해결하고자 하는 문제  
- **분산된 연구 분류**: 얼굴 관련 딥페이크 기술과 탐지 연구가 부분적으로 분절됨  
- **평가 일관성 부재**: 데이터셋·평가지표가 논문마다 달라 공정 비교 어려움  
- **최신 확산모델 미포함**: 기존 조사에서는 GAN/VAE 중심, 최근 확산모델 현황 미반영  

이를 위해 논문은 딥페이크 생성·탐지 태스크를 통합 정의하고, 대표 데이터와 지표로 일관된 벤치마크를 제시한다[1].  

## 2. 제안하는 방법론  
### 2.1 통합 과제 정의  
- **생성(Generation)**:

  $$I_o = \phi_G(I_t, C) $$
  
  여기서 $$I_t$$는 대상 입력 이미지, $$C$$는 소스 이미지·오디오·텍스트 등의 조건, $$\phi_G$$는 생성 네트워크.  
- **탐지(Detection)**:

  $$S_o = \phi_D(I_o) $$
  
  위변조 여부를 점수화하는 분류 네트워크 $$\phi_D$$.  

### 2.2 모델 구조 분류  
- **Face Swapping**: 전통 그래픽 → GAN 기반(Identity/Attribute 디소결) → 3DMM 통합 → 확산모델 기반  
- **Face Reenactment**: 3DMM 기반 → 랜드마크 매칭 → 특성 디소결 → self-supervised  
- **Talking Face Generation**: Audio/Text 주도 → 멀티모달 조건화 → 확산모델 적용 → NeRF/3D 모델  
- **Facial Attribute Editing**: GAN 기반 스타일·속성 분리 → Transformer 통합 → 확산모델·NeRF 결합  
- **Forgery Detection**: 공간·시간·주파수 도메인 단서 → 오디오-비주얼 불일치 → 데이터드리븐 학습  

# Deepfake Inspections: A Survey

딥페이크(DG) 생성 기술과 탐지 기술을 **기술적 관점**에서 체계적으로 조망하는 본 Survey의 3장(Deepfake Inspections)은 **5개 주요 분야**로 구성됩니다. 각 분야별 주요 접근법, 발전 흐름, 기법별 장·단점을 정리하면 다음과 같습니다.

## 3.1 딥페이크 생성(Deepfake Generation)

### 3.1.1 얼굴 교환 (Face Swapping)
-  전통 그래픽  
  – 키포인트 매칭∙혼합, 3DMM 기반(Blanz et al., Dale et al.)  
  – 제약: 포즈·조명 유사성 필요, 낮은 해상도, 잔여 인공물  
-  GAN 기반  
  – 디소결(InfoSwap), ID 주입(SimSwap), 3D 정합(HifiFace, FlowFace)  
  – 해상도 향상(MegaFS), 마스크 기반 스티칭(FSGAN)  
  – 한계: 일반화, Occlusion, ID-속성 균형 어려움  
-  확산(Diffusion) 기반  
  – 조건부 인페인팅(DiffSwap), 정체성·표정 균형(Liu et al.), 범용 모델(FaceX)  
  – 특징: 훈련 안정성, 고품질 세부 묘사, 무거운 연산  

### 3.1.2 얼굴 재연 (Face Reenactment)
-  3DMM 기반  
  – 모션 필드+렌더링(Face2Face, Head2Head)  
  – 계층적 모션 네트워크(PECHead)  
-  랜드마크 정합  
  – 메타-학습(Free-HeadGAN), 어텐션 블록(MarioNETte)  
  – 한계: 대각 포즈에서 왜곡  
-  특성 디소결  
  – 스타일GAN 잠재공간 활용(HyperReenact, StyleMask)  
  – NeRF 활용(HiDe-NeRF)  
-  Self-supervised  
  – 레이블 없는 실영상 활용(Zhang et al., Oorloff et al.)  
  – 장점: 레이블 비용 절감, 견고성 향상  

### 3.1.3 말하는 얼굴 생성 (Talking Face Generation)
-  오디오·텍스트 주도  
  – 피처 분리+LSTM(Wav2Lip, MakeItTalk)  
  – 감정 표현 통합(EMMN, AMIGO)  
  – 한계: 포즈·표정∙감정 제어 미흡  
-  멀티모달 조건화  
  – 음성·표정·포즈 독립 제어(GC-AVT, PC-AVS)  
  – 한계: 복잡 배경, 실시간성 낮음  
-  Diffusion 기반  
  – DAE-Talker, DreamTalk, EmoTalker  
  – 장점: 잠재공간 학습 통한 세부 묘사, 감정 편집성  
  – 단점: 모델 복잡도, 제어 신호 부족 시 왜곡  
-  3D 모델(NeRF 등)  
  – AD-NeRF, AE-NeRF, SyncTalk  
  – 장점: 3D 구조 일관성  
  – 단점: 제어 세밀도, 렌더링 비용  

### 3.1.4 얼굴 속성 편집 (Facial Attribute Editing)
-  GAN 기반 다속성 편집  
  – 잠재공간 분리(AttGAN, TransEditor)  
  – 속성 손실 최소화(HifaFace, GuidedStyle)  
-  텍스트 주도  
  – TextFace, TG-3DFace  
  – CLIP 기반 문장–속성 정합  
-  Diffusion 기반  
  – 협업 확산(Huang et al.), 3DMM+확산(DiffusionRig)  
-  GAN+NeRF  
  – FENeRF, CIPS-3D++  
  – 장점: 공간 일관성, 시맨틱 정합성  

## 3.2 위변조 탐지 (Forgery Detection)

딥페이크 탐지는 **공간∙시간∙주파수 도메인**과 **데이터 중심**으로 크게 나뉩니다.

### 3.2.1 공간 도메인 (Space Domain)
-  픽셀·텍스처 불일치  
  – 경계 아티팩트(Face X-ray), 텍스처 통계(Gram-Net)  
-  노이즈 패턴  
  – 국소 잡음 불균형(NoiseDF), 캡슐망(Nguyen et al.)  

### 3.2.2 시간 도메인 (Time Domain)
-  생체 신호 이상  
  – 눈 깜빡임(FRL), 시선일관성(Peng et al.)  
-  프레임간 불연속  
  – 픽셀∙스타일 변화(FTCN, Choi et al.)  
  – 그래프 분류(Yang et al.)  
-  멀티모달 불일치  
  – 음성–영상 불일치(AVoiD-DF, POI-Forensics)  

### 3.2.3 주파수 도메인 (Frequency Domain)
-  주파수 분포 차이(F3-Net, HFI-Net)  
-  공간–주파수 상호작용(Guo et al.)  

### 3.2.4 데이터 중심 (Data Driven)
-  모델 지문  
  – GAN fingerprint(Yu et al.)  
-  신경망 행태  
  – 뉴런 커버리지(FakeSpotter)  
-  자기일관성 학습  
  – Pairwise self-consistency(Zhao et al.)  

## 3.3 연관 분야 간략 언급

-  얼굴 초해상 (Face Super-resolution): CNN→GAN→Diffusion  
-  스타일 전송 (Portrait Style Transfer): GAN[1]→DiffusionGAN3D  
-  신체 애니메이션 (Body Animation): GAN→Diffusion[100, 메이크업 전이 (Makeup Transfer): 그래픽→GAN→Transformer  

--- 

 **각 태스크별 기술 분류**, **대표 기법·데이터셋·지표**, **장단점**을 종합해 **딥페이크 생성·탐지 기술의 기술적 지형도**를 그립니다. 이러한 분류와 분석은 새로운 기법 개발 시 **적합한 벤치마크 선택**, **이슈 극복 전략** 수립에 핵심적 인사이트를 제공합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f0f93c14-fb19-4189-ab0d-38a4b5bec48b/2403.17881v4.pdf

### 2.3 성능 향상 포인트  
- **확산 모델(diffusion)**: GAN 대비 고해상도·세부 표현력↑, 훈련 안정성↑  
- **디소결(Disentanglement)**: ID·속성 분리로 얼굴 교환 시 **정체성 보존**과 **속성 유지** 균형 향상  
- **3D 정보 통합**(3DMM·NeRF·Tri-plane): 다각도, 포즈 변화 대응력↑  
- **멀티모달 조건화**: 오디오·텍스트·표정·포즈 정보를 동시 활용해 자연스러운 동영상 생성  

## 3. 성능 향상 및 한계  
- **벤치마크 결과**:  
  - GAN–Diffusion 하이브리드 기법이 **FID**, **ID Ret.**, **Pose/Expression Error** 등 주요 지표에서 상향  
  - 탐지 모델은 다중 도메인(공간·시간·주파수·오디오-비주얼) 결합 시 **AUC/ACC** 전반적 개선  

- **한계**  
  - **일반화(Generalization)**: 훈련-테스트 도메인 불일치 시 성능 하락  
  - **벤치마크 통일성 부족**: 새로운 데이터·지표 등장 시 지속 업데이트 필요  
  - **실시간성·경량화**: 대용량 확산모델·3D구조 모델의 추론 비용 높음  

## 4. 일반화 성능 향상 방안  
- **도메인 적응(Domain Adaptation)**: 다양한 환경·압축률·조명 시나리오 학습  
- **멀티태스크 학습**: 여러 위변조 유형 동시 인식 → 특징 표현 범용성↑  
- **Self-supervised 학습**: 레이블 없는 실제 영상 활용해 robust 표현 학습  
- **주파수·공간 결합**: 소수 훈련 샘플로도 일반화 가능한 주파수 특성 병합  

## 5. 향후 연구 영향 및 고려사항  
- **표준 벤치마크 유지**: 지속적인 데이터·지표 업데이트로 공정 비교 체계 정립  
- **확산모델 최적화**: 실시간 적용 가능토록 경량화·속도 개선 연구  
- **보안·윤리 고려**: 생성과 탐지 기술 발전의 사회적·윤리적 영향 평가  
- **교차모달·교차도메인**: 얼굴 외 신체·장면 전반으로 확산 기술 및 탐지 기법 확장  

앞으로의 딥페이크 연구는 **강인한 일반화**, **실시간 처리**, **윤리·보안 조치**라는 세 축을 중심으로 발전할 것이다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f0f93c14-fb19-4189-ab0d-38a4b5bec48b/2403.17881v4.pdf
