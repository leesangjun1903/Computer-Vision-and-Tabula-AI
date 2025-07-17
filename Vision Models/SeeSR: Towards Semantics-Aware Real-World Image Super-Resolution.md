# SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution | Super resolution

**핵심 주장 및 주요 기여**  
SeeSR은 실제 저해상도(LR) 이미지에서 고해상도(HR) 복원을 수행할 때, 텍스트-투-이미지(T2I) 사전 학습된 확산 모델의 생성적 잠재력을 끌어내되, 왜곡된 입력으로 인한 의미 오류(semantic distortion)를 최소화하는 **의미 인지(semantics-aware)** 접근법을 제안한다[1].  
1. **열화 인지 프롬프트 추출기(DAPE)**: HR 이미지에서의 고품질 태그(prompt)를 LR 이미지에서도 신뢰성 있게 예측하도록 학습하여, 객체별 태그(“hard prompts”)와 연속 표현(“soft prompts”)을 함께 뽑아낸다[1].  
2. **LR 잠재 임베딩(LRE)**: 초기 샘플링 노이즈에 LR 이미지의 잠재 표현을 주입하여, 무작위 세부 묘사 생성 경향을 억제하고 스무스 영역의 부자연스러운 아티팩트를 줄인다[1].  
3. **통합 제어망(ControlNet + RCA)**: T2I 모델(Unet 기반 Stable Diffusion)의 인코더와 교차-어텐션 모듈에 DAPE가 제공하는 프롬프트를 결합, 세밀하고 의미 일관성 있는 SR을 유도한다[1].  

## 1. 해결 문제  
- **실제 세계 SR(Real-ISR)에서의 의미 오류**  
  - GAN 기반 Real-ISR은 과도한 아티팩트  
  - 기존 확산 기반 방법(StableSR, DiffBIR)은 LR만 제어 신호로 사용해 의미 왜곡 발생  
  - 간단 프롬프트(PASD)의 태깅·캡션은 강열화 시 오류 잦음  

## 2. 제안 방법

### 2.1. 열화 인지 프롬프트 추출기(DAPE)  
- **목표**: LR 입력 $$y$$의 임베딩 $$f_{\text{rep}}(y)$$와 로짓 $$f_{\text{logits}}(y)$$를 HR의 $$f_{\text{rep}}(x), f_{\text{logits}}(x)$$에 근접토록 학습  
- **손실**:  

$$
\mathcal{L}\_{\text{DAPE}} = \|f_{\text{rep}}(y)-f_{\text{rep}}(x)\|^2_2 +\lambda\mathrm{CE}(f_{\text{logits}}(y),f_{\text{logits}}(x))
$$

[1]  

- **Hard prompts**: 임계값 기반 태그(“airplane”, “building” 등)  
- **Soft prompts**: 연속 임베딩(f_rep)  

### 2.2. 제어 확산 모델  
- **ControlNet**: Stable Diffusion Unet 인코더 복제체로 LR 잠재 $$z_{\text{lr}}$$를 입력  
- **Representation Cross-Attention (RCA)**: Soft prompts를 Unet 내부의 텍스트 교차-어텐션 뒤에 결합[1]  
- **학습 목표**:  

$$
\mathcal{L} = \mathbb{E}\_{z_0,z_{\text{lr}},t,p_h,p_s,\epsilon}\bigl\|\epsilon-\epsilon_\theta(z_t,z_{\text{lr}},t,p_h,p_s)\bigr\|^2_2
$$

[1]  

### 2.3. LR 잠재 임베딩(LRE)  
- 훈련 시 사용된 노이즈 스케줄러에 따라 초기 샘플링 노이즈를  
$$\alpha_t z_0 + \sigma_t\epsilon$$ → $$\alpha_t z_{\text{lr}} + \sigma_t\epsilon$$  
로 교체, 불필요한 랜덤 디테일을 억제[1].  

## 3. 모델 구조  
1. **DAPE**  
   - Image encoder (LoRA로 RAM fine-tune)  
   - Tagging head  
2. **ControlNet + RCA**  
   - SD Unet encoder (trainable)  
   - RCA 모듈(soft prompts)  
3. **VAE 인코더**: HR→latent 변환  
4. **Diffusion U-Net**: 입력 = $$(z_t, z_{\text{lr}}, t, p_h, p_s)$$  

## 4. 성능 향상  
- **전반적 SR 품질**  
  - DIV2K-Val: FID 31.93 (최저), LPIPS·DISTS 최상[Table 2][1]  
  - RealLR200: NIQE 4.1620, MANIQA 0.6254, MUSIQ 69.71, CLIPIQA 0.6813 최고[1]  
- **의미 보존** (COCO-Val 기반)  
  - 물체 검출 AP 21.1 → LR 대비 4× 개선  
  - Panoptic PQ 30.0, 세그멘테이션 mIoU 41.3 등 최고[Table 3][1]  
- **사용자 선호도**  
  - 합성: GT와 혼동률 38.6%로 2위 대비 3배↑  
  - 실제: 57.1% 선택률로 2위 대비 3.5배↑[Table 4][1]  

## 5. 한계  
- **강렬한 열화 시 태그 오류** → 잘못 복원된 객체  
- **프롬프트-영역 정렬 불안정** → 추가 마스크 필요  
- **텍스트 영역 복원 한계**  

## 6. 일반화 성능 및 향후 연구 고려점  
- **DAPE의 도메인 적응력**: 다양한 열화 유형(모션 블러, 압축 노이즈) 학습으로 **일반화** 가능  
- **다중 스케일 태그·어텐션** 추가로 소·대 객체 모두 포착  
- **마스크·세그멘테이션 정보** 통합 시 복원 정확도↑  
- **다른 T2I 사전학습 모델**(e.g., Imagen) 활용 검토  
- **반자동 학습**: 불확실 프롬프트 검증 피드백 루프  

## 7. 향후 영향  
- **SR 연구**: 의미 인지 프롬프트와 확산 제어의 융합 방향 제시  
- **응용**: 자율주행·의료영상 SR에서 의미 보존 강화  
- **확산 모델 제어**: soft+hard 복합 제어 전략 일반화 가능  

SeeSR는 의미 인지 제어를 통해 실제 SR 작업에서 **시각 품질과 의미 보존**을 동시에 달성하는 새로운 패러다임을 제시하였다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/987eea2b-1609-4967-adab-1c51752463d4/2311.16518v2.pdf
