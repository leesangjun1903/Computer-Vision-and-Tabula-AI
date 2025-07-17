# ResShift: Efficient Diffusion Model for Image Super-Resolution by Residual Shifting | Super resolution

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
ResShift는 고해상도(HR)와 저해상도(LR) 이미지 간 잔차(residual)를 단계적으로 이동(shift)시키는 새로운 확산(diffusion) 모델을 도입하여, 기존 확산 기반 SR 기법들의 수백~수천 단계 샘플링에 따른 느린 추론 속도를 15단계로 대폭 단축하면서도 성능 저하를 방지한다고 주장한다.

**주요 기여**  
- HR–LR 이미지 간 짧은 마르코프 체인(Markov chain) 설계로 15단계만에 고품질 SR 달성  
- 잔차 이동(residual shifting) 전이 함수 및 유연한 노이즈 스케줄(ηₜ, κ) 도입  
- VQGAN 잠재공간 활용으로 연산량 절감  
- 합성 및 실제 데이터셋에서 기존 SOTA와 동등 이상 성능 입증  

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
- 기존 확산기반 SR은 가우시안 노이즈로부터 시작해 HR을 복원하므로 수천 단계 필요  
- 가속 샘플러(DDIM 등)는 단계 수를 줄여도 성능(샤프니스, 디테일)을 희생  

### 2.2 제안하는 모델  
#### Forward process  
- LR y₀, HR x₀ 간 잔차 $e₀ = y₀ – x₀$  
- 마르코프 체인 길이 T, 단계별 이동량 ηₜ 증가 순(monotonic), $αₜ=ηₜ–ηₜ₋₁$
- 전이 분포  

$$
q(x_t\mid x_{t-1},y_0)=\mathcal{N}\bigl(x_t;\,x_{t-1}+α_t e_0,\;κ^2α_t I\bigr)
$$ 

- 이로부터  

$$
q(x_t\mid x_0,y_0)=\mathcal{N}\bigl(x_t;\,x_0+η_t e_0,\;κ^2η_t I\bigr)
$$  

#### Reverse process  
- 목표 posterior p(x₀∣y₀) 추정  
- 학습 목표: 각 단계 KL 발산 최소화  

$$
\min_θ\sum_{t} \mathbb{E}_{q}\bigl\|f_θ(x_t,y_0,t)-x_0\bigr\|^2
$$

  – $f_θ$ 는 UNet 기반 네트워크, x₀ 예측  

### 2.3 노이즈 스케줄  
- κ: 전체 노이즈 강도 조절  
- $η₁≈0, η_T≈1$ , 중간 단계 $√ηₜ$ 기하급수적 증가  

$$
\sqrt{η_t} = \sqrt{η_1}\times b_0^{\,β_t},\quad β_t=\bigl(\tfrac{t-1}{T-1}\bigr)^p (T-1)
$$ 

- p로 잔차 이동 속도 제어 → 품질 대 현실감(perception–realism) 트레이드오프  

### 2.4 모델 구조  
- VQGAN latent space에서 확산 진행 (4배 해상도 축소)  
- UNet 기반 백본, Swin Transformer 블록으로 attention 대체  

## 3. 성능 향상 및 한계  

### 3.1 성능 개선  
- **단계 수**: 15단계만으로 기존 LDM-15 대비 PSNR↑, LPIPS↓, CLIPIQA↑[Table 3]  
- **속도**: LDM-100 대비 4배 빠른 추론(0.105s vs. 0.413s) 유지[Table 2]  
- **합성/실제 데이터**: ImageNet-Test, RealSR, RealSet65에서 CLIPIQA·MUSIQ 최고 성능  
- **Perception–Distortion**: LDM 대비 항상 더 낮은 왜곡에 더 나은 지각 품질 견지(Fig. 7)  

### 3.2 한계  
- **추론 속도**: 여전히 GAN계열보다 느림 (0.105s vs. 0.012s)  
- **극단적 열화**: 매우 심하게 손상된 실사진(만화, 문자 등) 복원 실패 사례 존재(Fig. 9)  

## 4. 일반화 성능 및 향후 연구 고려사항  
- **노이즈 스케줄 유연성**(κ, p)으로 다양한 degradation에 적응 가능  
- VQGAN 잠재공간 활용은 다른 잠재 generative 모델에도 적용 확장 가능  
- 실제 열화 모델링 한계 → 더 현실적·다양한 degradation 분포 학습 필요  
- 추론 속도 개선: 샘플링 스킴 최적화 및 네트워크 경량화 병행  

## 5. 향후 연구 영향  
- **고속 SR 패러다임 변화**: 낮은 단계 확산 설계 아이디어가 타 작업(영상 복원, 인페인팅) 확산 모델 효율화로 확장  
- **잔차 이동 원리**: SR 외에도 스타일 전이·노이즈 제거 등 다양한 inverse 문제 해결 메커니즘으로 활용  
- **Degradation 모델**: 실제 열화 분포 학습 강화 → 모델 일반화·강인성 제고에 기여  

**간단 고찰**: ResShift는 확산 모델의 ‘시작 분포’를 LR 기반으로 재설계함으로써 단계 수를 획기적으로 줄이고 성능 저하 없이 효율을 개선한 혁신적 접근이다. 후속 연구는 속도 최적화 및 실제 열화 대응력 강화를 중점적으로 다뤄야 할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f65563a6-0499-4c44-b917-db9fbf98a634/2307.12348v3.pdf
