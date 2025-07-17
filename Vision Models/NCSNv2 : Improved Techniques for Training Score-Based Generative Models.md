# NCSNv2 : Improved Techniques for Training Score-Based Generative Models | Image generation
# 핵심 요약 및 주요 기여

**Improved Techniques for Training Score-Based Generative Models** 논문은 고해상도 이미지 생성에서 기존 Score-Based Generative Model(NCSN)이 겪는 불안정성과 성능 한계를 극복하기 위해 다음의 다섯 가지 핵심 기법을 제안한다.

1. **초기 노이즈 스케일 선택 (Technique 1)**  
   – 훈련 데이터의 최대 유클리드 거리만큼 σ₁을 설정하여 Langevin Dynamics가 데이터 모드를 충분히 탐색하도록 함.  
2. **노이즈 스케일 집합 구성 (Technique 2)**  
   – $σ₁$부터 $σ_L$까지 기하급수적 감소 비율 γ를 이론적으로 유도해, 각 스케일 간 분포 중첩(overlap)을 일정 수준(≈0.5)으로 유지.  
3. **노이즈 조건화 간소화 (Technique 3)**  
   – Noise-Conditional Score Network를 $s_θ(x,σ)=s_θ(x)/σ$ 형태의 *unconditional* 네트워크로 재설계해, 스케일 정보 인코딩을 직관적 스케일링으로 대체.  
4. **Annealed Langevin Dynamics 파라미터 최적화 (Technique 4)**  
   – 계산 예산에 따라 반복 횟수 T를 정한 뒤, 이론식  

```math
   s_T^2/σ_i^2
   =
   \Bigl(1 - \frac{ϵ}{σ_L^2}\Bigr)^{2T}\bigl(γ^2 - \tfrac{2ϵ}{σ_L^2}-1\bigr)
   + \tfrac{2ϵ}{σ_L^2}\bigl(1 - \tfrac{ϵ}{σ_L^2}\bigr)^2
```
   
   을 1에 가깝게 만드는 ϵ을 그리드 탐색으로 선택.  
5. **모델 안정화를 위한 EMA (Technique 5)**  
   – 지수 이동 평균(m≈0.999)을 적용해 학습 중 FID 변동과 컬러 쉬프트를 크게 감소시키고 최종 샘플 품질 향상.  

이 다섯 기법을 통합 적용해 얻은 NCSNv2는 CIFAR-10/​CelebA 64×64에서 기존 대비 FID를 절반 이하로 낮추며, 최대 256×256 해상도에서도 안정적이고 시각적으로 선명한 샘플을 생성할 수 있다.

# 상세 분석

## 1. 문제 정의  
- **기존 한계**: NCSN(N(score)S(list)N(et))은 다중 노이즈 스케일로 학습한 score network와 Annealed Langevin Dynamics를 결합해 저해상도(~32×32) 이미지에서 우수한 성능을 보였으나,  
  - 고해상도(≥64×64)로 확대 시 σ₁의 부적절한 설정, 스케일 간 중첩 부족, 스케일 정보 인코딩 비효율, 샘플링 파라미터 미조정, 학습 불안정성 등으로 실패  
- **목표**: 이론적 근거에 기반한 하이퍼파라미터 및 구조 설계로 NCSN을 64–256×256 이미지에도 안정적으로 확장

## 2. 제안 방법 및 수식

### 2.1 초기 노이즈 σ₁  
- Empirical distribution $$\hat p_{σ₁}(x)=\frac1N\sum_i \mathcal N(x|x^{(i)},σ₁^2I)$$의 score $$\nabla_x\log\hat p_{σ₁}(x)=\sum_i r^{(i)}(x)\nabla_x\log p^{(i)}(x)$$이며,  
- $$E_{p^{(i)}}[r^{(j)}(x)]\le\tfrac12\exp\bigl(-\|x^{(i)}-x^{(j)}\|^2/(8σ₁^2)\bigr)$$로 σ₁≪max pairwise distance 시 모드 간 이동 확률이 극히 작아진다[Proposition 1].  
- **설정법**: σ₁ = 훈련 데이터의 최대 유클리드 거리.

### 2.2 노이즈 스케일 집합 $$\{σ_i\}$$  
- 고차원 가우시안 분포 $$\|x\|\sim\mathcal N(\sqrt{D}\,σ, σ^2/2)$$[Proposition 2]를 이용해,  
- 각 스케일 간 반경 분포 겹침 비율  

$$
  C=\Phi\bigl(\sqrt{2D}(γ-1)+3γ\bigr)-\Phi\bigl(\sqrt{2D}(γ-1)-3γ\bigr)\approx0.5
$$
  
  이 되도록 $γ=σ_{i-1}/σ_i$ 일정하게 하여 기하급수열 구성[Technique 2].

### 2.3 샘플링 파라미터 $$ϵ, T$$  
- Annealed Langevin Dynamics step:  

$$
  x_{t}\leftarrow x_{t-1} + α_i s_θ(x_{t-1},σ_i) + \sqrt{2α_i}\,z_t,\quad α_i=ϵ\,\frac{σ_i^2}{σ_L^2}
$$

- 한 스케일 단계 후 $$x_T\sim\mathcal N(0,s_T^2I)$$이고,  

$$
  \frac{s_T^2}{σ_i^2}
  =\Bigl(1-\frac{ϵ}{σ_L^2}\Bigr)^{2T}(γ^2-1)
  +\frac{2ϵ}{σ_L^2}\Bigl(1-\frac{ϵ}{σ_L^2}\Bigr)^2
$$

- **설정법**: 예산에 맞춰 T 결정 후 위 식을 1에 가깝도록 ϵ 그리드 탐색[Technique 4][Proposition 3].

### 2.4 네트워크 구조 및 노이즈 조건화  
- 기존 RefineNet 기반의 CondInstanceNorm++ → σ 별 scale/bias  
- **제안**: 불필요한 채널 분기 제거, *unconditional* score network $$s_θ(x)$$의 출력만 1/σ로 스케일[Technique 3]  

$$
  s_θ(x,σ)=\frac{s_θ(x)}{σ}.
$$

### 2.5 EMA 안정화  
- 학습 중 파라미터 $θ′←mθ′+(1–m)θ$ (m≈0.999) 유지하고, 샘플링 시 θ′ 사용으로 FID 변동 및 컬러 쉬프트 감소[Technique 5].

## 3. 성능 향상

| 데이터셋             | 모델              | FID ↓    | Inception ↑ | HYPE∞ (%) |
|----------------------|-------------------|----------|-------------|-----------|
| CIFAR-10 32×32       | NCSN              | 25.32    | 8.87±0.12   | —         |
|                      | **NCSNv2 (+denoise)** | **10.87** | 8.40±0.07   | —         |
| CelebA 64×64         | NCSN              | 25.30    | —           | 19.8      |
|                      | **NCSNv2 (+denoise)** | **10.23** | —           | **37.3**  |

- FID 절반 이하, HYPE∞(인간 평가) 기준 ProgressiveGAN 수준으로 개선.  
- 96·128·256 해상도에서도 고품질 샘플 생성 가능.

## 4. 일반화 및 한계

- **일반화**  
  - 다양한 해상도(32→256)·데이터셋(CIFAR-10, CelebA, LSUN, FFHQ)에서 일관된 성능 향상.  
  - 훈련/테스트 손실 일치, nearest-neighbor 검사로 모델이 단순 암기를 넘어 데이터 분포를 포괄적으로 학습함 입증.  
  - 외삽(interpolation)에서 부드러운 잠재 공간 표현 학습.

- **한계**  
  - EMA 적용·큰 σ₁·다수 스케일로 메모리·계산 비용 증가.  
  - Annealed Langevin Dynamics 단계 수(T×L)가 수천 단위로, 실시간 응용에 부적합.  
  - FID/Inception 지표의 한계(시각 품질과 상관 미흡), 추가 인간 평가 필요.

## 5. 향후 연구 영향 및 고려사항

- **영향**  
  - Score-Based 모델의 고해상도 이미지 생성 가능성을 열어, GAN 대안으로 주목.  
  - 노이즈 스케일 설계·샘플링 최적화 이론적 기틀 제공.  
  - *Unconditional* noise conditioning 등 간소화된 구조 설계 아이디어 확산.

- **고려사항**  
  1. 효율적 샘플링(단계 수·스케일 수 축소) 연구  
  2. 비가우시안 노이즈 분포 활용  
  3. EMA 대체 기법(예: SWA) 및 Adam 변형 적용  
  4. FID 외 perceptual/precision-recall 지표 보완  
  5. 멀티모달·비이미지 데이터(음성, 텍스트) 확장 검증

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8ba4604a-57e9-4e29-b4df-74676d35b5b2/2006.09011v2.pdf
