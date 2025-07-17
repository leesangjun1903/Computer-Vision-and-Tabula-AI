# Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model | Super resolution

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
사전 학습된 확산 모델(diffusion model)을 제너레이티브 프라이어로 활용해, 어떠한 선형 열화(linear degradation)에도 추가 학습 없이 “제로샷(zero-shot)”으로 고품질 이미지 복원을 수행할 수 있다는 점을 제안한다.  

**주요 기여**  
1. **통합 이론적 프레임워크**  
   - 임의의 선형 열화 연산자 $$A$$에 대해, 복원 결과 $$\hat x$$를  
     
$$
       \hat x = A^\dagger y + (I - A^\dagger A)\bar x
     $$ 

  형태로 분해하고, null-space $$(I - A^\dagger A)\bar x$$ 성분만 확산 모델의 역방향 과정에서 점진적으로 보정함으로써 데이터 일관성과 실제감(realness)을 동시에 보장한다.  

2. **DDNM 알고리즘**  
   - 기존 DDPM(denoising diffusion probabilistic model) 샘플링 과정에서, 매 단계 $$t$$마다  
     1. 노이즈 예측 $$\epsilon_t = Z_\theta(x_t, t)$$로부터 “깨끗한” 추정 $$x_{0|t}$$ 계산(식 12).  
     2. 범위 공간(range-space) 성분 $$A^\dagger y$$로 교체해 $$\hat x_{0|t}$$ 구성(식 13).  
     3. 이를 기반으로 $$x_{t-1}$$ 샘플링(식 14).  
   - 이 과정을 통해 제로샷으로 다양한 IR(inverse recovery) 과제를 해결.  

3. **확장판 DDNM+**  
   - **노이즈 보정**: 식 (17)–(19)에서 도입한 스케일 매트릭스 $$\Sigma_t$$, $$\Phi_t$$로 실제 관측 잡음 $$\sigma_y$$를 보정.  
   - **타임트래블(time-travel) 기법**: 중간 단계 $$t+\ell$$로 되돌아가 다시 역확산하여 전역 조화(global harmony) 성능 개선.  

4. **범용성 및 실험 검증**  
   - 단일 사전학습 모델로 슈퍼해상도, 색상화, 인페인팅, 압축센싱, 디블러링 등 5가지 전형적 IR 과제에서 최첨단 zero-shot 기법 대비 우수한 PSNR/SSIM/FID 성능 달성[1].  
   - 실제 노이즈나 복합 열화(old photo restoration)에도 견고하게 동작.

## 2. 문제 정의 및 제안 방법 상세

### 2.1 해결하고자 하는 문제  
- **선형 이미지 열화**:  
  
$$
    y = Ax + n,\quad A:\mathbb R^D\to\mathbb R^d,\quad n\sim\mathcal N(0,\sigma_y^2I).
  $$ 

- 목표: 데이터 일관성(Consistency, $$A\hat x=y$$)과 실제감(Realness, $$\hat x\sim q(x)$$)을 동시에 만족하는 복원 $$\hat x$$ 획득.

### 2.2 Null-Space 분해 이론  
- 선형 연산자 $$A$$의 의사역원(pseudo-inverse) $$A^\dagger$$를 써서  

$$
    x = A^\dagger Ax + (I - A^\dagger A)x
  $$ 
  
  로 분해.  
- 대응 복원 해는  

$$
    \hat x = A^\dagger y + (I - A^\dagger A)\bar x,
  $$ 
  
  여기서 $$\bar x$$를 잘 선택하면 $$\hat x\sim q(x)$$.

### 2.3 DDNM 알고리즘  
1. 초기: $$x_T\sim\mathcal N(0,I)$$.  
2. 단계 $$t=T,\dots,1$$ 반복:  
   a. $$\epsilon_t = Z_\theta(x_t,t)$$ 예측 →  

$$
     x_{0|t}=\frac{1}{\sqrt{\bar\alpha_t}}\Bigl(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_t\Bigr).
   $$ 
   
   b. 범위공간 교체:  
   
$$
     \hat x_{0|t} = A^\dagger y + (I - A^\dagger A)x_{0|t}.
   $$  
   
   c. 역확산 샘플링:  
   
$$
     x_{t-1}
       = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\,\hat x_{0|t}
       + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t
       + \sigma_t\epsilon,\quad \epsilon\sim\mathcal N(0,I).
   $$  
   
3. 결과 $$x_0$$ 반환.

### 2.4 DDNM+ 개선  
- **노이즈 보정**:  

$$
    \hat x_{0|t}
      = x_{0|t}
      - \Sigma_t A^\dagger\bigl(Ax_{0|t}-y\bigr),
    \quad \epsilon\sim \mathcal N(0,\Phi_tI),
  $$  
  
  $$\Sigma_t,\Phi_t$$는 $$\sigma_y$$ · $$\sigma_t$$ 관계로 유도된 대각 매트릭스.
- **타임트래블**: 일정 간격 $$s$$마다 과거 $$t+\ell$$ 단계로 되돌아가 재샘플링해 전역 구조 정합성 강화.

## 3. 모델 구조  
- 핵심은 사전학습된 **확산 네트워크** $$Z_\theta(·)$$만 활용. 네트워크 구조 변경 불필요.  
- 열화 행렬 $$A$$와 $$A^\dagger$$만 정의하면, 픽셀 마스크·평균 풀링·그레이스케일 등 다양한 선형 연산자 지원[1].

## 4. 성능 향상 및 한계  
- **성능**:  
  - ImageNet·CelebA 256×256에서 5개 IR 과제 비교: DDNM 평균 PSNR+0.1–1 dB, FID-10–20pt 개선[1].  
  - DDNM+는 노이즈 제거에서 PSNR +6 dB, FID 절반 수준으로 개선.  
- **한계**:  
  1. **추론 속도**: 확산 모델의 본질적 느린 속도.  
  2. **선형성 가정**: 비선형 열화엔 직접 적용 불가.  
  3. **$$A,A^\dagger$$ 지식 필요**: 실제 열화의 정확 모델링 어려움.  
  4. **사전학습 프라이어 한계**: 복원 성능은 $$Z_\theta$$의 학습 데이터·용량에 의존.

## 5. 일반화 성능 향상 가능성  
- **제로샷 범용성**: 한 번의 사전학습으로 모든 선형 IR 과제에 응용 가능.  
- **복합 열화 대응**: $$A=A_1A_2…$$ 분해해 복합 노이즈·마스킹·다운샘플링 병합 복원.  
- **확산 모델 발전 수혜**: $$Z_\theta$$ 성능 개선 시 즉시 복원 품질 향상.  
- **비디오·오디오 등 타 도메인 확장**: 선형 연산자 및 확산 프라이어만 준비되면 유사한 제로샷 IR 가능.

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **영향**: 선형 역문제에 대한 “사전학습 제너레이티브 프라이어 + null-space 보정” 패러다임 제시 → 다양한 비전·신호 복원 분야 재설계 촉진.  
- **고려 사항**:  
  1. **비선형 및 미지 열화 모델 학습**(Blind IR) 연동 방안.  
  2. **추론 가속화** 위한 경량화·DDIM 속도 최적화.  
  3. **적응적 노이즈 추정**으로 $$\sigma_y$$ 자동 조정 전략.  
  4. **확산 네트워크 개선**: 고해상도·다양 도메인 사전학습.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3b4a24f4-b810-4632-824f-bd984e983c96/2212.00490v2.pdf
