# Noise Flow: Noise Modeling with Conditional Normalizing Flows | Image denoising

**핵심 주장 및 주요 기여**  
Noise Flow는 단순 가우시안 또는 이질분산(heteroscedastic) 모델이 포착하지 못하는 실제 카메라 센서 노이즈 분포의 복잡도를 학습 기반의 정규화 흐름(normalizing flow)으로 정밀하게 모델링한다.  
- **통합 모델**: 신호 종속성(noise level function)과 ISO 게인, 그리고 고차원 픽셀 간 상관관계를 하나의 컴팩트한 흐름 모델(<2500개 파라미터)로 통합.  
- **정확도 개선**: 기존 카메라 보정 노이즈 레벨 함수 대비 로그 우도에서 0.42 nats/pixel 향상(52% 우도 개선), 주변 분포(KL) 기준 84% 개선.  
- **응용 가능성**: 실제 노이즈 합성기로 활용해 denoising CNN(DnCNN) 학습 시 PSNR·SSIM 성능이 실데이터 학습 모델보다 우수.

## 1. 해결하고자 하는 문제  
기존 노이즈 모델  

$$
n_i \sim \mathcal{N}(0,\,\beta_1 I_i + \beta_2)
$$  

– *가우시안(AWGN)*과 *신호 의존 이질분산(NLF)* 모델은 픽셀 간 독립성, 공간 비균질성(고정 패턴 등)·비선형성(증폭·양자화 등)을 반영하지 못함.  
– 실제 스마트폰 RAW 이미지 노이즈는 Poisson–Gaussian 혼합, 고정 패턴, 채널 간 교차 잡음 등 복합적.

## 2. 제안 방법  
### 2.1. 조건부 정규화 흐름 구조  
Noise Flow는 다음 순서의 가역 변환으로 구성된다:  
1. **Signal‐Dependent Layer**  

$$
   f_{\mathrm{SD}}(x) = s \odot x,\quad s = (\beta_1 I + \beta_2)^{1/2}
   $$  
  
   – 역변환 $$g(x)=s^{-1}\odot x$$, 로그-야코비안 $$\sum_i\log s_i$$.  

2. **Unconditional Flow Steps** (K번 반복)  
   – Glow 기반의 affine coupling + 1×1 convolution  
3. **Gain Layer**  

$$
   f_{\mathrm{G}}(x) = \gamma(\mathrm{ISO},m)\odot x,\quad
   \gamma = \psi_m\,u(\mathrm{ISO})\times\mathrm{ISO}
   $$  
  
   – ISO별·카메라별 학습 가능, 로그-야코비안 $$D\log\gamma$$.  
4. **Unconditional Flow Steps** (K번 반복)  

### 2.2. 학습 및 평가  
- **데이터**: SIDD RAW 패치 64×64, 약 500K개(70% 학습, 30% 시험).  
- **손실**: 음의 로그 우도(NLL).  
- **추가 지표**: 픽셀 주변 분포 히스토그램 KL divergence.

## 3. 모델 구조 및 성능 향상  
| 모델 구조                           | 테스트 NLL    | 주변 DKL   | 우도 개선율 |
|-------------------------------------|--------------:|-----------:|-----------:|
| Gaussian                            | –2.831        | 0.394      | –          |
| Camera NLF                          | –3.105        | 0.052      | +51.6%     |
| Noise Flow (K=4, CAM 포함)         | **–3.521**    | **0.008**  | **+99.4%** |

- **카메라별 파라미터** 추가 시 NLL –3.431→–3.511으로 개선.  
- **Affine coupling 레이어** 1→4 스텝 확장 시 추가 소폭 개선.  
- **DnCNN 학습**: Noise Flow 합성 노이즈로 학습한 모델이 실데이터 학습 모델보다 PSNR+1.44 dB, SSIM+0.003 우수 (48.52 dB, 0.992)  

## 4. 일반화 성능 향상 가능성  
- **카메라-ISO 조건부** 설계로 신규 장치·ISO 조합에 대해 소량 파인튜닝만으로 빠르게 적응 가능.  
- **데이터 기반** affine coupling이 고정 패턴·채널 상관 등 기기 특유 노이즈를 학습해, 단순 파라메트릭 모델보다 현장 데이터에 강건.  
- **컴팩트 파라미터**로 메모리·계산 효율이 높아, 모바일·엣지 기기에도 적용 여지.

## 5. 한계 및 고려 사항  
- **학습 데이터 의존성**: SIDD와 유사 환경 데이터가 필요. 완전히 새로운 노출·광학 환경엔 성능 저하 우려.  
- **복잡도**: 정규화 흐름 블록 수·파라미터 튜닝이 모델 크기와 학습 안정성에 민감.  
- **실시간 합성**: 추론 시 다수 흐름 스텝 비용 고려해야 하므로 고속화 필요.

## 6. 향후 연구에 미치는 영향 및 고려할 점  
- **강건한 노이즈 합성기**: 다양한 컴퓨터 비전(분류, 향상, 복원) 응용 시 노이즈 증강(Data Augmentation) 도구로 활용 가능.  
- **도메인 적응**: 적은 레이블로 신규 카메라나 광학 시스템에 파인튜닝하는 연구 확장.  
- **경량화**: 실시간 영상 처리 위해 흐름 단계 축소, 저비용 연산 모델 경량화 연구 필요.  
- **비정형 노이즈**: 고차원 노이즈(모션 블러+센서 노이즈) 복합 모델링 연구에도 본 방식을 확장 가능.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5aa2e49d-c85a-4a0e-9eb1-c29147d62e55/1908.08453v1.pdf
