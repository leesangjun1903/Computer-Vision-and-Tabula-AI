# A Deep Journey into Super-resolution: A Survey

**주요 주장 및 기여**  
이 논문은 단일 영상 초해상도(SISR)에 특화된 딥러닝 기반 기법들을 체계적으로 분류·분석하고, 네트워크 구조·학습 세부사항·성능을 비교함으로써 향후 연구 방향을 제시한다.  
1. **광범위한 비교 평가**: 30여 개 대표적 SISR CNN을 6개 데이터셋(Set5, Set14, Urban100, BSD100, DIV2K, Manga109)에 대해 PSNR·SSIM 성능과 모델 복잡도로 벤치마킹  
2. **새로운 분류 체계**: 선형(linear), 잔차(residual), 다중 분기(multi-branch), 재귀(recursive), 점진적(progressive), 주의(attention), 적대적(GAN) 등 9개 범주로 기법을 체계적 분류  
3. **설계·학습 비교**: 파라미터 수, 네트워크 깊이, 입력·출력 방식(early/late upsampling), 손실 함수(ℓ1·ℓ2·perceptual), 스킵 연결 유무 등을 표로 정리  
4. **성능·복잡도 분석**: PSNR 개선 추세와 함께 모델 복잡도(연산량·파라미터) 증가 경향, 고배율(8× 이상)에서의 성능 저하 문제 지적  
5. **미해결 과제 제시**: 실제 열화(degradation) 처리, 극고배율·임의 배율 SR, 평가 지표 한계, 통합 모델 필요성 등  

# 해결 과제, 제안 기법, 구조 및 한계  

## 해결하고자 하는 문제  
· LR→HR 복원은 비가역적(inverse ill-posed) 문제로, 같은 LR에 다수 HR 대응 가능  
· 고배율(scaling factor↑) 시 세부 정보 복원이 더욱 어려워짐  
· PSNR·SSIM 등의 정량 지표가 인간 시각(지각 품질)과 상관이 약함  

## 제안하는 방법  
논문 자체가 단일 모델 제안이 아닌 **분류(taxonomy)** 및 **비교 평가** 연구이나, 딥러닝 SR의 일반식과 주요 네트워크 구조는 다음과 같다.  

1. **열화 모델**  

   $$y = (x \otimes k)\downarrow_s + n $$
   
   – $$x$$: HR 이미지, $$y$$: 관측된 LR 이미지  
   – $$k$$: 블러 커널, $$\downarrow_s$$: 축소 연산(비율 $$s$$), $$n$$: AWGN  

2. **목표 함수**  

   $$J(\hat x) = \|x\otimes k - y\|^2 + \alpha\Psi(\hat x) $$
   
   – $$\Psi$$: 영상 사전(prior) (e.g., residual, attention)  
   – 최근 모델들은 픽셀 수준 ℓ1/ℓ2 외에 perceptual 손실($$\|\phi(\hat x)-\phi(x)\|_1$$)·GAN adversarial 손실을 병합  

3. **네트워크 구조 분류 및 예시**  
   -  **Linear**: SRCNN, FSRCNN, ESPCN  
   -  **Residual**: EDSR(단일 단계), MDSR(다중 배율), CARN, REDNet(인코더-디코더)  
   -  **Recursive**: DRCN, DRRN, MemNet (parameter 재사용)  
   -  **Progressive**: LapSRN (2×→4×→8× 단계별 복원)  
   -  **Dense**: SRDenseNet, RDN (local/global residual + dense 연결)  
   -  **Multi-branch**: CNF, CMSC, IDN (병렬 경로로 다양한 맥락 정보 학습)  
   -  **Attention**: RCAN (채널 주의), DRLN (라플라시안 주의)  
   -  **Multiple-degradation**: ZSSR (zero-shot), SRMD (degradation map 입력)  
   -  **GAN 기반**: SRGAN, EnhanceNet, ESRGAN (실제감 지각 품질 최적화)  

# 3. Single Image Super-Resolution (SISR)

## 개념과 목표  
단일 영상 초해상도(Single Image Super-Resolution, SISR)는 손실된 해상도의 저해상도(LR) 이미지를 입력으로 받아, 이를 가능한 한 원본 고해상도(HR) 이미지에 가깝게 복원하는 작업입니다.  
- “비가역적(inverse ill-posed) 문제”여서, 동일한 LR에 여러 HR이 대응될 수 있습니다.  
- 확대 배율이 커질수록(예: 2×→4×→8×) 픽셀 간격이 멀어져 복원이 어려워집니다.  
- 전통적 화질 지표(PSNR, SSIM)는 인간 시각 품질과 항상 일치하지 않습니다.  

## 딥러닝 기반 SISR 분류
딥러닝 네트워크 설계 특징에 따라 SISR 기법을 9가지 큰 범주로 나눕니다.  

1. **Linear Networks (단일 경로)**  
   - 순차적 컨볼루션만 쌓아 올린 구조  
   - Early vs. Late Upsampling  
     -  Early: 입력을 먼저 업샘플링(예: SRCNN)  
     -  Late: 특징 추출 후 마지막에 업샘플링(예: FSRCNN, ESPCN)  

2. **Residual Networks (잔차 학습)**  
   - 입력과 출력 간 차이(잔차)를 학습하여 세부 정보를 집중 복원  
   - Single-stage: 전체를 한 번에 처리(예: EDSR, CARN)  
   - Multi-stage: 단계별로 점진 복원(예: REDNet, BTSRN)  

3. **Recursive Networks (재귀 구조)**  
   - 동일 레이어를 여러 번 반복 적용하여 깊은 학습 효과  
   - DRCN, DRRN, MemNet, SRFBN 등  

4. **Progressive Reconstruction (점진적 복원)**  
   - 2×→4×→8× 단계별로 잔차 예측 후 누적  
   - LapSRN, SCN  

5. **Densely Connected Networks (조밀 연결)**  
   - DenseNet 스타일로 모든 이전 층의 출력을 연결  
   - SRDenseNet, RDN, D-DBPN 등  

6. **Multi-branch Designs (다중 경로)**  
   - 서로 다른 분기(branch)에서 다양한 크기·특징을 병렬 학습 후 융합  
   - CNF, CMSC, IDN, EBRN  

7. **Attention-based Networks (주의 메커니즘)**  
   - 채널·공간별 중요도를 동적으로 가중  
   - SelNet, RCAN, DRLN, SRRAM  

8. **Multiple-degradation Handling (다중 열화 대응)**  
   - 단일 bicubic 가정 벗어나 실제 blur·noise 복합 열화 처리  
   - ZSSR(Zero-Shot), SRMD  

9. **GAN Models (생성적 적대 신경망)**  
   - 생성자와 판별자 경쟁 학습으로 **인지적(Perceptual) 품질** 최적화  
   - SRGAN, EnhanceNet, SRFeat, ESRGAN  

## 공통 수식 및 학습 목표  
- **열화 모델(degradation)**  

  $$y = (x \otimes k)\downarrow_s + n $$  

  -  $$x$$: HR, $$y$$: 관측 LR, $$k$$: blur kernel, $$\downarrow_s$$: 축소 연산, $$n$$: AWGN  

- **최적화 목표(loss)**  
  -  복원 오차: $$\ell_1$$ 또는 $$\ell_2$$ 픽셀 손실  
  -  지각 손실: deep feature 차이 $$\|\phi(\hat x)-\phi(x)\|_1$$  
  -  GAN 손실: 실제와 구분 어려운 이미지 생성  

## 네트워크 구조의 핵심 아이디어  
- **잔여(residual) 학습**: 입력(LR)과 예측(HR) 차이를 학습해 어려운 세부 정보에 집중  
- **스킵 연결(skip connection)**: 정보 소실 방지·그래디언트 안정화  
- **조밀 연결(dense connection)**: 다층 특징 융합으로 표현력 강화  
- **주의(attention) 모듈**: 중요한 채널·공간 강조로 일반화 성능 향상  
- **재귀(recursion)**: 파라미터 공유로 더 깊은 표현 학습  

## SISR 설계 시 고려 사항  
- **배율(scale factor)**: 높을수록 단계적 복원이 권장  
- **모델 복잡도 vs. 실시간성**: 초경량(FSRCNN) ↔ 초정밀(EDSR/RCAN)  
- **학습 데이터 열화**: 실제 LR 데이터셋 확보·도메인 적응 필요  
- **평가 지표**: PSNR/SSIM 한계 보완 위해 지각 품질 측정 도입  

---  

위와 같이 SISR은 “LR→HR 복원” 문제를 다양한 네트워크 아키텍처(Linear, Residual, Recursive, Attention 등)와 학습 목표(픽셀 손실, perceptual 손실, GAN 손실)로 풀어내며, 실제 적용 시에는 배율, 모델 경량화, 실제 열화 대응, 평가 지표 선택 등 여러 요소를 동시에 고려해야 합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87ccc689-3f91-4bc3-aa63-1d2eb272f234/1904.07523v3.pdf

## 성능 향상 및 한계  
- **PSNR vs. 인지 품질**: ℓ2 최적화 모델(EDSR, RDN 등)은 PSNR 최고 기록. 반면 GAN 기반(ESRGAN)은 시각 품질 우수하나 PSNR 열위  
- **고배율 저하**: 8× 이상에서 균질 질감 복원 실패  
- **모델 복잡도**: SOTA 모델일수록 수천만 파라미터, 높은 FLOPs  
- **실제 LR 일반화 한계**: 대부분 bicubic 훈련, 실제 열화 일반화 어려움  

# 일반화 성능 향상 관점  

- **스킵 연결 & residual learning**: global/local skip 으로 잔류 정보 직접 학습 → 깊은 네트워크 안정화, 일반화 개선  
- **attention 메커니즘**: 채널·공간 중요도 동적 조절 → 범용 열화 상황에서 다양한 패턴 적응성 향상  
- **multi-degradation 모델**: degradation map 입력(SRMD)·image-specific 학습(ZSSR) → 비정형 노이즈·흐림에도 어느 정도 견고  
- **self-ensemble & multi-scale 학습**: MDSR, MS-LapSRN 등으로 여러 배율에 공통 표현 학습 → unseen 배율에 대한 일반화 잠재력  

# 향후 영향 및 연구 고려 사항  

**미래 영향**  
- SISR 연구를 구조·성능·학습 측면에서 통합적 조망하며, 다양한 모델 설계에 일관된 비교 기준 제공  
- residual/attention/recursive/dense 등 핵심 모듈화 아이디어가 이후 복합 모델 설계에 기여  
- 실제 열화·극고배율·평가 지표 개선 연구의 기반 마련  

**향후 연구 시 고려점**  
1. **실제 LR 일반화**: bicubic 벗어난 실제 데이터셋 구축 및 도메인 적응 방법  
2. **평가 지표 개발**: PSNR·SSIM 한계 보완할 학습 기반 지각 품질 측정 도입  
3. **경량·실시간 모델**: 모바일·임베디드 적용을 위한 파라미터·연산 절감 기법  
4. **극고·임의 배율 SR**: 점진적·메타 학습 기반 일반화 가능한 단일 모델 연구  
5. **통합 복원 솔루션**: SR·디노이즈·디블러링·아티팩트 제거를 하나의 네트워크로 처리하는 멀티태스크 학습

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87ccc689-3f91-4bc3-aa63-1d2eb272f234/1904.07523v3.pdf
