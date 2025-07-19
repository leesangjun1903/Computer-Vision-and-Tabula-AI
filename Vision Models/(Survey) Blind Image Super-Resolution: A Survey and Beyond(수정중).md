# Blind Image Super-Resolution: A Survey and Beyond

## 1. 핵심 주장과 주요 기여  
본 논문은 **알려지지 않은(degradation unknown) LR(low‐resolution) 이미지**에 대해 고품질 HR(high‐resolution) 영상을 복원하는 **Blind SR** 연구 동향을 체계적으로 정리하고,  
- **Taxonomy 제안**: Blind SR 기법을 ‘명시적 모델링 vs. 암시적 모델링’ 및 ‘외부 데이터 사용 vs. 단일 이미지 사용’이라는 축으로 분류  
- **각 분류별 대표 기법 리뷰**: Non-blind 대비 blind SR의 수식적 정의→외부 데이터 기반 explicit methods (SRMD, IKC 등)→단일 이미지 내부 통계 기반 methods (ZSSR, KernelGAN 등)→GAN 기반 implicit methods (CinCGAN, DASR 등)  
- **데이터셋·경진대회 정리**: Synthetic/Real 제작 방식, NTIRE/AIM 등 Blind SR 트랙 비교  
- **성능·한계 분석**: 정량·정성 비교를 통해 각 방식의 장·단점을 밝힘  

를 통해 “현재까지 Blind SR이 어디까지 왔고, 무엇이 남아 있는가”를 명확히 제시하였다.

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 문제 정의  
- **기존 Non-blind SR**: $$y = (x\otimes k)\downarrow_s + n$$ 수식에서 $$k$$ (blur kernel) 및 $$n$$ (noise) 가정(보통 bicubic $$k$$, 무시하거나 고정된 Gaussian $$n$$)  
- **Blind SR**: LR마다 **미지의** $$k,\;n$$ 를 복원→역함수 $$f^{-1}$$ 추정  
- **도전 과제**: 실제 촬영·저장 과정에서 발생하는 복합 degradations (sensor noise, ISP artifacts, compression, 옛 사진 필름 훼손)  

### 2.2 제안된 Taxonomy  
1. **Explicit Modelling (수식 기반)**  
  1.1. 외부 데이터 + degradation map 입력  
    - SRMD : PCA로 커널압축→$$H\times W$$ map과 concat  
    - USRNet : MAP 최적화 풀링→non-blind SR 네트워크로 풀기  
  1.2. 외부 데이터 + kernel estimation 내장  
    - IKC : **Iterative Kernel Correction** – SR 결과 artifact→kernel corrector 반복  
    - DAN : IKC end-to-end 학습, non-iterative  
  1.3. 단일 이미지 내부 통계  
    - KernelGAN : GAN으로 patch recurrence 분포 학습→커널 복원  
    - ZSSR : self-supervised CNN, LR 자체 downsample→학습→SR  
2. **Implicit Modelling (GAN/Distribution Learning)**  
  - CinCGAN : unpaired LR→bicubic LR→pretrained non-blind SR (CycleGAN×2)  
  - DASR : High-to-Low generator로 realistic LR 생성 + domain-aware SR 학습  
  - FSSR : high-frequency만 어드버서리얼 학습  

### 2.3 모델 구조 요약  
- **SRMD**: shallow/deep feature extraction + degradation map concat + residual blocks  
- **IKC**: predictor(커널)→SR 네트워크(Spatial Feature Transform)→corrector(잔차 커널) 반복  
- **KernelGAN**: generator(fully-conv chain)→patch discriminator→학습 후 합성 필터 추출  
- **CinCGAN**: LR→CleanLR CycleGAN + CleanLR→HR CycleGAN + pretrained SR  

# 6. 명시적 열화 모델링(Explicit Degradation Modelling)

명시적 열화 모델링은 저해상도(LR) 이미지가 어떻게 열화(degradation)되었는지를 **수식(모델)로 명확히 가정**하고, 그 가정에 따라 **학습된 SR 모델**을 이용해 고해상도(HR) 이미지를 복원하는 방식입니다.  
주로 **블러(blur) 커널** $$k$$와 **잡음(noise)** $$n$$ 두 가지 요인을 다루며, 외부에 준비된 데이터셋(쌍으로 구성된 HR–LR 이미지쌍)으로 학습합니다.

## 6.1 외부 데이터셋 기반 Explicit Modelling

대부분의 방법은 다음의 “클래식” 열화 모델을 가정합니다:  

$$
y = (x \otimes k)\!\downarrow_s + n
$$ 

- $$x$$: HR 원본  
- $$y$$: 생성된 LR 이미지  
- $$\otimes$$: 컨볼루션(블러)  
- $$\downarrow_s$$: 축소 비율 $$s$$로 다운샘플링  
- $$k$$: 미지의 블러 커널  
- $$n$$: 미지의 잡음

### 6.1.1. 커널·잡음 정보 입력형 (Image-Specific Adaptation without Kernel Estimation)  
1) SRMD  
   - **커널+잡음 맵**을 입력 이미지에 “채널로” 붙여 넣고(차원 확장) CNN에 함께 학습  
   - 단일 네트워크로 **다양한** $$k,n$$ 처리  
2) UDVD  
   - **동적 컨볼루션(dynamic conv.)** 적용  
   - 커널·잡음 맵에 따라 **채널별 가중치**를 학습  
3) DPSR / USRNet  
   - MAP(우도+사전) 프레임워크로 **블러 분리**(FFT) ↔ **SR+디노이즈** 단계 반복  
   - USRNet은 이를 **언폴딩(unfold)** 해 다단계 네트워크로 구현

**장점**  
- 명확한 수식 기반으로 학습 안정성  
- 알려진 범위 내 블러·잡음에서 우수한 성능  

**단점**  
- 실제 추론 시 정확한 $$k,n$$ 추정 필요  
- 추정 오차 시 심각한 화질 저하  

### 6.1.2. 커널 추정 통합형 (Image-Specific Adaptation with Kernel Estimation)  
**SR 과정 중에 자동으로 커널을 추정**해, 추정→SR→추정…을 반복하거나  
단일 네트워크 안에서 **추정기가 곧 SR 입력**이 되도록 설계합니다.

1) IKC (Iterative Kernel Correction)  
   - 초기 커널 예측 후 SR 수행  
   - SR 결과에서 생긴 “아티팩트”를 이용해 **커널 보정**  
   - 이를 **반복(iteration)** 해 점진적 개선  

2) DAN (Deep Alternating Network)  
   - IKC를 언폴딩(unfold)해 **end-to-end** 학습  
   - 커널 추정기와 SR 네트워크를 **동시 최적화**  

3) 기타  
   - VBSR: 판별자(discriminator)로 SR 오류 맵을 학습, 최적 커널 검색  
   - KOALAnet, AMNet-RL: **대역 학습**·**강화학습** 이용해 커널 인코더와 SR 네트워크 연결  

**장점**  
- 커널 추정 과정을 네트워크 내부에 통합해 추론 편의성 향상  
- 반복 학습으로 커널 미스매치 문제 완화  

**단점**  
- 반복 횟수·종료 조건 설정의 번거로움  
- 독립적인 블러가 아닌, **복합 열화**(ISP 아티팩트·압축노이즈 등)엔 여전히 취약  

## 6.2 단일 이미지 내부 통계 기반 Explicit Modelling

외부 데이터 없이 **한 장의 LR 이미지** 내에서 **패치 반복성(patch recurrence)** 만으로 SR을 수행하려는 방법들입니다.

- **NPBSR**: 반복되는 패치 간의 통계로 MAP 기반 커널 추정  
- **KernelGAN**: 작은 네트워크를 “GAN 생성기”로 보고, 다운샘플된 LR과 “원본 LR”의 패치 분포 일치를 통해 **블러 커널** 직접 학습  
- **ZSSR**: 입력 LR을 HR라 보고, 다시 다운샘플해 생성한 LR을 학습 데이터로 삼아 **Self-supervised CNN** 학습  
- **DGDML-SR**: depth map 기반으로 HR/LR 패치를 분리해 **CycleGAN** 유사 구조에서 내부 열화 모델과 SR 네트워크를 동시 학습  

**장점**  
- **외부 데이터 불필요**, 이미지 특화  
- 심각한 도메인 갭 없이 **유연한** SR 가능  

**단점**  
- 패치 반복성이 낮은(단조로운) 또는 복잡한(real-world) 장면에서는 **반복 통계가 부족**해 성능 저하  
- 잡음·압축 아티팩트 등 **복합 열화** 모델링 한계  

## 6.3 요약 및 활용 가이드

- **명시적 모델링**은 블러·잡음 등 **수식화 가능한 열화**에 강하지만, 실제 기기 ISP나 압축 노이즈처럼 **복합적이고 예측불가한** 열화에는 한계가 있습니다.  
- **외부 데이터 기반**은 다양한 열화 조건을 **학습 범위**로 끌어올 수 있으나, 학습하지 않은 열화에겐 무력합니다.  
- **단일 이미지 내부 통계 기반**은 외부 데이터 부담이 없지만 **패치 반복성**이 큰 전제 조건이므로 일반 자연 이미지 전반에 적용하기엔 어려움이 있습니다.

**적용 팁**  
1. 실제 촬영 이미지가 대부분 **Gaussian blur+noise**라면, SRMD·IKC·USRNet 등 외부 데이터 기반 방식을 추천  
2. **한 장의 이미지**로만 처리해야 하고, 패치 반복성이 높다면 KernelGAN+ZSSR 같은 내부 통계 방식을 고려  
3. 복합 열화(모바일 ISP·JPG 압축 등)에는 명시적 모델링만으로는 부족하며, **암시적(gan) 모델링**이나 향후 발전할 **단일 이미지 암시적 모델링** 기법 연구 필요  

이처럼 **명시적 열화 모델링**은 수식 가정의 명확성과 학습 안정성이 장점이지만, “모델 가정 밖” 열화엔 취약하다는 특성을 꼭 이해하고 활용해야 합니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6055dde7-45fe-4156-9d86-d9ded19e3fe2/2107.03055v1.pdf

# 7. 암시적 열화 모델링(Implicit Degradation Modelling)

암시적 열화 모델링은 **열화 과정을 수식으로 정의하지 않고**, 대신 **훈련용 데이터의 분포**를 학습하여 저해상도(LR)→고해상도(HR) 변환 모델을 만드는 접근입니다.  
즉, 블러 커널 $$k$$나 노이즈 $$n$$ 같은 명시적 파라미터를 추정하지 않고, **실제 LR–HR 쌍** 또는 **LR과 HR 도메인 분포**를 GAN 등의 모델로 직접 연결하여 SR 네트워크를 학습합니다.

## 7.1 주요 방식

### 7.1.1 페어드 데이터 기반 간접 학습  
- **훈련 데이터**: 실제 촬영한 HR–LR 쌍이 확보된 경우  
- **학습**: 일반적인 지도학습(Supervised)과 동일하게, LR→HR 매핑망을 직접 최적화  
- **장점**: 열화를 명시적으로 다루지 않아도 되고, 실제 환경에서 촬영된 다양한 LR 품질을 학습 가능  
- **단점**: 고품질 HR–LR 쌍을 구축하기 어렵고 비용이 큼  

### 7.1.2 언페어드 데이터 기반 도메인 적응  
- **훈련 데이터**: HR 도메인(예: 웹 상 고해상도 원본)과 LR 도메인(예: 스마트폰 사진) 두 그룹  
- **모델 구조**:  
  1. **High-to-Low Generator**: HR → “실제 같은” LR 생성  
  2. **Low-to-High SR Generator**: 생성된 LR → SR(고해상도)  
  3. 두 Generator를 **GAN**과 **Cycle-Consistency**로 연결[1]  
- **대표 기법**  
  - CinCGAN: LR→Clean LR→SR의 두 단계 CycleGAN[1]  
  - Degradation GAN + SRGAN: HR→LR 생성 후, 페어드 학습  
  - DASR: 생성 LR과 실제 LR을 함께 섞어 학습해 도메인 갭 최소화  
- **장점**: 실제 LR 도메인에서 수집한 이미지로 훈련하므로 현실 열화를 반영  
- **단점**:  
  - GAN 훈련의 불안정성(블링·가짜 텍스처)  
  - 생성된 LR과 진짜 LR 간 도메인 차이 완전 해소 어려움  

## 7.2 동작 원리

1. **생성 네트워크 학습**  
   - $$G_d$$: HR 이미지 $$x$$를 받아 “실제 LR”처럼 보이는 $$\hat y = G_d(x)$$ 생성  
   - **Adversarial Loss**로 $$\hat y$$ 분포를 실제 LR 데이터에 근접하도록 함  
2. **SR 네트워크 학습**  
   - 생성된 $$\hat y$$와 대응 HR $$x$$ 쌍을 사용해 $$G_s$$: $$\hat y \mapsto \hat x = G_s(\hat y)$$ 학습  
   - **Pixel Loss**($$\ell_1$$, $$\ell_2$$) + **Perceptual/GAN Loss** 결합  
3. **순환 일관성(Cycle)**  
   - 경우에 따라 $$G_s(G_d(x))\approx x$$ $$\Rightarrow$$ Cycle-Consistency Loss 추가[1]  

## 7.3 장·단점 비교

| 구분           | 장점                                                                 | 단점                                                       |
|---------------|--------------------------------------------------------------------|------------------------------------------------------------|
| 명시적 모델링 | – 블러·노이즈 수식화로 안정적 성능– 수학적 해석 가능               | – 복합 열화(ISP, 압축) 표현 불가– 파라미터 추정 필요    |
| 암시적 모델링 | – 수식 가정 불필요– 실제 LR 분포 반영– 단일 네트워크로 통합 가능| – GAN 훈련 불안정– 가짜 질감·아티팩트 위험– 고품질 데이터 필요 |

## 7.4 활용 팁

1. **실제 LR 데이터가 풍부**하다면 → 암시적 모델링 추천  
2. **명확한 블러·노이즈 환경**이라면 → 명시적 모델링으로 높은 PSNR/SSIM 확보  
3. **하이브리드**: 암시적으로 LR 분포 학습 후, 명시적 모델링을 SR 단계에 적용 가능  

## 7.5 결론

암시적 열화 모델링은 실제 환경의 복잡한 열화를 **직접 학습**한다는 점에서 강력하나, **GAN 훈련 안정성**과 **도메인 갭** 문제를 해결하는 것이 핵심 과제입니다. 앞으로는 GAN 대체 기법(예: Diffusion 모델)이나 도메인 적응 기법 고도화가 필수적일 것입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6055dde7-45fe-4156-9d86-d9ded19e3fe2/2107.03055v1.pdf

### 2.4 성능 향상 및 한계  
- **Explicit w/ 외부 데이터**  
  - Pros: 알려진 $$k,n$$ 범위 내에서 우수한 PSNR/SSIM  
  - Cons: 범위를 벗어나면 급락. 정확한 $$k,n$$ 추정 필요→실제 이미지엔 부정확  
- **Explicit w/ 단일 이미지**  
  - Pros: external data 불필요, self-supervised  
  - Cons: patch recurrence를 전제로→texture 다양/단순 장면에 취약  
- **Implicit w/ 외부 데이터**  
  - Pros: 복합 degradations 실 모델링 가능  
  - Cons: GAN artifacts (가짜 질감), domain gap 문제, paired data 구성 어려움  

## 3. 일반화 성능 향상 가능성

1. **Implicit + Single Image**  
   - 본 논문이 지적한 미개척 영역.  
   - **아이디어**: 단일 이미지에 내재된 **데이터 분포**를 학습(예: human-in-the-loop, modulation networks로 컨트롤 파라미터 제공)  
2. **Domain-aware 학습**  
   - DASR처럼 **real LR/distributed generated LR** 모두 사용해 domain gap 최소화  
3. **Contrastive Degradation Encoding**  
   - DRL-DASR : degradation encoder의 contrastive 학습으로 일반화 강화  
4. **Hybrid Approaches**  
   - explicit + implicit 융합: 내부 통계로 초기 추정→GAN으로 polishing  

## 4. 향후 영향 및 연구 시 고려점

- **Benchmark 구축**: 동일 데이터·평가환경 하에서 각 분류별 fair comparison 필요  
- **General SR Prior 설계**: 단일 이미지 암시적 모델링을 위한 **강건한 SR prior** 발굴  
- **Human-in-the-Loop 인터페이스**: 사용자 입력(참조 이미지·조절 파라미터) 활용  
- **안전한 GAN 대체 모델**: diffusion 모델·flow 기반 prior(NF) 활용  
- **실제 애플리케이션 맞춤화**: 스마트폰·감시카메라·옛 사진 각각에 특화된 복합 degradations 연구  

이 논문은 Blind SR 분야를 **명쾌한 Taxonomy**와 **깊이 있는 리뷰**로 재정비함으로써, 이후 연구자들이 **현재 위치**를 빠르게 파악하고, **미개척 문제(특히 단일 이미지 암시적 모델링)** 를 향해 나아갈 수 있는 든든한 출발점을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6055dde7-45fe-4156-9d86-d9ded19e3fe2/2107.03055v1.pdf

# Abs
열화를 알 수 없는 저해상도 이미지를 초해상도로 초해상도화하는 것을 목표로 하는 블라인드 이미지 초해상도(SR)는 실제 애플리케이션을 홍보하는 데 있어 중요성 때문에 점점 더 많은 관심을 받고 있습니다.  
특히 강력한 딥 러닝 기술을 사용하여 최근에 새롭고 효과적인 솔루션이 많이 제안되었습니다.  
수년간의 노력에도 불구하고 이는 여전히 어려운 연구 문제로 남아 있습니다.  
이 논문에서는 블라인드 이미지 SR의 최근 진행 상황에 대한 체계적인 검토 역할을 하며, 열화 모델링 방법과 SR 모델 해결에 사용되는 데이터에 따라 기존 방법을 세 가지 다른 클래스로 분류하는 분류법을 제안합니다.  
이 분류법은 기존 방법을 요약하고 구별하는 데 도움이 됩니다.  
저희는 현재 연구 상태에 대한 통찰력을 제공할 뿐만 아니라 탐구할 가치가 있는 새로운 연구 방향을 밝힐 수 있기를 바랍니다.  
또한 일반적으로 사용되는 데이터 세트와 블라인드 이미지 SR과 관련된 이전 대회에 대한 요약을 제공합니다.  
마지막으로 합성 및 실제 테스트 이미지를 모두 사용하여 장단점에 대한 자세한 분석과 함께 다양한 방법 간의 비교를 제공합니다.

# Introduction
SINGLE-IMAGE 초해상도(SISR)는 관찰된 저해상도(LR) 입력에서 고해상도(HR) 이미지를 복구하는 것을 목표로 오랫동안 저수준 비전의 근본적인 문제였습니다.  
연구 커뮤니티의 수년간의 노력은 특히 딥 러닝 기술 [1], [2], [3], [4], [5]의 호황과 함께 이 분야에서 괄목할 만한 발전을 가져왔습니다.  
그러나 대부분의 기존 방법은 HR 이미지에서 LR 이미지로의 사전 정의된 열화 프로세스(예: bicubic downsampling)를 가정하며, 이는 복잡한 열화 유형을 가진 실제 이미지의 경우 거의 적용되지 않습니다.  
이러한 격차를 메우기 위해 최근 몇 년 동안 알려지지 않은 열화, 즉 Blind SR에 대한 접근 방식에 대한 관심이 커지고 있습니다.  
많은 흥미로운 개선에도 불구하고 이러한 제안된 방법은 일반적으로 성능이 특정 유형의 입력으로 제한되고 다른 경우에는 극적으로 저하되기 때문에 많은 실제 시나리오에서 실패하는 경향이 있습니다.  
주요 이유는 입력 LR과 관련된 열화 유형에 대해 여전히 몇 가지 가정을 하기 때문입니다.  
그림 1(a)에서 일부 최첨단 방법의 가정된 열화 유형을 사용하지만 동일한 HR을 대상으로 하는 네 가지 다른 LR 입력을 보여주는 그림을 볼 수 있습니다.  

![]()

따라서 가정된 데이터 분포에서 벗어난 임의의 입력이 주어지면 이러한 방법은 훨씬 덜 만족스러운 결과를 생성할 수밖에 없습니다.  
그림 1(b)는 유명 영화 Forest Gump에서 잘라낸 실제 이미지에 대한 다른 SR 결과를 보여주며, 이는 네 가지 최신 방법으로 생성됩니다.  
이 실제 이미지가 입력에 대한 가정을 엄격하게 따르지 않기 때문에 이러한 방법 중 어느 것도 좋은 시청 경험에 대한 기대에 부응하지 못했다는 것을 발견할 수 있습니다.  
사실, 우리가 당면한 특정 이미지에 대해 어떤 방법을 선택해야 할지 또는 기존 방법을 사용하여 정말로 고품질의 결과를 얻을 수 있는지에 대해 혼란을 느끼는 것은 드문 일이 아닙니다.  

본 논문에서는 최근 Blind SR의 진행 상황에 대한 체계적인 조사를 통해 이러한 혼란을 해소하고자 합니다.  
또한 제안된 방법을 되돌아보고 성찰하여 현재 연구 상태와 남아있는 격차에 대해 명확하게 이해하는 것이 매우 필요합니다.  
위에서 언급한 바와 같이, 단일 이미지에 대한 KernelGAN[6]은 멋져 보이지만 반복 계획이 있는 IKC[7] 또는 짝을 이루지 않은 훈련 데이터가 있는 CinCGAN[8]은 어떻습니까?  
또한 모든 Blind SR 방법이 실제 이미지에 대해 잘 작동한다고 주장하더라도 그림 1의 경우와 마찬가지로 자신의 이미지에 대해 만족스러운 출력을 얻기 위해 여전히 어려움을 겪을 수 있습니다.  
이 개발 단계에서 다음과 같이 질문을 해야 할 때입니다.  
문제를 어느 정도 해결했습니까?  
무엇이 우리를 가로막고 있으며 향후 노력을 위해 어디로 가야 합니까?  

따라서 이 논문은 최근 진행 상황을 나열하는 것 이상의 역할을 하는 것을 목표로 합니다.  
특히 기존 접근 방식을 효과적으로 분류하기 위한 분류법을 제안하는데, 이는 다양한 방법을 명확하게 구분하고 연구 격차를 자연스럽게 드러내는 것입니다.  
이 분류법을 기반으로 각 방법이 기존 작업으로 구성된 넓은 그림 내에서 고유한 위치를 갖도록 하는 것이 목표입니다.  
이 그림은 향후 작업에서 다양한 종류의 방법 간의 합리적이고 공정한 비교에 대한 지침을 제공할 수 있습니다.  
또한 각 종류의 접근 방식의 한계와 함께 응용 범위를 요약하여 독자가 다양한 시나리오에 적합한 방법을 효율적으로 선택할 수 있도록 도울 것입니다.  
이 논문에서는 face SR 또는 depth map SR과 같은 도메인별 주제를 포함하지 않고 일반 이미지에 대한 SISR에 중점을 둡니다.

저희의 기여는 주로 세 가지입니다:  
1) 저희는 다양한 접근 방식의 개선점과 한계를 포함하여 Blind 이미지 초해상도의 최근 진행 상황에 대한 체계적인 조사를 제시합니다.  
2) 저희는 기존 방법을 효과적으로 분류하고 몇 가지 연구 격차를 드러내기 위한 분류법을 제안합니다.  
3) 저희는 현재 연구 상태와 유망한 미래 방향에 대한 깊은 통찰력을 제공합니다.  
다음 섹션에서는 먼저 2절에서 일반적으로 사용되는 일부 SR 모델의 수학적 공식을 소개하고 3절에서 블라인드 SR을 다룰 때 직면하는 실제 이미지의 과제에 대해 논의합니다.  
그런 다음 4절에서 제안된 분류법을 제시합니다.  
Non-Blind 환경의 연구 상태는 블라인드 SR의 기반을 마련했기 때문에 Non-Blind SISR에 대한 간략한 검토는 5절에서 제공됩니다.  
그런 다음 6절과 7절의 각 범주의 방법에 대해 자세히 설명한 다음 8절에서 일반적으로 사용되는 데이터 세트와 블라인드 SR 분야의 이전 경쟁 제품에 대한 요약을 설명합니다.  
일부 대표적인 방법 간의 정량적 및 정성적 비교는 9절에 포함되어 있습니다. 마지막으로, 저희는 이 설문 조사를 통한 통찰력과 10절에서 미래 방향에 대한 관점에 대한 결론을 도출합니다.

# PROBLEM FORMULATION
이 섹션에서는 SISR 문제의 수학적 공식을 소개합니다.  
구체적으로 SISR은 주어진 LR 입력, 특히 HR의 고주파수 콘텐츠에서 HR 이미지를 재구성하는 작업을 말합니다.  
HR에서 LR로의 기본 분해 과정은 일반적으로 다음 방정식으로 표현할 수 있습니다: 

y = f (x; s)

여기서 x, y는 각각 HR 영상과 LR 영상을 나타내며 f는 scale factor s를 갖는 열화 함수입니다.  
따라서 SR 문제는 역함수 f^-1을 모델링하고 푸는 것과 같습니다.  
Non-Blind SR의 배경에서 f는 일반적으로 bicubic downsampling으로 가정됩니다:

y = x ↓bic,s

또는 다운 샘플링과 커널 kg과 고정 가우시안 블러의 조합:

y = (x ⊗ kg) ↓s

여기서 ⊗는 컨볼루션 연산을 나타냅니다.  
두 가정 모두에서 해당 SR 모델은 이러한 특정 종류의 열화를 가진 LR 입력만 처리할 수 있습니다.  
열화 유형이 다른 다른 LR 이미지의 경우 SR 모델과 입력의 고유 열화 사이의 불일치로 인해 SR 결과가 심각하게 손상될 수 있습니다[7], [12].  
그림 2는 이미지 도메인 적응의 관점에서 이러한 불일치에 대한 예를 보여줍니다:  
사전 정의된 열화에 해당하는 SR 모델이 임의의 LR 입력에 적용되면 SR 출력과 대상 Natural HR 도메인의 원하는 이미지 샘플 사이에 큰 도메인 차이가 존재하여 품질이 저하될 수 있습니다.  

![]()

따라서 이러한 격차를 해소하기 위해 알려지지 않은 열화에 대한 Blind SR이라는 주제가 제안되었습니다.  
지금까지 Blind SR을 위한 열화 과정을 모델링하는 방법은 식 (3)의 확장을 기반으로 한 명시적 모델링과 외부 데이터 세트 내의 고유 분포를 통한 암시적 모델링 두 가지가 있었습니다.  
구체적으로 명시적 모델링은 일반적으로 식 (3)의 보다 일반적인 형태인 소위 고전적 열화 모델을 사용합니다:  

y = (x ⊗ k) ↓s +n

여기서 SR blur kernel k와 추가된 noise n은 열화 과정에 관여하는 두 가지 주요 요인이며, 임의의 LR 입력에 대해서는 이 두 요인과 관련된 매개변수를 알 수 없습니다.  
그림 3(a)는 서로 다른 k와 n을 가진 여러 이미지 예를 보여주는데, 이는 bicubic 다운샘플링된 대응물보다 훨씬 더 저하된 것입니다.  
일부 접근법은 외부 데이터 세트를 활용하여 IKC[7] 및 SRMD[13]와 같이 다양한 k 또는 n의 큰 세트에 잘 적응된 SR 모델을 학습합니다.  
블러링과 잡음 외에도 quality factor q[14]를 사용한 JPEG compression과 같이 더 복잡하고 현실적인 열화 유형도 공식에 포함될 수 있습니다:  

y = ((x ⊗ k) ↓s +n)JPEG_q

또 다른 방법은 고전적 열화 모델에서 파생된 단일 이미지 내의 내부 통계를 활용하므로 ZSSR[12] 및 DGDML-SR[15]과 같은 훈련을 위한 외부 데이터 세트가 필요하지 않습니다.  
사실 내부 통계적 정보는 이미지의 패치 재발 속성을 반영할 뿐이며, 독자는 그림 3(b)를 참조하여 설명을 얻을 수 있습니다.  

![]()

그럼에도 불구하고 실제 열화는 일반적으로 그림 3(c)와 같이 여러 열화 유형의 명시적 조합으로 모델링하기에는 너무 복잡합니다.  
따라서 implicit 모델링은 explicit 모델링 기능을 회피하려고 시도합니다.  
대신 데이터 배포를 통해 암묵적으로 열화 프로세스를 정의하며, 암시적 모델링을 사용하는 기존의 모든 접근 방식은 훈련을 위한 외부 데이터 세트가 필요합니다. 
 
일반적으로 이러한 방법은 CinCGAN[8]과 같은 훈련 데이터 세트 내에 있는 암시적 열화 모델을 파악하기 위해 생성적 적대 네트워크(GAN)[16]를 사용한 데이터 배포 학습을 활용합니다.  
Blind SR에 많은 모델이 제시되었지만, 실제 이미지의 작은 세트만 다루기 때문에 여전히 문제가 많습니다.  
기존 방법은 종종 실제 설정에 초점을 맞춘다고 주장하지만 실제로 일부 디지털 카메라[17], [18]에서 촬영한 이미지와 같은 특정 장면을 가정합니다.  
실제로 실제 이미지는 기본 열화 유형이 크게 다르며, 특정 유형을 위해 설계된 SR 모델은 다른 모델에 쉽게 실패할 수 있습니다.  
다음 섹션에서는 Blind SR 분야에 심각한 도전을 제기한 다양한 유형의 실제 이미지에 대해 간략하게 설명합니다.  

# CHALLENGES FROM REAL-WORLD IMAGES

# TAXONOMY
이 섹션에서는 검토 및 분석을 위한 지침 역할을 하기 위해 제안된 분류법에 대해 자세히 설명합니다.  
Sec.2에 따르면 블라인드 SR과 관련된 열화 과정을 모델링하는 두 가지 방법이 있습니다.  
고전적 열화 모델 또는 그 변형을 기반으로 한 명시적 모델링과 외부 데이터 세트 간의 데이터 분포를 사용한 암시적 모델링입니다.  
명시적 모델링의 기본 아이디어는 일반적으로 식 (4)에서 k와 n으로 매개변수화되는 큰 열화 세트를 포함하는 외부 학습 데이터를 사용하여 SR 모델을 학습하는 것입니다.  
대표적인 접근 방식에는 SRMD[13], IKC[7] 및 KMSR[23]이 있습니다.   KernelGAN[6] 및 ZSSR[12]과 같은 패치 재발의 내부 통계적 정보를 활용하는 방법을 제안하는 또 다른 그룹의 방법이 있습니다.  
이러한 유형의 모델링은 주로 고전적 열화 모델을 기반으로 합니다.  
반면, 암시적 모델링이 있는 방법은 명시적 매개변수화에 의존하지 않으며 일반적으로 외부 데이터 세트 내의 데이터 분포를 통해 기본 SR 모델을 암묵적으로 학습합니다.  
이러한 방법 중에는 CinCGAN[8] 및 FSSR[24]이 있습니다.  

따라서 저희는 열화 모델링 방법과 그림 7과 같이 SR 모델을 해결하는 데 사용되는 데이터에 따라 기존 접근 방식을 효과적으로 분류하는 분류법을 제안합니다. 

![]()
 
이 분류법을 채택하는 이유는 세 가지입니다.  
첫째, 명시적 모델링과 암시적 모델링을 구별하는 것은 특정 방법의 가정, 즉 이 방법이 처리하고자 하는 열화의 종류를 이해하는 데 도움이 됩니다.  
둘째, 외부 데이터 세트를 사용하든 단일 입력 이미지를 사용하든 명시적 모델링을 사용하든 이미지별 적응 전략이 다르든, 마지막으로 기존 접근 방식을 이러한 클래스로 분류한 후 하나의 남아 있는 연구 격차는 단일 이미지를 사용한 암시적 모델링으로 자연스럽게 드러납니다.  
저희는 이 방향이 다양한 콘텐츠를 가진 일반적인 실제 이미지를 다루는 측면에서 유망하다고 주장하며, 이 방향으로 새로운 솔루션에 대한 실현 가능한 제안도 제안하려고 노력할 것입니다.  
다음 섹션에서는 먼저 Blind SR 방법의 기반을 설정하는 Non-Blind SISR에 대해 간략하게 개요를 설명합니다.  
그런 다음 명시적 모델링이 있는 방법은 6장에서 소개하고, 암시적 모델링을 사용하는 방법은 7장에서 논의합니다.  
각 유형의 방법에 대해 개발 과정을 따라 검토를 전개하고, 그 한계에 대해 분석하여 향후 작업에 영감을 줄 것입니다.

# OVERVIEW OF NON-BLIND SINGLE-IMAGE SUPER-RESOLUTION
2절에서 설명한 바와 같이, Non-Blind SR은 HR 출력을 해결하기 위해 알려진 고정된 열화 과정을 가정합니다.  
딥 러닝 기술이 개발되기 전에는 많은 전통적인 기술이 예제 기반입니다.  
[25], [26], [27], [28]은 외부 HRLR 예시 쌍을 사용하여 LR에서 HR로의 매핑 함수를 학습하며, 매핑 학습은 일반적으로 compact dictionary 또는 다양체 공간을 기반으로 합니다.  
다른 일부 [29], [30]은 외부 데이터 세트를 사용하지 않고 단일 이미지 내에서 내부 self-similarity의 속성을 활용합니다.  
2014년 SRCNN[31]의 선구적인 작업은 이 작업을 해결하기 위해 컨볼루션 신경망(CNN)을 배포하는 새로운 시대를 열었고, 그림 8과 같이 이후 작업을 위한 기본 프레임워크도 설정했습니다.  

![]()

SISR 작업을 위해 일반적으로 채택되는 CNN 프레임워크에는 입력 LR 이미지를 feature map으로 변환하기 위한 shallow feature extraction, extracted shallow feature을 기반으로 한 deep feature extraction 또는 매핑, 마지막으로 SR 출력 재구성의 세 가지 주요 모듈이 포함됩니다.  
Residual learning은 또한 이미지 수준 [33] 또는 feature 수준 [34]에서 훈련 프로세스를 용이하게 하기 위해 널리 채택되었습니다.  
최근 몇 년 동안 residual block [34], [35], [36], recursive or recurrent structure [37], [38], attention mechanism [39], [40], sub-pixel convolution [41] 등을 도입하는 것과 같은 심층 특징 추출 및 SR 재구성 모듈이 많이 개선되었습니다.  
또한 SR 결과 [32], [42], [43]의 더 나은 지각 품질을 위해 multiple loss function도 제안됩니다.  
이러한 기술은 재구성 정확도와 효율성 측면에서 괄목할 만한 진전을 가져오며, bicubic-downsampling 가정을 사용한 Non-Blind SISR은 실제로 최종 결과에 도달합니다.  
그러나 이러한 non-Blind 모델은 일반적으로 가정된 열화에서 벗어나 더 복잡한 열화를 가진 입력 이미지로 일반화하는 데 어려움을 겪습니다.  
Non-Blind SR 네트워크의 일부 실패 사례는 그림 9에 나와 있으며, 여기서 네트워크는 가정된 열화 모델에 따라 bicubicly downsample된 깨끗한 입력에서 잘 수행되지만 흐릿하거나 노이즈가 많은 입력 이미지를 처리할 수 없습니다.  

![]()

따라서 이 조사의 주요 초점인 Blind SR 설정 방법을 제안하는 것이 요구되며, 다음 두 섹션에서 자세히 살펴보도록 하겠습니다.

# EXPLICIT DEGRADATION MODELLING
이 섹션에서는 일반적으로 식 (4)에 표시된 고전적인 열화 모델을 기반으로 열화 과정을 명시적으로 모델링하여 최근에 제안된 블라인드 SR 방법을 다룹니다.  
또한 이러한 접근 방식은 외부 데이터 세트를 사용하는지, SR 문제를 해결하기 위해 단일 입력 이미지에 의존하는지에 따라 두 가지 하위 클래스로 추가로 분류할 수 있습니다.  

## Classical Degradation Model with External Dataset
이러한 종류의 접근 방식은 외부 데이터 세트를 활용하여 variant SR blur kernel k와 노이즈 n에 잘 적응된 SR 모델, 특히 전자를 훈련합니다.  
일반적으로 SR 모델은 컨볼루션 신경망(CNN)으로 매개변수화되며, 특정 LR 이미지에 대한 k 또는 n에 대한 추정은 기능 적응을 위해 SR 모델에 대한 조건부 입력으로 사용됩니다.  
훈련 프로세스 후 모델은 훈련 데이터 세트에 포함된 열화 유형으로 LR 입력에 대해 만족스러운 결과를 생성할 수 있습니다.  
특정 접근 방식에 제안된 프레임워크에 열화 추정이 포함되어 있는지 여부에 따라 이러한 접근 방식을 커널 추정이 없는 이미지별 적응과 커널 추정을 통한 이미지별 적응의 두 가지 유형으로 더 나눕니다.  
좀 더 구체적으로 설명하자면, 첫 번째 유형은 추정된 열화 정보를 추가 입력으로 받고 이미지별 적응을 위해 추정 입력을 활용하는 방법에 중점을 둔 반면, 두 번째 유형은 SR 프로세스와 함께 커널 추정에 특별한 주의를 기울입니다. 전체 프레임워크에 대한 설명이 그림 10에 나와 있습니다.  

![]()

### Image-Specific Adaptation without Kernel Estimation
다양한 열화에 대한 초해상도(SRMD)[13]는 SR 모델에 대한 통합 입력으로 LR 입력 이미지와 열화 맵을 직접 연결하여 특정 열화에 따른 feature 조정을 허용하고 단일 모델에서 여러 열화 유형을 커버할 것을 제안합니다.  
LR 이미지와 동일한 차원의 열화 맵을 생성하기 위해 dimensionality stretching이라는 전략을 도입했습니다.  
특히 크기가 rxr인 SR blur 커널을 r^2 길이 벡터로 flatten하고 principal component analysis(PCA)을 사용하여 t차원으로 축소하여 kernel coding을 얻습니다.  
n과 관련된 추정된 노이즈 레벨 σ와 연결한 후 (t+1)차원 벡터를 수직 및 수평으로 모두 확장하여 최종 H×W×(t+1)차원 열화 맵을 얻습니다.  
이 전략은 공간적으로 변형된 열화에 대해 불균일한 맵으로 쉽게 확장할 수 있습니다.  
SRMD의 SR 재구성 네트워크는 Non-Blind SR에서 일반적으로 채택되는 것과 유사합니다.  
전체 파이프라인은 그림 11(a)에 나와 있습니다.

![]()

SRMD에 이어 UDVD[44]도 열화 맵을 SR 재구성을 위한 추가 입력으로 사용하지만 픽셀당 dynamic convolution을 사용하여 이미지 간의 variational degradation를 보다 효과적으로 처리함으로써 한 단계 더 발전했습니다.  
특히, dynamic convolution을 가진 여러 dynamic block으로 구성된 refinement network가 feature extraction module로 cascade되고, 이러한 각 블록은 이전 블록의 결과를 기반으로 반복적인 방식으로 SR 출력을 정제합니다.  
또한 variant blind SR[47]은 PCA 기법을 shallow neural network으로 대체하여 특정 SR 모델에 더 적합한 kernel mapping을 잠재적으로 학습할 수 있도록 커널 코딩 작업에 대한 개선을 제안합니다. 
SRMD는 SR 모델의 일반화 용량을 variant SR 커널 및 노이즈 수준으로 확장하지만, 특히 motion blur와 같은 불규칙한 패턴을 가진 경우 임의의 커널을 효과적으로 인코딩하고 단일 모델로 처리하는 것이 일반적으로 사소하지 않기 때문에 여전히 범위가 매우 제한적입니다.  
따라서 degradation map generation을 위해 kernel coding이 필요하지 않은 MAP 프레임워크를 기반으로 하는 또 다른 방법 그룹이 제안되었습니다.  
특히 deep plug-and-play super-resolution(DPSR)[45]는 SR 네트워크를 MAP 기반 반복 최적화 체계에 통합합니다.  
주로 매개변수 λ로 정규화된 데이터 항 D와 이전 항 P로 구성된 다음과 같은 objective function를 최소화하여 HR 이미지를 해결합니다:  

E(x) = 1 / 2σ2 ‖y − x ↓s ⊗k‖^2 + λΦ(x) = D + λP

해당 열화 모델은 식 (4)의 수정된 버전으로 다운 샘플링 프로세스를 블러링 작업에서 분리합니다:

y = (x ↓s ⊗k) + n

식 (6)에 표시된 objective function는 half-quadratic splitting(HQS) 알고리즘을 사용하여 두 가지 하위 문제로 나눌 수 있습니다.  
하나는 deblurring 작업을 다루고 다른 하나는 매개 변수 k를 가진 데이터 항 D와 관련이 있으며, 다른 하나는 일부 가상 노이즈 수준 μ를 가진 bicubic downsample 이미지를 초해상도하는 것을 목표로 하며 이전 항 P와 관련이 있습니다.  
다행히 첫 번째 하위 문제는 kernel coding 없이 Fast Fourier Transform으로 closed form 형태로 해결할 수 있으므로 모델이 더 복잡한 커널에 대처할 수 있습니다.  
또한 blurring and downsampling operation의 분리 덕분에 두 번째 하위 문제는 추가된 노이즈를 처리할 수 있는 non-blind SR 네트워크로 모델링할 수 있으며, 이 네트워크는 단일 노이즈 맵을 추가 입력으로 사용하여 SRMD 프레임워크에서 직접 적응할 수 있습니다.  
Unfolding super-resolution network(USRNet) [46]도 MAP 프레임워크를 채택하지만 식 (4)의 원래 열화 모델을 기반으로 하며, 해당하는 두 하위 문제는 kernel k에 의해 블러링된 LR 이미지를 초해상도하고 virtual noise level μ를 가진 HR 이미지를 노이즈 제거합니다.  
DPSR의 iterative optimization process를 반복 방식으로 end-to-end 학습 가능한 네트워크로 펼쳐 솔루션 프레임워크를 향상시켜 두 하위 문제 간의 joint optimization를 가능하게 합니다.  
DPSR과 USRNet의 솔루션 프레임워크를 비교하면 그림 11(b)과 (c)가 표시됩니다.  

![]()

이외에도 plug-and-play technique을 활용하는 몇 가지 다른 방법으로는 [51], [52], [53]이 있습니다.

한계점 : 앞서 언급한 진전에도 불구하고, 이러한 종류의 방법에는 한 가지 분명한 단점이 있습니다:  
이들은 모두 열화 추정의 추가된 입력, 특히 SR kernel k에 의존합니다.  
그러나 임의의 LR 이미지에서 올바른 kernel을 추정하는 것은 쉬운 작업이 아니며, 부정확한 추정된 입력은 kernel 불일치를 유발하고 SR 성능을 크게 저하시킬 것입니다[7], [12].  
그림 12는 SRMD 방법을 기반으로 SR 결과를 정확한 kernel과 부정확한 kernel 간의 비교를 보여줍니다.  

![]()

따라서 신뢰할 수 있는 열화 추정을 위한 방법이 있는 경우에만 만족스러운 SR 출력을 빠르게 얻을 수 있으며, 그렇지 않으면 더 나은 결과를 위해 적절한 추정 입력을 수동으로 선택하는 지루한 작업을 하게 될 수 있습니다.  

따라서 다음 부분에서는 kernel 추정을 SR 프레임워크에 통합하여 보다 강력한 성능을 제공하는 또 다른 종류의 접근 방식을 소개합니다.

### Image-Specific Adaptation with Kernel Estimation
Iterative kernel correction(IKC)[7]은 만족스러운 결과에 점진적으로 접근하기 위해 반복적인 방식으로 kernel estimation을 수정할 것을 제안합니다.  
이 방법의 백미는 kernel mismatch로 인해 SR 이미지 내의 결함이 규칙적인 패턴을 갖는 경향이 있기 때문에 중간 SR 결과를 활용하는 것입니다.  
특히 현재 커널에 조건화된 SR 이미지가 주어지면 corrector network를 사용하여 kernel correcting residual를 추정합니다.  
그런 다음 업데이트된 커널을 사용하여 더 적은 결함으로 새로운 SR 결과를 생성합니다.  
SR 네트워크에는 각 residual block에  spatial feature transform [54] layer가 포함되어 있으며 현재 커널은 feature adaptation을 위한 transforming parameter를 생성하는 데 사용되며, 이는 SRMD에서 제안한 입력의 direct concatenation보다 더 효과적일 수 있습니다.  
또한 입력 LR 이미지만 기반한 kernel initialization를 위해 predictor network를 적용하고 kernel coding을 위해 dimensionality stretching을 채택합니다.  
보다 최근의 작업인 deep alternating network(DAN)[48]는 IKC 프레임워크를 더욱 향상시킵니다.  
IKC와 같이 각 하위 네트워크를 별도로 훈련하는 대신 corrector 및 SR 네트워크를 end-to-end 훈련 가능한 네트워크로 통합합니다.  
이 joint training 전략은 두 네트워크를 서로 더 호환되도록 만들 수 있습니다.  
또한 corrector는 중간 SR 결과에 조건화된 kernel estimation에 원본 LR 입력을 사용하므로 보다 강력한 커널 추정 성능에 도움이 됩니다.  
IKC와 DAN의 전체 프레임워크는 그림 13(a)과 (b)에 나와 있습니다.  

![]()

커널 추정을 위해 SR 결함을 사용하는 아이디어는 variant blind SR(VBSR)[47]에도 사용되지만 커널 자체 대신 SR 출력의 error map을 추정하도록 커널 discriminator를 훈련하고 그림 13(c)와 같이 추론 단계에서 SR 출력의 오류를 최소화하여 최적의 커널을 찾습니다.  
SR 커널 외에도 더 많은 열화 유형에 대한 추정도 연구되었습니다.  
CBSR[14]은 노이즈 및 커널 추정을 위한 두 개의 하위 네트워크를 non-blind SR 네트워크와 결합하여 blind SR을 위한 unified cascaded architecture를 형성합니다.

실제로 IKC와 DAN이 채택한 반복 방식은 도메인 적응의 관점에서 잘 해석될 수 있습니다.  
SRMD와 같이 single stroke로 최종 SR 출력을 생성하는 대신 입력 LR에서 대상 Natural HR 도메인으로 이동하는 동안 여러 중간 SR 결과를 interchange station으로 선택하여 그림 2의 domain 차이를 단계적으로 통과합니다.  
이 두 가지 방법은 커널 추정 입력의 정확도에 따라 SRMD 프레임워크보다 더 강력한 성능을 가질 수 있습니다. 

그럼에도 불구하고 이러한 반복 방식은 일반적으로 더 많은 추론 시간을 소비하고 최적의 반복 횟수를 선택하기 위해 사람의 개입이 필요합니다.  
이러한 문제를 해결하기 위해 최근 일부 연구에서는 보다 정확한 열화 추정 또는 보다 효율적인 기능 적응 전략을 도입하여 non-iterative framework를 제안합니다. Blind SR을 위한 Unsupervised degradation representation learning(DRLDASR)[55]은 latent feature space에서 훈련 가능한 encoder로 열화 정보를 추정하려고 하며, 열화 인코더는 비지도 방식으로 대조적인 방식으로 학습됩니다.  
특히, query input과 동일한 열화를 가진 LR 샘플은 긍정적인 예시로 간주되고 다른 열화를 가진 샘플은 부정적인 예시로 간주됩니다.  
그런 다음 모든 샘플 간의 상호 정보가 latent space에서 최대화되어 content-invariant degradation representation으로 이어집니다.  
또한 추정된 열화 표현은 SR 네트워크에서 해당 convolutional kernel과 modulation coefficient를 생성하는 데 사용됩니다.  
이러한 프레임워크는 single forward pass로 만족스러운 SR 결과를 달성할 수 있습니다.  
Kernel-oriented adaptive local adjustment(KOalanet)[56]도 SR 네트워크를 특정 열화에 적응시키는 유사한 dynamic kernel 전략을 사용하며, local kernel estimation을 위해 다운샘플링 네트워크를 사용하여 non-iterative framework를 공간적으로 변형된 열화로 더욱 확장합니다.  
또 다른 작업인 adaptive modulation network with reinforcement learning (AMNet-RL) [57]는 수정된 버전의 adaptive instance norm(AdaIN)을 제안합니다[58].  
kernel estimation을 SR 네트워크에 통합하기 위해 강화 학습 프레임워크에서 차별화할 수 없는 지각 메트릭(예: NIQE [59])으로 blind SR 모델을 최적화하는 데 개척했습니다.

훈련 데이터 세트, 특히 실제 이미지에서 추정된 보다 현실적인 커널의 더 많은 열화를 커버하여 blind SR 모델을 학습하는 것을 제안하는 몇 가지 다른 접근 방식도 있습니다.  
예를 들어, kernel modelling super-resolution(KMSR)[23]는 실제 LR 이미지에서 추정된 일부 현실적인 SR 커널을 기반으로 data distribution learning을 통해 대규모 kernel pool을 구축합니다.  
그런 다음 이 풀의 커널을 사용하여 고전적인 열화 모델에 따라 HR-LR 훈련 쌍을 합성하고 훈련 프로세스는 지도 학습과 함께 non-blind 설정을 따릅니다.  
일반적으로 보다 일반적인 훈련 데이터 세트를 사용하면 SR 모델이 암시적으로 서로 다른 열화를 가진 LR 입력을 구별하고 적응적으로 처리할 수 있습니다.  
즉, SR 모델은 훈련 프로세스에서 커널 추정을 위한 더 많은 양의 데이터를 암묵적으로 부여받으므로 프레임워크에서 explicit kernel estimation을 피할 수 있습니다.  
그러나 이러한 직접적인 방법은 [13]에서 주장한 바와 같이 최고의 성능으로 이어지지 않을 수 있습니다.  
RealSR[49] 및 RealSRGAN[50]에서도 유사한 전략을 사용하여 보다 일반적인 훈련 데이터 세트를 보다 많은 종류의 현실적인 커널로 구축합니다.  
이 프로세스는 그림 13(d)에 나와 있습니다.  

![]()

이러한 방법 외에도 correction filter[60]는 SR 모델을 미리 정의된 열화와 일치하도록 LR 입력을 수정하도록 설계되었으며, 이는 주로 LR의 kernel estimation을 기반으로 합니다.

한계점 : kernel estimation이 없는 접근 방식과 비교했을 때, 이러한 방법은 특히 추론 단계에서 kernel estimation 알고리즘을 검색하는 노력을 실질적으로 절약할 수 있으며 인상적인 성능을 입증했습니다.  
그러나 여전히 explicit modelling의 본질적인 단점을 피할 수 없습니다.  
IKC와 같은 커널 k로 인한 열화에 초점을 맞춘 SR 모델의 경우 모델링 범위를 벗어난 열화가 있는 LR 입력을 거의 처리할 수 없습니다.  
이 제한은 복잡한 실제 이미지에 대해 매우 어렵습니다.  
더 많은 열화 유형으로 모델을 재교육할 의향이 있다고 해도, 3절에 명시된 바와 같이 임의의 LR에서 열화를 명시적으로 모델링하고 충분한 외부 교육 데이터를 수집하는 것은 비현실적입니다.  

다음으로, 이미지별 SR 모델링을 위해 단일 입력 이미지만 사용하는 다른 유형의 방법으로 들어가 보겠습니다.

## Single Image Modelling with Internal Statistics
단일 이미지를 사용한 SR 모델링은 자연 이미지의 내부 통계적 정보를 기반으로 합니다:  
단일 이미지의 패치는 이 이미지의 다양한 스케일 내에서 그리고 여러 스케일에 걸쳐 반복되는 경향이 있습니다.  
내부 통계는 정량화되었으며 많은 자연 이미지에 대한 외부 통계적 정보보다 더 많은 예측력을 갖는 것으로 입증되었습니다[61].  
이론적 공식은 [62]에 의해 제공됩니다.  
구체적으로, HR 이미지 h와 그 LR 대응물 l은 동일한 카메라에 의해 촬영되지만, 후자의 경우 latter에 s-scale zoom-in된다고 가정할 수 있습니다:

h[n] = ∫ I(x)bH ( n/s − x)dx  
l[n] = ∫ I(x)bL(n − x)dx

여기서 I(x)는 연속 공간 이미지, b는 카메라의 point spread function(PSF), bH는 optical zoom의 경우 bL의 다운스케일링 버전이어야 합니다:

bH (x) = sbL(sx)

또한 노이즈 n이 없는 식 (4)의 고전적 열화 모델링을 사용하면 이산 형태로 표현되는 h와 l의 관계는 다음과 같습니다:

l[n] = ∑m h[m]k[sn − m]

이제 주어진 LR 영상에 대해 q와 r이 연속되는 장면에서 반복되는 패턴 P(x)를 갖는 두 개의 로컬 패치라고 가정합니다.  
여기서 r은 q보다 s배 큽니다:

r[n] = ∫ P(x/s)bL(n − x)dx = ∫ sP (x)bL(n − sx)dx  
q[n] = ∫ P (x)bL(n − x)dx

식 (10)을 기준으로 최종적으로 도착할 수 있습니다:

q[n] = ∑m r[m]k[sn − m]

즉, 동일한 LR에서 q와 r 사이의 관계는 커널 k와 관련된 HR과 그 LR 버전의 두 패치와 동등합니다.  
이 속성은 k를 추정하고 미지의 HR을 푸는 데 사용될 수 있습니다. 

Glasner et al. [30]은 2009년에 단일 이미지에서 SISR 문제를 해결하기 위해 내부 통계적 정보를 도입하는 선구적인 작업을 수행했습니다.  
후자의 경우 nonparametric blind SR(NPBSR)[62]은 이 프레임워크를 blind SR 설정으로 더욱 확장합니다.  
구체적으로, 최적의 커널 k는 서로 다른 스케일에서 반복되는 패치 간의 유사성을 극대화하는 것이라는 식 (14)의 관찰을 기반으로 SR blur kernel을 추정하는 MAP 프레임워크를 제안합니다.  
또한 NPBSR은 최적의 k가 카메라의 PSF가 아니라 흔한 지표과 달리 폭이 작은 커널이어야 함을 증명합니다.  

최근 GAN의 발전으로 blind kernel estimation에 patch recurrence을 사용하는 새로운 실현이 이루어졌습니다.  
KernelGAN[6]은 단일 이미지 내의 patch recurrence을 극대화하는 것을 데이터 분포 학습 문제로 해석합니다.  
최적 k에 의해 생성된 LR 이미지의 다운샘플링된 버전이 원래 LR과 동일한 패치 분포를 공유해야 한다고 가정합니다.  
GAN 프레임워크에서는 deep linear network를 generator로 사용하여 기본 SR 커널을 매개변수화하고, 생성된 패치를 원래 LR 이미지의 패치와 구별합니다.  
훈련이 완료되면 generator의 모든 convolutional filter를 컨볼루션하여 커널 추정치를 명시적으로 얻을 수 있습니다.  
훈련 프로세스는 외부 데이터 세트 없이 입력 LR에만 의존하며, 이는 self-supervised learning으로 볼 수 있습니다.  
Flow-based kernel prior(FKP)[63]은 kernel optimization를 위한 보다 효과적인 접근 방식을 개발하며, 여기서 latent space의 kernel prior은 normalizing flow(NF)[64], [65] 기법으로 학습됩니다.  
NF에 의해 허용된 latent space과 pixel space 간의 invertible mapping 덕분에 학습된 kernel manifold에서 최적의 k를 검색할 수 있습니다.  
이 프로세스는 무작위로 초기화된 심층 네트워크를 직접 최적화하는 것보다 더 효율적일 수 있으므로 보다 강력한 커널 추정 결과를 얻을 수 있습니다.  

patch recurrence property을 기반으로 한 self-supervision 아이디어는 SR 수행에도 직접 적용될 수 있습니다.  
NPBSR 및 KernelGAN과 같은 그룹의 저자들이 개발한 Zero-shot super-resolution(ZSSR)[12]는 사전 훈련 단계 없이 각 입력 LR을 초해상도화하기 위해 이미지별 CNN을 훈련하는 최초의 시도를 했습니다.  
훈련은 단일 LR 입력 y에서 생성된 HR-LR 쌍으로 수행되며, 여기서 y는 HR로 간주되고 커널 k로 다운 샘플링하여 안 좋은 LR 이미지가 생성됩니다.  
Data augmentation은 입력 이미지에서만 정보를 최대한 활용하는 데 활용됩니다.  
이러한 이미지 쌍으로 훈련된 CNN은 y의 다양한 스케일에서 특정 관계를 추론할 수 있으며, 이는 y를 초해상도화하는 데 사용됩니다.  
또한 ZSSR은 상관된 이미지 콘텐츠만 노이즈가 아닌 스케일에 걸쳐 반복되는 경향이 있다고 주장하기 때문에 LR 훈련 샘플에 약간의 노이즈를 추가하여 주의를 산만하게 하는 결함(예: Gaussian noises, JPEG artifacts)에 더 강력할 수 있습니다.  

실제로 ZSSR은 여전히 blind 설정을 위해 잘 설계되지 않았습니다.  
훈련을 위해 거친 LR 이미지의 생성을 안내하기 위해 입력으로 추정된 SR 블러 커널 k가 필요합니다.  
따라서 학습 기반 SR을 위한 통합된 self-supervision 프레임워크가 DGDML-SR[15] - depth guided degradation model for learning-based SR에서 제안됩니다.  
열화 네트워크와 SR 네트워크를 단일 아키텍처로 결합하여 KernelGAN의 기능과 유사하게 열화 과정을 시뮬레이션하도록 훈련되고 후자는 ZSSR처럼 SR 작업을 수행하는 것을 목표로 합니다.  
이 공동 프레임워크를 사용하면 SR 커널의 명시적인 추출 없이 생성된 LR을 SR 네트워크의 입력으로 직접 사용할 수 있습니다.  
또한 DGDML-SR은 depth가 작은 패치가 HR과 동일하고 depth가 큰 패치가 LR과 동일하다고 가정하여 입력 이미지의 depth map에 따라 페어링되지 않은 방식으로 HR 및 LR 패치를 샘플링할 것을 제안합니다.  
CycleGAN[66] 구조와 유사한 두 개의 네트워크를 동시에 훈련하기 위해 사용되며(그림 15 참조), 여기서 페어링되지 않은 HR 및 LR 패치는 data distribution learning을 위한 실제 샘플로 사용됩니다.

![]()

한계점 : 내부 통계적 정보를 사용한 self-supervision 아이디어는 대규모 외부 훈련 데이터 세트를 수집하는 노력이 필요하지 않기 때문에 변형 열화 유형을 가진 LR의 SR 이미지를 해결하는 데 매력적인 것으로 보입니다.  
그럼에도 불구하고, 특히 다양한 콘텐츠(예: 동물)가 있는 자연 이미지 또는 단조로운 장면(예: 하늘)의 경우 이러한 종류의 입력 이미지로 SR을 강력하게 수행하기 위해 스케일 간 반복 정보를 활용하기 어렵기 때문에 기본적인 가정이 쉽게 실패할 수 있습니다.  
따라서 이러한 접근 방식은 스케일 간에 자주 반복되는 콘텐츠가 있는 매우 제한된 이미지 세트에 대해 유리한 SR 출력만 생성할 수 있으며, 보다 일반적인 자연 이미지에 대해 단일 이미지 모델링을 위한 새로운 방법이 모색되기를 기다리고 있습니다.  

지금까지 explicit degradation modelling을 사용한 접근 방식과 장단점에 대해 개요를 보았습니다.  
열화 프로세스의 explicit degradation modelling은 명확하고 간단하지만 카메라 센서에서 발생하는 실제 열화와 같은 블러링 및 가산 노이즈 외에 더 복잡한 열화를 모델링하는 것에 비해 너무 간단할 수 있습니다.  
실제로 실제 이미지에는 일반적으로 여러 열화가 포함되며, 이러한 얽힌 요소를 명시적으로 잘 정의된 함수로 표현하기는 거의 어렵습니다.  
따라서 다른 방법 그룹에서는 data distribution learning을 통해 열화를 암시적으로 모델링할 것을 제안합니다.  
저희가 아는 한, 지금까지 암시적 모델링을 위한 외부 데이터 세트를 기반으로 한 접근 방식만 있었고, 다음 섹션에서 이에 대해 이야기하겠습니다.

# IMPLICIT DEGRADATION MODELLING
## Learning Data Distribution within External Dataset
