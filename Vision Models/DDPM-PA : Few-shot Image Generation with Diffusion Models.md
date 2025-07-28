# DDPN-PA : Few-shot Image Generation with Diffusion Models | Image generation
## 핵심 주장

이 논문은 **DDPM(Denoising Diffusion Probabilistic Models)을 적은 양의 데이터로 학습할 때 발생하는 과적합과 다양성 저하 문제를 최초로 체계적으로 분석하고, 이를 해결하기 위한 DDMP 쌍별 적응(DDPM-PA) 접근법을 제안**한다[1]. 기존의 few-shot 이미지 생성 연구는 주로 GAN 기반이었으나, 이 연구는 **diffusion 모델을 few-shot 이미지 생성에 처음으로 적용한 선구적 연구**이다[2].

## 주요 기여

### 1. **과적합 분석과 평가 지표 제안**
- **DDPM의 과적합 패턴 분석**: 훈련 데이터가 10개 또는 100개로 제한될 때, DDPM이 훈련 샘플을 복제하거나 대칭적 변형만 생성하는 과적합 문제를 확인[1]
- **Nearest-LPIPS 지표 도입**: 생성 샘플과 가장 유사한 훈련 샘플 간의 다양성을 정량화하는 새로운 평가 지표 제안[1]

### 2. **DDPM-PA 방법론 개발** 
- **상대적 쌍별 거리 보존**: 생성 샘플 간의 상대적 거리를 유지하여 다양성을 보존하는 쌍별 유사성 손실 함수 설계[1]
- **고주파 세부사항 향상**: Haar 웨이블릿 변환을 활용하여 고주파 정보를 강화하는 두 가지 접근법 제안[1]

### 3. **성능 우위 입증**
- **GAN 기반 방법 대비 우수한 성능**: 기존 최신 GAN 기반 few-shot 방법들보다 생성 품질과 다양성에서 우수한 결과 달성[1]

## 해결하고자 하는 문제

### **핵심 문제: 제한된 데이터에서의 DDPM 과적합**

1. **훈련 샘플 복제 문제**: 10-100개의 제한된 데이터로 학습할 때, DDPM이 새로운 샘플을 생성하지 못하고 기존 훈련 샘플만 복제하는 현상[1]

2. **다양성 저하**: 직접 fine-tuning된 DDPM이 수렴은 빠르지만 여전히 다양한 특징을 보존하지 못하고 거친 이미지만 생성하는 문제[1]

3. **고주파 세부사항 손실**: fine-tuning 과정에서 고주파 디테일이 손실되어 생성 품질이 저하되는 문제[1]

## 제안하는 방법 (수식 포함)

### **1. 쌍별 유사성 손실 (Pairwise Similarity Loss)**

생성 샘플 간 상대적 거리를 보존하기 위해 다음 손실 함수를 제안:

**이미지 레벨 쌍별 손실:**

$$
L_{img}(\epsilon_{sou}, \epsilon_{ada}) = E_{t,x_0,\epsilon}\left[\sum_i D_{KL}(p_{ada}^i \| p_{sou}^i)\right]
$$

여기서 확률 분포는 다음과 같이 정의:

$$
p_{sou}^i = \text{sfm}\left(\{\text{sim}(\tilde{x}\_0^{i,sou}, \tilde{x}\_0^{j,sou})\}_{∀i≠j}\right)
$$

$$
p_{ada}^i = \text{sfm}\left(\{\text{sim}(\tilde{x}\_0^{i,ada}, \tilde{x}\_0^{j,ada})\}_{∀i≠j}\right)
$$

**예측된 깨끗한 이미지:**

$$
\tilde{x}\_0 = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\epsilon_\theta(x_t, t)
$$

### **2. 고주파 세부사항 향상**

**Haar 웨이블릿 변환:**

$$
L^T = \frac{1}{\sqrt{2}}[1][1], \quad H^T = \frac{1}{\sqrt{2}}[-1, 1]
$$

**고주파 성분 정의:**

$$
hf = LH + HL + HH
$$

**고주파 쌍별 손실:**

$$
L_{hf}(\epsilon_{sou}, \epsilon_{ada}) = E_{t,x_0,\epsilon}\left[\sum_i D_{KL}(pf_{ada}^i \| pf_{sou}^i)\right]
$$

**고주파 MSE 손실:**

$$
L_{hfmse} = E_{t,x_0,\epsilon}[\|hf(\tilde{x}_0) - hf(x_0)\|^2]
$$

### **3. 전체 최적화 목표**

$$
L = L_{simple} + \lambda_1 L_{vlb} + \lambda_2 L_{img} + \lambda_3 L_{hf} + \lambda_4 L_{hfmse}
$$

여기서 $\lambda_1 = 0.001$, $\lambda_2, \lambda_3 \in [0.1, 1.0]$, $\lambda_4 \in [0.01, 0.08]$로 설정[1].

## 모델 구조

### **DDPM-PA 프레임워크**

1. **소스 모델**: 대규모 데이터셋(FFHQ, LSUN Church)에서 사전 훈련된 DDPM
2. **적응 모델**: 소스 모델 가중치로 초기화되어 타겟 도메인으로 적응되는 모델
3. **참조 모델**: 적응 과정에서 소스 도메인 정보를 보존하기 위한 고정된 소스 모델[1]

**핵심 구성 요소:**
- **UNet 기반 아키텍처**: 기존 DDPM과 동일한 UNet 구조 사용
- **웨이블릿 변환 모듈**: 고주파 정보 추출 및 강화
- **쌍별 거리 계산 모듈**: 코사인 유사성 기반 확률 분포 생성[1]

## 성능 향상

### **정량적 성과**

1. **Intra-LPIPS 향상**: 기존 GAN 기반 방법들보다 일관되게 높은 다양성 점수 달성
   - FFHQ → Babies: 0.599 vs CDC(0.583), DCL(0.579)
   - FFHQ → Sunglasses: 0.604 vs CDC(0.581), DCL(0.574)[1]

2. **FID 개선**: 
   - Babies: 48.92 vs DCL(52.56)
   - Sunglasses: 34.75 vs DCL(38.01)[1]

### **정성적 개선**

1. **세부사항 보존**: 헤어스타일, 얼굴 표정 등 다양한 고주파 디테일 보존
2. **아티팩트 감소**: GAN 기반 방법에서 나타나는 흐림과 아티팩트 현상 현저히 감소
3. **자연스러운 도메인 적응**: 훈련 샘플과 다른 특징(모자 착용 등)을 가진 이미지 생성 가능[1]

## 한계점

### **1. 해상도 제약**
- **256×256 해상도로 제한**: 현재 실험은 256×256 해상도에서만 수행되어 더 높은 해상도로의 확장성 검증 필요
- **메모리 요구사항**: 배치 크기가 GPU당 3개로 제한되어 고해상도 확장이 어려움[1]

### **2. 추상적 도메인에서의 한계**
- **스타일 재현의 한계**: 일부 추상적인 타겟 도메인(예: 아티스트 그림)에서 완전한 스타일 재현과 다양성 유지의 균형 문제
- **고주파 성분이 많은 도메인**: 타겟 도메인이 소스 도메인보다 현저히 많은 고주파 성분을 포함할 때 개선 여지 존재[1]

### **3. 계산 비용**
- **24.14% 추가 훈련 시간**: 기존 DDPM 대비 약 24% 더 많은 계산 시간 필요
- **샘플링 시간**: 1000개 샘플 생성에 GPU당 약 21시간 소요[1]

## 일반화 성능 향상 가능성

### **1. 도메인 간 지식 전이**

**쌍별 거리 보존의 일반화 효과:**
- 소스 모델에서 학습한 **구조적 지식을 타겟 도메인으로 효과적 전이**
- 다양한 도메인 조합에서 일관된 성능 향상 확인: FFHQ→다양한 스타일, LSUN Church→건축 스타일[1]

**비관련 도메인에서의 성능:**
- FFHQ→LSUN Church, LSUN Church→Sunglasses 등 **비관련 도메인에서도 수렴 가속화** 확인
- 1000개 이미지 데이터셋에서 20K 반복으로 우수한 결과 달성[1]

### **2. 확장 가능성**

**다양한 태스크로의 적용:**
- **의료 이미지 분할**[3]: DDPM을 특징 추출기로 활용하여 few-shot 분할 성능 개선
- **SAR 이미지 인식**[4]: 제한된 샘플에서 산란 정보와 결합하여 인식 정확도 향상
- **현미경 이미지 증강**[5]: 희귀 특징 생성을 통해 클래스 불균형 문제 해결

**아키텍처 독립성:**
- 특정 모델 아키텍처에 의존하지 않는 **일반적 프레임워크** 제공[6]
- 다양한 diffusion 모델에 적용 가능한 범용성[1]

### **3. 제한적 데이터 환경에서의 강건성**

**적응적 학습 메커니즘:**
최근 연구들이 DDPM-PA의 아이디어를 발전시켜 더욱 적응적인 학습 방법 개발:
- **적응적 개인화 훈련(APT)**[7]: 과적합 지표를 통한 동적 데이터 증강
- **자기 증류 기반 fine-tuning(SDFT)**[6]: 소스 모델에서 일반적 특징 증류

## 향후 연구에 미치는 영향

### **1. Diffusion 모델의 Few-shot 학습 패러다임 확립**

**새로운 연구 분야 개척:**
- **Diffusion 기반 few-shot 생성의 선구적 연구**로서 후속 연구들의 기반 제공[8][9][10]
- GAN 중심이었던 few-shot 생성 연구에 새로운 방향성 제시[1]

**방법론적 영향:**
- **쌍별 거리 보존 개념**이 다른 생성 모델로 확장 적용[11][12]
- **고주파 정보 강화 기법**이 이미지 복원 분야로 파급[13][14]

### **2. 실용적 응용 분야 확장**

**의료 AI 분야:**
- 제한된 의료 데이터로 고품질 합성 데이터 생성 기술의 기반 마련[3][15]
- 희귀 질환 데이터 증강을 통한 진단 모델 성능 향상[5]

**산업 응용:**
- **개인화된 콘텐츠 생성**: 소수 이미지로 개인 맞춤형 이미지 생성 시스템 개발
- **제품 디자인**: 제한된 참조 이미지로 다양한 디자인 변형 생성

### **3. 기술적 발전 방향 제시**

**메모리 효율적 훈련:**
- **경량화 연구 필요성** 제기: 현재 방법의 메모리 요구사항이 실용화에 제약[1]
- **압축 모델과 혼합 증강 전략** 개발로 이어짐[16]

**고해상도 확장:**
- 256×256 해상도 제한 극복을 위한 **점진적 해상도 증가 방법** 연구 필요성 제시[1]

## 앞으로 연구 시 고려할 점

### **1. 확장성 개선**

**해상도 스케일링:**
- **메모리 효율적 아키텍처** 설계를 통한 고해상도 이미지 생성 지원
- **점진적 학습 전략**을 통한 단계적 해상도 증가 방법 개발

**계산 효율성:**
- **어댑터 기반 fine-tuning**[17]: 전체 모델이 아닌 일부 파라미터만 업데이트
- **증류 기반 경량화**: 고성능 teacher 모델에서 경량 student 모델로 지식 전이[18]

### **2. 평가 지표 표준화**

**다양성 측정의 한계:**
- **Intra-LPIPS와 FID의 한계** 인식: 소규모 데이터셋에서 FID의 불안정성[1]
- **인간 평가와 일치하는 지표** 개발 필요성
- **도메인별 특화 평가 지표** 개발

### **3. 도메인 적응 범위 확대**

**추상적 도메인 처리:**
- **스타일 전이와 다양성 보존의 균형** 최적화
- **도메인 간 유사성 측정** 방법 개선

**다중 모달리티:**
- **텍스트-이미지 조건부 생성**으로의 확장
- **3D 생성 모델**에의 적용 가능성 탐구

### **4. 이론적 기반 강화**

**과적합 메커니즘 분석:**
- **diffusion 과정에서의 과적합 수학적 모델링**
- **최적 훈련 스케줄링** 이론 개발

**일반화 성능 보장:**
- **PAC-Bayes 이론 적용**을 통한 일반화 오차 바운드 도출
- **도메인 적응의 이론적 한계** 분석

이 논문은 diffusion 모델을 few-shot 이미지 생성에 처음 적용한 선구적 연구로서, 제한된 데이터 환경에서의 생성 모델 학습에 대한 새로운 패러다임을 제시했다. 특히 쌍별 거리 보존과 고주파 세부사항 강화라는 핵심 아이디어는 후속 연구들에 지속적인 영향을 미치고 있으며, 실용적 응용 분야에서의 활용 가능성을 크게 확장시켰다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2f4b87b0-1bbc-47a0-be0a-a04c496faebe/2211.03264v3-abcugdoem.pdf
[2] https://arxiv.org/abs/2211.03264
[3] https://pubmed.ncbi.nlm.nih.gov/37778140/
[4] https://ieeexplore.ieee.org/document/10689422/
[5] https://www.nature.com/articles/s41598-025-93954-x
[6] https://arxiv.org/abs/2311.01018
[7] https://arxiv.org/html/2507.02687v1
[8] https://arxiv.org/abs/2205.15463
[9] https://github.com/georgosgeorgos/few-shot-diffusion-models
[10] https://openreview.net/forum?id=rqKTms-YHAW
[11] https://www.themoonlight.io/en/review/pairwise-alignment-improves-graph-domain-adaptation
[12] https://www.themoonlight.io/en/review/tuning-timestep-distilled-diffusion-model-using-pairwise-sample-optimization
[13] https://arxiv.org/abs/2308.13442
[14] https://arxiv.org/abs/2404.00279
[15] https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Ran_Few-Shot_Depth_Completion_Using_Denoising_Diffusion_Probabilistic_Model_CVPRW_2023_paper.pdf
[16] https://icml.cc/virtual/2025/poster/45514
[17] https://openreview.net/forum?id=0J6afk9DqrR
[18] https://arxiv.org/abs/2412.06243
[19] https://ieeexplore.ieee.org/document/10658528/
[20] https://ieeexplore.ieee.org/document/10382547/
[21] https://arxiv.org/abs/2307.00522
[22] https://www.semanticscholar.org/paper/04cf695745efa8b3645c01b731f09f038470921a
[23] https://openaccess.thecvf.com/content/WACV2024/papers/Hur_Expanding_Expressiveness_of_Diffusion_Models_With_Limited_Data_via_Self-Distillation_WACV_2024_paper.pdf
[24] https://arxiv.org/abs/2410.03190
[25] https://pure.kaist.ac.kr/en/publications/expanding-expressiveness-of-diffusion-models-with-limited-data-vi
[26] https://openreview.net/forum?id=fXnE4gB64o
[27] https://michalakelis.eu/wp-content/uploads/2024/12/06_Forecasting-with-limited-data_Combining-ARIMA-and-diffusion-models.pdf
[28] https://cs.brown.edu/people/pfelzens/papers/prop.pdf
[29] https://arxiv.org/html/2407.07249v1
[30] https://ieeexplore.ieee.org/document/11037402/
[31] https://www.frontiersin.org/articles/10.3389/fnbot.2023.1182375/full
[32] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12602/2668222/Salient-object-detection-via-high-frequency-edge-detail-enhancement/10.1117/12.2668222.full
[33] https://ieeexplore.ieee.org/document/10440188/
[34] https://www.mdpi.com/2306-5354/11/7/646
[35] https://openreview.net/pdf/9ab72ba49975b6226005c3ecfede31e347d78ec6.pdf
[36] https://zilliz.com/ai-faq/how-does-overfitting-manifest-in-diffusion-model-training
[37] https://arxiv.org/abs/2402.04929
[38] https://openreview.net/pdf/61526944a8edb7c265eee8c67728b2a289dd1cfe.pdf
[39] https://arxiv.org/abs/2403.13652
[40] https://www.themoonlight.io/en/review/apt-adaptive-personalized-training-for-diffusion-models-with-limited-data
[41] https://openreview.net/forum?id=ancAesl2LU
[42] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ddpm-pa/
[43] https://openaccess.thecvf.com/content/WACV2024/papers/Song_StyleGAN-Fusion_Diffusion_Guided_Domain_Adaptation_of_Image_Generators_WACV_2024_paper.pdf
[44] https://arxiv.org/html/2403.01497v2
[45] https://www.reddit.com/r/MachineLearning/comments/1br15cv/d_why_dont_diffusion_models_overfit/
[46] https://papers.miccai.org/miccai-2024/226-Paper0195.html
[47] https://raw.githubusercontent.com/mlresearch/v235/main/assets/wang24ap/wang24ap.pdf
[48] https://cvpr.thecvf.com/virtual/2025/poster/32599
[49] https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Benigmim_One-Shot_Unsupervised_Domain_Adaptation_With_Personalized_Diffusion_Models_CVPRW_2023_paper.pdf
[50] https://ieeexplore.ieee.org/document/10476867/
[51] https://ieeexplore.ieee.org/document/9577580/
[52] https://arxiv.org/pdf/2311.16353.pdf
[53] http://arxiv.org/pdf/2404.16556.pdf
[54] https://arxiv.org/html/2407.18125
[55] https://arxiv.org/html/2312.03046v2
[56] https://arxiv.org/html/2405.19201
[57] https://arxiv.org/html/2407.05875v1
[58] https://arxiv.org/html/2409.16488v1
[59] https://arxiv.org/html/2412.14422
[60] https://arxiv.org/pdf/2106.03802.pdf
[61] https://arxiv.org/pdf/2212.00793.pdf
[62] https://proceedings.neurips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf
[63] https://openaccess.thecvf.com/content/ICCV2023/papers/Hu_Phasic_Content_Fusing_Diffusion_Model_with_Directional_Distribution_Consistency_for_ICCV_2023_paper.pdf
[64] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11133.pdf
[65] https://ojs.aaai.org/index.php/AAAI/article/view/28214
[66] https://ieeexplore.ieee.org/document/9857454/
[67] http://arxiv.org/pdf/2403.11078.pdf
[68] https://arxiv.org/pdf/2306.07440.pdf
[69] https://arxiv.org/pdf/2301.13362.pdf
[70] https://arxiv.org/pdf/2306.02929.pdf
[71] http://arxiv.org/pdf/2208.11284.pdf
[72] http://arxiv.org/pdf/2102.09672.pdf
[73] http://arxiv.org/pdf/2403.17870.pdf
[74] https://arxiv.org/pdf/2310.19460.pdf
[75] https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf
[76] https://arxiv.org/abs/2507.02687
