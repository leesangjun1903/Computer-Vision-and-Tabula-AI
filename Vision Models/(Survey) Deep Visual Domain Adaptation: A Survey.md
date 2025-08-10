# Deep Visual Domain Adaptation: A Survey 

## 1. 핵심 주장과 주요 기여
“Deep Visual Domain Adaptation: A Survey”는  
- **도메인 편차(domain shift)를 극복**하기 위해 전통적 얕은(shallower) 방법 대신 **딥러닝 기반 기법**을 도입한 연구 동향을 체계적으로 정리  
- 기존 설문들이 얕은 기법만 다룬 데 반해, **딥 도메인 적응(deep domain adaptation)** 방법들을 **분류·비교·종합**하여 비전 분야 전반에 걸친 응용 사례까지 포괄적으로 제시함  

**주요 기여**  
1. **시나리오별 분류 체계**: 도메인 간 분포 차이(동질 vs. 이질)와 레이블 가용성(감독/반감독/비감독) 기준으로 **도메인 적응 설정**을 체계화  
2. **기법 카테고리화**:  
   - *Discrepancy-based* (통계·클래스·아키텍처·기하 기준)  
   - *Adversarial-based* (생성모델 vs. 비생성모델)  
   - *Reconstruction-based* (엔코더–디코더 vs. 적대적 재구성)  
3. **다중 단계(Transitive) 적응**: 한 걸음 접근이 어려운 이질적 도메인을 연결할 중간 도메인 선택·활용 전략(수공·인스턴스·표현 기반) 제시  
4. **응용 사례 총망라**: 이미지 분류, 객체 검출, 얼굴 인식, 의미 분할, 스타일 변환, 인물 재식별 등 다양한 비전 과제에서 딥 DA 성과 분석  

***

## 2. 문제 정의, 제안 기법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제
- *소스 도메인*의 풍부한 레이블 데이터로 학습한 모델이,  
- 분포 $$P_s(X)\neq P_t(X)$$ 혹은 피처 공간 $$X_s\neq X_t$$가 다른 *타깃 도메인*에선 성능이 급락하는 **도메인 시프트** 문제  

### 2.2 제안하는 방법
딥 네트워크 학습 과정에 **추가 손실**을 삽입하거나 구조를 변경하여,  
- **도메인 불변 표현(domain-invariant features)**을 학습  
- **분포 간 차이(discrepancy)**를 최소화  

#### 2.2.1 Discrepancy-based
- **MMD(Maximum Mean Discrepancy)**  

$$
    \mathrm{MMD}^2 = \Big\|\frac{1}{n_s}\sum_{i=1}^{n_s}\phi(x^s_i)-\frac{1}{n_t}\sum_{j=1}^{n_t}\phi(x^t_j)\Big\|^2_{\mathcal H}
  $$  

- **CORAL(Correlation Alignment)**  

$$
    L_{\mathrm{CORAL}}=\frac{1}{4d^2}\|C_s - C_t\|^2_F,\quad C=\mathrm{cov}(X)
  $$  

- **Adaptive BatchNorm**: 도메인별 배치 정규화 통계(평균·분산)를 분리 적용

#### 2.2.2 Adversarial-based
- **Non-Generative**  
  - *DANN*: 피처 추출기와 도메인 분류기 사이에 **Gradient Reversal Layer** 삽입  
- **Generative**  
  - *CoGAN, CycleGAN 등*: GAN 구조로 픽셀 수준에서 소스↔타깃 이미지 변환  

#### 2.2.3 Reconstruction-based
- **Encoder–Decoder**: 소스·타깃 모두에서 입력 재구성 손실  
- **Cycle-consistency**: $$\ell_1$$ 순환 재구성 손실과 GAN 대립 손실 결합  

### 2.3 모델 구조 예시
- **Deep Adaptation Network (DAN)**: 마지막 여러 레이어에 MMD 손실 추가  
- **Residual Transfer Network (RTN)**: MMD + 레지듀얼 분류기 조정  
- **Domain Separation Network (DSN)**: Private/Shared 인코더 병렬 구성 후 재구성과 분리 손실  

### 2.4 성능 향상
- Office-31(A→W) 비교:  
  - 프리트레인 AlexNet 61.6% → DAN 68.5% → RTN 73.3% → JAN 75.2%  
- SVHN→MNIST: VGG-16 60.1% → DANN 73.9% → ADDA 76.0%  

### 2.5 한계
- **레이블 공간 가정**: 대부분 $$Y_s=Y_t$$ 동질 레이블 가정  
- **이질 도메인**: 서로 다른 미디어(텍스트↔이미지) 간 적응 방법 상대적으로 미비  
- **이론적 보장 부족**: 일반화 성능에 대한 엄밀한 이론·경계 미흡  

***

## 3. 일반화 성능 향상 관점
- **공통 표현 학습**: MMD·CORAL로 소스·타깃 분포 정렬 시, 높은 레이어일수록 도메인 특이성이 커지므로 **하위 계층 고정**, 상위 계층에 적응 손실 적용  
- **적대적 학습**: DANN처럼 도메인 분류기를 속여서 **도메인-무관 피처** 획득  
- **다중 손실 통합**: 분포 정렬 + 재구성 + 적대 손실 병합으로 **다각도 일반화** 유도  
- **BatchNorm 변형**: 도메인별 통계 사용, 레이어 내에서 자연스럽게 도메인 차이 제거  

***

## 4. 향후 연구 영향 및 고려 사항
- **이질 도메인 전이**: 서로 다른 피처·레이블 공간 간 딥 적응, 멀티모달 DA 연구 확대  
- **부분 레이블 불일치**: 타깃에만 존재하는 클래스 혹은 비공통 레이블 문제(부분 전이) 해결  
- **일반화 이론 강화**: 도메인 간 간극(Generalization Bound) 이론적 분석과 최적화  
- **효율적 학습**: 대용량 데이터 및 실시간 응용을 위한 경량·온라인 DA 메커니즘 필요  

이 설문은 딥 기반 도메인 적응 연구의 전개를 체계적으로 조망하고, 주요 기법들의 구조·수식·성능을 통합 정리함으로써, 향후 **강건한 일반화**와 **이질 도메인 전이** 방향 연구의 토대를 마련했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c754d71c-a116-4a8d-850b-e2c054b621dc/1802.03601v4.pdf

# III. 딥 도메인 적응(Deep Domain Adaptation)

도메인 적응은 “소스(domain S)”와 “타깃(domain T)” 간에 분포 차이(또는 피처 공간 차이)가 존재할 때, S에서 학습한 모델이 T에서도 잘 동작하도록 만드는 기술입니다. 딥 도메인 적응은 깊은 신경망 구조 안에 적응 메커니즘을 통합하여, **피처 표현(feature representation)을 학습 단계에서부터 도메인 불변(invariant) 성질을 갖도록 강화**합니다. III장에서는 크게 세 가지 **One-step** 기법과, 도메인 간 직접적 연결이 불가능할 때 사용하는 **Multi-step** 기법으로 분류합니다.

***

## 1. One-step 딥 도메인 적응  
소스와 타깃이 충분히 관련 있을 때 “한 번에” 적응을 수행합니다. 세 가지 접근 방식을 통해, 피처 추출기(feature extractor)를 소스·타깃 모두에 걸쳐 **도메인-무관(domain-invariant)** 하도록 학습시킵니다.

### 1.1 Discrepancy-based (분포 차이 정량화)  
피처 공간에서 소스와 타깃 분포 차이를 측정하고 최소화하는 **추가 손실(loss)** 을 네트워크에 더합니다.

  -  **Statistic Criterion**  
    – **MMD**(Maximum Mean Discrepancy):  

$$
        \mathrm{MMD}^2 = \Big\|\frac{1}{n_S}\sum_{i}\phi(x^S_i) \;-\;\frac{1}{n_T}\sum_{j}\phi(x^T_j)\Big\|^2_{\mathcal H}
      $$  
      
  커널 함수 φ를 통해 두 분포 간 평균 차이를 줄입니다.  
     – **CORAL**(Correlation Alignment):  

$$
        L_{\mathrm{CORAL}}=\frac{1}{4d^2}\|C_S - C_T\|^2_F
      $$  
      
  공분산 행렬 $$C=\mathrm{cov}(X)$$ 간 Frobenius 노름을 최소화합니다.

  -  **Class Criterion**  
    타깃에 일부 레이블이 있을 때, 소스와 타깃의 **클래스 예측(confidence)** 을 함께 학습해 결정경계(decision boundary)를 정렬합니다.  
    – *Soft label* 속성을 도입해, 소프트맥스 확률분포를 온도 T로 부드럽게 하여 클래스 간 관계를 유지합니다.

  -  **Architecture Criterion**  
    네트워크 구조나 파라미터 공유 방식을 조절해 **전달 가능한(transferable)** 표현을 강화합니다.  
    – *Adaptive BatchNorm*: 도메인별 통계(평균·분산)를 분리해 배치 정규화  
    – *Weakly-shared layers*: 소스·타깃 가중치 간 가벼운 규제(Reg. term)로 유사도 유지  

  -  **Geometric Criterion**  
    Grassmann manifold 상의 지오데식(geodesic) 경로를 따라 **중간 서브스페이스**를 샘플링해 분포를 매끄럽게 연결합니다.

***

### 1.2 Adversarial-based (적대적 학습)  
생성적 적대 신경망(GAN) 개념을 차용하여, 피처 추출기가 “도메인 분류기(domain discriminator)”를 속이도록 학습합니다.  

  -  **Non-Generative**  
    – **DANN**:  
      피처망(feature extractor) ↔ 도메인 분류기 사이에 **Gradient Reversal Layer** 삽입 → 피처망은 도메인 예측 정확도를 낮추며, 분포 차이를 제거  

  -  **Generative**  
    – **CoGAN, CycleGAN** 등:  
      소스 이미지를 타깃처럼 보이게 픽셀 단위로 변환(역변환 포함)해, **레이블 정보(annotations)를 보존**하며 가상 타깃 데이터를 생성  
    – 타깃 데이터 생성을 통해 “실제 타깃처럼” 학습시키거나, 생성된 이미지로 분포 정렬  

***

### 1.3 Reconstruction-based (재구성)  
인코더–디코더(autoencoder) 구조나 적대적 재구성(adversarial reconstruction)을 통해, **도메인 간 공유 표현(shared features)** 을 학습하면서 **도메인 특이 정보(private features)** 를 보존합니다.

  -  **Encoder–Decoder**  
    – **Deep Reconstruction Classification Network (DRCN)**:  
      소스에서 분류용 분기, 타깃에서 재구성 분기(decoder) 공유 인코더 사용  
      $$\min\;\lambda\,L_\text{class} + (1-\lambda)\,L_\text{recon}$$  
    – **Domain Separation Network (DSN)**:  
      “공유 인코더” vs. “도메인 전용 인코더” 병렬 → 재구성 손실 + 공유/전용 공간 **절연(orthogonality)** 제약  

  -  **Adversarial Reconstruction**  
    – **CycleGAN**: 두 세트의 생성기 G, F와 두 세트의 판별기 DX, DY 학습  
      - **Adversarial Loss** + **Cycle-consistency Loss**(L1 재구성)  
      → “한 번 변환 후 역변환”이 원본을 복원하도록 압박  

***

## 2. Multi-step 딥 도메인 적응  
소스와 타깃 간 거리가 너무 멀어 One-step이 어려울 때, **중간 도메인(Intermediate Domain)** 을 경유해 전이를 수행합니다.  

  -  **Hand-crafted**: 경험 기반으로 중간 도메인 선정  
  -  **Instance-based**: 소스·타깃 데이터 일부를 선택해 “가교(instance bridge)” 형성  
  -  **Representation-based**: 먼저 학습된 네트워크의 중간 피처를 **고정(freeze)** 후, 이를 입력으로 새 네트워크 학습  

***

### 핵심 포인트  
- **One-step**: 분포 정렬(MMD/CORAL), 적대학습(DANN/GAN), 재구성(AE/CycleGAN)  
- **Multi-step**: 중간 도메인 도입으로 큰 도메인 시프트 해결  
- **통합**: 서로 다른 손실과 구조를 결합해 **강건한 불변 표현** 획득

이처럼 III장에 제시된 다양한 기법들은, **딥 네트워크 학습과정에 DA 모듈을 끼워 넣어** 소스·타깃 분포 차이를 줄이고, 타깃 도메인에서 일반화 성능을 높이기 위한 주요 전략을 포괄적으로 다룹니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c754d71c-a116-4a8d-850b-e2c054b621dc/1802.03601v4.pdf

# IV. One-Step 도메인 적응(One-Step Domain Adaptation)

One-Step 도메인 적응은 소스(Source)와 타깃(Target) 도메인이 충분히 관련되어 있어, **중간 브릿지 없이** 한 번에 적응을 수행할 수 있는 상황을 다룹니다. 이 방식은 크게 **동질적(homogeneous)** 도메인 적응—소스와 타깃의 피처 공간이 동일(Xs = Xt)—과 **이질적(heterogeneous)** 도메인 적응으로 나뉘지만, 대부분의 연구는 동질적 비감독(unsupervised) 설정을 중심으로 진행됩니다.  

***

## 1. 동질적 도메인 적응(Homogeneous DA)

### 1.1 데이터 가정  
- 소스 도메인 Ds = {Xs, P_s(X)}에 풍부한 레이블이 있고,  
- 타깃 도메인 Dt = {Xt, P_t(X)}에는 레이블이 없거나 일부만 존재(Dtl ⊂ Dt)  

### 1.2 설정  
- **감독(supervised)**: 타깃에 소량의 레이블 Dtl 사용  
- **비감독(unsupervised)**: 타깃 레이블 없이 Dt 전체(Dtu)만 사용  

### 1.3 주요 기법  
#### A. Discrepancy-based Approaches  
피처망의 **중간 또는 최상위 레이어**에 분포 차이 측정 손실을 추가하여 P_s ≈ P_t가 되도록 학습  

  1. **통계 기준(Statistic Criterion)**  
     - MMD(Maximum Mean Discrepancy):  

$$
         \mathrm{MMD}^2 = \Big\|\frac{1}{n_s}\sum_i\phi(x^s_i)-\frac{1}{n_t}\sum_j\phi(x^t_j)\Big\|^2
       $$  
     
  - CORAL(Correlation Alignment):  

$$
         L_{\mathrm{CORAL}}=\frac1{4d^2}\|C_s-C_t\|_F^2,\quad C=\mathrm{cov}(X)
       $$  
  
  2. **클래스 기준(Class Criterion)**  
     - 타깃 레이블(Dtl)이 있을 때 소스+타깃의 **소프트맥스 손실**을 함께 최소화  
     - *Soft label* 기법: 온도 $$T$$를 높여 부드러운 확률분포로 학습  
  3. **아키텍처 기준(Architecture Criterion)**  
     - **Adaptive BatchNorm**: 도메인별 배치 정규화 통계(평균·분산) 분리  
     - **Weakly-shared Layers**: 소스·타깃 레이어 가중치 간 유사도 규제  
  4. **기하 기준(Geometric Criterion)**  
     - Grassmann manifold상 지오데식 경로의 중간 서브스페이스를 샘플링해 투영  

#### B. Adversarial-based Approaches  
피처망이 **도메인 분류기(domain discriminator)** 를 속이도록 적대적(adversarial) 손실을 추가  

  1. **비생성(Non-Generative)**  
     - **DANN**(Domain-Adversarial Neural Network):  
       - 피처 추출기 ↔ 도메인 분류기 사이 **Gradient Reversal Layer** 삽입  
       - 피처망은 도메인 예측 정확도를 **최대화**하려는 분류기를 **최소화**  
  2. **생성(Generative)**  
     - **CoGAN, CycleGAN** 등:  
       - 소스 이미지를 타깃처럼 변환 및 역변환 → **레이블 보존** 가상 타깃 데이터 생성  
       - 픽셀 단위 적대 학습으로 “실제 타깃과 유사”한 합성 이미지 활용  

#### C. Reconstruction-based Approaches  
입력 재구성 자체를 보조 과제(auxiliary task)로 삼아, **공유 표현(shared feature)** 은 획득하고 **도메인 특이 표현(private feature)** 은 유지  

  1. **Encoder–Decoder**  
     - **DRCN**: 소스→분류 분기, 타깃→재구성 분기 공유 인코더  

       $$\min\;\lambda L_\text{class}+(1-\lambda)L_\text{recon}$$  

     - **DSN**(Domain Separation Network):  
       - “공유 인코더” vs. “도메인 전용 인코더” 병렬  
       - 재구성 손실 + 공유/전용 공간 **절연 손실(orthogonality)**  
  2. **Adversarial Reconstruction**  
     - **CycleGAN**: 두 쌍의 생성기/판별기를 이용해 G: X→Y, F: Y→X 학습  
       - 적대 손실 + 순환 일관성(cycle-consistency) 손실 = L1 재구성  

***

## 2. 이질적 도메인 적응(Heterogeneous DA)

피처 공간 Xs ≠ Xt일 때, 동일 차원으로 **크기 변경(resize)** 가능하다면 동질적 기법을 그대로 적용하거나, 그렇지 않으면 **픽셀 수준 변환(GAN)** 혹은 **쌍별 대응(pairwise) SAE→공유층** 구조를 사용합니다.

***

### 요약

- **One-Step DA**는 추가 손실(분포 정렬, 적대, 재구성)을 네트워크에 통합해 **소스→타깃** 간 피처를 한 번에 정렬  
- **Discrepancy-based**: MMD/CORAL, Soft label, Adaptive BN  
- **Adversarial-based**: DANN, CoGAN/CycleGAN  
- **Reconstruction-based**: DRCN, DSN, CycleGAN  
- **Heterogeneous**: 이미지→이미지 변환(GAN), SAE 기반 쌍별 입력  

이처럼 IV장 기법은 “한 걸음(one-step)”으로 도메인 간 간극을 좁히며, **타깃에서 레이블이 없거나 극히 적은 상황**에서도 **피처 추출 단계**에서부터 도메인 불변성을 확보해 일반화 성능을 높입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c754d71c-a116-4a8d-850b-e2c054b621dc/1802.03601v4.pdf
