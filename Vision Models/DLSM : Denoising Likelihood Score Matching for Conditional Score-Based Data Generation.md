# DLSM : Denoising Likelihood Score Matching for Conditional Score-Based Data Generation | Image generation

**핵심 주장 및 기여 요약**  
본 논문은 **조건부 확률 분포**를 생성하는 기존의 **score-based** 모델들이 분류기(classifier)로부터 얻은 **우도(likelihood) 스코어**와 실제 우도 스코어 사이의 편차(스코어 미스매치)로 인해 샘플 품질이 저하된다는 문제를 지적한다. 이를 해결하기 위해, 우도의 로그-그래디언트를 직접 매칭하는 새로운 손실함수인 **Denoising Likelihood Score Matching (DLSM)** 을 제안한다. DLSM은 분류기 학습 시 cross-entropy와 함께 적용되어, 조건부 스코어 모델의 샘플링 품질을 크게 향상시킨다.

## 1. 문제 정의  
조건부 score-based 모델(classifier guidance)은 Bayes’ 정리로 분해된 posterior log-density gradient  

$$
\nabla_x \log p_\sigma(x\mid y)
= \nabla_x \log p(y\mid x) + \nabla_x \log p_\sigma(x)
$$ 

를 score 모델과 분류기로 학습해 활용한다.  
그러나 분류기를 **cross-entropy**만으로 학습할 때 $$\nabla_x\log p(y\mid x)$$가 실제 $$\nabla_x\log p_\sigma(y\mid x)$$와 크게 어긋나며, 이로 인해 최종 샘플링 품질이 저하된다.

## 2. 제안 방법  
### 2.1. Denoising Likelihood Score Matching 손실  
우도 스코어 매칭 손실 (Explicit Likelihood Score Matching):  

$$L_{\mathrm{ELSM}}(\theta)
= \mathbb{E}\_{\tilde x,y} \Big[\tfrac12\big\|\nabla_x\log p(y\mid x;\theta) - \nabla_x\log p_\sigma(y\mid x)\big\|^2\Big]$$  

직접 계산 불가능하므로, DSM 유도와 Bayes 분해를 결합하여 다음과 같은 **근사** DLSM 손실을 도출:  

$$
L_{\mathrm{DLSM}}'(\theta) = \mathbb{E}_{\tilde x,x,y}\Big[\tfrac12\big\|\nabla_x\log p(y\mid x;\theta) + s(x;\phi) - \tfrac1{\sigma^2}(x - \tilde x)\big\|^2\Big],
$$  

여기서 $$s(x;\phi)\approx\nabla_x\log p_\sigma(x)$$ 는 DSM 손실로 미리 학습된 prior score 모델이다.

### 2.2. 전체 학습 목표  
분류기는 **DLSM’**과 **cross-entropy**를 결합하여 학습:  

$$
L_{\mathrm{Total}}(\theta)= L_{\mathrm{DLSM}}'(\theta) + \lambda\,L_{\mathrm{CE}}(\theta),
$$

λ는 교차 엔트로피의 비중을 조절하는 하이퍼파라미터이다.

### 2.3. 모델 구조 및 학습 절차  
1. **Stage 1 (Prior score 학습)**  
   - DSM 손실 $$L_{\mathrm{DSM}}(\phi)$$로 $$s(x;\phi)$$ 학습  
2. **Stage 2 (분류기 학습)**  
   - $$s(x;\phi)$$ 고정  
   - $$L_{\mathrm{Total}}(\theta)$$로 분류기 학습  

최종 샘플링 시, classifier·score 합산 벡터를 사용해 조건부 Langevin diffusion 수행.

## 3. 성능 향상  
- **CIFAR-10:**  
  - FID 4.10→2.25, IS 9.08→9.90  
  - Precision/Recall 증가 및 class-wise metrics 개선  
- **CIFAR-100:**  
  - FID 4.52→3.86, IS 11.53→11.62  
  - Class-wise fidelity/diversity 동시 향상  

스케일링(α) 기법 대비 **다양성(Recall, Coverage)** 유지하면서 품질(FID, IS) 제고.  

## 4. 한계 및 일반화 성능  
- **한계**:  
  - DSM 기반 prior score 학습 품질에 의존  
  - λ 및 σ-스케줄링 민감도 존재  
- **일반화**  
  - 다양한 조건부 생성 작업(colorization, inpainting)에 DLSM 도입 가능  
  - 분류기 guidance 외 다른 조건부 모델(텍스트-투-이미지)에도 적용 여지  
  - 다중 조건(속성, 클래스) 시 복합적 우도 매칭 필요  

## 5. 향후 연구 영향 및 고려사항  
- **영향**:  
  - conditional diffusion 전반에 우도-스코어 매칭 개념 확산  
  - classifier guidance 방법론 개선 방향 제시  
- **고려사항**:  
  - **다중 조건** 및 **비이산 y**(연속 속성)로 확장  
  - prior score 모델과 분류기 간 **공동 학습** 전략  
  - 다른 분포 근사 기법(e.g., implicit score matching)과의 통합 검토  
  - λ, σ-스케줄링 최적화 및 자동화  

DLSM은 score-based 조건부 생성 모델의 **스코어 정밀도**를 크게 향상시켜, 후속 연구에서 보다 안정적이고 고품질의 조건부 샘플링 기법 개발로 이어질 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6d6e3ae5-c1e8-4e33-9594-7675494f053d/2203.14206v1.pdf
