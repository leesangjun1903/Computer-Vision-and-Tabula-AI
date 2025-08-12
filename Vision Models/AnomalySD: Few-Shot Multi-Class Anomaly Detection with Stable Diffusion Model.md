# AnomalySD: Few-Shot Multi-Class Anomaly Detection with Stable Diffusion Model | Anomaly detection

## 1. 핵심 주장 및 주요 기여
**AnomalySD**는 Stable Diffusion(SD) 모델의 강력한 **inpainting** 능력을 활용하여, 단 몇 장의 정상 이미지(few-shot)만으로도 여러 클래스에 걸친 이상(anomaly)을 검출·위치(localization)할 수 있는 **통합 프레임워크**를 제안한다.  
- **Few-shot·Multi-class** 환경에서 단일 모델로 다수 클래스 처리  
- SD 모델의 **fine-tuning**을 위한 계층적 텍스트 프롬프트(prompt)와 객체 포그라운드 기반 마스크(mask) 설계  
- 추론 시 **다중 크기 마스크(multi-scale mask)** 및 **프로토타입 기반 마스크(prototype-guided mask)** 전략을 통해 잠재적 이상 영역을 효과적으로 탐지  

## 2. 문제 정의 및 제안 기법
### 2.1 해결하고자 하는 문제
- 기존 비지도(an unsupervised) 이상 탐지 기법들은  
  1) **풍부한 정상 데이터** 가정  
  2) **클래스별 별도 모델** 트레이닝  
  → 실제 산업현장에선 정상 샘플이 극히 적고(0~4장), 다양한 부품을 하나의 시스템에서 검사해야 함

### 2.2 제안 모델 구조 및 학습 수식
#### 2.2.1 Fine-tuning 단계
- **입력**  
  - 정상 이미지 $$x$$  
  - 포그라운드 기반 랜덤 마스크 $$m$$  
  - 계층적 텍스트 프롬프트 $$y$$ (전체 수준 “A perfect [c].”, 세부 수준 “A [c] with intact ….”)
- **인페인팅 SD 초깃값**  
  - VAE 인코더로 원본 $$\,z=E(x)$$, 마스크 적용된 $$\,z^\circ=E((1-m)\odot x)$$ 얻음  
  - 노이즈 첨가:  
    
$$ z_t = \sqrt{\bar\alpha_t}\,z + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I) $$
  
  - 입력 결합: $$\tilde z_t=[z_t;z^\circ;\tilde m]$$
- **손실함수**  
  - **디노이징 네트워크**  
    
$$ L_\theta = \mathbb{E}\_{x,y,\epsilon,t}\|\epsilon - \epsilon_\theta(\,\tilde z_t,\,t,\,\tau(y)\,)\|^2_2 $$
  
  - **디코더(VAE)**  
    
$$ L_D = \|x-\hat x\|^2_2 + \beta\,L_{\mathrm{LPIPS}}(x,\hat x) $$

#### 2.2.2 추론(Inference) 단계
1. 입력 이미지 $$x$$와 **다중 크기 마스크** $$M^{(k)}_{i,j}$$ 및 **프로토타입 기반 마스크** $$M^{(p)}$$ 생성  
2. SD inpainting 수행 (노이즈 시작 단계 $$\tilde T=\lambda T$$ 제어)  
3. **이상 스코어 맵**  
   - 각 마스크별 reconstruction 차이 $$\;D(x,\hat x)$$ 계산 (AlexNet 특징 공간)  
   - 다중 크기 맵 조합:  
     
$$ S_{ms} = \big(|K|^{-1}\sum_{k\in K} S^{(k)^{-1}}\big)^{-1} $$
   
   - 최종 맵:  
     
$$ S_{\mathrm{map}}=(1-\alpha)\,S_{ms} + \alpha\,S_{pg} $$

### 2.3 성능 향상 및 한계
- **MVTec-AD 1-shot**: 이미지 AUROC 93.6%, 픽셀 AUROC 94.8%  
- **VisA 1-shot**: 이미지 AUROC 80.9%, 픽셀 AUROC 92.8%  
- 4-shot 설정에서는 **95.6%/96.2%** 달성하며 기존 few-shot 기법 대비 우수  
- **제한점**:  
  - 프롬프트·마스크 설계에 여전히 수작업 하이퍼파라미터 존재  
  - 대규모 정상샘플이 없는 환경에서 프로토타입 정확도 저하 가능  
  - 추론 속도가 inpainting 반복으로 비교적 느림

## 3. 모델 일반화 성능 향상 가능성
- **계층적 텍스트 지시어**를 범용화하여 새로운 클래스로 확장 가능  
- 프로토타입 기반 마스크는 **다양한 배경·조명** 조건에서도 이상 패턴 탐지에 유연  
- 노이즈 시작 단계($$\lambda$$)와 프롬프트 가중치($$\alpha$$)의 카테고리별 자동 최적화 연구로 **적응형 일반화** 달성 가능  
- SD 기반 inpainting이 다양한 도메인(의료·위성·자율차 센서)으로 전이 학습하기 유리

## 4. 향후 연구 영향 및 고려 사항
- **자율화된 프롬프트 학습(prompt tuning)** 기법과의 결합으로 수작업 의존도 감소  
- **노이즈 및 마스크 생성 정책**을 강화학습으로 설계하여 자동 탐지 최적화  
- **실시간 추론 가속화**를 위한 경량화 모델(distillation·pruning) 연구 필요  
- 타 도메인으로의 전이 가능성을 검증하여 범용 이상 탐지 플랫폼 구축 방안 고려

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5413a863-2354-499d-8bd0-ff55e3bec1ba/2408.01960v1.pdf
