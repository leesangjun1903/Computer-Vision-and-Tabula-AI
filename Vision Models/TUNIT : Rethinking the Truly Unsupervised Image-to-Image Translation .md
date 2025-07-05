# TUNIT : Rethinking the Truly Unsupervised Image-to-Image Translation | Image generation
**Kyungjune Baek et al., TUNIT (2021)**

## 1. 핵심 주장 및 주요 기여  
본 논문은 *어떠한 형태의 지도(supervision)* 없이 이미지 간 변환을 수행하는 **진정한 의미의** 완전 비지도(Unsupervised) 이미지-투-이미지 번역 방법(TUNIT)을 제안한다[1].  
- **진정한 비지도 정의**: 입력-출력 쌍이나 도메인 레이블 없이, 혼합된 여러 도메인의 이미지로부터 도메인을 스스로 추정하여 번역한다[1].  
- **주요 기여**:  
  - **Guiding Network** 설계: 이미지 클러스터링(도메인 추정)과 스타일 인코딩(대조학습) 모듈을 하나의 네트워크에서 공유 표현으로 동시 학습[1].  
  - **상호 피드백 훈련**: 클러스터링↔스타일 인코딩↔GAN 사이에 상호 작용을 도입하여 분리된 도메인 구분 및 스타일 보존을 강화[1].  
  - **반-지도(semi-supervised) 확장**: 극소수의 라벨만 있어도 간단히 확장 가능하며, FUNIT 대비 향상됨[1].  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- 기존 비지도 번역 기법은 **도메인 레이블(set-level labels)**을 필요로 함  
- 대규모 데이터에서 도메인 구분 레이블 확보는 비용·효율성 측면에서 한계  

### 2.2 제안 방법  
TUNIT은 세 가지 손실로 구성된 **공동 최적화** 구조이다[1]:
1. **Mutual Information 기반 클러스터링**
 
$$L_{MI} = -I\bigl(p, p^+\bigr) = -\sum_{i,j} P_{ij}\ln\frac{P_{ij}}{P_iP_j},$$

   여기서 $$p = E_C(x)$$는 도메인 확률 분포, $$x^+$$는 증강 영상이다[1]. 
   
2. **Style Contrastive Loss** (Guiding Network)

$$L_{E}^{style} = -\log\frac{\exp(s\cdot s^+/\tau)}{\sum\nolimits_i \exp(s\cdot s_i^-/\tau)},$$

   $$s = E_S(x)$$는 스타일 코드로, MoCo 큐에서 음성 예시를 활용하여 표현 학습 강화[1].  
   
3. **Translation Losses** (Generator & Discriminator)  
   - **Adversarial Loss**: 멀티태스크 판별기로 도메인별 진위 판단  
   - **Style Contrastive Loss**:
   
  $$L_{G}^{style} = -\log\frac{\exp(s'\cdot \tilde s/\tau)}{\sum\nolimits_i \exp(s'\cdot s_i^-/\tau)}$$
     ($$s' = E_S(G(x,\tilde s))$$, $$\tilde s$$는 참조 스타일)  
     
   - **Reconstruction Loss**:

$$L_{rec} = \|x - G(x,s)\|_1$$ 으로 내용 보존  

전체 목적함수:

$$
\begin{aligned}
L_D &= -L_{adv}, \\
L_G &= L_{adv} + \lambda_{G}^{style} L_{G}^{style} + \lambda_{rec}L_{rec}, \\
L_E &= L_G - \lambda_{MI}L_{MI} + \lambda_{E}^{style}L_{E}^{style}.
\end{aligned}
$$  

(하이퍼파라미터 $$\lambda$$들은 실험적으로 설정)[1].

### 2.3 모델 구조  
| 구성 요소         | 아키텍처 요약                                       |
|------------------|----------------------------------------------------|
| **Guiding Network** (E)  | VGG11 베이스 공유 인코더 + 두 개 분기 (도메인 분류기, 스타일 인코더)[1] |
| **Generator** (G) | AdaIN 기반 ResNet 블록으로 스타일 및 내용 융합[1]        |
| **Discriminator** (D) | Filter Response Normalization 적용한 다중 도메인 판별기[1] |

### 2.4 성능 향상  
- **mFID**: 감독 학습 기반 FUNIT(레퍼런스) 대비 동등~우수 달성[1].  
- **Density & Coverage**: 샘플 다양성에서 오히려 상위권[1].  
- **Semi-Supervised**: 0.1% 라벨만 사용해도 FUNIT·SEMIT 대비 20% 이상 mFID 개선[1].  
- **Hyperparameter** $$\hat K$$ (클러스터 수) 변화에 **강인**하며, 실제 도메인 수보다 크게 설정해도 성능 유지[1].  

### 2.5 한계  
- **고해상도 적용 한계**: 256×256 이상 해상도의 효율적 처리 미검증  
- **도메인 수 과도 추정 시** 일부 클러스터가 의미상 중첩  
- **복잡한 도메인 정의**: 얼굴 속성처럼 연속적·극단적 특성 구분 어려움  

## 3. 일반화 성능 향상 가능성  
- **공동 학습 프레임워크**: 클러스터링 ↔ 대조 학습 ↔ GAN 상호 피드백이 **표현 일반화**에 기여[1].  
- **대조학습 기반 스타일 인코더**: 다양한 이미지 도메인 간 스타일 분리 강화로 **초도 미관측 도메인**에 대한 적응력 우수[1].  
- **클러스터 수 과잉 설정**: 세부 속성 단위로 도메인 분할하여 **미세 스타일** 번역 가능성 제시[1].  

## 4. 앞으로의 연구 영향 및 고려사항  
- **세미·비지도 모델의 표준**: TUNIT이 제시한 “진정한 무라벨(Un-labeled)” 번역 정의는 후속 연구의 **강력 baseline**이 될 전망.  
- **고해상도·실시간 번역**: 대형 네트워크 경량화 및 효율적 훈련 기법 필요.  
- **도메인 추정 자동화 심화**: 연속적 속성(예: 얼굴 표정)을 위한 **계층적 클러스터링** 통합 과제.  
- **다중 모달·다양성 보장**: 스타일·내용 외 추가 속성(음향, 텍스트) 융합으로 **크로스모달 번역** 확장 여지[1].  

**요약**: TUNIT은 완전 비지도 환경에서 이미지-투-이미지 번역 문제를 정의-해결하고, 클러스터링과 대조학습을 통한 공유 표현 학습으로 **지도학습 성능**을 무라벨 상태에서 달성한 혁신적 방법론이다[1]. 앞으로의 연구는 **고해상도, 계층적 도메인 추정, 크로스모달** 영역으로 발전할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e0707c69-5690-40bf-be55-dad778d026c2/2006.06500v2.pdf
