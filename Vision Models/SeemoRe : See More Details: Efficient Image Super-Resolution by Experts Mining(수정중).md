# SeemoRe : See More Details: Efficient Image Super-Resolution by Experts Mining | Super resolution

# Efficient SR + Expert Mining

# 핵심 주장 및 주요 기여

**핵심 주장:** SeemoRe는 협업하는 전문가(Experts)를 저사양 환경에서도 효과적으로 활용해, 기존 CNN 및 경량 Transformer 방식보다 뛰어난 효율성과 화질을 동시에 달성하는 새로운 이미지 초해상도 모델이다.  

**주요 기여:**  
1. **협업 전문가 구조**  
   - **Rank Modulating Expert (RME):** 서로 다른 저차원(rank) 전문화 모델(Mixture of Low-Rank Expertise, MoRE)을 동적으로 선택해 전역 텍스처를 효율적으로 복원.  
   - **Spatial Modulating Expert (SME):** 대규모 스트라이프(depth-wise) 합성곱을 이용해 국소 공간 정보를 보강하는 Spatial Enhancement Expertise(SEE) 도입.  

2. **효율성-화질 균형**  
   - **실험 결과:** Manga109, Urban100 등 주요 벤치마크에서 기존 경량 모델 대비 GMACS를 최대 51% 절감하면서 PSNR 0.16–0.18dB 상승(×2) (Figure 1).  
   - **모델 스케일링:** Tiny(Base), Large(scale) 모델 모두 동일 아키텍처로 효율성과 성능을 동시 확보.  

3. **동적 전문가 선택**  
   - **Top-1 Routing:** 입력 및 네트워크 깊이에 따라 최적의 저차원 전문가를 단일(top-1)으로 선택, 추론 시 불필요 연산 제거.  

4. **간결한 설계**  
   - 모놀리식 블록 대신 일관된 Residual Group 구성, RME→SME 순서로 배치해 구조적 단순성 유지.  

# 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

## 1. 문제 정의  
저해상도(LR) 입력으로부터 고해상도(HR) 이미지를 재구성할 때, 제한된 연산 자원(GMACS) 내에서 “전역 문맥”과 “국소 공간” 특징을 모두 효율적으로 모델링하는 것이 어려움.  

## 2. 제안 방법  
### 2.1 Mixture of Low-Rank Expertise (MoRE)  
- 입력 피처 $$x\in \mathbb{R}^{H\times W\times C}$$를 두 갈래로 분기:  
  $$\hat{x}_a$$ (공간 정보)와 $$\hat{x}_b$$ (문맥 피라미드) 생성  
- 각 전문가 $$E_i$$는 저차원(rank $$R_i$$) 연산으로 두 피처를 결합:  

$$
    E_i = W_{3}^{R_i \to C}\bigl(W_{1}^{C \to R_i}\hat{x}\_a \;\odot\; W_{2}^{C \to R_i}\hat{x}_b\bigr)
$$

- 라우터 $$G(\hat{x}_a)$$가 Softmax로 전문가별 가중치 $$g_i$$ 생성, Top-1 전문가만 활성화  
- 출력  

$$
    y = \sum_{i=1}^n g_i E_i(\hat{x}_a,\hat{x}_b) + \hat{x}_a
$$

### 2.2 Spatial Enhancement Expertise (SEE)  
- 입력 $$x_{in}$$에 대해 두 개의 투영 $$W_4, W_5$$ 후 대형 스트라이프 합성곱 적용  
- 공간 피처 보강 및 원본 피처와 Hadamard 곱:  

$$
    x_{out} = \text{DConv}\_{k\times k}(W_4\,x_{in}) \;\odot\; W_5\,x_{in}
$$

## 3. 모델 구조  
- **Shallow Feature:** 3×3 Conv → 초기 피처  
- **Residual Groups (RGs):** 순서대로 RME → GatedFFN → SME → GatedFFN  
- **Upsampler:** 3×3 Conv + Pixel Shuffle  
- 스케일별 RG 수: Tiny 6, Base 8, Large 16; 채널 36–48  

## 4. 성능 향상  
- ×2, ×3, ×4 모든 배율에서 PSNR/SSIM에서 기존 CNN·경량 Transformer 상회 (Table 1–2)  
- 예: Manga109 ×2에서 Tiny 모델이 PSNR 39.01dB로 SAFMN 대비 +0.30dB, GMACS −37%[1]  
- 메모리·추론 속도: Tiny 모델이 DDistill-SR 대비 GPU 메모리 −3%, 속도 2배 이상  

## 5. 한계  
- **블록 수 증가 시** 성능 향상 둔화  
- **전문가 수 확장 어려움:** 저차원 공간 급증  
- **실제 블러 현상** 완전 제거 불가(Transformer 대비 여전히 일부 존재)  

# 일반화 성능 향상과 전망

- **동적 전문가 선택**은 입력 복잡도에 유연 대응 가능 → 저조도·노이즈 제거 등 다양한 복원 과제로 확장 여지  
- **MoRE 구조**는 다른 저정보 도메인(저조도, 압축 해제)에도 활용 가능  
- **대규모 전문가 집합** 도입 시 더 세분화된 패턴 학습 기대, 다만 연산·메모리 비용 관리 필요  
- **추가 제약(orthogonality, diversity loss)**을 통해 전문가 간 중복 최소화 및 일반화 강인성 제고  

# 향후 연구 영향 및 고려 사항

- **영향:**  
  - 효율적 SR 연구에 “저차원 전문가 선택” 패러다임 제시  
  - CNN 기반 모델의 Transformer 대비 경쟁력 재확인  

- **고려 사항:**  
  1. 전문가 수·차원 확장에 따른 자원 최적화  
  2. 훈련 시 전문가 다양성 제약 기법 도입  
  3. 실제 저해상도 영상(실사, 의료) 일반화 성능 검증  
  4. 블러·노이즈 대응을 위한 주파수 영역 손실 결합 전략  

[1] Table 1, Figure 1  Table 3

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a1ebf23a-4051-47fc-9519-4a5a39e1776e/2402.03412v2.pdf

# Abs
저해상도(LR) 입력에서 고해상도(HR) 이미지를 재구성하는 것은 이미지 초해상도(SR)에서 중요한 과제입니다.  
최근의 접근 방식은 다양한 목표에 맞게 맞춤화된 복잡한 연산의 효율성을 입증했지만, 이러한 서로 다른 연산을 단순하게 쌓는 것은 상당한 계산 부담을 초래하여 실용성을 방해할 수 있습니다.  
이에 저희는 전문가 마이닝(expert mining)을 사용하는 효율적인 SR 모델(efficient SR model)인 SeeMoRe를 소개합니다.  
저희의 접근 방식은 다양한 수준의 전문가를 전략적으로 통합하여 협업 방법론을 채택합니다.  
거시적 규모(macro scale)에서 저희의 전문가는 순위별(rank-wise) 및 공간별(spatial-wise) 정보들의 feature들을 다루며 전체적인 이해를 제공합니다.  
그 후 이 모델은 낮은 순위(low-rank)의 전문가를 혼합하여 순위 선택의 미묘한 부분까지 살펴봅니다.  
정확한 SR에 중요한 개별 주요 요소에 전문화된 전문가를 활용하여 저희 모델은 복잡한 기능 내 세부 사항을 파악하는 데 탁월합니다.  
이 협업 접근 방식은 "see more"의 개념을 연상시켜 효율적인 설정에서 최소한의 계산 비용으로 최적의 성능을 달성할 수 있도록 해줍니다.  

# Introduction
단일 이미지 초해상도(SR)는 성능이 저하된 저해상도(LR)에 대응하여 고해상도(HR) 이미지의 재구성을 추구하는 오랜 비전에서 시도되는 방법입니다.  
이 어려운 작업은 초고화질 장치 및 비디오 스트리밍 애플리케이션의 신속한 개발로 인해 상당한 주목을 받았습니다(Khani et al., 2021; Zhang et al., 2021a).  
리소스 제약을 미리 고려하여 이러한 장치 또는 플랫폼에서 고해상도 이미지를 완벽하게 시각화하기 위한 효율적인 초해상도 모델을 설계하고자 합니다.  
고해상도 픽셀이 누락될 가능성이 가장 높은 후보를 식별하는 것은 초해상도로 이어질 수 있는 단계를 이어지게 합니다.  
외부 지식이 없는 경우 초해상도에 대한 주요 접근 방식은 재구성을 위해 인접 픽셀 간의 복잡한 관계를 탐색하는 것을 포함합니다.  
최근 초해상도 모델은 (a) attention(Liang et al., 2021; Zhou et al., 2023; Chen et al., 2023), (b) feature mixing(Hou et al., 2022; Sun et al., 2023), (c) global-local context modeling(Wang et al., 2023; Sun et al., 2022)과 같은 방법을 통해 이를 예시하여 놀라운 정확도를 제공합니다.

이 작업의 다른 접근 방식과 달리, 저희는 특정 요소에 초점을 맞춘 복잡하고 연결되지 않은 블록을 피하고 대신 모든 측면에 특화된 통합 학습 모듈을 선택하는 것을 목표로 합니다.  
그러나 효율성 요구 사항으로 인해 특히 리소스가 제한된 장치의 맥락에서 방대한 매개 변수를 통한 암시적 학습을 실현할 수 없게 만드는 추가적인 문제가 발생합니다.  
이러한 효율적인 통합을 달성하기 위해 다양한 전문가의 시너지를 활용하여 기능 내 얽힘을 극대화하고 LR 픽셀 간의 응집력 있는 관계를 협력적으로 학습하는 SeeMoRe를 소개합니다.  
저희의 동기는 이미지 feature들이 종종 다양한 패턴과 구조를 표시한다는 관찰에서 비롯됩니다. 단일 모델로 이러한 모든 패턴을 캡처하고 모델링하려고 시도하는 것은 어려울 수 있습니다.  
반면에 협력 전문가(Collaborative experts)는 네트워크가 입력 공간의 다양한 영역이나 측면에 특화되어 다양한 패턴에 대한 적응력을 향상시키고 "See More"와 유사하게 LR-HR 종속성의 모델링을 용이하게 합니다.  

기술적으로 저희 네트워크는 두 가지 다른 측면에 초점을 맞춰 전문가를 통해 중추적인 기능을 동적으로 선택하기 위한 stacked residual groups(RG)로 구성되어 있습니다.  
매크로 수준에서 각 RG는 (a) 낮은 순위 변조를 통해 가장 유익한 기능을 처리하는 데 전문가인 Rank modulating expert(RME)와 (b) 효율적인 공간 향상에 전문가인 Spatial modulating expert(SME)의 두 가지 연속적인 전문가 블록을 구현합니다.  
마이크로 수준에서는 글로벌 컨텍스트 관계를 암시적으로 모델링하면서 다양한 입력과 다양한 네트워크 깊이에서 가장 적합하고 최적의 순위를 동적으로 선택하기 위해 RME 내의 기본 구성 요소로 Mixture of Low-Rank Expertise(MoRE)를 고안합니다.  
또한 공간별 로컬 집계 기능을 크게 개선하기 위해 SME 내의 복잡한 self-attention에 대한 효율적인 대안으로 Spatial Enhancement Expertise(SE)를 설계합니다.  
이러한 조합은 기능 속성 내의 상호 종속성을 효율적으로 변조하여 모델이 초해상도의 핵심 측면인 높은 수준의 정보를 추출할 수 있도록 합니다.  
서로 다른 전문 지식을 위해 서로 다른 세분성(granularity)에서 전문가를 명시적으로 마이닝함으로써 네트워크는 공간 과 채널 feature들간의 복잡성을 탐색하여 시너지 기여도를 극대화하고 더 많은 세부 정보를 정확하고 효율적으로 재구성합니다.

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2010.48.55.png)
그림 1에서 볼 수 있듯이, 당사의 네트워크는 Ddistill-SR(Wang et al., 2022) 또는 SAFMN(Sun et al., 2023)과 같은 최첨단(SOTA) 효율적인 모델을 상당한 차이로 능가하면서도 GMACS의 절반 또는 그 이하만 활용합니다.  
당사의 모델은 효율적인 SR을 위해 특별히 설계되었지만, 더 큰 모델은 성능 면에서 SOTA lightweight transformer를 능가하면서도 계산 비용은 절감하기 때문에 확장성이 분명합니다.  
전반적으로 당사의 주요 기여는 다음의 세 가지입니다:
- 우리는 Transformer 기반 방법의 다양성과 CNN 기반 방법의 효율성에 부합하는 SeemoRe를 제안합니다.
- 관련 기능 예측 간의 복잡한 상호 의존성을 효율적으로 조사하기 위한 Rank modulating expert(RME)이 제안됩니다.
- 로컬 컨텍스트 정보를 인코딩하여 SME에서 추출한 보완 기능을 통합하는 Spatial modulating expert(SME)이 제안됩니다.

# Related Works

# Methodology
이 섹션에서는 효율적인 초해상도를 위해 조정된 제안된 모델의 기본 구성 요소를 공개합니다.  
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2010.58.02.png)
그림 2에서 볼 수 있듯이, 저희의 전체 파이프라인은 N개의 residual groups(RG) 시퀀스와 upsampler layer를 구현합니다.  
초기 단계에는 입력 저해상도(LR) 이미지에서 얕은 특징을 생성하기 위해 3x3 convolution layer을 적용하는 것이 포함됩니다.  
이후 여러 개의 적층된 RG를 배포하여 심층 특징을 개선하여 고해상도(HR) 이미지의 재구성을 용이하게 하고 효율성을 유지합니다.  
각 RG는 RME(Rank Modulation Expert)와 SME(Spatial Modulation Expert)로 구성됩니다.  
마지막으로, global residual connection는 얕은 특징을 high-frequency details을 캡처하기 위한 deep features의 출력과 연결하고 더 빠른 재구성을 위해 up-sampler layer(3x3 및 픽셀 shuffle(Shi et al., 2016)를 배포합니다.

## Rank Modulating Expert
LR-HR 종속성을 모델링하기 위해 행렬 연산에 의존하는 대규모 커널 컨볼루션(Hou et al., 2022) 또는 self-attention(Vaswani et al., 2017)과 달리, 저희는 효율성을 추구하기 위해 낮은 순위에서 가장 관련성이 높은 상호 작용을 변조하는 것을 선택했습니다.  
저희가 제안한 Rank modulating expert(RME)(그림 2 참조)은 관련 글로벌 정보 특징을 효율적으로 모델링하기 위해 Mixture of Low-Rank Expertise(MoRE)을 사용하는 Transformer과 정제된 컨텍스트 특징 집계를 위한 GatedFFN(Chen et al., 2023)을 사용하는 유사한 아키텍처를 탐구합니다.

## Mixture of Low-Rank Expertise
