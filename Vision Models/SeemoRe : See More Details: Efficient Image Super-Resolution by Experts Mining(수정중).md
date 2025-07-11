# SeemoRe : See More Details: Efficient Image Super-Resolution by Experts Mining | Super resolution

# Efficient SR + Expert Mining

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
