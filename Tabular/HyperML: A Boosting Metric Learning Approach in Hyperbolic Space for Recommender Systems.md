# HyperML: A Boosting Metric Learning Approach in Hyperbolic Space for Recommender Systems

## 1. 핵심 주장과 주요 기여

**HyperML**은 추천 시스템에서 유클리드 공간이 아닌 **쌍곡선 공간(hyperbolic space)**에서 사용자-아이템 표현을 학습하는 최초의 메트릭 러닝 접근법입니다. 이 연구의 핵심 아이디어는 쌍곡선 공간의 **지수적 확장 특성**을 활용하여 기존 유클리드 공간 기반 메트릭 러닝의 성능을 크게 향상시키는 것입니다.[1]

**주요 기여:**

- **최초의 쌍곡선 메트릭 러닝**: 추천 시스템에서 쌍곡선 공간을 활용한 메트릭 러닝의 첫 번째 시도[1]
- **Möbius gyrovector spaces 활용**: 유클리드 벡터 연산을 쌍곡선 공간으로 일반화[1]
- **Distortion 페널티 도입**: 정확도와 거리 보존 간의 균형을 제어하는 새로운 손실 함수[1]
- **다중 벤치마크에서 SOTA 달성**: 10개 공개 데이터셋에서 기존 방법들 대비 최대 32.32% 성능 향상[1]

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의
기존 추천 시스템의 메트릭 러닝 방법들(CML, LRML 등)은 **유클리드 공간**에서만 동작하여 복잡한 계층적 사용자-아이템 관계를 효과적으로 모델링하지 못합니다. 특히 협업 필터링에서 사용자의 선호도와 아이템 간의 거리 관계를 표현할 때 한계가 있었습니다.[1]

### 제안 방법

**1) 쌍곡선 거리 함수:**
쌍곡선 공간에서의 일반화된 거리는 다음과 같이 정의됩니다:[1]

$$d_c(x,y) = \frac{2}{\sqrt{c}} \tanh^{-1}(\sqrt{c}\|(-x) \oplus_c y\|)$$

여기서 $$c \geq 0$$는 곡률 매개변수이며, $$\oplus_c$$는 Möbius 덧셈입니다.

**2) Pull-and-Push 손실 함수:**
메트릭 러닝의 핵심인 긍정 쌍은 가깝게, 부정 쌍은 멀게 하는 손실 함수:[1]

$$L_P = \sum_{(i,j) \in S} \sum_{(i,k) \notin S} [m + d_D^2(i,j) - d_D^2(i,k)]_+$$

**3) Distortion 최적화:**
거리 보존을 위한 정규화 항:[1]

$$L_D = \sum_{(i,j) \in S} \left|\frac{|d_D(f(i), f(j)) - d_E(i,j)|}{d_E(i,j)}\right|_+ + \sum_{(i,k) \notin S} \left|\frac{|d_D(f(i), f(k)) - d_E(i,k)|}{d_E(i,k)}\right|_+$$

**4) 다중 태스크 학습:**
최종 목적 함수는 정확도와 거리 보존 간의 균형을 맞춥니다:[1]

$$\min_{\Theta} L = L_P + \gamma L_D$$

## 3. 모델 구조

HyperML은 두 가지 최적화 방식을 제공합니다:[1]

1. **직접 최적화**: 단위볼 내에서 직접 임베딩을 최적화
2. **접선공간 최적화**: 지수/로그 매핑을 통해 접선공간에서 최적화 후 매니폴드로 투영

모델은 사용자-아이템-부정아이템 삼조(triplet)를 입력으로 받아 쌍곡선 공간에서 계층적 구조를 학습합니다. 사용자는 원점 근처에, 아이템들은 사용자 주변의 구면에 배치되어 자연스러운 계층 구조를 형성합니다.[1]

## 4. 성능 향상 및 한계

### 성능 향상
- **전체적 우수성**: 모든 10개 벤치마크 데이터셋에서 기존 방법들을 일관되게 outperform[1]
- **특히 희소 데이터에서 효과적**: Sports & Outdoors에서 32.32%, Automotive에서 21.06% 향상[1]
- **메트릭 러닝 baseline 대비**: CML과 LRML을 모든 데이터셋에서 능가[1]

### 한계점
- **매개변수 민감성**: 곡률 매개변수 c와 다중태스크 가중치 γ에 대한 신중한 조정 필요[1]
- **계산 복잡도**: 쌍곡선 연산으로 인한 추가적인 계산 비용
- **소규모 데이터셋**: 매우 작은 데이터셋(예: Automotive)에서는 간단한 MF-BPR이 때로 경쟁력 있는 성능을 보임[1]

## 5. 일반화 성능 향상 가능성

HyperML은 여러 측면에서 **뛰어난 일반화 성능**을 보여줍니다:[2][3]

### 계층적 구조 모델링
쌍곡선 공간의 **지수적 확장 특성**은 복잡한 사용자-아이템 계층 관계를 더 효과적으로 포착할 수 있습니다. 이는 새로운 사용자나 아이템에 대해서도 기존 계층 구조를 기반으로 한 일반화가 가능함을 시사합니다.[2][1]

### 저차원에서의 효율성
실험 결과, HyperML은 **낮은 차원에서도 우수한 성능**을 보입니다. 이는 모델의 일반화 능력이 높고 overfitting 위험이 낮다는 것을 의미합니다.[4]

### Cross-Domain 적용성
최근 연구들에서 쌍곡선 임베딩이 **다양한 도메인**에서 일관된 성능 향상을 보이고 있어, HyperML의 접근법이 다른 추천 도메인으로도 확장 가능함을 보여줍니다.[5][6][2]

## 6. 연구에 미치는 영향과 향후 고려사항

### 학술적 영향

**1) 새로운 연구 방향 개척**
- **LLM과의 결합**: HyperLLM과 같은 최신 연구에서 대규모 언어모델과 쌍곡선 공간을 결합하여 40% 이상의 성능 향상을 달성[2]
- **다양한 아키텍처로의 확장**: Vision Transformer 기반 모델, Graph Neural Networks 등으로 확장[7][8]

**2) 메트릭 러닝 패러다임 변화**
기존 유클리드 공간 중심의 메트릭 러닝에서 **기하학적 특성을 고려한 설계**로의 전환을 촉진했습니다.[9][10]

### 향후 연구 고려사항

**1) 확장성 문제**
- **대규모 데이터**: 수백만 사용자/아이템을 가진 대규모 시스템에서의 효율성 개선 필요
- **실시간 추론**: 온라인 서비스에서의 응답 시간 최적화

**2) 이론적 깊이 확장**
- **수렴성 보장**: 쌍곡선 공간에서의 최적화 수렴성에 대한 이론적 분석 필요
- **일반화 이론**: 쌍곡선 임베딩의 일반화 능력에 대한 PAC-Bayes 이론 등 적용[11]

**3) 실용적 고려사항**
- **해석가능성**: 쌍곡선 공간에서 학습된 표현의 해석 방법 개발
- **하이브리드 접근법**: 유클리드와 쌍곡선 공간의 장점을 결합한 적응적 방법론[8]

**4) 새로운 응용 분야**
- **멀티모달 추천**: 텍스트, 이미지 등 다양한 모달리티를 쌍곡선 공간에서 통합[2]
- **동적 추천**: 시간에 따른 사용자 선호도 변화를 쌍곡선 공간에서 모델링[12]

HyperML은 추천 시스템 분야에서 **기하학적 관점**의 중요성을 입증했으며, 향후 연구는 이론적 깊이와 실용적 확장성을 동시에 추구하는 방향으로 발전할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/68df3639-bb7d-4dcd-9792-5d333b8e8de8/1809.01703v3.pdf)
[2](https://arxiv.org/abs/2504.05694)
[3](https://dl.acm.org/doi/10.1145/3336191.3371850)
[4](https://neurips.cc/virtual/2024/poster/93170)
[5](https://pubsonline.informs.org/doi/10.1287/isre.2022.0202)
[6](http://link.springer.com/10.1007/978-3-030-65351-4_11)
[7](https://www.nature.com/articles/s41598-023-38320-5.pdf)
[8](https://arxiv.org/html/2509.05757v1)
[9](https://arxiv.org/pdf/2411.06374.pdf)
[10](https://www.sciencedirect.com/science/article/abs/pii/S095219762401621X)
[11](http://proceedings.mlr.press/v80/sala18a/sala18a.pdf)
[12](https://www.koreascience.kr/article/JAKO202204859393346.page)
[13](https://ieeexplore.ieee.org/document/10598032/)
[14](https://link.springer.com/10.1007/s10489-023-05045-x)
[15](https://www.semanticscholar.org/paper/6990a2fbced2c509ffae1f85113690c7a400aa98)
[16](https://dl.acm.org/doi/10.1145/3631700.3664872)
[17](https://ieeexplore.ieee.org/document/9835344/)
[18](https://ieeexplore.ieee.org/document/10001755/)
[19](https://arxiv.org/abs/1809.01703v3)
[20](https://arxiv.org/pdf/2411.13865.pdf)
[21](http://arxiv.org/pdf/2204.08176.pdf)
[22](https://arxiv.org/html/2504.01541v2)
[23](https://arxiv.org/pdf/2308.15244.pdf)
[24](https://arxiv.org/pdf/2106.07720.pdf)
[25](http://arxiv.org/pdf/2406.17289.pdf)
[26](http://arxiv.org/pdf/1902.08648.pdf)
[27](https://arxiv.org/pdf/2207.09051.pdf)
[28](https://arxiv.org/html/2403.20298)
[29](https://arxiv.org/html/2504.05694v1)
[30](https://arxiv.org/abs/1809.01703)
[31](https://www.sciencedirect.com/science/article/pii/S1110016824004952)
[32](https://personal.ntu.edu.sg/xlli/publication/WSDM.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2024W/CVFAD/papers/Shimizu_A_Fashion_Item_Recommendation_Model_in_Hyperbolic_Space_CVPRW_2024_paper.pdf)
[34](https://dl.acm.org/doi/10.1145/3480651.3480695)
[35](https://arxiv.org/abs/2412.01023)
[36](https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412219)
[37](https://arxiv.org/html/2306.12680)
[38](https://openreview.net/forum?id=wBtmN8SZ2B)
[39](https://openreview.net/forum?id=0TZs6WOs16)
[40](https://openaccess.thecvf.com/content_CVPR_2020/papers/Khrulkov_Hyperbolic_Image_Embeddings_CVPR_2020_paper.pdf)
