# Learning to Compare: Relation Network for Few-Shot Learning | Image classification, Few-shot learning, Zero-shot learning

## 핵심 주장과 주요 기여

### 핵심 주장
이 논문의 핵심 주장은 기존의 고정된 거리 메트릭(유클리드, 코사인 거리 등) 대신 **학습 가능한 깊은 비선형 메트릭**을 사용하여 few-shot learning 성능을 크게 향상시킬 수 있다는 것입니다[1]. 저자들은 임베딩 모듈과 관계 모듈을 함께 end-to-end로 학습시켜 상호 보완적으로 작동하도록 설계했습니다[1].

```
## 1. 학습 가능한 깊은 비선형 메트릭이란?

### 정의와 개념
학습 가능한 깊은 비선형 메트릭은 전통적인 고정된 거리 함수(유클리드, 코사인 거리 등) 대신 **신경망을 통해 학습되는 복잡한 유사도 함수**입니다[1]. 이는 데이터의 비선형적 관계를 포착할 수 있어 단순한 선형 변환보다 훨씬 강력합니다[2][3].

### 기존 연구 사례들

**1. 화자 인식 분야[3]:**
- Deep Restricted Boltzmann Machine을 사용한 비선형 메트릭 학습
- i-vector 공간에서 두 음성의 유사도를 계산하는 명시적 매핑 학습

**2. 얼굴 인식 분야[4]:**
- Independent Subspace Analysis를 활용한 깊은 비선형 메트릭
- 얼굴 특징의 복잡한 변형을 포착하는 비선형 변환 학습

**3. 하이퍼스펙트럴 영상[5]:**
- 스펙트럼 변동성과 비선형 혼합 문제를 해결하기 위한 깊은 판별적 메트릭 학습
- 계층적 비선형 매핑을 통한 목표물과 배경 간의 판별 정보 활용

**4. Potential Field 기반 메트릭 학습[6]:**
- 각 샘플의 영향을 연속적인 잠재 필드로 표현
- 같은 클래스는 인력장, 다른 클래스는 척력장으로 모델링하여 거리에 따른 영향력 감소 적용
```

### 주요 기여
1. **통일된 프레임워크**: Few-shot learning과 zero-shot learning을 모두 처리할 수 있는 단일 프레임워크 제공[1]

```
## 2. Few-shot Learning vs Zero-shot Learning 차이점과 통합 가능성

### 차이점 비교

| 구분           | Few-shot Learning             | Zero-shot Learning |
|---------------|-------------------------------|-------------------|
| **훈련 예시 수** | 클래스당 적은 수의 예시 (1-10개)[7] | 목표 클래스의 예시 없음[7] |
| **의존하는 정보** | 제한된 레이블된 데이터[7]          | 의미적 설명, 속성 벡터[8] |
| **학습 방식**    | 메타러닝, 프로토타입 네트워크[7]     | 교차 모달 임베딩, 속성 매핑[9] |
| **예시**        | 희귀 동물 3장으로 새 종 인식       | "줄무늬 말"이라는 설명으로 얼룩말 인식 |

### 이해하기 쉬운 비유
- **Few-shot Learning**: 새로운 언어를 배울 때 각 단어의 예시를 2-3개씩만 보고 학습하는 것
- **Zero-shot Learning**: 단어의 뜻을 직접 설명해주고, 그 설명만으로 새로운 문장에서 그 단어를 찾아내는 것

### 논문에서 두 방식을 모두 처리할 수 있는 이유

**통합된 아키텍처 설계:**
1. **공통 관계 모듈**: 두 경우 모두 동일한 관계 모듈 gφ를 사용하여 유사도 점수 계산[1]
2. **다른 입력 모달리티 처리**: 
   - Few-shot: 이미지 + 이미지 비교
   - Zero-shot: 이미지 + 의미적 설명 비교[1]
3. **서로 다른 임베딩 모듈**: Zero-shot에서는 fφ1(이미지용)과 fφ2(의미적 벡터용) 두 개의 임베딩 모듈 사용[1]
```

2. **학습 가능한 메트릭**: 고정된 거리 함수 대신 CNN 기반의 학습 가능한 관계 모듈 도입[1]
3. **단순성과 효율성**: RNN이나 fine-tuning 없이도 최고 성능 달성[1]
4. **우수한 실험 결과**: 5개 벤치마크에서 기존 방법들을 능가하는 성능 입증[1]

## 해결하고자 하는 문제

### 문제 정의
Deep learning 모델들이 대량의 레이블된 데이터와 많은 훈련 반복을 필요로 하여, 새로운 클래스나 희귀한 카테고리에 확장성이 제한된다는 문제를 해결하고자 합니다[1]. 특히 클래스당 하나 또는 매우 적은 수의 예시만으로 새로운 시각적 카테고리를 인식해야 하는 few-shot learning의 도전과제를 다룹니다[1].

기존 방법들의 한계:
- 복잡한 추론 메커니즘 필요[1]
- 복잡한 RNN 구조 요구[1]
- Target 문제에 대한 fine-tuning 필요[1]

## 제안하는 방법

### 모델 구조
Relation Network는 두 개의 주요 모듈로 구성됩니다[1]:

1. **임베딩 모듈 (fφ)**: Query 이미지와 support 이미지를 특징 맵으로 변환
2. **관계 모듈 (gφ)**: 결합된 특징 맵에서 관계 점수를 계산

### 핵심 수식

**One-shot 설정에서의 관계 점수 계산:**

$$ r_{i,j} = g_\phi(C(f_\phi(x_i), f_\phi(x_j))), \quad i = 1, 2, \ldots, C $$ [1]

여기서:
- $$C(\cdot, \cdot)$$는 특징 맵의 깊이 방향 연결(concatenation)
- $$r_{i,j}$$는 query $$x_j$$와 support example $$x_i$$ 간의 관계 점수
- C는 클래스 수

```
## 3. 관계 모듈의 관계 점수 산출 과정

### 단계별 설명

**1단계: 특징 추출**
- Query 이미지 xj와 Support 이미지 xi를 각각 임베딩 모듈 fφ에 입력
- 특징 맵 fφ(xj)와 fφ(xi) 생성[1]

**2단계: 특징 결합**
- 두 특징 맵을 깊이 방향으로 연결(concatenation): C(fφ(xi), fφ(xj))[1]
- 결합된 특징 맵이 관계 모듈의 입력이 됨

**3단계: 관계 점수 계산**
- 관계 모듈 gφ가 결합된 특징 맵을 처리
- 최종적으로 0-1 범위의 스칼라 값 출력: ri,j = gφ(C(fφ(xi), fφ(xj)))[1]

**실제 구현:**
- 2개의 convolutional block + 2개의 fully-connected layer로 구성
- 최종 출력층은 Sigmoid 함수를 사용하여 0-1 범위 보장[1]
```

**K-shot 설정 (K > 1):**
각 훈련 클래스의 모든 샘플에 대한 임베딩 모듈 출력을 element-wise로 합하여 클래스 수준의 특징 맵을 형성합니다[1].

```
## 4. K-shot 설정 이해하기 쉬운 설명

### K-shot의 정의
K-shot 설정은 **각 클래스당 K개의 훈련 예시**를 사용하는 방식입니다[1].

### 구체적인 예시
- **1-shot (One-shot)**: 고양이 사진 1장, 강아지 사진 1장으로 학습
- **5-shot**: 고양이 사진 5장, 강아지 사진 5장으로 학습
- **C-way K-shot**: C개 클래스, 각 클래스당 K개 예시

### K-shot에서의 처리 방식
**K > 1일 때의 특별한 처리:**
1. 같은 클래스의 모든 K개 샘플을 임베딩 모듈에 통과시킴
2. **Element-wise 합산**으로 클래스 수준의 특징 맵 생성[1]
3. 이 풀링된 특징 맵을 query 이미지와 비교

**장점:**
- 더 많은 정보로 더 안정적인 클래스 표현 가능
- 각 클래스의 변동성을 더 잘 포착
```

**손실 함수:**
Mean Square Error (MSE) 회귀 손실을 사용합니다[1]:

$$ \phi, \varphi \leftarrow \arg\min_{\phi,\varphi} \sum_{i=1}^{m} \sum_{j=1}^{n} (r_{i,j} - \mathbf{1}(y_i == y_j))^2 $$

매칭되는 쌍은 유사도 1, 매칭되지 않는 쌍은 유사도 0을 목표로 합니다[1].

**Zero-shot Learning 확장:**

$$r_{i,j} = g_\phi(C(f_{\phi_1}(v_c), f_{\phi_2}(x_j))), \quad i = 1, 2, \ldots, C $$ [1]

여기서 $$v_c$$는 의미적 클래스 임베딩 벡터이고, 두 개의 서로 다른 임베딩 모듈을 사용합니다[1].

```
## 5. 의미적 클래스 임베딩 벡터와 서로 다른 임베딩 모듈

### 의미적 클래스 임베딩 벡터 예시

**동물 데이터셋 (AwA):**
- **85차원 속성 벡터 사용**[1]
- 예시 속성: "털이 있음", "4개 다리", "육식성", "큰 크기" 등
- 각 동물 클래스는 이러한 속성들의 연속적 값으로 표현

**조류 데이터셋 (CUB):**
- **312차원 속성 벡터 사용**[1]
- 예시 속성: "부리 모양", "깃털 색상", "크기", "서식지" 등

### 서로 다른 임베딩 모듈의 필요성

**구조적 차이:**

이미지 임베딩: fφ1 (CNN 기반, 4개 convolutional block)
의미적 임베딩: fφ2 (MLP 기반, FC1 + FC2 구조)


**처리 과정:**
1. **이미지**: Inception-V2/ResNet101로 1024/2048차원 벡터 생성[1]
2. **의미적 벡터**: MLP로 이미지 임베딩과 같은 차원으로 변환[1]
3. **정규화**: Hubness 문제 해결을 위해 L2 regularization 적용[1]

```

### 네트워크 아키텍처

**임베딩 모듈:**
- 4개의 convolutional block (각각 64개 필터, 3×3 convolution)
- Batch normalization과 ReLU 활성화 함수
- 처음 두 블록에만 2×2 max-pooling 적용[1]

**관계 모듈:**
- 2개의 convolutional block + 2개의 fully-connected layer
- 각 convolutional block: 3×3 convolution (64 필터) + batch norm + ReLU + 2×2 max-pooling
- FC layer: 8차원, 1차원 (최종 출력은 Sigmoid로 0-1 범위의 관계 점수 생성)[1]

```
## 6. 관계 모듈의 FC Layer 차원 설계 (8차원 → 1차원)

### 설계 근거
**8차원 히든 레이어:**
- **정보 압축**: 연결된 특징 맵의 복잡한 정보를 효율적으로 압축
- **비선형 변환**: ReLU 활성화를 통한 충분한 표현력 확보
- **계산 효율성**: 너무 크지 않아 연산 부담을 줄임

**1차원 출력:**
- **단일 관계 점수**: 두 샘플 간의 유사도를 하나의 스칼라 값으로 표현
- **Sigmoid 활성화**: 0-1 범위의 확률적 해석 가능한 값 생성[1]

### 실험적 최적화
논문에서는 다양한 히든 레이어 크기에 대한 실험을 통해 8차원이 성능과 효율성의 적절한 균형점임을 확인했을 것으로 추정됩니다.
```

## 성능 향상

### 주요 실험 결과

**Omniglot 데이터셋:**
- 5-way 1-shot: 99.6 ± 0.2% (기존 최고 대비 향상)
- 5-way 5-shot: 99.8 ± 0.1%
- 20-way 1-shot: 97.6 ± 0.2%
- 20-way 5-shot: 99.1 ± 0.1%[1]

**miniImageNet 데이터셋:**
- 5-way 1-shot: 50.44 ± 0.82% (당시 최고 성능)
- 5-way 5-shot: 65.32 ± 0.70%[1]

**Zero-shot Learning:**
- CUB 데이터셋에서 62.0% 정확도로 기존 방법들 대비 상당한 성능 향상[1]

## 모델의 일반화 성능 향상

### 핵심 일반화 메커니즘

1. **학습 가능한 메트릭의 장점**: 고정된 메트릭(유클리드, 코사인)과 달리, 데이터 기반으로 최적의 유사도 함수를 학습하여 다양한 도메인에 적응 가능[1]

2. **End-to-end 학습**: 임베딩과 관계 모듈을 동시에 최적화하여 상호 보완적 성능 향상[1]

3. **Episode-based 훈련**: Meta-learning 단계에서 실제 few-shot 상황을 시뮬레이션하여 일반화 능력 향상[1]

4. **비선형 관계 학습**: 단순한 선형 분류기나 nearest neighbor와 달리, 복잡한 비선형 관계를 학습하여 더 정교한 매칭 수행[1]

### 시각화를 통한 일반화 증명
논문에서 제시한 합성 예제와 실제 Omniglot 데이터 시각화를 통해, 기존의 고정된 메트릭으로는 해결할 수 없는 복잡한 관계를 Relation Network가 학습할 수 있음을 보여줍니다[1]. 관계 모듈이 매칭/비매칭 쌍을 선형적으로 분리 가능한 공간으로 매핑한다는 것을 확인했습니다[1].

## 한계점

1. **제한된 아키텍처 다양성**: 주로 CNN 기반 구조에 국한되어 다른 아키텍처와의 호환성 제한
2. **계산 복잡도**: 모든 query-support 쌍에 대해 관계 점수를 계산해야 하므로 클래스 수가 많을 때 계산 비용 증가
3. **MSE 손실의 한계**: 분류 문제임에도 회귀 손실을 사용하는 것에 대한 이론적 근거 부족[1]
4. **메트릭 특성 보장 부재**: 자기 유사성과 대칭성 등 형식적 유사도 함수의 속성을 보장하지 않음[1]

```
## 7. 자기 유사성과 대칭성 보장 방법

### 현재 한계
논문에서 인정한 바와 같이, 현재 아키텍처는 **형식적 유사도 함수의 속성을 보장하지 않습니다**[1]:
- **자기 유사성**: d(x,x) = 0
- **대칭성**: d(x,y) = d(y,x)  
- **삼각부등식**: d(x,z) ≤ d(x,y) + d(y,z)

### 보장 방법들

**1. 대칭성 강제 방법:**
def symmetric_relation_score(x, y):
    r_xy = relation_module(concat(embed(x), embed(y)))
    r_yx = relation_module(concat(embed(y), embed(x)))
    return (r_xy + r_yx) / 2

**2. 자기 유사성 보장:**

def self_similarity_loss(x):
    r_xx = relation_module(concat(embed(x), embed(x)))
    return (r_xx - 1.0) ** 2  # 자기 자신과의 유사도는 1이어야 함

**3. 정규화 기반 접근:**
- **Metric Learning 손실함수 추가**: Triplet loss나 Contrastive loss와 결합[10]
- **대칭성 제약조건**: 손실함수에 대칭성 페널티 추가
- **후처리 정규화**: 출력된 관계 점수를 메트릭 속성을 만족하도록 변환

**4. 아키텍처 수정:**
- **Siamese 구조**: 입력 순서에 관계없이 동일한 출력 보장
- **거리 기반 출력**: 유사도 대신 거리를 직접 학습하여 메트릭 속성 자연스럽게 만족

이러한 방법들을 통해 Relation Network의 학습된 메트릭이 수학적으로 더 엄밀한 유사도 함수가 되도록 개선할 수 있습니다.
```

## 향후 연구에 미치는 영향

### 긍정적 영향
1. **메트릭 러닝 패러다임 변화**: 고정된 메트릭에서 학습 가능한 메트릭으로의 전환 촉진
2. **통합 프레임워크**: Few-shot과 zero-shot learning을 하나의 프레임워크로 처리하는 접근법 제시
3. **단순성과 효율성**: 복잡한 RNN이나 fine-tuning 없이도 우수한 성능 달성 가능성 입증

### 향후 연구 고려사항
1. **더 효율적인 관계 모듈**: 계산 복잡도를 줄이면서도 성능을 유지하는 구조 연구 필요
2. **다양한 도메인 적용**: 시각적 인식 외의 다른 모달리티나 태스크로의 확장 연구
3. **이론적 분석**: 왜 MSE 손실이 효과적인지, 학습된 메트릭의 수학적 특성 등에 대한 깊이 있는 분석 필요
4. **대규모 데이터셋**: 더 큰 규모의 few-shot learning 벤치마크에서의 성능 검증
5. **메타러닝 최적화**: Episode-based 훈련 전략의 개선과 다른 메타러닝 접근법과의 결합 연구

이 논문은 few-shot learning 분야에서 중요한 전환점을 제공했으며, 학습 가능한 메트릭의 중요성을 입증하여 후속 연구들의 방향성을 제시했습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87f066a2-673d-4833-8b32-1ab6bde42038/1711.06025v2.pdf


[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87f066a2-673d-4833-8b32-1ab6bde42038/1711.06025v2.pdf
[2] http://ieeexplore.ieee.org/document/7801132/
[3] https://www.jstage.jst.go.jp/article/transinf/E100.D/1/E100.D_2016EDL8106/_article
[4] https://dl.acm.org/doi/10.1145/2393347.2396303
[5] https://ieeexplore.ieee.org/document/9904945/
[6] https://arxiv.org/abs/2405.18560
[7] https://milvus.io/ai-quick-reference/how-do-fewshot-learning-and-zeroshot-learning-differ
[8] https://onlinelibrary.wiley.com/doi/10.1111/add.16427
[9] https://ieeexplore.ieee.org/document/9895459/
[10] https://velog.io/@sjina0722/%EA%B0%95%EC%9D%98-%EC%A0%95%EB%A6%AC-Deep-Metric-Learning
[11] https://ieeexplore.ieee.org/document/9157478/
[12] https://ieeexplore.ieee.org/document/10460313/
[13] https://dl.acm.org/doi/10.1145/3626772.3657687
[14] https://www.mdpi.com/2079-9292/12/15/3315
[15] https://ieeexplore.ieee.org/document/9878537/
[16] https://arxiv.org/pdf/2110.08607.pdf
[17] https://arxiv.org/pdf/2004.14681.pdf
[18] https://arxiv.org/pdf/1312.6120.pdf
[19] https://arxiv.org/pdf/1410.0440.pdf
[20] https://arxiv.org/pdf/1810.09274.pdf
[21] https://arxiv.org/pdf/2107.14324.pdf
[22] https://arxiv.org/pdf/2201.09267.pdf
[23] https://arxiv.org/html/2503.08952v1
[24] https://arxiv.org/pdf/2310.11439.pdf
[25] https://royalsocietypublishing.org/doi/10.1098/rspa.2021.0830
[26] https://arxiv.org/abs/1909.09427
[27] https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Deep_Relational_Metric_Learning_ICCV_2021_paper.pdf
[28] https://www.ijcai.org/proceedings/2018/0680.pdf
[29] https://arxiv.org/pdf/1508.01534.pdf
[30] http://cvlab.postech.ac.kr/lab/papers/CVPR19_metric_learning.pdf
[31] https://arxiv.org/abs/2010.13511
[32] https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Deep_Meta_Metric_Learning_ICCV_2019_paper.pdf
[33] https://arxiv.org/abs/2312.10046
[34] https://openreview.net/forum?id=Skvd-myR-
[35] https://scispace.com/pdf/non-linear-metric-learning-3njz18dbtw.pdf
[36] https://www.cs.jhu.edu/~kevinduh/papers/tsubaki16nonlinear.pdf
[37] https://www.mdpi.com/1424-8220/21/9/3241
[38] https://untitledtblog.tistory.com/164
[39] https://ojs.aaai.org/index.php/AAAI/article/view/10356
[40] https://github.com/qdrant/awesome-metric-learning
[41] https://doug.tistory.com/63
[42] https://www.sciencedirect.com/science/article/abs/pii/S0893608007002481
[43] https://www.sciencedirect.com/science/article/pii/S0098135422004379
[44] https://techy8855.tistory.com/18
[45] http://medrxiv.org/lookup/doi/10.1101/2025.07.27.25332255
[46] https://www.ijcai.org/proceedings/2023/123
[47] https://ieeexplore.ieee.org/document/9763640/
[48] https://ieeexplore.ieee.org/document/9792444/
[49] https://arxiv.org/abs/2305.14045
[50] https://ieeexplore.ieee.org/document/10415463/
[51] https://arxiv.org/abs/2404.17807
[52] https://ieeexplore.ieee.org/document/10724558/
[53] https://arxiv.org/pdf/2112.10006.pdf
[54] http://arxiv.org/pdf/1711.06025v2.pdf
[55] http://arxiv.org/pdf/2406.16143.pdf
[56] https://arxiv.org/pdf/2007.15484.pdf
[57] http://arxiv.org/pdf/2401.05010.pdf
[58] https://arxiv.org/html/2404.09778v1
[59] https://arxiv.org/pdf/2301.00998.pdf
[60] https://arxiv.org/pdf/2109.04898.pdf
[61] https://arxiv.org/pdf/2205.06743.pdf
[62] http://arxiv.org/pdf/2009.08449.pdf
[63] https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.pdf
[64] https://foxylearning.com/modules/vb-s/lessons/lesson-9-formal-similarity/topics/9-5-formal-similarity-example-3/
[65] https://www.reddit.com/r/MachineLearning/comments/boitjj/d_what_is_the_difference_between_few_one_and/
[66] https://cvlab.postech.ac.kr/research/PMCNet/
[67] https://home.ttic.edu/~avrim/Papers/similarity.pdf
[68] https://www.geeksforgeeks.org/machine-learning/zero-shot-vs-one-shot-vs-few-shot-learning/
[69] https://cis.temple.edu/~latecki/Papers/isvc_submissionv_camera.pdf
[70] https://en.wikipedia.org/wiki/Similarity_measure
[71] https://cartinoe5930.tistory.com/entry/Zero-shot-One-shot-Few-shot-Learning%EC%9D%B4-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C
[72] https://www.arxiv.org/pdf/2507.17785.pdf
[73] https://passthebigabaexam.com/dana-dos-the-defining-features-of-verbal-behavior-explained/
[74] https://learn.microsoft.com/en-us/dotnet/ai/conceptual/zero-shot-learning
[75] https://www.mdpi.com/2073-8994/11/9/1066
[76] https://www.sciencedirect.com/science/article/pii/S1571066107004781
[77] https://www.ultralytics.com/blog/understanding-few-shot-zero-shot-and-transfer-learning
[78] https://openaccess.thecvf.com/content/ICCV2023/papers/Wewer_SimNP_Learning_Self-Similarity_Priors_Between_Neural_Points_ICCV_2023_paper.pdf
[79] http://papers.neurips.cc/paper/4508-supervised-learning-with-similarity-functions.pdf
[80] https://labelbox.com/guides/zero-shot-learning-few-shot-learning-fine-tuning/
[81] https://arxiv.org/abs/2505.23057
