# Reparameterizing Convolutions for Incremental Multi-Task Learning without Task Interference | Image classification, Semantic segmentation

# 논문의 핵심 주장과 기여 요약  
* **주장**: 표준 CNN의 각 합성곱을 (1) *학습이 불가능한* 공유터 뱅크 $$W_s$$와 (2) *학습 가능한* 작업별 모듈레이터 $$W_t^{(i)}$$로 재매개화(reparameterization) 하면,  
  - 새로운 작업을 추가해도 이전 작업의 성능 저하(​catastrophic forgetting·task interference​)가 없고  
  - 매번 전체 모델을 재훈련하지 않는 *점진적(incremental) 다중-작업 학습*이 가능하다는 것이 입증된다[1][2].  
* **주요 기여**[1]:  
  1. **Reparameterized Convolution(RC)**: 모든 $$k \times k$$ Conv를 “고정된 $$W_s$$ + 1 × 1 Conv $$W_t^{(i)}$$”로 분해.  
  2. **Normalized Feature Fusion(NFF)**: $$W_t^{(i)}$$의 각 행을 $$w_t=g_t\frac{v_t}{\lVert v_t\rVert}$$로 정규화해 학습 안정화.  
  3. **Response Initialization(RI)**: 사전학습 가중치 $$W^m$$의 응답 공분산 $$U\Sigma U^\top$$로부터 $$W_s = U^\top W^m$$, $$W_t^{(i)} = U$$를 초기화해 일반화 향상.  
  4. **경량 매개변수 확장**: 매 작업당 파라미터 수가 $$O(c_{out}^2)$$로 완만히 증가(독립 모델의 $$O(k^2c_{in}c_{out})$$ 대비).  
  5. **성능**: PASCAL-Context 5 task 및 NYUD 4 task 실험에서 기존 task-conditional MTL·IL 기법 대비 평균 성능 손실을 0.99%/1.48% 수준으로 최소화[1].  

## 1. 해결하려는 문제점  
1. **Incremental MTL**: 새 작업을 순차적으로 학습할 때, 이전 작업을 그대로 유지하면서도 *재훈련 없이* 확장되어야 함.  
2. **Task Interference**: 다중 작업이 동일 가중치를 공유할 때, 그래디언트 방향 충돌로 각 개별 작업 성능이 악화됨[1][3].  

기존 기법은 *가중치 공유 비율·손실 가중치* 조정[4] 등으로 “충돌을 완화”하려 했으나, 공유 파라미터를 계속 갱신하므로 근본적 충돌을 제거하지 못했다[1][3].

## 2. 제안 방법 (수식 포함)  

### 2.1 Reparameterized Convolution (RC)  
단일 작업 $$i$$에서의 기존 합성곱  

$$
y_i = f(x;W^{(i)})\quad (W^{(i)} \in \mathbb{R}^{c_{out}\times k^2c_{in}})
$$  

을 다음과 같이 재매개화한다:  

$$
y_i = f\!\bigl(x;W_s,W_t^{(i)}\bigr)= (W_t^{(i)}W_s)x 
$$  

* $$W_s\in\mathbb{R}^{c_{out}\times k^2c_{in}}$$: **공유 필터 뱅크**, 학습하지 않음.  
* $$W_t^{(i)}\in\mathbb{R}^{c_{out}\times c_{out}}$$: **작업별 1 × 1 Conv**, 해당 작업만 학습.  

### 2.2 Normalized Feature Fusion (NFF)  
각 행 $$w_t$$를  

$$
w_t = g_t\,\frac{v_t}{\lVert v_t\rVert} 
$$  

로 분해하여 *방향* $$v_t/\lVert v_t\rVert$$와 *스케일* $$g_t$$를 분리, 학습 안정성과 표현력을 동시에 확보[1].

### 2.3 Response Initialization (RI)  
사전학습 가중치 $$W^m$$으로부터 응답 행렬 $$Y$$를 수집하여  

$$Y Y^\top = U\Sigma U^\top$$.  

$$
W_s = U^\top W^m,\qquad W_t^{(i)} = U 
$$  

로 초기화하면, 사전학습 표현 공간에 맞춰진 필터 뱅크를 구성해 초기 일반화를 크게 높인다[1].

### 2.4 모델 구조  
ResNet-18/34 + DeepLabv3+ 인코더-디코더를 사용하되, **모든 Conv를 RC로 교체**. 디코더는 작은 작업별 헤드만 포함된다. 그래프상 공유 경로에서 역전파가 일어나지 않으므로 작업 간 그래디언트 충돌이 원천 차단된다[1].

## 3. 성능·일반화 향상 분석  

| 데이터셋 | 벤치마크 | 기존 최고 기법 | 평균 성능 손실 | RC-NFF-RI(제안) | 평균 손실 | 개선폭 |
|----------|---------|---------------|----------------|-----------------|-----------|--------|
| PASCAL-Context 5 task | ASTMT(R-26+SE)[5] | 4.12% | 0.99%[1] | **–3.13 pp** |
| NYUD 4 task | Parallel RA | 5.02% | **1.48%**[1] | –3.54 pp |

* **일반화 효과**: RI+NFF 결합이 없을 때의 손실 2.13% → 0.99%로 감소, *filter-bank 초기화*와 *정규화*가 일반화·수렴 모두에 기여[1].  
* **Task interference 제거**: Representation Similarity Analysis 결과, 기존 공유-가중치 MTL은 서로 음의 상관을 보였지만 RC 모델은 작업별 그래디언트가 독립적이어서 충돌이 사라짐[1].  
* **Incremental Learning**: “저수준 → 고수준” 혹은 역순으로 작업을 순차 추가해도 초기 작업 성능 하락이 0.5–1.3% 이내로 억제되며, 기존 방법(11–12% 손실) 대비 크게 우수[1].  
* **모델 크기**: 작업 수 $$P$$에 대해 $$O(c_{out}^2P)$$로 증가해, 독립 모델 대비 파라미터·메모리 절감폭이 크다 (예: R-18 기준 10 task에서 64% 절감)[1].  

## 4. 한계 및 고려 사항  
1. **여전히 선형 증분**: $$W_t^{(i)}$$는 작업 수에 비례해 늘어나므로, *수백* 작업 규모에서는 메모리 이슈가 남음.  
2. **공유 필터의 경직성**: $$W_s$$가 고정이므로, 이전에 존재하지 않던 도메인(예: 의료·멀티모달 입력)에는 추가 조정이 필요할 수 있다.  
3. **비전 중심 평가**: 실험이 모두 픽셀-단위 예측 과제에 국한돼, NLP·음성 등 다른 도메인으로의 일반화는 미검증.  
4. **고차 특성 학습**: 1 × 1 모듈레이터가 공간적 패턴까지 충분히 보정할 수 있는지에 대한 이론적 분석은 미비하다.

## 5. 일반화 성능 향상 관점의 시사점  
* **고정 특질 + 가변 어댑터**라는 분할 학습 전략은 *domain-agnostic core representation*을 유지하면서도, 작업별 분산표본에 대한 과적합을 방지해 *out-of-distribution* 일반화를 돕는다.  
* NFF와 RI는 파라미터 스케일·방향을 분리해 탐색 공간을 축소함으로써 *학습 안정성과 초기 성능*을 동시에 확보해, 적은 데이터로도 빠르게 일반화한다[1].

## 6. 앞으로의 연구 영향 및 고려할 점  
1. **확장형 필터 뱅크**: 완전 고정 대신 *저-학습률 미세 조정* 또는 *성장형 필터 셀프-디스틸*을 도입해, 전혀 새로운 도메인까지 포괄하도록 연구 가능.  
2. **Transformer·Mamba 블록 적용**: RC 아이디어를 선형 어텐션 커널이나 순환-기반 구조의 프로젝션 행렬에도 이식해 파라미터 효율적 MTL을 탐색할 수 있다.  
3. **이론적 분석**: 모듈레이터의 용량이 충분한지, 필터 뱅크의 표현 범위와 일반화 한계에 대한 *capacity-generalization trade-off* 연구가 필요.  
4. **메타-러닝 결합**: 작업 간 관계를 활용해 $$W_t^{(i)}$$를 메타-초기화하면, 새로운 작업에서 학습 샷 수를 더욱 줄일 수 있다.  
5. **대규모 태스크 환경**: 수백-수천 작업으로 확장 시, 파라미터 검색·메모리 관리·지식 선택(​task routing​)이 필수 과제가 될 전망이다.

**요약**: 이 논문은 “공유는 고정, 적응은 초소형”이라는 단순한 재매개화를 통해 *점진적·무간섭* 다중-작업 학습을 달성했다. 필터 뱅크 구조는 향후 대규모 멀티태스크·멀티도메인 시대에 파라미터 효율성과 일반화 성능을 동시에 보장하는 핵심 설계 원칙으로 자리매김할 것으로 보인다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fe079734-826e-402e-8427-d8d4e20c7269/2007.12540v1.pdf
[2] https://link.springer.com/10.1007/978-3-030-58565-5_41
[3] https://openaccess.thecvf.com/content/CVPR2023/papers/Ding_Mitigating_Task_Interference_in_Multi-Task_Learning_via_Explicit_Task_Routing_CVPR_2023_paper.pdf
[4] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12291
[5] https://ffighting.net/deep-learning-paper-review/incremental-learning/all-about-incremental-learning/
[6] https://ieeexplore.ieee.org/document/10403885/
[7] https://academic.oup.com/bib/article/doi/10.1093/bib/bbae043/7606633
[8] https://arxiv.org/abs/2405.13771
[9] https://ieeexplore.ieee.org/document/10123087/
[10] https://arxiv.org/abs/2301.03461
[11] https://ieeexplore.ieee.org/document/10058002/
[12] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650681.pdf
[13] https://www.eneuro.org/content/12/3/ENEURO.0542-24.2025
[14] https://arxiv.org/abs/2106.11437
[15] https://arxiv.org/abs/2007.12540
[16] https://arxiv.org/abs/2304.04854
[17] https://arxiv.org/abs/2408.09453
[18] https://arxiv.org/abs/2308.02066
[19] https://www.datacamp.com/blog/what-is-incremental-learning
[20] https://skk095.tistory.com/15
[21] https://www.jstage.jst.go.jp/article/jjptf/24/1/24_46/_article/-char/en
[22] https://www.nature.com/articles/s42256-022-00568-3
[23] https://velog.io/@hsbc/reparameterized-convolution
[24] https://pmc.ncbi.nlm.nih.gov/articles/PMC10942291/
[25] https://en.wikipedia.org/wiki/Incremental_learning
[26] https://dl.acm.org/doi/10.1007/978-3-030-58565-5_41
[27] https://ieeexplore.ieee.org/document/10401928/
[28] https://www.mdpi.com/2075-4418/13/2/262
[29] http://arxiv.org/pdf/1906.00097.pdf
[30] https://arxiv.org/pdf/1810.10703.pdf
[31] https://arxiv.org/html/2401.11124v1
[32] https://aclanthology.org/2022.findings-emnlp.124.pdf
[33] http://arxiv.org/pdf/2301.03461.pdf
[34] http://arxiv.org/pdf/2410.05975.pdf
[35] https://arxiv.org/pdf/1903.12117.pdf
[36] https://arxiv.org/abs/2408.16939
[37] http://arxiv.org/pdf/2501.19067.pdf
[38] https://arxiv.org/pdf/2009.09139.pdf
[39] https://openreview.net/forum?id=rTDyN8yajn
[40] https://github.com/thuml/awesome-multi-task-learning
[41] https://www.sciencedirect.com/science/article/pii/S0749596X05000574
