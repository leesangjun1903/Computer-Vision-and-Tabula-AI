# Deep Learning of Partial Graph Matching via Differentiable Top-K | Graph matching

# 핵심 주장 및 주요 기여 요약

**“Deep Learning of Partial Graph Matching via Differentiable Top-K”** 논문은 그래프 매칭(Graph Matching) 문제에서완전 대응(partial matching) 상황, 즉 양쪽 그래프에 아웃라이어가 존재할 때 발생하는 불필요한 매칭을 방지하기 위해 다음을 제안한다.

1. **Top-K 선택 기반의 미분 가능한 그래프 매칭 프레임워크**  
   -  매칭 확률 행렬을 벡터화한 뒤, 상위 *k*개의 매칭만 보존하는 최적 수송(Optimal Transport) 문제로 재정의.  
   -  엔트로피 정규화를 도입한 Sinkhorn 알고리즘을 활용해 연속화하고, 역전파가 가능한 형태로 구현.  
   -  테스트 시에는 Hungarian 알고리즘을 적용하여 1:1 매칭을 확보한 후 상위 *k*개를 선택.

2. **자동 *k* 추정용 Attention-Fused Aggregation (AFA) 모듈**  
   -  AFA-I: 개별 그래프의 노드 특징을 intra-/cross-graph 어텐션으로 집계하여 그래프 유사도를 추정하고, 이를 *k* 비율로 변환.  
   -  AFA-U: GM 네트워크의 doubly-stochastic 매칭 행렬을 양자화한 이분 그래프로 보고, 그래프 어텐션을 통해 글로벌 *k* 비율 추정.  
   -  학습 시에는 MSE 손실로 실제 매칭 수와 예측 수를 회귀 학습.

3. **대규모 부분 매칭 벤치마크 IMC-PT-SparseGM**  
   -  IMC-PT(Structure-from-Motion) 데이터셋에서 50/100개의 3D 앵커를 선정하여, 부분 가시성(occlusion)에 따른 부분 매칭 인스턴스 생성.  
   -  평균 노드 수 21.4/44.5, 부분율(partial rate) 57.3%/55.5%로 기존 벤치마크 대비 크게 확장.

4. **성능 및 확장성**  
   -  SOTA GM 네트워크(NGMv2, GCAN)에 플러그인 형태로 통합 가능.  
   -  Pascal VOC, Willow, IMC-PT-SparseGM에서 기존 thresholding·dummy-node 기법 대비 F1 평균 1–3% 향상.  
   -  NGMv2의 sparse 구현으로 노드 수 110에서도 24 GB 한계 돌파.

# 문제 정의 및 제안 방법

## 1. Partial Graph Matching 문제
- 그래프 $$G_1$$, $$G_2$$에 아웃라이어(불일치 노드)가 존재할 때, 유효 대응만 골라내야 함.  
- 전통적 방법은 임계치 기반 thresholding 또는 dummy-node 삽입으로 비차별적(discrete) 처리.

## 2. Differentiable Top-K 기반 프레임워크

1) **매칭 확률 행렬 벡터화**  

$$
   S \in \mathbb{R}^{n_1 \times n_2}
   ,\quad
   \mathbf{d} = \mathrm{vec}(S) \in \mathbb{R}^{n_1n_2}
   $$

2) **Optimal Transport 재정의**  
   - 이동 대상(destination): $$\{d_{\min}, d_{\max}\}$$  
   - 공급량(marginal)  

$$
       r = \frac{1}{n_1n_2}\mathbf{1},\quad
       c = \begin{bmatrix}n_1n_2 - k \\ k\end{bmatrix}
     $$
   - 비용 행렬

$$
     D = 
     \begin{bmatrix}
       \mathbf{d} - d_{\min}\,\mathbf{1}^\top \\
       d_{\max}\,\mathbf{1}^\top - \mathbf{d}
     \end{bmatrix}
     $$

3) **Entropic Regularized Sinkhorn**  

$$
     \min_{\Gamma \ge 0}\; \langle\Gamma, D\rangle \;+\;\varepsilon\sum_{ij}\Gamma_{ij}\log\Gamma_{ij}
     \quad
     \text{s.t.}\;\Gamma \mathbf{1}=r,\;\Gamma^\top\mathbf{1}=c
   $$
   -  교대로 행/열 정규화(iterative row/column normalization)  
   -  수렴 후 상위 행($$\Gamma_{:,2}$$)이 선택 확률  

4) **테스트 시 검출**  
   - $$\Gamma_{:,2}$$를 $$n_1\times n_2$$로 복원  
   - Hungarian + greedy top-k → 최종 이산 매칭

## 3. AFA-I 및 AFA-U 구조

### A. AFA-I (Individual-Graph)

1) **Intra-/Cross-Graph Attention Aggregation**  

$$\ell$$ 레이어 노드 특징 $$x_i^\ell$$ 업데이트  

2) **Attention Pooling**  

$$
     w = \sigma(X W_{\mathrm{att}}\,\bar{x}),\quad
     x_{\mathrm{glb}}=\sum_i w_i x_i
   $$

3) **Neural Tensor Network**  

$$
     s = x_{\mathrm{glb},1}^\top W_1 x_{\mathrm{glb},2} + W_2
     \begin{bmatrix}x_{\mathrm{glb},1} \\ x_{\mathrm{glb},2}\end{bmatrix} + b
   $$
   
   → MLP+Sigmoid → $$k$$ 비율

### B. AFA-U (Unified-Bipartite)

1) **Graph Attention on $$S$$**  
   -  양쪽 파트셋을 Q/K/V로 Transformer-style 어텐션  
   -  원본 어텐션 가중치와 $$S$$를 2-layer MLP로 융합  
2) **Max-Pooling & MLP** → $$k$$

# 성능 향상 및 한계

- **향상폭**: Pascal VOC키포인트 평균 F1 +1.4~2.7%, Willow +1.1~2.3%, IMC-PT-SparseGM +1.0~2.4%  
- **k 추정 정확도**: 평균 노드 20대 기준 ±5 이내 오차 시 성능 급락 없으며, AFA 모듈 오차는 허용 범위 내.  
- **한계**:  
  1. *k* 그라운드 트루스 없을 때, AFA 예측 오차 의존  
  2. Sinkhorn 반복 횟수·정규화 파라미터 $$\varepsilon$$ 민감  
  3. 대규모 노드(수백~천 단위)로 확장 시 계산 비용 증가 가능성

# 모델 일반화 성능 향상 가능성

- **도메인 불변 피처 학습**: CNN 초기 특징추출기 부분에 도메인어댑테이션을 도입하면, AFA-I의 글로벌 유사도 추정이 더욱 견고해질 수 있음.  
- **Self-Supervised AFA 학습**: 실 라벨 없는 상황에서는 예측한 매칭 행렬 $$\Gamma$$의 엔트로피 최소화 등을 목표로 *k*를 자가추정하는 무라벨 학습 전략 고려.  
- **Adaptive $$\varepsilon$$**: 그래프 크기·아웃라이어 비율에 따라 Sinkhorn 정규화 강도를 동적으로 조정하면 다양한 스케일에 일반화.

# 향후 연구에 미치는 영향 및 고려 사항

- **미분가능한 선택 메커니즘**: Top-K를 최적 수송 문제로 풀어낸 방식은 그래프 매칭을 넘어, *subset selection*이 필요한 다양한 비전·추천 시스템 등에 적용 가능.  
- **AFA 모듈 확장**: Graph-level 회귀(log *k*)에서, uncertainty 추정(Bayesian NN)으로 신뢰 범위 제공하도록 발전 여지.  
- **실제 대규모 매칭**: IMC-PT-SparseGM보다 더 방대한 스케일(수백 개 앵커, 수천 이미지)에서 AFA/Sinkhorn의 계산 효율화 및 근사 기법 연구 필요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d332ba6e-315f-4585-8954-cdafb354621f/Wang_Deep_Learning_of_Partial_Graph_Matching_via_Differentiable_Top-K_CVPR_2023_paper.pdf
