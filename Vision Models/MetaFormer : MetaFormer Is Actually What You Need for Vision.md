# MetaFormer Is Actually What You Need for Vision | Image classification

## 핵심 주장

본 논문의 핵심 주장은 **Transformer 모델의 성공 요인이 attention 메커니즘 자체가 아니라 전반적인 아키텍처 구조(MetaFormer)에 있다**는 것입니다. 저자들은 attention module을 단순한 pooling 연산으로 대체해도 경쟁력 있는 성능을 달성할 수 있음을 입증하여 이 가설을 검증했습니다.[1][2]

## 주요 기여

### 1. MetaFormer 개념 제안
- **MetaFormer**: Transformer에서 token mixer(attention)를 특정하지 않은 일반화된 아키텍처
- Token mixer 부분을 추상화하여 다양한 연산(attention, MLP, pooling 등)이 대체 가능한 범용 프레임워크 제시[1]

### 2. PoolFormer 모델 개발
- Token mixer로 **embarrassingly simple** pooling 연산만 사용하는 모델 제안
- ImageNet-1K에서 82.1% top-1 정확도 달성[1]
- DeiT-B/ResMLP-B24 대비 0.3%/1.1% 높은 정확도를 35%/52% 적은 파라미터로 달성[1]

## 해결하고자 하는 문제

### 기존 연구의 한계점
1. **Attention 중심주의**: Transformer의 성공을 attention 메커니즘에만 귀속시키는 일반적 믿음
2. **복잡한 token mixer 설계 경향**: 성능 향상을 위해 점점 복잡해지는 attention 변형들
3. **아키텍처 자체의 중요성 간과**: 전반적인 구조보다 특정 모듈에만 집중하는 연구 경향

### 핵심 연구 질문
"Vision Transformer의 성능은 정말로 attention 메커니즘 때문인가, 아니면 전반적인 아키텍처 구조 때문인가?"[1]

## 제안하는 방법 및 수식

### MetaFormer 아키텍처
MetaFormer 블록의 수학적 표현:

**첫 번째 서브블록 (Token Mixing)**:

$$Y = \text{TokenMixer}(\text{Norm}(X)) + X$$[3]

**두 번째 서브블록 (Channel MLP)**:

$$Z = \sigma(\text{Norm}(Y)W_1)W_2 + Y$$[3]

여기서:
- $X \in \mathbb{R}^{N \times C}$: 입력 토큰 임베딩
- $W_1 \in \mathbb{R}^{C \times rC}$, $W_2 \in \mathbb{R}^{rC \times C}$: MLP 가중치
- $r$: MLP 확장 비율
- $\sigma(\cdot)$: 활성화 함수 (GELU)

### PoolFormer의 Pooling 연산
PoolFormer에서 사용하는 pooling 연산:

$$T'\_{:,i,j} = \frac{1}{K \times K} \sum_{p,q=1}^{K} T_{:, i+p-\frac{K+1}{2}, j+q-\frac{K+1}{2}} - T_{:,i,j}$$[3]

여기서:
- $K$: pooling 크기 (기본값 3)
- 입력 자체를 빼는 이유: MetaFormer 블록에 이미 residual connection이 존재하기 때문

## 모델 구조

### 계층적 구조
PoolFormer는 4단계 계층 구조를 채택:
- **Stage 1**: $\frac{H}{4} \times \frac{W}{4}$ 토큰, 임베딩 차원 64/96
- **Stage 2**: $\frac{H}{8} \times \frac{W}{8}$ 토큰, 임베딩 차원 128/192  
- **Stage 3**: $\frac{H}{16} \times \frac{W}{16}$ 토큰, 임베딩 차원 320/384
- **Stage 4**: $\frac{H}{32} \times \frac{W}{32}$ 토큰, 임베딩 차원 512/768[3]

### 블록 배치
총 L개 블록을 각 단계에 L/6, L/6, L/2, L/6 비율로 배치[3]

## 성능 향상

### ImageNet-1K 분류 성능
| 모델 | 파라미터 (M) | MACs (G) | Top-1 정확도 (%) |
|------|-------------|----------|------------------|
| PoolFormer-S12 | 11.9 | 1.8 | 77.2 |
| PoolFormer-S24 | 21.4 | 3.4 | 80.3 |
| PoolFormer-S36 | 30.8 | 5.0 | 81.4 |
| PoolFormer-M36 | 56.1 | 8.8 | 82.1 |
| DeiT-B | 86 | 17.5 | 81.8 |
| ResMLP-B24 | 116 | 23.0 | 81.0 |[3]

### 다운스트림 태스크 성능
- **COCO 객체 탐지**: RetinaNet 기준 ResNet 대비 일관된 성능 향상[3]
- **ADE20K 의미분할**: FPN 기준 ResNet/PVT 대비 높은 mIoU 달성[3]

## 일반화 성능 향상 가능성

### 1. 아키텍처 유연성
- **임의의 token mixer 호환성**: Identity mapping, random matrix 등 다양한 연산자와 호환[4]
- **IdentityFormer**: Token mixer를 identity mapping으로 설정해도 74.3% 정확도 달성[3]
- **RandFormer**: Random matrix token mixer로 75.8% 정확도 달성[3]

### 2. 계산 효율성
- **선형 복잡도**: Pooling 연산은 토큰 수에 대해 선형 복잡도 (attention/MLP는 제곱)
- **파라미터 효율성**: 추가 학습 파라미터 없이 token mixing 수행

### 3. 하이브리드 모델 가능성
Pooling과 attention을 조합한 하이브리드 모델에서 더욱 향상된 성능:
- 하위 단계에서 pooling, 상위 단계에서 attention 사용 시 81.0% 정확도 달성[3]

## 한계점

### 1. 토큰 믹싱 능력의 제한
- **로컬 정보 처리**: Pooling 연산은 본질적으로 로컬한 spatial modeling만 수행
- **글로벌 의존성 모델링 부족**: Long-range dependency 포착에 한계

### 2. 태스크별 최적화 부족
- **특정 도메인 특화 어려움**: 복잡한 형태나 세밀한 특징이 중요한 태스크에서 제한적[5][6]
- **의료 영상 등 특수 도메인**: DeformableFormer 등 개선된 모델이 필요[6][5]

### 3. 암묵적 토큰 믹싱 의존성
- **Overlapping patch embedding**: 실제로는 patch embedding 단계에서 암묵적 토큰 믹싱 발생[7]
- **순수한 비교의 어려움**: 완전히 독립적인 토큰 믹싱 비교에 한계

## 미래 연구에 미치는 영향

### 1. 연구 패러다임 전환
- **아키텍처 중심 연구**: 복작한 attention 설계보다 전반적 구조 개선에 집중[1]
- **MetaFormer 기반 모델들**: CAFormer, ConvFormer 등 후속 연구 활발[4]

### 2. 효율성 중심 모델 개발
- **경량화 모델**: MicroViT, ParFormer 등 edge device용 모델 개발 영감 제공[8][9]
- **계산 효율성**: 성능 대비 계산량 최적화 연구 방향 제시

### 3. 의료/특수 도메인 적용
- **MetaSwin**: 의료 영상 분할을 위한 MetaFormer 기반 모델[10]
- **도메인 특화 개선**: 각 분야별 MetaFormer 적용 및 개선 연구 확산

## 앞으로 연구 시 고려할 점

### 1. 아키텍처 설계 원칙
- **범용성과 특수성 균형**: MetaFormer의 일반화 능력과 태스크 특화 요구사항 조화
- **계층적 구조 최적화**: 각 단계별 적절한 token mixer 선택 및 배치

### 2. 효율성 고려사항
- **파라미터-성능 트레이드오프**: 실용적 배포 환경에서의 효율성 중시
- **메모리 및 계산 제약**: Edge computing 환경에서의 활용 가능성 고려

### 3. 평가 방법론
- **공정한 비교**: 암묵적 토큰 믹싱 효과를 고려한 객관적 성능 평가
- **다양한 태스크 검증**: 단순 분류를 넘어선 복합적 비전 태스크에서의 검증 필요

이 연구는 Vision Transformer 분야에 패러다임 전환을 가져왔으며, 복잡한 모듈 설계보다 전반적인 아키텍처 구조의 중요성을 강조하여 더욱 효율적이고 실용적인 모델 개발의 방향을 제시했습니다.

[1] https://ieeexplore.ieee.org/document/9879612/
[2] https://arxiv.org/abs/2111.11418
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/46f84fe9-10fb-491d-aa6f-aa8111f780f9/2111.11418v3.pdf
[4] https://arxiv.org/html/2210.13452v4
[5] https://ieeexplore.ieee.org/document/10662921/
[6] https://ieeexplore.ieee.org/document/10473485/
[7] https://daebaq27.tistory.com/118
[8] https://www.semanticscholar.org/paper/57dbc164be141f365b3b331edad24e79ebdc3860
[9] https://ieeexplore.ieee.org/document/11043206/
[10] https://peerj.com/articles/cs-1762
[11] https://dl.acm.org/doi/10.1145/3569966.3570099
[12] https://ieeexplore.ieee.org/document/10304335/
[13] https://dl.acm.org/doi/10.1145/3627915.3627926
[14] https://www.semanticscholar.org/paper/c307564a2ae003d070eefab276ffa6bcbc3b18f2
[15] https://arxiv.org/pdf/2111.11418.pdf
[16] https://arxiv.org/pdf/2203.02751.pdf
[17] https://arxiv.org/html/2411.18995
[18] https://arxiv.org/pdf/2210.13452.pdf
[19] http://arxiv.org/pdf/2312.00412.pdf
[20] https://arxiv.org/pdf/2203.00131.pdf
[21] https://arxiv.org/pdf/2208.00713.pdf
[22] http://arxiv.org/pdf/2110.13083.pdf
[23] https://arxiv.org/pdf/2307.10802.pdf
[24] https://arxiv.org/pdf/2106.12011.pdf
[25] https://huggingface.co/docs/transformers/en/model_doc/poolformer
[26] https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.pdf
[27] https://mmpretrain.readthedocs.io/en/latest/papers/poolformer.html
[28] https://github.com/sail-sg/poolformer
[29] https://vds.sogang.ac.kr/wp-content/uploads/2023/01/2022-%ED%95%98%EA%B3%84-%EC%84%B8%EB%AF%B8%EB%82%98-%EC%8B%AC%EC%9E%AC%ED%97%8C.pdf
[30] https://www.nature.com/articles/s41598-024-63623-6
[31] https://docsaid.org/en/papers/vision-transformers/poolformer/
[32] https://openreview.net/forum?id=RpKA1wqgk0
[33] https://velog.io/@qtly_u/paper-MetaFormer-Is-Actually-What-You-Need-for-Vision
[34] https://jiankim3293.tistory.com/5
[35] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/poolformer/
[36] https://www.sciencedirect.com/science/article/abs/pii/S0143816625000624
[37] https://omocomo.tistory.com/entry/VisionTransformer-MetaFormer-is-Actually-What-You-Need-for-Vision
[38] https://jungeun-park.tistory.com/3
[39] https://velog.io/@es_seong/Transformer-MetaFormer-Baselines-for-Vision-%EB%A6%AC%EB%B7%B0


# MetaFormer : MetaFormer Is Actually What You Need for Vision | Image classification, Object detection, Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- Vision Transformer(ViT) 계열 모델의 성능은 **어텐션 기반 토큰 믹서(token mixer)**보다는, 어텐션·MLP·풀링 등 구체적 모듈에 무관한 **MetaFormer**라는 **일반화된 아키텍처**(Residual + Norm + TokenMixer + MLP 블록)가 주로 견인한다.  
- 극단적으로 단순한 풀링(pooling) 연산을 토큰 믹서로 사용한 PoolFormer도 ViT·MLP-like 모델과 비슷하거나 더 우수한 성능을 낸다.

**주요 기여**  
1. **MetaFormer 개념 제안**: Transformer를 ‘토큰 믹서’를 지정하지 않은 일반 아키텍처로 추상화.  
2. **PoolFormer 모델**: 풀링 연산만으로 토큰을 섞는 극단적 단순화 예시 제시. ImageNet-1K에서 ViT-B 대비 파라미터 35%↓, MACs 50%↓에도 Top-1 82.1% 달성.  
3. **다양한 태스크 검증**: 분류, 물체 검출‧인스턴스 분할(COCO), 의미 분할(ADE20K)에서 경쟁력 입증.  
4. **아키텍처 중심 연구 제안**: 토큰 믹서 연구를 넘어 MetaFormer 자체 개선 방향 제시.

## 2. 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
- ViT와 MLP-like 모델 성능의 주 원인으로 **어텐션**이나 **공간 MLP** 모듈이 지목되지만, 실제로는 **전체 블록 구조(Residual + Norm + TokenMixer + MLP)**가 핵심인지 불명확.  
- 따라서 극단적으로 단순한 토큰 믹서를 써도 MetaFormer 구조만 지키면 충분히 높은 성능을 낼 수 있는지 검증 필요.

### 2.2 제안 방법  
1. **MetaFormer 블록**  
   - 입력 $$X\in\mathbb{R}^{N\times C}$$에 대해  

  $$
       Y = \mathrm{TokenMixer}(\mathrm{Norm}(X)) + X,\quad
       Z = \mathrm{MLP}(\mathrm{Norm}(Y)) + Y.
     $$

2. **풀링 기반 TokenMixer**  
   - $$K\times K$$ 평균풀링으로만 토큰 혼합:  

  $$
       T'\_{:,i,j}
       = \frac1{K^2}\sum_{p,q=1}^K T_{:,\,i+p-\frac{K+1}2,\,j+q-\frac{K+1}2}
         - T_{:,i,j}.
     $$
   - PyTorch로는 `AvgPool2d(pool_size, stride=1, padding=…) – identity` 구현.

3. **계층적 구조**  
   - CNN·PVT 유사한 4단계: 해상도 $$\frac H4$$, $$\frac H8$$, $$\frac H{16}$$, $$\frac H{32}$$ 토큰.  
   - 블록 수 비율 $$[L/6,\,L/6,\,L/2,\,L/6]$$, MLP 확장비율 4, Modified LayerNorm(토큰+채널 정규화).

### 2.3 모델 구조  
| 모델        | 파라미터(M) | MACs(G) | Top-1 Accuracy(%) |
|-------------|-------------|---------|-------------------|
| PoolFormer-S24 | 21.4        | 3.4     | 80.3              |
| DeiT-S    | 22.0        | 4.6     | 79.8              |
| ResMLP-S24| 30.0        | 6.0     | 79.4              |
| PoolFormer-M36| 56.1        | 8.8     | 82.1              |
| DeiT-B    | 86.0        | 17.5    | 81.8              |
| ResMLP-B24| 116.0       | 23.0    | 81.0              |

### 2.4 성능 향상  
- **경량화 대비 우수**: PoolFormer-S24는 DeiT-S 대비 MACs 26%↓, 파라미터 유사하면서 0.5%↑.  
- **범용성**: COCO 물체 검출·분할, ADE20K 의미 분할에서도 ResNet 대비 AP·mIoU 3-4점씩 상회.  
- **아블레이션**:  
  - TokenMixer 없이 Identity만 써도 74.3%.  
  - 풀링 → 랜덤 매트릭스, DW-Conv, 어텐션 혼합 등 다양한 믹서 대체 실험에서도 모두 합리적 성능 유지.  

### 2.5 한계  
- **풀링 단독**은 전역 정보 학습 능력 한계(특히 크고 복잡한 패턴).  
- **자연어·다른 도메인 검증 부족**: NLP나 비디오 등에서 MetaFormer 범용성 추가 검증 필요.  
- **하이브리드 설계 최적화**: 풀링+어텐션/MLP 조합에 대한 구조 탐색 미완.

## 3. 모델 일반화 성능 향상 관점  
- **Residual 연결과 Norm** 중심의 블록이 안정적 학습과 기울기 전달 보장.  
- **TokenMixer 다양화**(풀링, 어텐션, MLP)에도 높은 성능 유지→**오버피팅 방지** 및 **도메인 적응력** 기대.  
- **하이브리드 Stage 설계**: 저단계 풀링→고단계 어텐션 조합으로 전역·지역 표상 균형 확보 시 일반화 개선 가능.

## 4. 논문의 향후 영향 및 고려사항  
- **아키텍처 우선 접근**: 토큰 믹서 설계보다 **MetaFormer 블록 구조**(정규화·잔차·MLP) 최적화 연구 가치 강조.  
- **효율적 하이브리드 모델** 개발: 풀링·어텐션·MLP 조합으로 성능·효율 균형 달성.  
- **도메인·태스크 확장**: NLP, 비디오, 3D 등 다양한 입력에 MetaFormer 적용성·일반화 검증 필요.  
- **학습·정규화 기법**: Modified LayerNorm, LayerScale, Stochastic Depth 등 안정화 기법이 핵심—추가 개선 여지.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/36683c49-3e5a-4ed8-ab6f-1ba7d0ea51d1/2111.11418v3.pdf
