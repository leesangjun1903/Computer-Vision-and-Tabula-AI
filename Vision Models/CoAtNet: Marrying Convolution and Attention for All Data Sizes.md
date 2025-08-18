# CoAtNet: Marrying Convolution and Attention for All Data Sizes | Image classification

## 1. 핵심 주장 및 주요 기여 (간결 요약)
CoAtNet은 **Convolutional Networks**의 강력한 **일반화(generalization)** 성능과 **Transformer**의 **고용량(model capacity)** 특성을 결합한 하이브리드 비전 모델이다.  
주요 기여:
- **Depthwise Convolution**과 **Self-Attention**을 하나의 연산 블록으로 통합(절대적 위치 정보와 적응형 가중치 학습)  
- **단계별(vertical) 레이아웃** 설계를 통해 저해상도에서는 Convolution, 고해상도·후반부에서는 Attention 블록 활용  
- 다양한 데이터 규모(ImageNet-1K, ImageNet-21K, JFT-300M/3B)에서 SOTA 성능 달성  

## 2. 문제 정의 및 제안 방법
### 2.1. 해결하고자 하는 문제  
- Transformer는 대규모 데이터에서 뛰어난 용량을 보이지만, 소규모 데이터에서 일반화 능력이 부족  
- ConvNet은 소규모 데이터에서 뛰어난 일반화 성능을 보이나, 대규모 데이터에서 용량 한계  

### 2.2. 제안 방법
1) **Convolution + Relative Self-Attention 통합 블록**  
   - Depthwise Convolution:  

$$
       y_i = \sum_{j \in L(i)} w_{i-j} \odot x_j
     $$
   
   - Self-Attention:

$$
       y_i = \sum_{j \in G} \frac{\exp(x_i^\top x_j)}{\sum_{k\in G}\exp(x_i^\top x_k)}\,x_j
     $$
   
   - 두 연산을 **pre-normalization relative attention**으로 결합:

$$
       y_i = \sum_{j\in G}
         \frac{\exp\bigl(x_i^\top x_j + w_{i-j}\bigr)}
              {\sum_{k\in G}\exp\bigl(x_i^\top x_k + w_{i-k}\bigr)}\,x_j
     $$
  
   - 이를 통해 **translation equivariance**, **input-adaptive weighting**, **global receptive field**를 동시에 획득  

2) **Vertical Multi-Stage Layout**  
   - **S0 (Stem)**: 2-layer Conv  
   - **S1**: MBConv (depthwise conv + SE)  
   - **S2**: MBConv  
   - **S3, S4**: Relative-Attention Transformer  
   - Convolution 단계가 앞에 위치할수록 일반화 성능↑, Attention 단계가 뒤로 갈수록 용량↑  
   - 가장 균형 잡힌 **C–C–T–T**(MBConv×2 → MBConv×2 → TFMRel×2 → TFMRel×2) 구조 채택  

3) **Pre-activation & Normalization**  
   - 모든 블록에 **pre-activation** 구조 적용:  

$$
       x \leftarrow x + \mathrm{Module}\bigl(\mathrm{Norm}(x)\bigr)
     $$
  
   - MBConv: **BatchNorm + GELU**, Transformer: **LayerNorm + GELU**  

### 2.3. 성능 향상
- **ImageNet-1K만 학습**: ConvNet·ViT·합성모델 대비 최고 동급 성능 (예: CoAtNet-2 84.1% vs. NFNet-F3 85.7%)  
- **ImageNet-21K→1K 전이**: CoAtNet-4 88.56% 달성, ViT-Huge(JFT-300M pre-train)와 동일 성능이지만 데이터·연산량 획기적 절감  
- **JFT-300M/3B pre-train**: CoAtNet-7 90.88%로 ViT-G/14(90.45%) 제치며 SOTA 경신  

| 데이터 규모        | 모델          | Top-1 Acc. |
|-------------------|--------------|-----------|
| ImageNet-1K only  | CoAtNet-2    | 84.1%     |
| 21K→1K finetune   | CoAtNet-4    | 88.56%    |
| JFT-300M pre-train| CoAtNet-5    | 89.77%    |
| JFT-3B pre-train  | CoAtNet-7    | 90.88%    |

### 2.4. 한계
- **계산 복잡도**: Attention 블록은 여전히 전역 연산으로 고비용  
- **구조 탐색(search space)**: C–C–T–T 구조는 최적이지만, 다른 태스크(객체 탐지·분할) 적용 시 재검증 필요  
- **추론 메모리**: 높은 해상도 지원 시 상대 편향 행렬 크기 증가  

## 3. 일반화 성능 향상 관점
- **Convolution의 inductive bias**(translation equivariance)가 소규모 데이터에서 overfitting 완화  
- **Relative Attention**: 정적 편향 $$w_{i-j}$$ 추가로 입력 독립적 지역 정보 보존 → 일반화 갭 감소  
- 실험: ImageNet-1K만 학습 시, Relative-Attn 적용 모델이 84.1% vs. 표준 Attn 83.8%[ablation]  
- **Multi-stage 배치**: 초기 단계 Convolution 집중 → 저수준 피처 안정적 추출, 후반부 Global Attention 활용 → 심층 표현력  

## 4. 향후 연구에의 영향 및 고려 사항
- **확장성**: CoAtNet의 하이브리드 블록은 객체 탐지, 분할, 영상 처리에 응용 가능  
- **구조 자동 탐색(NAS)**: C–C–T–T 외 다양한 스테이지 배치 최적화 연구 필요  
- **효율적 Attention**: 지역화(local window) 또는 선형 복잡도 어텐션 도입으로 비용 절감  
- **도메인별 튜닝**: 의료 영상, 위성 사진 등 데이터 특성에 맞춘 inductive bias 조정 고려  

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f1550a1b-20a1-4e10-abe6-784567e377ce/2106.04803v2.pdf
