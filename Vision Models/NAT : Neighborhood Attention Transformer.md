# NAT : Neighborhood Attention Transformer | Image classification, Object detection, Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
Neighborhood Attention Transformer(NAT)는 **전통적인 Self-Attention의 고비용·고메모리 문제**와 **Swin Transformer의 불완전한 국소성**을 동시에 해결하기 위해, 각 토큰이 오직 인접 이웃(neighborhood)만을 대상으로 슬라이딩 윈도우 형태로 어텐션을 수행하는 **Neighborhood Attention(NA)** 메커니즘을 도입한다[1].  
- NA는 이웃 크기 $$k$$에 따라 선형 복잡도를 유지하면서도, $$k$$를 늘리면 원래의 Self-Attention에 수렴하도록 설계되었다.  
- 이를 효율적으로 구현한 **NATTEN** 라이브러리는 CUDA/C++ 커널 및 타일드 연산을 활용하여 Swin의 윈도우 Self-Attention 대비 최대 40% 빠른 처리 속도와 25% 낮은 메모리 사용을 달성한다[1].  
- NA를 기반으로 한 **계층적 Vision Transformer 아키텍처(NAT)**는 ImageNet 분류, COCO 객체 검출, ADE20K 분할 등 주요 벤치마크에서 Swin 및 ConvNeXt 대비 최대 1.9% Top-1 정확도, 1.0% mAP, 2.6% mIoU 상승을 보인다[1].

## 2. 해결하고자 하는 문제  
1. **Quadratic 복잡도**: 전통적 Self-Attention은 토큰 수 $$n$$에 대해 $$\mathcal{O}(n^2)$$의 계산·메모리 복잡도를 갖는다.  
2. **국소성 및 번역 불변성**: Convolution의 지역적 inductive bias와 translational equivariance는 유지하면서, Swin의 블록 분할 방식이 깨뜨리는 전역 정보 융합을 보완해야 한다.  
3. **실용적 구현**: SASA 같은 슬라이딩 윈도우 어텐션은 이론상 선형이지만, 효율적 구현 부재로 실제 GPU 환경에서 느리다.

## 3. 제안 방법  
### 3.1 Neighborhood Attention(NA) 정의  
입력 $$X\in\mathbb{R}^{hw\times d}$$에 대해, 토큰 $$i$$의 쿼리 $$Q_i$$가 $$k$$개의 최근접 이웃 키 $$K_{\rho_j(i)}$$에만 닷 프로덕트를 수행:  

$$
A_i = \bigl[\,Q_iK^T_{\rho_1(i)} + B(i,\rho_1(i)),\,\dots,\,Q_iK^T_{\rho_k(i)} + B(i,\rho_k(i))\bigr]\in\mathbb{R}^{k},
$$  

$$
V_i = [\,V_{\rho_1(i)},\dots,V_{\rho_k(i)}]^\top\in\mathbb{R}^{k\times d},
$$  

$$
\mathrm{NA}_k(i) = \mathrm{softmax}\!\bigl(A_i/\sqrt{d}\bigr)\,V_i.
$$  

이로써 복잡도는 $$\mathcal{O}(hwdk^2)$$로 선형을 유지하며, $$k\to hw$$ 시 원래 Self-Attention에 수렴한다[1].

### 3.2 NATTEN: 효율적 구현  
- **타일드 연산**: 쿼리 타일과 그 이웃 키 타일을 공유 메모리에 로드하여 연산, 글로벌 메모리 접근 최소화  
- **FP16 벡터화**: half2 연산으로 대역폭 활용 최적화  
- 결과: Swin WSA+SWSA 대비 최대 40%↑ 처리량, 25%↓ 메모리[1].

## 4. 모델 구조  
- **계층적 디자인**: 입력 4× 다운샘플러(2×2 스트라이드 중첩 Conv) → 4단계 NAT 블록(NA + MLP + LayerNorm + skip) → 특성 피라미드  
- **변형**  
  - Mini/Tiny/Small/Base: 블록 수·헤드 수·채널 수 조합  
  - 중첩 Conv 토크나이저로 inductive bias 강화[1].

## 5. 성능 향상  
| 과제          | Swin 대비 향상               | 주요 지표                                   |
|--------------|-----------------------------|---------------------------------------------|
| ImageNet-1K  | +1.9% Top-1                 | NAT-Tiny 83.2% vs Swin-Tiny 81.3%[1]         |
| MS-COCO      | +1.0% box mAP               | NAT-Tiny 51.4 vs Swin-Tiny 50.4[1]           |
| ADE20K       | +2.6% mIoU                  | NAT-Tiny 48.4 vs Swin-Tiny 45.8[1]           |

## 6. 한계 및 일반화 성능  
- **모서리 이웃 중복**: corner token이 반복된 이웃을 가져 translational equivariance 일부 완화  
- **커널 크기 민감도**: 7×7에서 최적 성능, 과도한 확장 시 속도 저하  
- **데이터 효율성**: 중소규모 학습에서 inductive bias 덕분에 일반화 성능 개선 여지[1]  
- **추가적 제약**: 높은 해상도·대형 모델에서의 메모리·속도 병목 가능성.

## 7. 향후 영향 및 고려사항  
- **슬라이딩 윈도우 어텐션 연구**: NA가 효율성을 입증함에 따라, 다양한 국소 어텐션 메커니즘 재조명  
- **하드웨어 최적화**: CUTLASS 등 라이브러리 기반 implicit GEMM 구현으로 추가 성능 개선  
- **하이브리드 아키텍처**: convolution과 어텐션의 inductive bias 결합 방식 다변화  
- **범용성 평가**: 의료 영상·시계열 등 다양한 비전·비전 외 태스크에 대한 일반화 실험 필요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7de0cdc1-7e20-4ca1-8b03-4622d78f5d07/2204.07143v5.pdf
