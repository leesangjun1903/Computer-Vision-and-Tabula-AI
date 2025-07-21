# MobileViTv2 : Separable Self-Attention for Mobile Vision Transformers | Image classification

**핵심 주장 및 기여**  
이 논문은 모바일 비전 트랜스포머의 효율성을 근본적으로 개선하기 위해, 토큰 간 상호작용 연산을 O(k²)에서 O(k)로 줄이고 하드웨어 최적화가 용이한 요소별(element-wise) 연산만을 활용하는 **Separable Self-Attention** 모듈을 제안한다. 이를 통해 MobileViT 아키텍처를 MobileViTv2로 개량하여, ImageNet–1k 분류에서 1% 이상의 정확도 향상과 3.2× 추론 속도 개선을 동시에 달성하였다.

## 1. 문제 정의  
- **기존 MHA의 한계**: 토큰 수 k에 대해 O(k²) 연산 복잡도를 가지며, 배치 행렬 곱(batch-wise mm), softmax 등 고비용 연산으로 모바일·엣지 디바이스에서 병목 발생.  
- **목표**: 연산 복잡도를 O(k)로 줄이고, 저전력·저메모리 디바이스에서 실시간 추론 가능토록 설계.

## 2. 제안 기법: Separable Self-Attention  

1) **Context Score 계산**  
   입력 토큰 행렬 $$x \in \mathbb{R}^{k\times d}$$를 스칼라 맵으로 투영:  

$$
   cs = \mathrm{softmax}(x W_I),\quad W_I \in \mathbb{R}^{d\times 1}
   $$  

   — $$O(kd)$$  

2) **Context Vector 생성**  
   Key-branch 투영 $$x_K = x W_K,\, W_K\in\mathbb{R}^{d\times d}$$ 후,  

$$
   cv = \sum_{i=1}^k cs_i \, x_{K,i},\quad cv\in\mathbb{R}^d
   $$  
  
   — 가중합으로 글로벌 정보 집약  

3) **Context 확산 및 출력**  
   Value-branch 투영 $$x_V = \mathrm{ReLU}(x W_V)$$ 후,  

$$
   y = \bigl(cv \,\ast\, x_V\bigr)\,W_O,\quad W_V,W_O\in\mathbb{R}^{d\times d}
   $$  
   
   — 브로드캐스트 곱으로 모든 토큰에 글로벌 정보 병합  

└— 총 연산 복잡도 $$O(kd + d^2)\approx O(k)$$로 기존 MHA의 $$O(k^2d)$$ 대비 대폭 절감.

## 3. MobileViTv2 아키텍처  
- MobileViTv1의 각 트랜스포머 블록 내 MHA를 본 기법으로 교체.  
- 블록 반복 수 및 채널 수를 폭(α) 변수로 조절해 다양한 모델 크기(0.5×–2.0×) 제공.  
- 입력 해상도 및 파라미터 수, FLOPs 대비 성능이 대폭 향상.

## 4. 성능 향상  
- **ImageNet–1k 분류**:  
  - MobileViTv1 대비 +1% Top-1 정확도, 3.2× 모바일 추론 속도 개선  
  - MobileViTv2-1.0 (4.9 M) → 78.1% Top-1, 3.4 ms (iPhone12)  
- **MS-COCO 검출**:  
  - +5.4% mAP, 3× 빠른 추론  
- **PASCAL VOC 분할**:  
  - +6.7% mIoU, 3.1× 속도 향상  

## 5. 한계 및 일반화 가능성  
- **한계**:  
  - 단일 잠재 토큰 사용 시 복잡한 장면에서 세밀한 상호작용 정보 손실 가능성  
  - 브로드캐스트 연산이 최적화되지 않은 일부 하드웨어에서 병목 소지  
- **일반화 가능성**:  
  - 선형 시간 복잡도로 패치 기반·픽셀 기반 트랜스포머 모두에 적용 가능.  
  - 여러 태스크(분류·검출·분할)에서 일관된 성능 향상 입증.  
  - 복합적 구조에도 무리 없이 통합 가능, 대규모 ViT 모델의 스케일업에도 잠재적 이득.

## 6. 미래 연구 방향 및 고려사항  
- **다중 잠재 토큰**: 개수 확장 실험 및 동적 토큰 선택으로 표현력 강화  
- **하드웨어 최적화**: 브로드캐스트 연산용 커널 개발 및 모바일 가속기 대응  
- **조합 기법**: 로컬 윈도우 어텐션, 저랭크 근사 등과 통합해 성능–효율 극대화  
- **비전·언어 융합**: 멀티모달 트랜스포머에도 동일 원리 적용 가능성 탐색  

이 연구는 **트랜스포머의 핵심 병목**인 MHA를 선형 시간, 요소별 연산으로 대체함으로써, **모바일·엣지 환경**에서도 **고성능 비전 트랜스포머**를 구동할 수 있음을 입증하였다. 차세대 경량·고효율 비전 모델 설계의 **새로운 패러다임**을 제시하며, 다양한 비전·멀티모달 응용으로 발전할 여지를 남긴다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d56a10af-e69b-4ed4-a613-4ebc31ceae21/2206.02680v1.pdf
