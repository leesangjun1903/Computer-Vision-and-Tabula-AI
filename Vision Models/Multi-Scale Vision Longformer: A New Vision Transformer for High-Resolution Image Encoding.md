# Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding | Image classification, Object detection, Semantic segmentation

## 1. 핵심 주장 및 주요 기여
Multi-Scale Vision Longformer(이하 ViL)는  
- **고해상도 이미지**를 효율적으로 처리하기 위해  
- **다중 해상도 구조**(Multi-Scale)와  
- **2D 로컬+글로벌 메모리 어텐션**(Vision Longformer)  
를 결합하여  
기존의 ViT 대비 메모리·연산량을 선형 수준으로 줄이면서도  
이미지 분류·객체 검출·세분화 성능을 모두 크게 향상시킨다.

주요 기여  
1. **다중 해상도 스택 구조**: 입력 해상도를 점진적으로 줄이면서 채널 수를 늘려 CNN과 유사한 피처 피라미드를 형성  
2. **Vision Longformer 어텐션**: 윈도우 기반 로컬 어텐션과 소수(global) 토큰의 전역 어텐션을 결합해 토큰 수에 선형 복잡도 확보  
3. **종합 실험**: ImageNet 분류에서 +1.8%p, COCO 검출·세분화에서 기존 ResNet/PVT 대비 3–6 AP 차이의 성능 향상  

***

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제
- ViT는 패치 수에 따라 어텐션 연산이 $$O(N^2)$$이며, 이는 2D 영상에서는 해상도에 대해 **4차 복잡도**로 증가  
- 따라서 고해상도 이미지에 적용 시 **메모리·연산량 폭발**로 실용성 저하  

### 2.2 제안 모델 구조
전체 구조는 **4단계 스택**으로 구성되며, 각 단계 $$s$$는 다음을 포함  
- 패치 임베딩: 입력을 $$p_s\times p_s$$ 크기 패치로 분할  
- E-ViT 모듈: $$n_s$$개의 MSA 블록, 헤드 수 $$h_s$$, 은닉 차원 $$d_s$$, 글로벌 토큰 $$n_{g,s}$$  
- 해상도 ↓, 채널 ↑ 스케줄: $$H\times W\to H/2\times W/2$$, $$d\to2d$$

```math
\begin{aligned}
z^0 &= [x^g_1;\ldots;x^g_{n_g}; \mathrm{LN}(x^p_1E);\ldots] + E_{\text{pos}}\\
z^{\prime k} &= \mathrm{MSA}_a(\mathrm{LN}(z^{k-1})) + z^{k-1},\quad k=1,\dots,n\\
z^k &= \mathrm{MLP}(\mathrm{LN}(z^{\prime k})) + z^{\prime k}
\end{aligned}
```
- $$a=\text{ViL}$$: 각 토큰은 윈도우 반경 $$w$$ 내 이웃과 글로벌 토큰만 어텐션  
- 복잡도: $$O(n_l w^2 + n_g(n_l+n_g))$$ → $$n$$-선형

### 2.3 성능 향상
- **ImageNet-1K 분류**: 동일 파라미터·FLOPs 대비 +1–2%p Top-1↑  
- **COCO 검출**(RetinaNet): Tiny 모델 기준 AP 31.8→40.8 (+9.0)  
- **COCO 세분화**(Mask R-CNN): Small 모델 AP$$_m$$ 35.1→41.0 (+5.9)  

### 2.4 한계
- 윈도우 크기, 글로벌 토큰 수 등 하이퍼파라미터 민감도  
- 커스텀 CUDA 커널 혹은 복잡한 슬라이딩 청크 구현 필요  
- 대규모 사전학습 전 데이터 의존성

***

## 3. 일반화 성능 향상 관점
- **피라미드 구조**로 다양한 규모 객체에 대응 가능  
- **글로벌 토큰**이 전역 문맥 캡처, 로컬 어텐션이 세밀한 지역 정보 보전  
- COCO에서 작은 객체(AP$$_S$$) 성능도 대폭 개선  
- **윈도우 크기**(15)와 **글로벌 토큰 수**(1) 조합이 전이 학습 시 최적화됨  

이로써 다양한 해상도·도메인의 다운스트림 과제에서 **사전학습된 ViL**이 높은 일반화력을 보임.

***

## 4. 향후 연구 영향 및 고려 사항
- ViT 기반 모델의 **고해상도 확장성** 해결책 제시  
- 차후 연구에서 윈도우·메모리 하이퍼파라미터 자동 최적화,  
  어텐션 구현 효율화,  
  텍스트-이미지 멀티모달 확장에 활용 가능  
- **제한점**으로는 복잡한 구현·메모리 트레이드오프가 있으므로,  
  경량화 및 효율적 하드웨어 매핑 연구 병행 필요  

ViL의 개념은 **다중 해상도 + 효율적 어텐션** 패러다임을 제시하며,  
고해상도 비전 과제의 새로운 표준이 될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/11562379-bdea-4ac4-8c7b-06aa12c325b6/2103.15358v2.pdf
