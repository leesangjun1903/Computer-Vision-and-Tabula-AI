# Transformers in Vision: A Survey
# Abstract
자연어 처리에 대한 트랜스포머 모델의 놀라운 결과는 비전 커뮤니티에서 컴퓨터 비전 문제에 대한 트랜스포머의 적용을 연구하는 데 흥미를 불러일으켰습니다.  
트랜스포머는 중요한 이점 중 하나로 입력 시퀀스 요소 간의 긴 의존성을 모델링하고 반복 네트워크(예: Long short-term memory(LSTM)와 비교하여 시퀀스의 병렬 처리를 지원합니다.  
컨볼루션 네트워크와 달리 트랜스포머는 설계에 최소한의 귀납적 편향이 필요하며 자연스럽게 set-function으로 적합합니다.  
또한 트랜스포머의 간단한 설계를 통해 유사한 처리 블록을 사용하여 여러 양식(예: 이미지, 동영상, 텍스트 및 음성)을 처리할 수 있으며, 대용량 네트워크와 대규모 데이터 세트에 대한 뛰어난 확장성을 보여줍니다.  
이러한 강점은 트랜스포머 네트워크를 사용하는 여러 비전 작업에 대한 흥미로운 진전으로 이어졌습니다.  
이 설문조사는 컴퓨터 비전 분야에서 트랜스포머 모델에 대한 포괄적인 개요를 제공하는 것을 목표로 합니다.  
우리는 자기 주의, 대규모 사전 훈련 및 양방향 특징 인코딩과 같은 트랜스포머의 성공에 대한 근본적인 개념에 대한 소개부터 시작합니다.  
그런 다음 인기 있는 recognition tasks(예: 이미지 분류, 객체 감지, 동작 인식 및 분할), generative modeling, multi-modal tasks(예: 시각적 질문 응답, 시각적 추론 및 시각적 접지), video processing(예: activity recognition, visual grounding), low-level vision(예: image super-resolution, image enhancement, and colorization) 및 3D 분석(예: point cloud classification 및 분할)을 포함한 비전 분야의 트랜스포머의 광범위한 응용 분야를 다룹니다.  
우리는 아키텍처 설계와 실험적 가치 측면에서 인기 있는 기법의 각각의 장점과 한계를 비교합니다.  
마지막으로, 개방형 연구 방향과 향후 가능한 작업에 대한 분석을 제공합니다.  
이러한 노력이 컴퓨터 비전 분야에서 트랜스포머 모델의 적용에 대한 현재의 과제를 해결하기 위한 커뮤니티의 관심을 더욱 불러일으킬 수 있기를 바랍니다.

# Introduction
![](https://hoya012.github.io/assets/img/Visual_Transformer/1.PNG)

위의 그림을 보시면 알 수 있듯이 매년 Top-tier 학회, arxiv에 Transformer 관련 연구들이 빠른 속도로 늘어나고 있고 작년(2020년)에는 거의 전년 대비 2배 이상의 논문이 제출이 되었습니다.  
바야흐로 Transformer 시대가 열린 셈이죠.  
근데 주목할만한 점은 Transformer가 자연어 처리 뿐만 아니라 강화 학습, 음성 인식, 컴퓨터 비전 등 다른 task에도 적용하기 위한 연구들이 하나 둘 시작되고 있다는 점입니다.

논문에서는 컴퓨터 비전에 Transformer을 적용시킨 연구들을 크게 10가지 task로 나눠서 정리를 해두었습니다.

1. Image Recognition (Classification)
2. Object Detection
3. Segmentation
4. Image Generation
5. Low-level Vision
6. Multi-modal Tasks
7. Video Understanding
8. Low-shot Learning
9. Clustering
10. 3D Analysis

![](https://hoya012.github.io/assets/img/Visual_Transformer/3.png)

# Foundations
![](https://hoya012.github.io/assets/img/Visual_Transformer/2.PNG)

Transformers의 성공 요소는 크게 Self-Supervision 과 Self-Attention 으로 나눌 수 있습니다.  
세상엔 굉장히 다양한 데이터가 존재하지만, Supervised Learning으로 학습을 시키기 위해선 일일이 annotation을 만들어줘야 하는데, 대신 무수히 많은 unlabeled 데이터들을 가지고 모델을 학습 시키는 Self-Supervised Learning을 통해 모델을 학습 시킬 수 있습니다.  
자연어 처리에서도 Self-Supervised Learning을 통해 주어진 막대한 데이터 셋에서 generalizable representations을 배울 수 있게 되며, 이렇게 pretraining시킨 모델을 downstream task에 fine-tuning 시키면 우수한 성능을 거둘 수 있게 됩니다.  

또 다른 성공 요소인 Self-Attention은 말 그대로 스스로 attention을 계산하는 것을 의미하며 CNN, RNN과 같이 inductive bias가 많이 들어가 있는 모델들과는 다르게 최소한의 inductive bias를 가정합니다.  
Self-Attention Layer를 통해 주어진 sequence에서 각 token set elements(ex, words in language or patches in an image)간의 관계를 학습하면서 광범위한 context를 고려할 수 있게 됩니다.  


# Reference
- https://hoya012.github.io/blog/Vision-Transformer-1/

# "Transformers in Vision: A Survey" 

## 1. 핵심 주장과 주요 기여

**"Transformers in Vision: A Survey"**는 자연어처리에서 혁신적인 성과를 보인 Transformer 모델의 컴퓨터 비전 분야 적용에 대한 포괄적인 조사 논문입니다[1]. 

### 핵심 주장
- **Transformer의 범용성**: Self-attention 메커니즘은 CNN의 한계를 극복하고 다양한 비전 태스크에서 우수한 성능을 달성할 수 있음
- **통합 아키텍처의 가능성**: 이미지, 비디오, 텍스트, 음성 등 다양한 모달리티를 처리할 수 있는 통합된 처리 블록의 실현 가능성
- **확장성의 중요성**: 대규모 용량과 데이터셋에 대한 뛰어난 확장성을 통해 기존 CNN 모델을 능가하는 성능 달성

### 주요 기여
1. **체계적인 분류 체계**: Vision Transformer를 단일 헤드/다중 헤드 자기주의, 균등/다중 스케일, 하이브리드 설계로 체계적 분류[1]
2. **포괄적인 응용 분야 커버**: 이미지 분류, 객체 탐지, 분할, 생성 모델링, 멀티모달 태스크, 비디오 처리, 저수준 비전, 3D 분석 등 광범위한 적용 사례 분석
3. **장단점 비교 분석**: 아키텍처 설계와 실험적 가치 측면에서 각 기법의 장단점 체계적 비교
4. **미래 연구 방향 제시**: 현재 한계점과 향후 연구 과제에 대한 심층적 분석

## 2. 해결하고자 하는 문제와 제안 방법

### 해결하고자 하는 문제
1. **NLP에서 비전으로의 지식 전이**: 자연어처리에서 성공한 Transformer 모델을 컴퓨터 비전 태스크에 효과적으로 적용하는 방법
2. **장거리 의존성 모델링**: 기존 CNN의 제한된 수용 영역(receptive field) 문제 해결
3. **계산 효율성 vs 성능**: 자기주의 메커니즘의 이차 복잡도 문제와 실용적 적용 간의 균형
4. **데이터 효율성**: CNN에 비해 부족한 귀납적 편향으로 인한 대용량 데이터 요구 문제

# 2 FOUNDATIONS

## 2.1 Self-Attention in Transformers  
Self‐attention은 길이 $$n$$인 입력 시퀀스 $$X\in\mathbb{R}^{n\times d}$$의 모든 위치 간 관계를 학습하여  
출력 $$Z\in\mathbb{R}^{n\times d_v}$$를 생성한다. 세 개의 학습 가능한 투영 행렬을 통해 쿼리, 키, 값을 얻는다:

$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V,
$$  

여기서 $$W^Q,W^K\in\mathbb{R}^{d\times d_q}$$, $$W^V\in\mathbb{R}^{d\times d_v}$$, $$d_q = d_k$$.  
어텐션 출력은  

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_q}}\Bigr)\,V.
$$  

Multi‐head attention은 $$h$$개의 독립 헤드를 사용하고,  

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O,\quad
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

### Masked &amp; Bidirectional Variants  
- **Masked self‐attention**: 상삼각행렬 마스크 $$M$$로 미래 위치 차단

$$
\mathrm{softmax}\bigl((QK^\top/\sqrt{d_q})\circ M\bigr)V.
$$  

- **Bidirectional encoding (BERT)**: 입력 토큰의 15% 무작위 마스킹 후 복원 예측을 학습해 양방향 문맥 인코딩.

## 2.2 (Self‐)Supervised Pre‐training  
Transformer는 대규모 데이터 사전학습이 필수.  
- **Supervised**: ImageNet 등 라벨 데이터  
- **Self‐supervised**:  
  1. Generative (masked image modeling)  
  2. Context‐based (jigsaw, 회전 예측)  
  3. Contrastive (Augmentation 간 일치 학습)

## 2.3 Original Transformer Architecture  
인코더‒디코더 스택($$N$$개 블록). 각 블록은  
1. 다중 헤드 Self‐Attention  
2. Position‐wise Feed‐Forward  
3. Residual + LayerNorm  
시네소이드 위치 인코딩을 입력에 더함.

# 3 SELF‐ATTENTION &amp; TRANSFORMERS IN VISION

## 3.1 Single‐Head Self‐Attention in CNNs  
- **Non‐local block**: 특징맵 모든 위치 간 가중합

  $$\;y_i=\sum_j\frac{\exp(f(x_i,x_j))}{\sum_k\exp(f(x_i,x_k))}g(x_j)$$.
  
- **Criss‐cross attention**: 행‒열 경로만 attend, 복잡도 $$O(2\sqrt N)$$로 감소.  
- **Local relation networks**: 국소 윈도우 내 어댑티브 필터 학습.

## 3.2 Multi‐Head Self‐Attention (Vision Transformers)

### 3.2.1 Uniform‐Scale ViTs  
- **ViT**: $$P\times P$$ 패치를 벡터화해 위치 인코딩 부착 후 Transformer 인코더 적용.  
- **DeiT**: CNN 교사 모델 지식 증류, distillation token 도입.  
- **T2T‐ViT**: 토큰 재귀 결합으로 구조 정보 강화.

### 3.2.2 Multi‐Scale ViTs  
계층적 디자인: 패치 수 감소, 채널 폭 증가  
- **PVT**: 공간 축소 어텐션, 피라미드 토큰  
- **Swin**: 윈도우 기반 어텐션 + 시프팅  
- **CrossFormer**: 크로스 스케일 임베딩 + 다중 거리 어텐션

### 3.2.3 Hybrid ViTs with Convolutions  
- **CvT**: 합성곱 기반 패치 임베딩  
- **CoAtNet**: Depthwise convolution + self‐attention  
- **LeViT**: CNN 스템 후 계층적 Transformer

## 3.3 Detection &amp; Segmentation

### Object Detection  
- **DETR**: CNN 백본→Transformer → 박스 세트 예측, Hungarian 매칭 손실  
- **Deformable DETR**: 드리블러블 어텐션으로 비용↓, 수렴↑

### Segmentation  
- **Axial‐attention**: 2D→2×1D 어텐션  
- **SegFormer**: 피라미드 ViT + MLP 디코더

## 3.4 Generative Vision Transformers  
- **Image Transformer**: 오토회귀 픽셀 생성  
- **iGPT**: GPT‐style 플래튼 토큰으로 무조건 생성

## 3.5 Low‐Level Vision  
- **IPT**: 다작업 복수 헤드‒테일 + Transformer 공용 본체, 1천만 이미지 사전학습  
- **TTSR**: 참조 이미지 질감 전송 Transformer

## 3.6 Multi‐Modal &amp; Video  
- **ViLBERT/LXMERT**: 두 스트림 BERT + co‐attention  
- **UNITER/OSCAR**: 단일 스트림, 객체 태그로 정렬  
- **VideoBERT/PEMT**: 음성+비디오 순차 Transformer  
- **TimeSFormer/ViViT**: 순차적 시공간 어텐션

# 4 OPEN CHALLENGES &amp; FUTURE DIRECTIONS

1. **계산 복잡도**: $$O(n^2)$$ self‐attention → 효율화 연구(로컬 윈도우, 희소/저랭크 어텐션)  
2. **대규모 데이터 필요**: ViT→수억장, 데이터 효율적 학습(DeiT, T2T, SAM 등)  
3. **비전 특화 설계**: 시공간 구조 반영 self‐attention, 벡터 어텐션  
4. **NAS for ViT**: 구조 탐색 자동화(AutoFormer, BossNAS)  
5. **해석 가능성**: 어텐션 롤아웃·분석, relevancy 전파  
6. **경량·하드웨어 최적화**: FPGA, HAT  
7. **모달리티 불문 통합**: Perceiver IO

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/881412c8-b7c0-49cb-ba4c-43ca61259193/2101.01169v5.pdf

## 4. 성능 향상 및 한계

### 성능 향상
**정량적 성과**:
- ViT-L: ImageNet에서 88.55% top-1 정확도 (JFT-300M 사전훈련 시)[1]
- 사전훈련 vs 비사전훈련: 13% 절대적 성능 향상
- CLIP: ImageNet에서 75% 제로샷 분류 정확도[1]

**질적 개선**:
- 강건성 향상: 이미지의 80% 가려짐에도 60% 정확도 유지[1]
- 텍스처 변화에 대한 강한 내성
- 적대적 공격에 대한 향상된 강건성

### 주요 한계
**1. 높은 계산 비용**
- 자기주의의 이차 복잡도 O(n²)
- GPT-3 훈련 비용: 약 460만 달러[1]
- 고해상도 이미지 처리 시 메모리 제약

**2. 대용량 데이터 요구**
- ViT는 수억 장의 이미지 데이터 필요
- CNN 대비 부족한 귀납적 편향
- ImageNet만으로는 경쟁력 있는 성능 달성 어려움

**3. 해석가능성 문제**
- 레이어 간 어텐션 맵의 복잡한 상호작용
- 의사결정 과정의 시각화 어려움

## 5. 일반화 성능 향상 가능성

### 강력한 일반화 특성

**1. 강건성 특성**
- **텍스처 변화 저항성**: CNN 대비 텍스처 변화에 강한 내성 보여[1]
- **가림 현상 대응**: 80% 가려진 이미지에서도 60% 정확도 유지
- **적대적 공격 내성**: 동적 어텐션으로 인한 향상된 강건성

**2. 전이학습 우수성**
- **크로스 도메인 전이**: 다양한 벤치마크에서 우수한 전이 성능
- **자기지도학습 효과**: DINO 등 자기지도 ViT의 뛰어난 전이 특성
- **제로샷 성능**: CLIP의 30개 컴퓨터 비전 벤치마크에서 강력한 제로샷 성능

**3. 확장성과 데이터 효율성**
- **모델 스케일링**: 큰 모델일수록 추가 데이터를 효과적으로 활용
- **멀티모달 일반화**: 통합 아키텍처를 통한 다중 모달리티 처리
- **자기지도학습**: 라벨 없는 대용량 데이터로부터 효과적 학습

### 일반화 성능 향상 전략
1. **대규모 사전훈련**: JFT-300M 등 대용량 데이터셋 활용
2. **자기지도학습**: 마스크드 이미지 모델링, 대조학습 등
3. **지식 증류**: CNN 교사 모델로부터의 지식 전이
4. **멀티스케일 설계**: 계층적 특징 표현을 통한 다양한 추상화 수준 학습

## 6. 앞으로의 연구에 미치는 영향

### 연구 패러다임 변화
**1. 통합 아키텍처의 부상**
- NLP와 Computer Vision의 경계 해소
- 멀티모달 AI 연구의 가속화
- 범용 인공지능 연구 방향 제시

**2. 스케일링 법칙의 중요성**
- 모델 크기, 데이터 규모, 계산 예산의 체계적 관계 규명
- 대규모 모델의 우수성 입증
- 효율적 스케일링 전략 연구 촉진

### 향후 연구 시 고려사항

**1. 효율성과 실용성**
- **계산 효율성**: O(n²) 복잡도 문제 해결이 최우선 과제
- **하드웨어 최적화**: 엣지 디바이스 배포를 위한 경량화 연구
- **메모리 효율성**: 고해상도 이미지 처리를 위한 효율적 어텐션 설계

**2. 비전 특화 설계**
- **도메인 특화 귀납적 편향**: 이미지의 공간적 관계, 계층적 구조 반영
- **하이브리드 접근법**: CNN의 장점을 활용한 설계
- **Neural Architecture Search**: 자동화된 ViT 아키텍처 탐색

**3. 데이터 효율성과 일반화**
- **자기지도학습**: 라벨 의존성 완화를 위한 고도화된 사전 태스크
- **Few-shot Learning**: 적은 데이터로 빠른 적응이 가능한 메타학습
- **전이학습**: 도메인 간 지식 전이 효율성 향상

**4. 신뢰성과 해석가능성**
- **어텐션 시각화**: 모델 의사결정 과정의 투명성 확보
- **강건성 분석**: 다양한 공격과 노이즈에 대한 내성 연구
- **편향성 검증**: 공정하고 편향되지 않은 모델 개발

이 조사 논문은 Vision Transformer 연구의 현재 상태를 체계적으로 정리하고, 향후 10년간 컴퓨터 비전 연구의 방향성을 제시하는 중요한 기준점 역할을 할 것으로 예상됩니다. 특히 효율성, 일반화 성능, 실용성의 삼각 균형을 맞추는 것이 향후 연구의 핵심 과제가 될 것입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/881412c8-b7c0-49cb-ba4c-43ca61259193/2101.01169v5.pdf





# A Survey on Visual Transformer

# Reference
- https://jihyeonryu.github.io/2021-04-02-survey-paper1/

