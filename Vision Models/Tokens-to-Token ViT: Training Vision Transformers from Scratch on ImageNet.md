# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet

## 1. 핵심 주장 및 주요 기여
T2T-ViT는 **Vision Transformer(ViT)가 가진 ‘단순 패치 분할(tokenization)로 인한 지역 구조 정보 상실’**과 **주의(attention) 백본의 불필요한 채널 중복** 문제를 해결하고자 한다.  
- **Tokens-to-Token 모듈**로 이미지의 주변 토큰 간 지역 구조(엣지, 선 등)를 재귀적으로 통합해 샘플 효율성과 표현력 개선.  
- **Deep–Narrow 백본 설계**로 채널 수를 줄이는 대신 층을 늘려 feature richness를 극대화, 파라미터·MACs 절반 하면서 성능↑.  
- ImageNet에서 ResNet50급(≈25M param) 모델 대비 +2.4%p, ViT-S/16 대비 +3.4%p 개선.  

## 2. 문제 해결 방법
### 2.1 해결하고자 하는 문제
1. **지역 구조 정보 손실**: ViT는 이미지를 고정 크기 패치(예: 16×16)로 나누어 토크나이즈(tokenize)하기에 엣지·텍스처 같은 로컬 패턴 인식이 약함.  
2. **백본의 채널 중복·효율 저하**: ViT의 넓은 채널 구조(shallow-wide)는 limited training samples에서 feature richness가 낮고 일부 채널이 무효(값 0)임.

### 2.2 제안하는 방법
#### Tokens-to-Token 모듈
각 단계에서  
1) **Re-structurization**: Transformer 출력 토큰 $$T_i\in\mathbb{R}^{l_i\times c}$$을 공간 텐서 $$I_i\in\mathbb{R}^{h_i\times w_i\times c}$$로 재배열.  
2) **Soft Split (SS)**: 중첩(overlap) 패치 $$k\times k$$, stride $$k-s$$, padding $$p$$로 분할해 주변 패치 간 지역 상관을 내장한 토큰 생성.

$$
\ell_{i+1} = \Big\lfloor\frac{h_i+2p-k}{k-s}+1\Big\rfloor \times \Big\lfloor\frac{w_i+2p-k}{k-s}+1\Big\rfloor
$$

$$
T_{i+1}\in\mathbb{R}^{\ell_{i+1}\times (c\,k^2)}
$$

이를 $$n$$회 반복하여 토큰 길이를 점진 축소.  

#### 백본 구조
- 입력 최종 토큰 $$T_f$$에 클래스 토큰·Positional Embedding 추가.  
- **Deep–Narrow 구조**: 은닉 차원 $$d$$ 감소, 층수 $$b$$ 증가.

$$
T^0 = [t_{\text{cls}};T_f]+E,\quad
T^i = \mathrm{MLP}(\mathrm{MSA}(\mathrm{LN}(T^{i-1}))),\;i=1\ldots b
$$

- MLP 크기, 헤드 수 등은 ViT 대비 절반 수준으로 경량화.

## 3. 모델 구조
| 모델            | T2T 반복 | T2T 채널 | 백본 층수 $$b$$ | 은닉 dim $$d$$ | Params(M) | MACs(G) | Top-1 (%) |
|----------------|----------|----------|---------------|---------------|----------|---------|----------|
| T2T-ViT-14     | 2        | 64       | 14            | 384           | 21.5     | 4.8     | 81.5     |
| ResNet50       | –        | –        | 50            | –             | 25.5     | 4.3     | 79.1*    |
| ViT-S/16       | –        | –        | 8             | 768           | 48.6     | 10.1    | 78.1     |

\*동일 학습 스킴 적용 결과.  

## 4. 성능 향상 및 한계
- **ImageNet**: ResNet50 대비 +2.4%p, ViT-S/16 대비 +3.4%p Top-1↑[Tab.2][Tab.3].
- **경량 모델**: MobileNetV2-1.4×(6.9M) 75.6% vs. T2T-ViT-12(6.9M) 76.5%↑.
- **한계**:  
  - Transformer 특성상 still 높은 MACs; 연산 집약적.  
  - 소규모 데이터셋 일반화 실험(CIFAR-10/100)에서 우수하지만, 더 다양한 도메인에서 검증 필요[Tab.5].  

## 5. 일반화 성능 향상 가능성
- **지역 구조 학습**: T2T 모듈은 엣지·텍스처 등 로컬 패턴 우선 학습을 유도해 소량 데이터 조건에서도 과적합 완화.  
- **깊고 좁은 백본**: 잦은 레이어에서 다양한 표현 학습, 일반화 경향 우수.  
- **Transfer**: CIFAR-10/100 전이학습에서 ViT보다 0.4–1.9%p 높은 성능, 튜닝 여력 시사.

## 6. 향후 연구에 미치는 영향 및 고려사항
- **Transformer for Vision**: T2T 모듈 개념은 영상기반 전처리(tokenization) 개선 연구 확장에 기여.  
- **경량화·효율화**: Deep–Narrow 설계 원칙이 다른 Vision Transformer 변형에도 적용 가능.  
- **고려점**:  
  - 연산량·메모리 절감 위한 추가적 효율화(예: 효율적 어텐션, 동적 토크나이즈).  
  - 더 복합한 도메인(의료영상, 원격탐사) 일반화 검증.  
  - 대규모 비라벨 데이터 활용 자율 학습(self-supervised)과의 결합.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4dc4566b-0e9e-4279-9044-9cbf2b5258e3/2101.11986v3.pdf
