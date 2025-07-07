# MLP-Mixer : An all-MLP Architecture for Vision | Image classification

MLP-Mixer는 **컨볼루션(convolution)**이나 **어텐션(attention)** 없이 전적으로 **다층 퍼셉트론(MLP)** 만으로 비전 모델을 구성하여, 기존 CNN 및 Transformer와 **동등한 정확도**를 달성하면서 연산 복잡도를 줄이는 것을 목표로 한다[1].  
주요 기여:
- **모델 구조**: 토큰-믹싱(token-mixing) MLP와 채널-믹싱(channel-mixing) MLP를 교차 배치한 심플한 블록 설계  
- **효율성**: 입력 패치 수 및 채널 수에 선형(線形) 복잡도  
- **성능**: ImageNet-21k 사전학습 시 ViT-L/16과 유사한 정확도, JFT-300M 대규모 학습에서 최고 수준의 ImageNet top-1 87.94% 달성[1]
- **인덕티브 바이어스 분석**: 픽셀 순서 셔플 실험으로 CNN 대비 순서 무관 특성 확인  

# 1. 해결하려는 문제  
기존 CNN은 지역적 수용영역(local receptive field)에, Vision Transformer는 쌍별 토큰간 어텐션에 의존한다. MLP-Mixer는  
1) 이들 **핸드크래프트된 바이어스(inductive bias)** 없이도 대규모 데이터에서 경쟁력 있는 성능을 내는지,  
2) 구조를 단순화하여 연산·메모리 효율을 향상시킬 수 있는지를 규명하고자 한다[1].

# 2. 제안 방법  
## 2.1 입력 및 토큰화  
이미지를 $$H\times W$$ 해상도에서 $$P\times P$$ 크기 패치로 분할하여 $$S=\frac{HW}{P^2}$$개의 토큰으로 변환 후, 각 토큰을 $$C$$차원 임베딩으로 선형 투영한다.

## 2.2 Mixer 블록  
각 블록은 두 개의 MLP 서브블록—토큰-믹싱 MLP와 채널-믹싱 MLP—을 포함한다[1].

1. **토큰-믹싱 MLP** (spatial mixing):

$$
   U_{:,i} = X_{:,i} + W_2\,\sigma\bigl(W_1\,\mathrm{LayerNorm}(X)_{:,i}\bigr)
   \quad\text{for }i=1,\dots,C
$$

   여기서 $$X\in\mathbb{R}^{S\times C}$$, $$\sigma$$는 GELU, $$W_1\in\mathbb{R}^{D_S\times S}$$, $$W_2\in\mathbb{R}^{S\times D_S}$$.

2. **채널-믹싱 MLP** (feature mixing):

$$
   Y_{j,:} = U_{j,:} + W_4\,\sigma\bigl(W_3\,\mathrm{LayerNorm}(U)_{j,:}\bigr)
   \quad\text{for }j=1,\dots,S
$$

   여기서 $$W_3\in\mathbb{R}^{D_C\times C}$$, $$W_4\in\mathbb{R}^{C\times D_C}$$.

각 서브블록 뒤에 **스킵 커넥션(skip-connection)**과 **레이어 정규화(layer norm)**가 적용된다.

## 2.3 모델 전체 구성  
초기 패치 임베딩 → $$N$$개의 Mixer 블록 반복 → 전역 평균 풀링 → 분류 헤드 구조[1].

# 3. 성능 향상 및 한계  
| 모델          | 사전학습 데이터       | ImageNet top-1 [%] | Throughput [img/s/core] |
|--------------|---------------------|--------------------|-------------------------|
| Mixer-L/16   | ImageNet-21k        | 84.15               | 105                     |
| ViT-L/16     | ImageNet-21k        | 85.30               | 32                      |
| Mixer-H/14   | JFT-300M            | 87.94               | 40                      |
| ViT-H/14     | JFT-300M            | 88.55               | 15                      |

- **연산 대비 정확도**: Mixer-H/14는 ViT-H/14 대비 2.5배 빠른 추론 속도에 거의 동등한 정확도를 보임[1].  
- **데이터 규모 민감도**: 데이터가 클수록 Mixer의 성능 향상 폭이 ViT, ResNet보다 커짐[1].  
- **제한점**: 소규모 데이터(overfitting)에 취약하며, 작은 모델(scale-down) 구간에서 ViT 대비 성능이 낮음[1].

# 4. 일반화 성능 향상 관련 고찰  
- **토큰 및 픽셀 순서 셔플 실험**:  
  - 패치·픽셀 셔플 시에도 성능 유지, 전역 픽셀 셔플에도 CNN 대비 큰 폭 감소 없음 → **순서 무관성(order-invariance)**·**높은 로버스트성** 확인[1].  
- **특성 시각화**:  
  - 토큰-믹싱 MLP의 필터는 전역·지역 패턴을 모두 학습, 위상 반전 필터 쌍 관찰 → **다양한 공간 스케일 일반화** 가능성 시사[1].

# 5. 향후 연구 영향 및 고려 사항  
- **이론적 연구**: MLP만으로도 학습 가능한 함수 클래스의 **인덕티브 바이어스 분석**이 필요.  
- **NLP 등 타 도메인 적용**: 순서 무관성·선형 복잡도를 바탕으로 자연어·음성 처리에의 확장 가능성.  
- **모델 경량화 및 Pyramidal 구조**: 작은 스케일 성능 격차 해소를 위한 계층적(pyramidal) 설계 실험.  
- **데이터 효율성 연구**: 소규모 데이터에서의 일반화 능력 개선을 위한 정교한 정규화·증강 기법 연구.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0d35acc3-3633-4421-9f45-98ec2a1a1939/2105.01601v4.pdf
