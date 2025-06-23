이 논문은 "Vision Transformer (ViT)"라는 모델을 제안하는데, 이 모델은 기존의 합성곱 신경망(CNN) 대신 순수한 트랜스포머 아키텍처를 이미지 패치 시퀀스에 직접 적용합니다.  
ViT는 큰 데이터셋에서 사전학습(pre-training)된 후, 이미지넷(ImageNet), CIFAR-100 등 다양한 이미지 인식 작업에 활용할 수 있으며, 뛰어난 성능을 보여줍니다.  
또한, 전통적인 CNN과 비교했을 때 학습에 사용하는 계산 자원이 훨씬 적고 확장성도 높다는 것이 핵심 장점입니다.

1. 입력 처리: 이미지를 일정 크기의 격자(패치)로 나누고, 각 패치를 1D 시퀀스로 변환합니다. 각 패치는 (P, P) 크기의 작은 이미지 블록이고, 이들을 평평하게 펼쳐서 벡터로 만든 후, 선형 투영 계층을 통해 고정 크기(D)의 벡터(패치 임베딩)로 만듭니다.

2. 포지셔닝 임베딩: 이미지 내 위치 정보를 보존하기 위해, 위치 정보를 나타내는 포지셔닝 임베딩을 사용하며, 사전 학습된 임베딩을 고해상도 이미지를 위해 선형 보간을 통해 조정할 수 있습니다.

3. 트랜스포머 인코더: 이렇게 만들어진 시퀀스(패치 임베딩)를 표준 트랜스포머 인코더에 입력합니다. 이 인코더는 여러 층의 셀프 어텐션과 피드포워드 네트워크로 구성되어 있으며, 언어 모델에서 사용하는 것과 거의 동일합니다.

4. 클래스 토큰: 시퀀스의 처음에 특별한 "클래스 토큰"을 넣고, 최종 출력에서 이 토큰의 표현을 사용해 분류 결과를 얻습니다.

5. 학습 및 fine-tuning: 이 모델은 대규모 데이터셋에서 사전학습(pre-training, 예를 들어, 마스킹 없는 언어 모델처럼 다수의 이미지를 사용)된 후, 특정 분류 태스크에 맞춰 미세 조정(fine-tuning)됩니다.

이러한 구조는 CNN과 달리 이미지 내 지역적 구조를 별도로 명시하지 않고도 뛰어난 성능을 발휘하며, 전체적으로 매우 간단하고 확장 가능하다는 장점이 있습니다.

# “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” 

## 1. 개요  
“An Image is Worth 16×16 Words”는 Google Research 팀이 발표한 Vision Transformer(ViT) 논문으로, 이미지 분류에 순수 Transformer 구조만을 적용해도 우수한 성능을 낼 수 있음을 보입니다[1].

## 2. 모델 구조  
### 2.1 입력 처리  
- **패치 분할**: 원본 이미지를 $$P\times P$$ 크기의 패치 $$N = \tfrac{H \times W}{P^2}$$개로 자릅니다.  
- **선형 임베딩**: 각 패치를 펼쳐(flatten) $$D$$차원 벡터로 변환하는 선형층으로 투영합니다[1].  

### 2.2 위치 정보  
- **포지셔널 임베딩**: 패치 순서 정보를 보존하기 위해 학습 가능한 1D 위치 임베딩을 더합니다[1].

### 2.3 분류 토큰  
- **[class] 토큰**: BERT의 [CLS] 토큰과 유사하게, 학습 가능한 분류용 토큰을 시퀀스 앞에 추가하고, 최종 인코더 출력에서 이 토큰을 분류용 표현으로 사용합니다[1].

### 2.4 Transformer 인코더  
- **Multi-Head Self-Attention (MSA)**: 패치 간 전역 상호작용을 수행합니다.  
- **MLP 블록**: 두 개의 완전연결층과 GELU 활성화로 구성됩니다.  
- **LayerNorm & Residual**: 각 MSA/MLP 앞에 LayerNorm, 뒤에 잔차 연결을 사용합니다[1].

### 2.5 분류 헤드  
- **사전학습**: MLP 한 개 숨은층을 붙입니다.  
- **파인튜닝**: 단일 선형층을 사용해 최종 클래스 확률을 출력합니다[1].

## 3. 하이브리드 구조  
CNN 특성 맵(feature map)을 입력으로 받아 Transformer에 연결하는 방식으로, 초기 특징 추출에 CNN의 로컬 바이어스를 활용할 수 있습니다[1].

## 4. 주요 실험 결과  
### 4.1 데이터 규모와 성능  
- **작은 데이터(ImageNet 1.3M장)**: ResNet 계열에 비해 성능이 낮음  
- **중간 데이터(ImageNet-21k, 14M장)**: Base와 Large 모델 간 성능 비슷  
- **대규모 데이터(JFT-300M장)**: 대형 ViT 모델이 ResNet 기반 BiT 모델을 뛰어넘음[1]

### 4.2 SOTA 비교  
| 모델               | 사전학습 데이터  | ImageNet Top-1(%) | CIFAR-100(%) | VTAB(19 task)(%) | TPUv3-days |
|-------------------|-------------|-----------------|-------------|----------------|-----------|
| ViT-H/14          | JFT-300M    | 88.55           | 94.55       | 77.63          | 2.5k      |
| ViT-L/16          | JFT-300M    | 87.76           | 93.90       | 76.28          | 0.68k     |
| BiT-L(ResNet152x4)| JFT-300M    | 87.54           | 93.51       | 76.29          | 9.9k      |
| Noisy Student     | JFT-300M    | 88.5            | –           | –              | 12.3k     |

ViT 모델들은 CNN 대비 훨씬 적은 연산 비용으로 동등하거나 우수한 성능을 냈습니다[1].

### 4.3 컴퓨트 대비 성능  
- 동일 예산에서 ViT는 ResNet보다 2–4× 적은 FLOPs로 비슷한 성능 달성  
- 하이브리드는 소형 모델 예산에서 다소 유리하나, 대형 모델에서는 순수 ViT와 차이 감소[1]

## 5. 내부 동작 분석  
- **포지셔널 임베딩**: 공간 내 거리와 행·열 구조를 임베딩 유사도로 학습[1].  
- **어텐션 거리**: 일부 헤드는 초기 레이어부터 전역 어텐션을 활용, 다른 헤드는 국소적 정보 집중. 깊이가 깊어질수록 어텐션 범위 확장[1].  
- **시각화**: 입력 토큰이 분류 토큰에 어떻게 기여하는지 시각화해, 의미론적으로 중요한 영역을 학습함을 확인[1].

## 6. 결론 및 시사점  
순수 Transformer만으로 이미지 분류가 가능하며, 대규모 데이터 사전학습 시 전통적 CNN을 뛰어넘는 성능을 보였습니다. 향후 물체 검출·분할, 자기지도 학습 등 다양한 비전 과제에 ViT를 확장하는 연구가 기대됩니다[1].

[1] https://arxiv.org/pdf/2010.11929.pdf
[2] https://arxiv.org/abs/2208.01618
[3] https://arxiv.org/html/2405.02793
[4] https://arxiv.org/pdf/2311.11919.pdf
[5] https://arxiv.org/pdf/2206.01843.pdf
[6] http://arxiv.org/pdf/1811.00491.pdf
[7] https://arxiv.org/html/2408.04909v1
[8] https://arxiv.org/abs/2106.13445
[9] https://arxiv.org/abs/2010.11929
[10] https://velog.io/@sjinu/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0AN-IMAGE-IS-WORTH-16X16-WORDS-TRANSFORMERS-FOR-IMAGE-RECOGNITION-AT-SCALE-Vi-TVision-Transformer
[11] https://openreview.net/forum?id=YicbFdNTTy
[12] https://arxiv.org/abs/2103.13915
[13] https://www.semanticscholar.org/paper/An-Image-is-Worth-16x16-Words:-Transformers-for-at-Dosovitskiy-Beyer/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a
[14] https://webisoft.com/articles/vision-transformer-model/
[15] https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
[16] https://openreview.net/pdf?id=YicbFdNTTy
[17] https://arxiv.org/pdf/2306.03168.pdf
[18] https://arxiv.org/abs/2209.14491
[19] https://www.ultralytics.com/glossary/vision-transformer-vit
[20] https://www.pinecone.io/learn/series/image-search/vision-transformers/
