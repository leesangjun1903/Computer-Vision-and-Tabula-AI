# Masked autoencoders
# Masked Autoencoders Are Scalable Vision Learners (MAE) 

## 개요

**Masked Autoencoders (MAE)**는 2021년 Facebook AI Research(FAIR)에서 Kaiming He를 비롯한 연구진이 발표한 혁신적인 컴퓨터 비전 모델입니다[1][2]. MAE는 자연어 처리 분야에서 큰 성공을 거둔 BERT의 마스킹 기법을 컴퓨터 비전 영역에 성공적으로 적용한 자기지도학습(Self-Supervised Learning) 모델로, 이미지의 일부 패치를 가리고 나머지 부분으로부터 가려진 영역을 복원하는 방식으로 학습합니다[3][4].

이 모델의 핵심 아이디어는 매우 간단합니다: 입력 이미지의 일부분(예: 75%)을 임의로 마스킹한 후, 남은 25%의 정보만으로 전체 이미지를 복원하도록 학습시키는 것입니다[1][5].

## 왜 MAE가 필요했을까?

### 기존 문제점들

딥러닝 모델이 점점 커지면서 수백만 장의 레이블된 이미지가 필요하게 되었습니다[6][7]. 하지만 대량의 데이터에 레이블을 붙이는 작업은 매우 비싸고 시간이 많이 걸리는 작업입니다[8][9].

자연어 처리 분야에서는 BERT와 GPT 같은 모델이 자기지도학습을 통해 이 문제를 성공적으로 해결했습니다[10][9]. 하지만 컴퓨터 비전에서는 다음과 같은 차이점들 때문에 동일한 접근법이 쉽게 적용되지 못했습니다[1][2]:

1. **아키텍처의 차이**: 기존 CNN 구조에서는 마스크 토큰이나 위치 임베딩을 적용하기 어려웠습니다[11][12]
2. **정보 밀도의 차이**: 언어는 매우 의미론적이고 정보 밀도가 높지만, 이미지는 공간적 중복성이 큽니다[1][13]
3. **디코더 역할의 차이**: 언어에서는 의미 있는 단어를 예측하지만, 이미지에서는 픽셀을 복원해야 합니다[1][2]

## MAE의 핵심 설계

### 1. 비대칭 인코더-디코더 구조

MAE의 가장 중요한 특징은 **비대칭적인 인코더-디코더 구조**입니다[1][4]:

**인코더 (Encoder)**:
- 마스킹되지 않은 패치들만 처리합니다[2][5]
- Vision Transformer(ViT) 구조를 기반으로 합니다[11][14]
- 전체 패치의 25%만 처리하므로 연산량이 크게 줄어듭니다[4][15]

**디코더 (Decoder)**:
- 인코더보다 훨씬 가벼운 구조입니다[1][7]
- 인코더의 출력과 마스크 토큰을 함께 사용하여 전체 이미지를 복원합니다[5][4]
- 일반적으로 8개 블록, 512차원으로 구성됩니다[15][16]

### 2. 높은 마스킹 비율 (75%)

MAE는 매우 높은 마스킹 비율을 사용합니다[1][15]:

- **75%의 패치를 마스킹**: BERT의 15%보다 훨씬 높은 비율입니다[15][7]
- **공간적 중복성 제거**: 단순히 주변 패치를 복사하는 것을 방지합니다[1][13]
- **의미론적 이해 필요**: 전체적인 맥락 이해가 필요한 어려운 과제를 만듭니다[3][4]

## 작동 원리

### 단계별 과정

MAE의 학습 과정은 다음과 같습니다[4][7]:

1. **패치 분할**: 입력 이미지를 고정 크기 패치(예: 16×16)로 나눕니다[5][12]
2. **랜덤 마스킹**: 75%의 패치를 임의로 선택하여 마스킹합니다[15][16]
3. **인코딩**: 남은 25%의 패치만 인코더에 입력합니다[2][4]
4. **디코딩**: 인코더 출력과 마스크 토큰으로 전체 이미지를 복원합니다[5][7]
5. **손실 계산**: 원본 이미지와 복원된 이미지 간의 차이를 MSE(Mean Squared Error)로 계산합니다[4][17]

### 수식 표현

입력 이미지를 패치로 나누면: H × W × C → N × (P × P × C)[12]
- H, W: 이미지의 높이와 넓이
- C: 채널 수
- N: 패치 개수
- P: 패치 크기

## 주요 장점

### 1. 연산 효율성

MAE는 기존 방법보다 **3배 이상 빠른 학습 속도**를 제공합니다[1][6]:
- 인코더가 전체 패치의 25%만 처리하므로 연산량이 대폭 감소합니다[4][15]
- 마스크 토큰은 가벼운 디코더에서만 사용됩니다[1][2]

### 2. 뛰어난 성능

**ImageNet-1K 실험 결과**[15][6]:
- ViT-Huge 모델로 **87.8%의 정확도** 달성[1][18]
- 200 에폭 학습한 일반 ViT보다 50 에폭만으로 2.4%p 높은 성능[15][6]
- 기존 지도학습 방법보다 우수한 성능[6][19]

### 3. 전이학습 성능

MAE는 다양한 다운스트림 태스크에서 뛰어난 성능을 보입니다[1][19]:
- **객체 탐지 (Object Detection)**
- **인스턴스 분할 (Instance Segmentation)**  
- **의미론적 분할 (Semantic Segmentation)**[19][20]

## 실험 결과 및 분석

### 마스킹 비율에 따른 성능

연구진의 실험에 따르면[15][7]:
- **75% 마스킹**에서 선형 프로빙(Linear Probing)과 파인튜닝 모두 최고 성능
- 40%~80% 범위에서 안정적인 성능 유지
- 85% 마스킹에서도 어느 정도 복원 가능[15]

### 디코더 설계의 중요성

디코더의 깊이가 성능에 미치는 영향[15][16]:
- **선형 프로빙**: 디코더 깊이가 충분해야 좋은 성능
- **파인튜닝**: 디코더가 얕아도 성능 유지 가능
- 최적 구성: 8개 블록, 512차원[15][7]

## 다른 모델과의 비교

| 구분 | MAE | BEiT | SimMIM | iGPT |
|------|-----|------|--------|------|
| 방식 | 이미지 패치 마스킹 후 복원 | 시각 토큰 예측 | 픽셀 단위 복원 | 픽셀 순차 예측 |
| 구조 | ViT 기반, 인코더-디코더 분리 | BERT 구조 차용 | ViT + 픽셀 손실 | GPT 구조 |
| 특징 | 고효율, 대용량 학습 적합 | 이산 토큰 사용 | MAE 유사, 단순함 | NLP 구조 차용 |

[4]

## 한계점과 개선 방향

### 현재 한계점

1. **복원 중심의 학습**: 픽셀 복원에 초점을 맞춰 고차원적 의미 정보 추출에 한계[4]
2. **마스킹 전략 의존성**: 마스킹 방법에 따른 성능 편차 존재[4]
3. **디코더 역할의 모호성**: 학습 후 디코더는 폐기되어 활용도가 제한적[4]

### 개선 연구

최근에는 **MLO-MAE**와 같은 후속 연구들이 다운스트림 태스크 성능을 직접 고려한 마스킹 전략을 학습하는 방향으로 발전하고 있습니다[20].

## 결론

MAE는 컴퓨터 비전 분야에서 자기지도학습의 새로운 패러다임을 제시한 혁신적인 연구입니다[1][3]. 간단하면서도 효과적인 접근 방식으로 기존의 지도학습 방법을 뛰어넘는 성능을 달성했으며, 연산 효율성까지 크게 개선했습니다[6][13].

특히 Vision Transformer와 결합하여 이미지를 "단어"처럼 취급하는 접근법은 향후 멀티모달 AI 발전에도 중요한 기여를 할 것으로 예상됩니다[11][14]. MAE의 성공은 컴퓨터 비전 분야에서도 대규모 무라벨 데이터를 활용한 사전학습이 매우 효과적임을 보여주었으며, 이는 AI 연구의 새로운 방향성을 제시했다고 평가할 수 있습니다[9][13].

[1] https://arxiv.org/abs/2111.06377
[2] https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf
[3] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/mae/
[4] https://gnuhcjh.tistory.com/79
[5] https://developers-shack.tistory.com/13
[6] https://lcyking.tistory.com/entry/dff
[7] https://www.themoonlight.io/ko/review/downstream-task-guided-masking-learning-in-masked-autoencoders-using-multi-level-optimization
[8] https://daeun-computer-uneasy.tistory.com/37
[9] https://wikidocs.net/167320
[10] https://www.ibm.com/think/topics/masked-language-model
[11] https://daebaq27.tistory.com/108
[12] https://hipgyung.tistory.com/entry/%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EB%8A%94-ViTVision-Transformer-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale
[13] https://davidlds.tistory.com/31
[14] https://velog.io/@leehyuna/Vision-TransformerViT
[15] https://cryptosalamander.tistory.com/179
[16] https://velog.io/@kbm970709/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Masked-Autoencoders-Are-Scalable-Vision-Learners
[17] https://www.ibm.com/kr-ko/think/topics/autoencoder
[18] https://huggingface.co/papers/2111.06377
[19] https://dhk1349.tistory.com/5
[20] https://arxiv.org/abs/2406.10973
[21] https://wikidocs.net/235826
[22] https://paperswithcode.com/paper/masked-autoencoders-are-scalable-vision
[23] https://dongwoo-im.github.io/papers/review/2023-03-08-MAE/
[24] https://daebaq27.tistory.com/122
[25] https://discuss.pytorch.kr/t/vision-transformer-a-visual-guide-to-vision-transformers/4158
[26] https://www.ultralytics.com/ko/glossary/vision-transformer-vit
[27] https://velog.io/@rcchun/%EB%94%A5%EB%9F%AC%EB%8B%9D-Vision-Transformer%EC%97%90-%EB%8C%80%ED%95%9C-%EC%8B%9C%EA%B0%81%EC%A0%81-%EC%84%A4%EB%AA%85
[28] https://blog.outta.ai/202
