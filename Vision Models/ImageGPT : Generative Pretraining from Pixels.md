# Generative Pretraining from Pixels | Image classification

**핵심 주장**  
“Generative Pretraining from Pixels”(iGPT) 논문은 자연어처리에서 성공을 거둔 대규모 언어 모델의 *사전학습*(pretraining) 아이디어를 이미지 도메인에 적용하여, 2D 공간적 구조를 전혀 인코딩하지 않은 순차적 오토회귀(pixel-by-pixel) Transformer가 레이블 없이 학습된 경우에도 강력한 이미지 표현을 획득할 수 있음을 보인다.  

**주요 기여**  
1. 2D inductive bias 없이 픽셀을 1D 시퀀스로 처리하는 *Transformer decoder* 아키텍처 적용  
2. 저해상도 ImageNet 및 웹 이미지(100M장)로 1차 사전학습, CIFAR-10/100, STL-10, ImageNet 상의 선형 프로브(linear probe)와 전수(fine-tuning) 평가에서 기존 자가-지도학습(self-supervised)·감독학습(supervised) 벤치마크를 능가  
3. 모델 규모(76M→455M→1.4B→6.8B 파라미터) 및 입력 해상도(32²×3→48²×3→64²×3)를 늘릴수록 표현 품질과 분류 정확도에서 일관된 향상 관찰  

## 1. 해결하고자 하는 문제  
- **레이블 획득 비용**: 대규모 이미지 분류를 위해 수백만 개의 레이블을 수집하는 비용과 노력이 막대  
- **비지도 학습 한계**: 기존 GAN·VAE 기반 비지도 모델은 분류용 표현 학습에서 최첨단(self-supervised) 대비 뒤처짐  
- **인풋 구조 의존**: 대부분의 자가-지도학습은 CNN 특유의 2D 구조 가정을 활용  

→ *질문*: 2D 구조 가정 없이 순차적 픽셀 예측만으로도 강력한 범용 이미지 표현을 얻을 수 있는가?

## 2. 제안 방법

### 2.1. 입력 전처리 및 시퀀스화  
- 원본 이미지 → 짧은 변형 경로(resize + 9-bit 컬러 클러스터링) → 32², 48², 64² 해상도  
- RGB 채널을 분리하지 않고 픽셀별로 R→G→B 순서(raster order)로 1D 시퀀스 생성  

### 2.2. 학습 목표  
1) **오토회귀(next-pixel prediction)**  

$$L_{\mathrm{AR}} = -\mathbb{E}\_{x \sim X} \sum_{i=1}^n \log p(x_i \mid x_ { < i }; \theta) $$

2) **BERT-스타일 마스킹(masked pixel prediction)**  

$$
L_{\mathrm{BERT}} = -\mathbb{E}\_{x\sim X}\,\mathbb{E}\_{M}\sum_{i\in M}\log p(x_i\mid x_{\setminus M};\theta)
$$

- 마스킹 확률 15% (±실험적 조정)  

### 2.3. 모델 구조  
- **Transformer Decoder**  
  - 깊이 $$L$$=24–60층, 임베딩 차원 $$d$$=512–3072  
  - 각 블록: LayerNorm → (Masked) Multi-Head Attention → Residual → MLP → Residual  
  - 포지션 임베딩 독립 학습 → 2D 공간 정보 무가정(permutation-invariant)  
- **특징 추출**  
  - 사전학습 후: 각 레이어 또는 중앙 레이어의 출력에 평균 풀링 → 선형 분류기(frozen) 학습  
  - 전수(fine-tuning): 마지막 레이어 출력에 분류기 부착 → 전체 네트워크 재학습  

## 3. 성능 향상 및 한계

| 데이터셋    | 방식             | 정확도 (%)                          |
|-------------|------------------|-------------------------------------|
| CIFAR-10    | Linear Probe     | 96.3 (iGPT-L) *최고치*              |
| CIFAR-10    | Fine-tuning      | 99.0 (iGPT-L) **AutoAugment 동등** |
| CIFAR-100   | Linear Probe     | 82.8 (iGPT-L)                       |
| CIFAR-100   | Fine-tuning      | 88.5 (iGPT-L)                       |
| ImageNet    | Linear Probe     | 72.0 (iGPT-XL, 계층 합산)           |
| ImageNet    | Fine-tuning      | 72.6 (iGPT-L, MR=48²)               |

- **모델 및 해상도 증대**가 곧 표현 품질 향상으로 직결  
- **비모노톤적(depth-wise) 특징 품질**: 중간 레이어 최적 (supervised와 대조적으로)  
- **한계**  
  - 높은 메모리·계산 비용(Quadratic self-attention)  
  - 저해상도 제한 → 고해상도 처리 부진  
  - 대형 모델일수록 저데이터(fine-tuning) 과적합 어려움  

## 4. 일반화 성능 향상 가능성

- **스케일 효과**: 모델 크기·훈련 스텝·데이터량 증대로 표현 분리도 및 선형 분류 성능 지속 상승  
- **비모노톤적 레이어 활용**: 중간 레이어 피처 조합(concatenation)으로 더 풍부한 표현 → 일반화 개선  
- **BERT 목표**: 오토회귀 대비 선형 프로브 저조하나, 전수 이후 수렴하여 일반화 동등  

→ 향후, 더 많은 데이터·효율적 어텐션·다중 레이어 피처 융합으로 **일반화 성능 강화** 가능

## 5. 향후 연구에 미치는 영향 및 고려 사항

- **영향**  
  - 이미지 도메인에서도 대규모 언어 모델 사전학습 패러다임의 실용성 입증  
  - 순차 모델의 표현 학습 잠재력 재조명 → 비감독 영상·단백질·시계열 등 타 도메인 확대  
- **고려 사항**  
  1. **효율적 어텐션**: 메모리 절감형 Sparse/Local/LSH 어텐션 연구  
  2. **고해상도 스케일업**: 멀티스케일·패치 기반 모델과 통합  
  3. **저데이터 시나리오**: 대형 사전학습 모델의 소량 라벨 일반화 기법 (정규화, 증강, 프롬프트)  
  4. **다양한 생성모델 비교**: 플로우·VAE·GAN과 representation learning 성능 비교  

**요약**: iGPT는 픽셀 단위 오토회귀 Transformer를 통해 2D 구조 가정 없이도 강력한 이미지 표현을 획득함을 입증했으며, 모델·데이터·계산량의 스케일링이 표현 학습의 핵심임을 강조한다. 향후 효율적 구조와 고해상도 처리, 저데이터 일반화 강화 연구가 이어질 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c267a78f-1996-4676-8360-ccfeea47309e/Generative_Pretraining_from_Pixels_V2.pdf
