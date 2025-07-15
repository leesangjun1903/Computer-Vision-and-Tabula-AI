# MaskGIT: Masked Generative Image Transformer | Image generation, Image manipulation, Image reconstruction

**핵심 주장**  
MaskGIT은 기존의 순차적(autogressive) 이미지 생성 방식을 버리고, 비순차적(non-autoregressive) 병렬 예측을 통해 빠르고 높은 품질의 이미지 생성을 달성하는 새로운 패러다임을 제시한다.  

**주요 기여**  
1. **Masked Visual Token Modeling (MVTM):**  
   – BERT의 마스킹 학습을 이미지 생성에 적용하여, 랜덤하게 마스킹된 시각 토큰을 양방향(attention in all directions)으로 예측하도록 모델을 훈련.  
2. **Iterative Parallel Decoding:**  
   – 초깃값으로 모든 토큰을 마스킹한 뒤, 매 반복(iteration)마다 마스킹 비율 γ(t/T)에 따라 토큰을 점진적으로 채워 나가는 스케줄링(8∼12회 반복)으로, 기존 256단계가 필요하던 순차 생성 대비 최대 64× 속도 향상.  
3. **높은 생성 품질 및 다양성:**  
   – ImageNet 256×256: FID 6.18, IS 182.1(기존 VQGAN 대비 FID 대폭 개선)  
   – ImageNet 512×512: FID 7.32로 GAN·확산모델과 견줄 만한 성능 달성  
   – Classification Accuracy Score(CAS) top-1 63.43%, top-5 84.79%로 다양성 지표에서도 우수함.  
4. **범용적 이미지 편집:**  
   – 단일 모델로 클래스 조건부(object replacement), 인페인팅, 아웃페인팅, 엑스트라폴레이션(extrapolation) 등 다양한 편집 작업을 “초기 마스크”만 변경하여 처리 가능.

# 상세 설명

## 1. 해결하고자 하는 문제  
– **순차 생성의 비효율성:**  
  Autoregressive 모델은 이미지의 토큰 수(N≈256∼1024)에 비례해 긴 생성 시퀀스를 가지며, 32×32 크기 이미지를 생성하는 데만 30초가 소요됨.  
– **이미지의 2D 구조 미반영:**  
  기존 작업은 래스터 스캔(raster-scan) 순서로 1D 시퀀스처럼 생성하나, 실제 이미지는 순차적이지 않고 전역적(context)이 중요.

## 2. 제안 방법

### 2.1 Masked Visual Token Modeling (MVTM)  
– 입력 영상 x를 VQGAN 인코더 E로 압축해 토큰 시퀀스 $$Y=\{y_i\}_{i=1}^N$$ 획득  
– 무작위 마스크 $$M\in\{0,1\}^N$$ 적용:  

$$
    Y^M_i = 
      \begin{cases}
        [\mathrm{MASK}], & \text{if } M_i = 1\\
        y_i, & \text{if } M_i = 0
      \end{cases}
  $$
  
– 양방향(transformer) 디코더를 학습해, 마스크된 위치에 대한 확률 $$p(y_i\,|\,Y^M)$$ 를 예측.  
– 손실 함수:  
  
$$
    \mathcal{L}\_\text{mask} = -\mathbb{E}\_{Y,M}\Bigl[\sum_{i:M_i=1}\log p(y_i\,|\,Y^M)\Bigr].
$$

### 2.2 Iterative Parallel Decoding  
– **초기화:** $$Y^{(0)}$$의 모든 토큰 마스킹  
– **t번째 반복(총 T회):**  
  1. **예측:** 모델이 모든 마스크 위치에 확률 분포 $$\mathbf{p}^{(t)}\in\mathbb{R}^{N\times K}$$ 출력  
  2. **샘플링 & 신뢰도:** 각 위치 i에서 $$y_i^{(t)}$$ 샘플링, confidence $$c_i$$ 계산  
  3. **마스크 비율 결정:** $$n = \lceil \gamma(t/T)\,N\rceil$$, 여기서 γ는 **cosine 스케줄**  
  4. **재마스킹:** confidence가 낮은 n개 위치를 다시 마스킹  
– **γ(t/T) = $$\frac{1+\cos(\pi\,t/T)}{2}$$**, t=0→1 전체 마스킹→0 단계적 해제.  

### 2.3 모델 구조  
– 24-layer bidirectional Transformer  
– 8 attention heads, 임베딩 차원 768, hidden 3072  
– learnable positional embedding, LayerNorm, Adam optimizer.  
– 비디오나 다른 해상도에도 확장 가능(공유된 토크나이저 재사용).

## 3. 성능 향상  
| 해상도 | 모델     | FID↓   | IS↑    | #steps | CAS top-1↑ | CAS top-5↑ |
|-------|----------|-------|-------|-------|------------|-----------|
|256×256| VQGAN    | 15.78 | 78.3  | 256   | 53.10      | 76.18     |
|       | MaskGIT  | 6.18  | 182.1 | 8     | 63.14      | 84.45     |
|512×512| BigGAN   | 8.43  | 232.5 | 1     | 44.02      | 68.22     |
|       | MaskGIT  | 7.32  | 156.0 | 12    | 63.43      | 84.79     |

- **속도:** 8∼12회 반복으로 대폭 가속(30×∼64×)  
- **다양성:** Precision/Recall 지표에서 높은 Recall, CAS에서도 10% 이상 개선  

## 4. 한계  
– **경계 일관성 부족:** 아웃페인팅 시 색상·구조 shift  
– **세밀 구조 복원 한계:** 작은 텍스트나 대칭 구조에서 artifacts  
– **Mask 스케줄 민감도:** T 반복횟수·스케줄 함수 최적화 필요  

# 일반화 성능 향상 가능성  
- **전역 문맥 활용:** Bi-directional attention이 세밀-전역 특징 모두 학습해, 다양한 편집(task)에서 튼튼한 성능.  
- **마스크 스케줄링의 확장:** 학습 시 다양한 마스킹 비율을 랜덤 샘플링해, 마스크 비율 변화에 강인한 모델을 학습.  
- **토크나이저 공유:** VQGAN 토크나이저를 재사용해, 도메인 변경(자연→실사→의료) 시 레이어 파인튜닝만으로 적응 가능.  

# 향후 연구 영향 및 고려사항  
- **대체 디코딩 패러다임 제시:** 기존 GAN·확산 기반 생성 방식에 비순차·병렬 접근이 유망함을 입증.  
- **마스크 스케줄 최적화 연구:** 다양한 함수(Concave, Convex 등)의 성능 차 분석 및 자동화 스케줄 탐색 필요.  
- **대규모 도메인 일반화:** 의료·위성 이미지 등 특수 도메인에 MVTM+parallel decoding 구조 적용 및 안정성 검증.  
- **편집·합성 확장:** 텍스트-이미지 멀티모달, 3D 합성, 동영상 생성 분야에서 마스킹 기반 비순차 디코딩 활용.  

MaskGIT은 이미지 생성·편집 전반에 새로운 비순차적 병렬 디코딩 패러다임을 제시하며, 향후 다양한 비전 태스크의 실시간·고품질 생성에 중추적 역할을 할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/88ca7a20-589f-4a7f-a8b6-f24d6be936e3/2202.04200v1.pdf
