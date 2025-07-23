# BEIT: BERT Pre-Training of Image Transformers | Image classification, Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
**BEIT**(Bidirectional Encoder representation from Image Transformers)는 BERT의 *masked language modeling*을 이미지 영역으로 확장한 **Masked Image Modeling (MIM)** 과정을 통해, 레이블 없이도 비전 Transformer를 효과적으로 사전학습하는 방법을 제안한다.  
- **시각적 토큰화**: dVAE로 학습한 비주얼 코드북(8,192개 어휘)을 이용해 이미지 전체를 토큰 시퀀스(14×14)로 변환.  
- **MIM 과제**: 입력 패치(16×16 픽셀) 중 약 40%를 블록 단위로 마스킹한 후, Transformer로부터 얻은 인코딩 벡터로부터 원래의 비주얼 토큰을 예측.  
- **성과**:  
  - ImageNet-1K fine-tuning에서 BEIT-B(224²) 83.2%, BEIT-L(224²) 85.2%, 고해상도(384²)에서 최대 86.3% 달성.  
  - ADE20K semantic segmentation에서 45.6% → intermediate fine-tuning 후 47.7% mIoU.  
- **기여**:  
  1. 이미지용 **BERT 스타일 MIM** 과제 도입  
  2. dVAE 기반 **비주얼 토큰** 활용  
  3. 블록 마스킹 및 Transformer 안정화 기법 제시  

## 2. 문제 정의, 방법론, 모델 구조, 성능, 한계  

### 2.1 해결하고자 하는 문제  
- **데이터 효율성**: Vision Transformer는 대규모 레이블된 데이터 없이는 수렴이 느리고 일반화 성능 저하.  
- **픽셀 복원 한계**: 단순 픽셀 회귀는 단기 의존성과 고주파 세부 묘사에 치중하여 고수준 특징 학습 부진.  

### 2.2 제안 방법  
#### 2.2.1 비주얼 토큰화  
- dVAE(Discrete VAE)로 $$x\in\mathbb{R}^{H\times W\times C}$$를 비주얼 코드북 $$|V|=8192$$로부터 시퀀스 $$z\in V^N$$로 인코딩.  

#### 2.2.2 Masked Image Modeling (MIM)  
- 입력 패치 수 $$N=14\times14$$. 마스킹 비율 $$0.4N$$. 블록 단위로 크기·종횡비 랜덤 지정.  
- 마스킹된 입력 $$\tilde x$$를 Transformer 인코더에 통과시켜 최종 히든 $$\{h_i^L\}$$ 획득.  
- 각 마스크 위치 $$i\in M$$에 대해 소프트맥스 예측:  

$$
\max_\theta \sum_{i\in M} \log p(z_i\mid \tilde x)
= \sum_{i\in M} \log \mathrm{softmax}(W_c h_i^L + b_c)_{z_i}.
$$  

### 2.3 모델 구조  
- **백본**: ViT-Base/Large (12/24 layer, hidden 768/1024, head 12/16)  
- **입력**: 16×16 패치 + [M]·[S] 토큰 + 위치 임베딩  
- **사전학습**: ImageNet-1K 1.2M 이미지, 800 epochs, batch 2K, Adam, cosine LR, stochastic depth  

### 2.4 성능 향상  
| 모델            | ImageNet-1K Top-1 | ADE20K mIoU (UperNet) |
|-----------------|-------------------|-----------------------|
| Supervised ViT-L (22K)  | 85.2%              | 45.3%                |
| DINO-B (SS)     | 82.8%              | 44.1%                |
| **BEIT-B (SS)** | **83.2%**          | **45.6%**            |
| BEIT-L (SS)     | 85.2%              | 47.7%†               |

† Intermediate fine-tuning on ImageNet-1K 이후[1].  

### 2.5 한계  
- **토크나이저 의존**: dVAE 학습 품질에 성능 민감.  
- **계산 비용**: 고해상도·대형 모델 MIM 사전학습 시 GPU 자원 요구량 큼.  
- **레이블 전이 한계**: 라벨 풍부한 중간 데이터 필요성 존재.  

## 3. 일반화 성능 및 적용 가능성  
- **자연 이미지 일반화**: MIM으로 학습된 셀프어텐션은 객체 경계·의미 영역 구분 능력 획득[image:Figure2], 소규모 데이터셋 전이 시 과적합 완화.  
- **다양한 태스크 확장**: 분할·검출·초해상도 등 픽셀 수준 태스크에 불용 토큰 예측 대신 특화된 헤드만 추가하여 확장 용이.  
- **멀티모달 융합**: BERT-스타일 공통 프레임워크로 텍스트·이미지 통합 모델 설계 가능.  

## 4. 향후 연구 영향 및 고려사항  
- **코드북·토크나이저 개선**: 보다 표현력이 풍부한 비주얼 어휘 개발 연구 필요.  
- **경량화 사전학습**: MIM 효율화·자원 저감형 알고리즘 고안  
- **대규모·도메인 특화**: 의료·위성·상업 이미지처럼 라벨 희소 도메인에 BEIT 사전학습 적용  
- **통합 멀티모달**: CLIP, Flamingo 등과 결합하여 공통 Transformer 인코더 정합성 강화  

[1]: Table 1, Table 3  
: Figure 2, Section 3.2

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9de02785-465a-4d13-bd0b-72d08afa2996/2106.08254v2.pdf
