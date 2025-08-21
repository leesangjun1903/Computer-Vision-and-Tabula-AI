# TokenCut: Segmenting Objects in Images and Videos with Self-supervised Transformer and Normalized Cut | Unsupervised object discovery

TokenCut은 **사전 학습된 Self-Supervised Vision Transformer**의 패치 임베딩을 그래프의 노드로, 패치 간 유사도를 그래프의 에지로 정의하여, 고전적인 **Normalized Cut** 알고리즘만으로 **학습 없이(run-time만으로)** 이미지와 비디오의 주요 객체를 탐지·분할하는 단일 통합 방법을 제안한다.  
- 이미지/비디오 세그멘테이션을 하나의 파이프라인으로 통합  
- 객체 발견(Object Discovery), 무감독 Saliency Detection, 무감독 비디오 세그멘테이션에서 기존 최상위 기법 대비 최대 6.1% CorLoc, 5.6% IoU 향상  
- 추가적인 미세 경계 정제를 위해 CRF나 Bilateral Solver 사용 가능  

## 2. 해결 문제 및 제안 방법  
### 2.1 문제 정의  
- **Unsupervised Object Discovery**: 레이블 없이 단일 영상에서 주요 객체 경계 상자(localization)를 얻는 문제  
- **Unsupervised Saliency Detection**: 레이블 없이 영상 내 두드러진(foreground) 객체 마스크(segmentation)를 얻는 문제  
- **Unsupervised Video Segmentation**: 레이블 없이 연속 프레임에서 움직이는 객체 마스크를 추출하는 문제  

### 2.2 제안 기법(TokenCut)  
1. **그래프 구성(Graph Construction)**  
   - 이미지/비디오 프레임을 $$N$$개의 $$K\times K$$ 패치로 나누고, Vision Transformer(DINO)로부터 각 패치 임베딩 $$v_i\in\mathbb{R}^d$$ 추출  
   - 유사도 함수:  

$$
       S(v_i, v_j)=\frac{v_i^\top v_j}{\|v_i\|\|v_j\|},
     $$  
     
  비디오의 경우 RGB 임베딩 $$v^I$$와 Optical Flow 임베딩 $$v^F$$의 평균 사용  
   
   - 에지 가중치:

$$
       E_{ij} = 
       \begin{cases}
         1, & \frac{S(v_i,v_j)+S(v_i^F,v_j^F)}{2}\ge\tau,\\
         \epsilon, & \text{otherwise}.
       \end{cases}
     $$

2. **그래프 분할(Graph Cut)**  
   - **Normalized Cut** 최적화:

$$
       \min_{A,B}\; \frac{\mathrm{cut}(A,B)}{\mathrm{assoc}(A,V)} + \frac{\mathrm{cut}(A,B)}{\mathrm{assoc}(B,V)}
       \;\Longleftrightarrow\;
       (D - E)y = \lambda D y,
     $$  
     
  두 번째 고유벡터 $$y_1$$를 계산하여 값의 평균을 기준으로 bi-partition  
   
  - foreground 파티션을 $$\max|y_1|$$ 값을 가진 쪽으로 선택 후, 연결 컴포넌트 중 최대 크기만 객체로 취함  

3. **에지 정제(Edge Refinement)**  
   - Coarse한 패치 단위 마스크 경계를 **Conditional Random Field(CRF)** 또는 **Bilateral Solver**로 정교화  

### 2.3 모델 구조  
- 입력 영상 → 비감독 Vision Transformer(DINO-S/16) → 패치 임베딩  
- Fully-connected 그래프 구축 → Normalized Cut(Generalized Eigen-decomposition) → Coarse 분할  
- CRF/BS 정제 → 최종 마스크  

## 3. 성능 향상 및 한계  
### 3.1 주요 성능  
- **Unsupervised Object Discovery** (CorLoc):  
  VOC07 61.9%→68.8% (↑6.9%), VOC12 64.0%→72.1% (↑8.1%), COCO20K 50.7%→58.8% (↑8.1%)  
- **Unsupervised Saliency Detection** (maxF$$_\beta$$/IoU):  
  ECSSD 75.8%/65.4%→87.4%/77.7% (↑4.4% IoU), DUTS 61.1%/51.8%→75.7%/62.8% (↑5.6% IoU)  
- **Unsupervised Video Segmentation** (Jaccard):  
  DAVIS 76.2%→76.7%, FBMS 59.8%→66.6%, SegTV2 57.2%→61.6%  

### 3.2 한계  
- **단일 객체 가정**: 복수 객체나 서로 겹치는 객체가 있으면 탐지 실패  
- **Optical Flow 품질**: 저품질 비디오(작은 움직임)에서는 Flow 유사도 기여 미미  
- **가장 “Salient” 패치 우선**: 가장 두드러진 부분이 반드시 사용자가 원하는 객체가 아닐 수 있음  

## 4. 일반화 성능 향상 관점  
- **Self-Supervised 백본 중요성**: DINO 등의 Self-Supervised ViT가 Supervised ViT보다 일반화 및 강인성 우수  
- **Threshold $$\tau$$ 민감도 낮음**: 이미지 $$\tau=0.2$$, 비디오 $$\tau=0.3$$ 전 범위에서 성능 변화 적음  
- **백본·패치 크기·모델 크기** 실험: ViT-S/8, ViT-B/16 등 다양한 설정에서 일관된 개선 효과 관찰  
- **학습 불필요**: 추가 훈련 없이 도메인 변경 시 즉시 적용 가능, 비디오 길이(최대 90프레임)만 고려  

## 5. 향후 연구에 미치는 영향 및 고려 사항  
- **학습 없는 범용 세그멘테이션**: 대규모 레이블 불필요 세그멘테이션 기법 연구 촉진  
- **다중 객체 분할 확장**: bi-partition에서 spectral clustering 확장, 다중 고유벡터 활용이 후속 연구로 유망  
- **Flow-Appearance 융합 최적화**: 저품질 Flow 보완을 위한 self-supervised flow 학습 또는 적응 전략 필요  
- **연산 효율성 개선**: $$O(N^2)$$ 복잡도를 낮추기 위한 근사 그래프 또는 샘플링 기법 연구  

TokenCut은 Self-Supervised Transformer의 패치 임베딩을 활용한 그래프 기반 분할과 고전 Ncut을 결합하여, **라벨 없이 즉시 적용 가능한** 강력한 객체 분할 프레임워크로서, 후속 학습-없는 컴퓨터 비전 연구의 새로운 패러다임을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/dceb0dec-fb21-49e0-be51-626fcb9da273/2209.00383v3.pdf
