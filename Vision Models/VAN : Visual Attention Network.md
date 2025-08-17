# VAN : Visual Attention Network | Image classification, Object detection, Semantic segmentation, Pose estimation

## 1. 핵심 주장 및 주요 기여  
Visual Attention Network(VAN)은 기존 CNN과 ViT(self-attention)의 장점을 결합하여 다음을 핵심 기여로 제안한다.  
- 대규모 커널 어텐션(Large Kernel Attention, LKA) 모듈: 2D 이미지의 국소 구조 정보, 장거리 종속성(long-range dependence), 공간 및 채널 적응성(adaptability)을 모두 확보하면서 계산 복잡도를 선형으로 유지  
- LKA 기반의 비전 백본(VAN): 단순 구조에도 불구하고 유사 규모의 ViT 및 CNN 대비 이미지 분류, 객체 검출, 세그멘테이션, 포즈 추정 등 다양한 비전 과제에서 일관된 성능 우위 달성  

## 2. 논문 상세 설명

### 2.1 해결하고자 하는 문제  
- CNN: 고정된 컨볼루션 커널로 채널·공간 적응성 부족  
- ViT(Self-Attention):  
  1. 2D 이미지를 1D 시퀀스로 처리해 공간 구조 파괴  
  2. 고해상도에서 제곱 복잡도로 연산 비용 급증  
  3. 채널 차원 적응성 미흡  

### 2.2 제안하는 방법

#### 2.2.1 Large Kernel Attention (LKA)  
- $$K\times K$$ 대규모 컨볼루션을 세 모듈로 분해  
  1. DW-Conv: $$(2d-1)\times(2d-1)$$ depth-wise convolution  
  2. DW-D-Conv: $$\lceil K/d\rceil \times \lceil K/d\rceil$$ dilated depth-wise convolution  
  3. Pointwise Conv: $$1\times1$$ convolution  
- LKA 연산:  

$$
    \mathrm{Attention} = \mathrm{Conv}_{1\times1}(\mathrm{DW\text{-}D\text{-}Conv}(\mathrm{DW\text{-}Conv}(F)))
  $$

$$
    \mathrm{Output} = \mathrm{Attention} \odot F
  $$
  
여기서 $$F\in\mathbb{R}^{C\times H\times W}$$, $$\odot$$는 원소별 곱셈  
- 선형 복잡도 $$O(n)$$ 유지, 별도 정규화(sigma/softmax) 불필요  

#### 2.2.2 Visual Attention Network 구조  
- **4단계 계층적 디자인**: 입력 해상도 $$\tfrac{H}{4}\times\tfrac{W}{4}$$ → $$\tfrac{H}{8}\times\tfrac{W}{8}$$ → $$\tfrac{H}{16}\times\tfrac{W}{16}$$ → $$\tfrac{H}{32}\times\tfrac{W}{32}$$  
- 각 단계:  
  - 다운샘플링 → L개 반복 $$[{\tt BN}\to1\times1\;{\tt Conv}\to{\tt GELU}\to{\tt LKA}\to{\tt FFN}]$$  
- 모델 크기: B0–B6 일곱 가지 변형 (파라미터 4.1M–200M, FLOPs 0.9G–38.4G)

### 2.3 성능 향상  
- **ImageNet-1K 분류**: VAN-B2(26.6M, 5.0G) Top-1 82.8% → Swin-T 대비 +1.5%, ConvNeXt-T 대비 +0.7%  
- **COCO 객체 검출**: VAN-B2 RetinaNet AP44.9 vs. ResNet50 AP36.3, Mask R-CNN AP46.4 vs. ResNet101 AP40.4  
- **ADE20K 세그멘테이션 (UPerNet)**: VAN-B2 mIoU50.1% vs. Swin-T 46.1%  
- **COCO 파놉틱 세그멘테이션**: VAN-B6 PQ58.2%로 당시 최상위  
- **COCO 포즈 추정**: VAN-B2 AP74.9% vs. Swin-T AP72.4%  

### 2.4 한계  
- LKA는 고정 커널 크기(K=21)와 팽창 비율(d=3)를 기본으로 사용하나, 다양한 입력 크기나 멀티스케일 구조와의 융합이 미검증  
- 복잡한 멀티브랜치나 자율 학습(self-supervised) 도입 시 추가 실험 필요  

## 3. 모델의 일반화 성능 향상 가능성  
- **채널·공간 적응성 결합**: 입력 특징에 동적으로 반응해 다양한 도메인 변화에 강건  
- **선형 복잡도**: 고해상도 및 배치 크기에 따라 확장성 우수, 전이 학습 시 효율적  
- **2D 구조 보전**: 자연 이미지의 공간 구조를 보존하므로 도메인 간 분포 차이 완화에 유리  

## 4. 향후 연구 영향 및 고려 사항  
VAN과 LKA는 비전 모델 설계에서 ‘순수’ self-attention에 대한 대안으로 자리매김할 전망이다.  
- **영향**:  
  - 대규모 커널 어텐션의 범용성 검증  
  - CNN·Transformer 하이브리드 연구 촉진  
- **고려 점**:  
  - 멀티스케일 및 멀티브랜치 구조와의 결합 가능성  
  - 자율 학습·도메인 적응 등 전이 학습 전략 적용  
  - 경량화 또는 모바일 환경 최적화 방안 모색

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87f0fa3f-c2b2-4a3a-8bc8-2db46f1c4566/2202.09741v5.pdf
