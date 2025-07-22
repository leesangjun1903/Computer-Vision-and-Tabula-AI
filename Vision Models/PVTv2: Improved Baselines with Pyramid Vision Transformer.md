# PVTv2: Improved Baselines with Pyramid Vision Transformer | Image classification, Object detection, Semantic segmentation

## 핵심 주장 및 주요 기여  
**PVT v2**는 기존 Pyramid Vision Transformer (PVT v1)의 한계를 세 가지 설계 개선을 통해 극복함으로써, 선형 복잡도의 효율성과 향상된 성능을 동시에 달성하는 범용 비전 트랜스포머 백본을 제안한다[1].  
1. **Linear Spatial Reduction Attention (Linear SRA)**: 입력 특성맵을 고정 크기 $$P\times P$$로 평균 풀링한 뒤 어텐션을 수행하여 계산 비용을 $$O(hw)$$ 수준으로 감소시킴.  
2. **Overlapping Patch Embedding (OPE)**: 이미지 토큰화를 위해 패치 간 절반 겹침을 도입, 지역 연속성을 보존하여 표현력을 향상.  
3. **Convolutional Feed-Forward Network (CFFN)**: 기존 FFN 내부에 zero-padding depth-wise convolution을 삽입, 위치 정보를 암묵적으로 학습케 함.  

이 세 가지 개선을 결합한 PVT v2는 PVT v1 대비 유사한 파라미터·FLOPs 상황에서 이미지 분류, 객체 검출, 의미 분할 등 주요 비전 과제에서 일관되게 성능 향상을 이룬다[1].

## 1. 문제 정의  
PVT v1은  
- 고해상도 처리 시 어텐션의 $$O((hw)^2)$$ 계산 복잡도  
- 비중첩(non-overlapping) 패치로 인한 지역 정보 손실  
- 고정 크기 위치 임베딩의 유연성 부족  

세 가지 제약으로 인해 최첨단 CNN 및 트랜스포머에 비해 실제 비전 과제 적용 시 한계가 있었다[1].

## 2. 제안 방법  

### 2.1 Linear Spatial Reduction Attention  
기존 SRA의 계산 복잡도:  

$$
\Omega(\text{SRA}) = \frac{2h^2w^2c}{R^2} + hwc^2R^2
$$  

제안된 Linear SRA:  

$$
\Omega(\text{Linear SRA}) = 2hwP^2c
$$  

여기서 $$P$$는 pooling 크기(논문 기본값 7), $$h, w$$는 특성맵 높이·너비, $$c$$는 채널 수이다[1].

### 2.2 Overlapping Patch Embedding  
스트라이드 $$S$$, 커널 크기 $$2S-1$$, 제로 패딩 $$S-1$$인 컨볼루션을 통해 패치를 겹치게(token overlap) 추출, 토큰 간 지역 연속성을 유지한다[1].

### 2.3 Convolutional Feed-Forward Network  
전통적 FFN 구조(FC–GELU–FC)에 깊이별 분리 컨볼루션(depth-wise conv) 레이어를 삽입, 위치 정보와 지역 정보를 보강하며 고정 크기 위치 임베딩을 제거해 입력 해상도 유연성을 확보한다[1].

### 2.4 전체 모델 구조  
PVT v2는 B0–B5까지 6가지 규모로 확장되며, 각 스테이지별  
- 패치 임베딩 스트라이드 $$S_i$$  
- 채널 수 $$C_i$$  
- 어텐션 풀링 크기 $$P_i$$ 및 축소비 $$R_i$$  
- 헤드 수 $$N_i$$  
- FFN 확장비 $$E_i$$  
- 레이어 수 $$L_i$$  
등을 ResNet형 원칙(깊어질수록 공간 축소·채널 증가)에 따라 설계한다[1].

## 3. 성능 향상 및 한계  

| 과제             | 백본           | 주요 비교 대상 | 성능 차이                     |
|----------------|--------------|-------------|----------------------------|
| 이미지 분류      | PVT v2-B5    | Swin-B      | +0.5% Top-1 Acc (83.8% vs. 83.3%)[1] |
| 객체 검출 (COCO)| PVT v2-B2    | Swin-T      | +2.7 AP (49.9 vs. 47.2) on ATSS[1]  |
| 의미 분할 (ADE) | PVT v2-B4    | PVT v1-Large| +5.8 mIoU (47.9% vs. 42.1%)[1]     |

- **계산 효율성**: Linear SRA 적용 시 FLOPs 22% 절감, 입력 해상도 확장 시 선형적 비용 증가[1].  
- **지역성 보존 효과**: OPE + CFFN 결합으로 PVT v1 대비 COCO AP +4.2, ImageNet Acc +2.2% 향상[1].  
- **한계**:  
  - 여전히 대규모 데이터셋(ImageNet-22K 등)에서의 사전학습 효과 연구 미흡.  
  - 비전 트랜스포머의 일반적 과제인 사소한 오버피팅 가능성 및 메모리 사용량에 대한 세부 분석 부족.  

## 4. 일반화 성능 향상 가능성  
OPE와 CFFN가 도입하는 지역적 연속성 및 내재적 위치 표현은 도메인 전이(domain transfer)나 소량 데이터 학습(small-data regime)에서도 **특징 일반화**를 촉진할 수 있다. 선형 어텐션 구조는 고해상도 연산 비용을 억제하여, 자율주행·위성 영상 등 대규모 입력에도 확장성을 제공한다. 이로써 다양한 해상도·스케일 변동 환경에서의 일반화 가능성이 높아진다.

## 5. 향후 연구 영향 및 고려 사항  
- **확장성 연구**: 더 큰 사전학습 데이터셋과 다양한 도메인(의료 영상, 위성 영상)에서 PVT v2의 견고성 검증이 필요하다.  
- **경량화 결합**: MobileNet-류 경량 트랜스포머와의 하이브리드 설계를 통해 엣지 디바이스 적용 가능성 모색.  
- **메모리 및 효율성 최적화**: 어텐션 메커니즘의 추가 최적화 및 양자화·프루닝 전략 적용 연구.  
- **세밀한 일반화 평가**: 소량 학습, 도메인 변화, 잡음 강도 변화에 대한 종합적 일반화 성능 분석.  

PVT v2는 비전 트랜스포머의 효율성·성능·유연성을 동시에 개선함으로써 차세대 백본 설계에 중요한 기준점을 제공하며, 다양한 응용 및 후속 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3db19a4f-5b9e-428e-b384-b46eadd0b130/2106.13797v7.pdf
