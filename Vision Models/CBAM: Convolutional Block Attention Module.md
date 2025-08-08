# CBAM: Convolutional Block Attention Module | Image classification, Object detection

## 1. 핵심 주장 및 주요 기여  
**CBAM**(Convolutional Block Attention Module)은 기존 CNN 블록에 “채널”과 “공간” 두 차원의 경량화된 어텐션 서브모듈을 순차적으로 결합하여 특징 맵을 동적으로 보강함으로써, 네트워크의 표현력을 크게 향상시킨다.  
주요 기여  
- 채널 어텐션: 평균 풀링(avg)과 최대 풀링(max)을 병렬로 활용하여 유의미한 채널 정보를 정교하게 추출 (식 (2)).  
- 공간 어텐션: 채널 풀링(avg, max)으로 획득한 2D 특징 맵에 $$7\times7$$ 컨볼루션을 적용하여 중요한 공간 위치를 선택 (식 (3)).  
- 경량성: 오버헤드가 거의 없으면서 다양한 네트워크(ResNet, WideResNet, ResNeXt, MobileNet 등)에 손쉽게 통합 가능.  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- CNN이 깊이(depth), 너비(width), 카드널리티(cardinality)만으로 성능을 개선해 왔으나, “어텐션” 메커니즘을 통해 중요한 피처를 강조·불필요한 피처를 억제하는 접근이 부족했다.  
- 기존 Squeeze-and-Excitation(SE) 모듈은 채널 간 상관관계만 반영하고, 공간 정보는 무시함.  

### 2.2 제안 방법  
#### 2.2.1 채널 어텐션 (Channel Attention)  
주어진 특징 맵 $$F\in\mathbb{R}^{C\times H\times W}$$에서  
1) 평균 풀링, 최대 풀링으로 채널별 요약 벡터

```math
F\_{\mathrm{avg}}, F\_{\mathrm{max}}\in\mathbb{R}^{C\times1\times1}
```

계산  
2) 공유 MLP($$W_0,W_1$$)를 통과시켜 시그모이드 $$σ$$로 정규화  
3) 두 결과를 합산하여 채널 어텐션 맵 $$M_c(F)$$ 생성  
   
$$
M_c(F)=σ\bigl(W_1(W_0(F_{\mathrm{avg}}))+W_1(W_0(F_{\mathrm{max}}))\bigr)
$$  

#### 2.2.2 공간 어텐션 (Spatial Attention)  
채널 어텐션 적용 이후 특징 $$F'$$에 대하여  
1) 채널 축으로 평균, 최대 풀링하여 $$F'\_{\mathrm{avg}},F'_{\mathrm{max}}\in\mathbb{R}^{1\times H\times W}$$ 계산  
2) 두 맵을 채널축으로 연결(concat)  
3) $$7\times7$$ 컨볼루션 $$f^{7\times7}$$ 및 시그모이드 $$σ$$로 공간 어텐션 맵 $$M_s(F')$$ 생성  

$$
M_s(F')=σ\bigl(f^{7\times7}([F'\_{\mathrm{avg}};F'_{\mathrm{max}}])\bigr)
$$  

#### 2.2.3 모듈 결합 구조  
채널→공간 순차 적용이 최적이며, 병렬 결합이나 공간→채널 순서 대비 성능 우수.  
전체 프로세스:  

$$
F'=M_c(F)\otimes F,\quad F''=M_s(F')\otimes F'.
$$  

### 2.3 성능 향상  
- ImageNet‐1K 분류: ResNet-50 기준 Top-1 오류율 24.56%→22.66% (SE 대비 0.48% 추가 개선)  
- 다양한 백본(ResNet-18/34/101, WideResNet, ResNeXt, MobileNet)에서 일관된 향상  
- MS COCO 검출(mAP@[.5:.95]): ResNet-50+Faster R-CNN 기준 27.0→28.1, ResNet-101 기준 29.1→30.8  
- VOC2007 검출(SSD/StairNet): VGG16+StairNet 78.9%→79.3%, MobileNet+StairNet 70.1%→70.5%  

### 2.4 한계  
- 추가된 풀링·컨볼루션 연산으로 약간의 GFLOPs 증가  
- 순차적 어텐션 적용만 검증, 더 복잡한 다중 분기 혹은 다단계 적용은 미실험  
- 동영상, 3D 인풋 등 다른 도메인 적용 문헌 부족  

## 3. 일반화 성능 향상 관점  
- 채널과 공간 어텐션이 상호 보완적으로 작용하여, 다양한 아키텍처·작업(domain) 전반에 걸쳐 성능 안정적 향상  
- Grad-CAM 시각화에서 CBAM을 갖춘 모델이 객체 영역에 더욱 집중함을 확인하여, 노이즈 억제 및 특징 표현 정교화에 기여  
- 경량 모듈 설계로 모바일·임베디드 환경에서도 적용 가능성 확인  

## 4. 향후 연구 영향 및 고려 사항  
- **영향**: 어텐션 모듈의 범용성 입증으로, 이후 다양한 네트워크 설계에 CBAM 구조 혹은 변형 채널·공간 어텐션 모듈 통합 연구 확산  
- **고려점**:  
  1) **다중 스케일 어텐션**: 다양한 해상도·스케일 정보 결합 실험  
  2) **동적 구조 학습**: 입력에 따라 어텐션 경로·연산량을 조절하는 가변적 모듈 설계  
  3) **비전 외 도메인**: 자연어·음성 처리 등 시퀀스 모델에 공간·채널 대응 어텐션 확장  
  4) **효율 최적화**: 하드웨어 가속기 대응, 연산·메모리 오버헤드 추가 절감 방안 탐색

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4204ac7a-9ab4-4a38-88a7-d384d6e164b9/1807.06521v2.pdf
