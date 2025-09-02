# Mask R-CNN | Object detection, Semantic segmentation

**핵심 주장**  
Mask R-CNN은 객체 검출(object detection)과 인스턴스 분할(instance segmentation)을 단일 네트워크 프레임워크에서 동시에 처리하며, 단순한 구조 변경만으로 최첨단 성능을 달성할 수 있음을 보인다.  

**주요 기여**  
1. Region-of-Interest Alignment (RoIAlign): RoIPool의 양자화 오차를 제거하고 정확한 픽셀 정렬을 수행하여 마스크 및 키포인트 예측의 위치 정확도를 크게 향상.  
2. 클래스-독립적 마스크 분리(branch): 분류(classification)와 마스크 예측(mask prediction)을 완전히 분리하여, 각 클래스별 독립적인 이진 마스크를 예측함으로써 학습 안정성과 성능 개선.  
3. 범용성 및 확장성: Faster R-CNN 구조에 마스크 분기(branch)만 추가하는 간단한 방식으로 인스턴스 분할, 사람 자세 추정(keypoint detection) 등 다양한 작업에 쉽게 적용 가능.  

***

# 1. 해결하고자 하는 문제  
인스턴스 분할은 객체 검출과 의미론적 분할(semantic segmentation)을 결합한 과제로, 각 인스턴스의 정확한 경계(mask)를 요구한다. 기존 방법들은 복잡한 파이프라인(분할 먼저 → 분류, 또는 다단계 캐스케이드)을 사용해 느리고 구현이 까다롭다. Mask R-CNN은 이 문제를 간단한 2단계 구조에서 해결하고자 한다.  

# 2. 제안 방법  
## 2.1 네트워크 개요  
- 기본 구조: Faster R-CNN  
- 추가 분기: 각 RoI에 대해 **클래스 분류**, **바운딩 박스 회귀**, **인스턴스 마스크 예측** 을 병렬 처리  

## 2.2 RoIAlign  
RoIPool의 좌표 양자화 과정  

$$
x_{\text{quant}} = \lfloor x / s \rfloor \times s
$$

를 제거하고, 각 RoI bin 내 샘플 지점의 피처 값을 **bilinear interpolation**으로 추출하여 정확도를 높인다.  

## 2.3 손실 함수  
각 RoI에 대하여 다중 작업 손실을 적용:  

$$
\mathcal{L} = \mathcal{L}\_{\text{cls}} + \mathcal{L}\_{\text{box}} + \mathcal{L}_{\text{mask}}
$$  

- $$\mathcal{L}_{\text{cls}}$$: 다중 클래스 분류용 softmax + cross-entropy  
- $$\mathcal{L}_{\text{box}}$$: bounding-box 회귀용 Smooth L1  
- $$\mathcal{L}_{\text{mask}}$$: 클래스별 **독립** 이진 마스크 $$m\times m$$에 대한 per-pixel sigmoid + binary cross-entropy  

```math
\mathcal{L}_{\text{mask}} = - \frac{1}{m^2} \sum_{i,j} \bigl[y_{ij} \log \hat{y}_{ij} + (1 - y_{ij}) \log (1 - \hat{y}_{ij})\bigr]
```

(각 RoI는 정답 클래스 $$k$$에 해당하는 마스크만 손실 계산에 기여)  

## 2.4 모델 구조  
- 백본(backbone): ResNet-50/101 또는 ResNeXt-101 + Feature Pyramid Network (FPN)  
- 박스/분류 헤드: 기존 Faster R-CNN 구조(ResNet-C4의 Res5 스테이지 또는 FPN의 4-conv head)  
- 마스크 헤드: RoIAlign 후 4×4 deconv 포함한 **fully convolutional** 마스크 분기  

***

# 3. 성능 향상 및 한계  
## 3.1 성능 향상  
- COCO 인스턴스 분할: mask AP 35.7 → ResNeXt-101-FPN에서 37.1 (기존 SOTA 대비 +3.5 이상)  
- 바운딩 박스 검출: box AP 39.8 (Mask R-CNN multi-task) vs. 36.2 (FPN only)로 +3.6 향상  
- 사람 자세 추정: keypoint AP 63.1 (단일 모델, 5fps)로 2016년 대회 우승자 대비 +0.9  

## 3.2 한계  
1. 연산 비용: 마스크 분기로 인한 ∼20% 추가 연산(ResNet-101-C4 기준)  
2. 작은 물체 분할: m×m(기본 28×28) 해상도 한계로 APS (small) 에서 개선 여지  
3. 다중 인스턴스 과밀 상황: RoI 수 제한(상위 100개)으로 누락 가능성  

***

# 4. 일반화 성능 향상 가능성  
- **다중 작업 학습(multi-task learning)**이 바운딩 박스, 마스크, 키포인트 모두 성능 향상에 기여함을 보임.  
- **RoIAlign**으로 픽셀 정렬 편차를 제거하여, 해상도가 더욱 중요한 키포인트 검출에서도 큰 성능 상승을 확인.  
- FPN 및 ResNeXt 같은 강력한 백본과 결합 시, 대규모 데이터셋(COCO) 외에도 소규모-저데이터(Cityscapes)에서 COCO 사전학습 활용 시 AP 26.2 → 32.0으로 대폭 향상.  

*결론적으로, Mask R-CNN 프레임워크는 다양한 백본, 추가 분기(branch), 확장된 작업(task)에 유연하게 적용되어 일반화 성능을 지속해서 끌어올릴 수 있는 강건한 구조를 제공한다.*  

***

# 5. 향후 연구에 미치는 영향 및 고려사항  
Mask R-CNN은 인스턴스 수준 인식의 **단일 프레임워크 표준**이 되었으며, 이후 연구들은 본 구조를 기반으로 다음을 고찰해야 한다.  
- **고해상도 마스크** 및 **세밀한 객체 분할**을 위한 multi-scale 마스크 해상도 향상  
- **실시간 처리 최적화**: lightweight 백본, 효율적 RoI 처리  
- **셀프-슈퍼비전 & 옴니-학습**: unlabeled data 활용(data distillation) 및 약지도 학습  
- **크로스 도메인 일반화**: 소규모 데이터셋에 대한 사전학습과 도메인 적응 기법  

Mask R-CNN이 제시한 RoIAlign, 분리된 마스크 분기 등의 핵심 아이디어는 후속 인스턴스 분할, 자세 추정, 비디오 분할, 3D 인스턴스 예측 등 다양한 영역에 기초 기술로 폭넓게 활용될 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7050691a-72f2-43fe-8d2b-d1467ac3e5bf/1703.06870v3.pdf)
