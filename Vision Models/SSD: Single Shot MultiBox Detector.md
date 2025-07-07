# SSD: Single Shot MultiBox Detector | Object detection

**핵심 주장 및 주요 기여**  
SSD는 단일 신경망만으로 객체 탐지(Object Detection)를 수행하여 기존의 다단계(region proposal + pooling) 방식보다 훨씬 간단하고 빠르면서도 높은 정확도를 달성한다. 주요 기여는 다음과 같다:  
- **Single-Shot Detection**: 앵커(anchor) 역할의 기본 박스(default boxes)를 여러 크기·비율로 갖춘 단일 네트워크가 한 번의 순전파로 모든 객체 후보를 예측.  
- **Multi-Scale Feature Maps**: 서로 다른 해상도의 여러 피처 맵에서 예측을 수행해 다양한 크기의 객체에 대응.  
- **다양한 종횡비 처리**: 각 위치마다 다양한 종횡비의 기본 박스를 사용하여 객체 형태 다양성을 포착.  
- **End-to-End 학습**: 제안 단계 없이 위치(regression)와 분류(classification)를 동시에 최적화하는 단일 손실 함수 적용.  

## 1. 문제 정의  
기존 최첨단 방법(Selective Search + R-CNN 계열)은  
1) 객체 후보(region proposal) 생성  
2) 각 후보에 대한 특성(pooling)  
3) 분류 및 위치 보정  
의 다단계 파이프라인으로 복잡하고 연산량이 매우 크다. SSD는 이 **제안 생성(proposal) 및 후처리 단계를 제거**하여 단일 네트워크로 실시간·고정밀 탐지를 목표로 한다.

## 2. 제안 방법  
### 2.1 네트워크 구조  
- **Base Network**: VGG-16의 Conv5_3까지 사용.  
- **Extra Feature Layers**: Conv7, Conv8_2, …, Conv11_2 등 여러 해상도의 피처 맵 추가.  
- **Detection Heads**: 각 피처 맵의 각 위치마다  
  - *c*개 클래스 확률(confidence)과  
  - 4개의 박스 오프셋(offsets)  
  를 예측하는 $$3\times3$$ 컨볼루션 필터.  
- **Default Boxes**: k개의 종횡비(ar ∈ {1,2,3,½,⅓} 및 추가 정사각형)와 스케일($$s_k$$)을 가지는 박스를 픽셀 좌표가 아닌 상대 좌표로 정의.  

### 2.2 손실 함수  
전체 손실은 분류 손실(confidence loss)과 위치 손실(localization loss)의 가중합:  

$$
L(x,c,l,g) = \frac{1}{N}\bigl(L_{\mathrm{conf}}(x,c) + \alpha\,L_{\mathrm{loc}}(x,l,g)\bigr)
$$  

- $$N$$: 매칭된 기본 박스 수
- $$\displaystyle L\_{\mathrm{loc}} = \sum_{i\in Pos}\sum_{m\in\{cx,cy,w,h\}}x_{ij}^\ast \,\mathrm{smooth}_{L1}(l_i^m - \hat{g}_j^m)$$  

$$\displaystyle L\_{\mathrm{conf}} = -\sum\_{i\in Pos}x_{ij}^\ast\log(\hat{c}_i^p) - \sum\_{i\in Neg}\log(\hat{c}_i^0)$$  

- Hard Negative Mining: 부정 예시 비율을 최대 3:1로 제한  

## 3. 성능 향상 및 한계  
### 3.1 성능  
| 모델     | 입력 크기 | VOC2007 mAP | COCO $$0.5\!:\!0.95$$ AP | FPS(Titan X) |
|---------|----------|-------------|---------------------------|-------------|
| SSD300  | 300×300  | 74.3%       | 23.2                      | 59          |
| SSD512  | 512×512  | 76.8%       | 26.8                      | 22          |
| Faster R-CNN | ∼1000×600 | 73.2%       | 24.2                      | 7           |

- **Real-Time**: SSD300은 59FPS로 실시간 탐지 가능하면서 Faster R-CNN 대비 정확도 우위.  
- **Scale 민감도**: 작은 객체 성능은 낮으나, 입력 해상도를 키우거나 데이터 증강(expansion)으로 개선 가능[1].  

### 3.2 한계  
- **작은 객체 탐지**: 최상위 피처 맵에서 작은 객체 정보 소실 → 성능 저하.  
- **비교적 단순한 기본 박스 디자인**: 최적 종횡비·스케일 배치 연구 필요.  
- **배후 네트워크 의존**: VGG-16 기반으로 전체 속도의 80% 차지 → 경량화 네트워크 적용 과제.  

## 4. 일반화 성능 향상 가능성  
- **데이터 증강 강화**: “확장(expansion)” 기법으로 작은 객체 학습 데이터 생성 시 mAP 2–3% 상승.  
- **더 다양한 스케일·종횡비**: 기본 박스 설계를 receptive field에 맞춰 최적화하면 일반화 성능 개선 여지.  
- **백본 네트워크 교체**: ResNet, MobileNet 등 최근 아키텍처로 교체 시 탐지 성능 및 속도 모두 상승 기대.  
- **Multi-Task 학습**: 분할(segmentation), 포즈 추정(joint detection) 등과 결합하여 특징 일반화 강화.  

## 5. 향후 연구에의 영향 및 고려 사항  
SSD는 “단일단 네트워크” 탐지 패러다임을 확립해 추후 YOLO 계열 및 FPN(FPN+ RetinaNet) 등의 발전에 중추적 기여를 했다.  
- **실시간 시스템 적용**: 드론·자율주행·로봇 비전에서 표준 모듈로 채택.  
- **네트워크 경량화**: 경량화 백본(예: MobileNetV3) + SSD 구조 조합 연구 필수.  
- **작은 객체 대응**: 피처 피라미드(FPN), 어텐션 메커니즘과 SSD 결합 가능성.  
- **End-to-End 학습 심화**: 비전-언어 멀티모달, 비디오 탐지 연속성 제약 등으로 확장 고려.  

> SSD는 단순함과 속도, 정확도의 균형을 실용적으로 제시했으며, 향후 멀티스케일·멀티태스크 딥러닝 비전 연구의 토대가 될 것임.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d86e00b0-4420-4882-bbdf-d1fe8090f223/1512.02325v5.pdf
