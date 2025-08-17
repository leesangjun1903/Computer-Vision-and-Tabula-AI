# Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism | Object detection

**핵심 주장 및 주요 기여**  
Gold-YOLO 시리즈는 기존 YOLO 계열의 FPN 기반 neck 구조가 겪는 정보 손실 문제를 해결하기 위해, 모든 스케일의 피처를 전역적으로 “수집(gather)”하고 “분배(distribute)”하는 새로운 Gather-and-Distribute(GD) 메커니즘을 제안한다. 이로써 multi-scale 피처 퓨전 성능을 대폭 향상시키면서도 지연(latency)은 최소화한다. 또한 MAE 스타일의 MIM(Masked Image Modeling) 비지도 사전학습을 YOLO 계열에 도입해 작은 모델에서도 일반화 성능을 높인다.

***

## 1. 해결하고자 하는 문제  
- **전통적 FPN 구조의 정보 손실**  
  - 인접 스케일 간 재귀적 피처 전송 과정에서 원본 고수준·저수준 피처 중 일부가 폐기됨  
  - 비인접 레벨 간 직접 정보 교환 불가로 성능 한계  
- **작은 모델의 일반화 한계**  
  - YOLO-N/S와 같은 경량 모델에서 복잡한 시멘틱 정보 획득 부족

***

## 2. 제안하는 방법

### 2.1 Gather-and-Distribute(GD) 메커니즘 개요  
- **Gather**: 모든 스케일 피처를 동일 해상도로 정렬(Feature Alignment Module, FAM) 후, 전역 퓨전(Information Fusion Module, IFM)  
- **Distribute**: 각 스케일 브랜치별로 fused global 피처를 attention 기반으로 inject  

### 2.2 Low-stage 및 High-stage 브랜치  
- Low-stage GD: 백본 피처 $$B_2,B_3,B_4,B_5$$를 평균 풀링으로 $$R_{B4}$$ 해상도로 정렬 후, RepConv 블록으로 퓨전  

```math
    F_{\text{align}} = \text{Low\_FAM}([B_2,B_3,B_4,B_5]),\quad
    F_{\text{fuse}} = \text{RepBlock}(F_{\text{align}}),
```  

```math
    [F_{\text{inj\_P3}},F_{\text{inj\_P4}}] = \text{Split}(F_{\text{fuse}}).
```

- High-stage GD: Low-stage 출력 $$\{P_3,P_4,P_5\}$$를 $$R_{P5}$$ 해상도로 정렬 후, transformer 기반 퓨전  

```math
    F_{\text{align}} = \text{High\_FAM}([P_3,P_4,P_5]),\quad
    F_{\text{fuse}} = \text{Transformer}(F_{\text{align}}),
```

```math
    [F_{\text{inj\_N4}},F_{\text{inj\_N5}}] = \text{Split}(\text{Conv}_{1\times1}(F_{\text{fuse}})).
```

### 2.3 Information Injection (Inject)  
각 스케일 $$i$$에 대해  

```math
  F_{\text{act}}^i = \text{resize}\bigl(\sigma(\text{Conv}_{\text{act}}(F_{\text{inj}}^i))\bigr),\quad
  F_{\text{embed}}^i = \text{resize}(\text{Conv}_{\text{embed}}(F_{\text{inj}}^i)),
```

```math
  F_{\text{att}}^i = \text{Conv}_{\text{local}}(F_{\text{local}}^i)\odot F_{\text{act}}^i + F_{\text{embed}}^i,\quad
  F_{\text{out}}^i = \text{RepBlock}(F_{\text{att}}^i).
```

### 2.4 경량 Adjacent-Layer Fusion (LAF)  
인접 레벨 간 bilinear up/down-sampling과 $$1\times1$$ Conv로 로컬 피처를 추가 병합하여 전역 GD와 결합, 정확도를 더욱 향상시킴.

***

## 3. 모델 구조 및 성능  

| 모델           | AP〈val〉 | FPS (bs=32) | Params | FLOPs |
|---------------|----------|-------------|-------|-------|
| Gold-YOLO-N   | 39.9%    | 1030        | 5.6M  | 12.1G |
| Gold-YOLO-S   | 46.4%⋆   | 446         | 21.5M | 46.0G |
| Gold-YOLO-M   | 51.1%⋆   | 220         | 41.3M | 87.5G |
| Gold-YOLO-L   | 53.3%⋆   | 116         | 75.1M |151.7G |

⋆: MIM 사전학습 적용, †: self-distillation 사용.  
- YOLOv6-3.0-N 대비 +2.4% AP 개선, 유사 FPS 유지  
- YOLOv8 시리즈 대비 모든 스케일에서 AP↑, Speed–Accuracy trade-off 최적화  

***

## 4. 일반화 성능 및 한계  
- **일반화 성능**  
  - Mask R-CNN, PointRend, EfficientDet 등 다양한 검출·분할 모델에 GD 적용 시 모두 성능 상승  
  - COCO instance seg.: FPN→GD 시 Bbox mAP +2.5%, Segm mAP +1.3%  
  - Cityscapes seg.: mIoU +2.07%(ResNet50), +1.71% (ResNet101)  
- **한계**  
  - 트랜스포머 블록 사용으로 구형 하드웨어에서 호환성 제약  
  - GD 모듈 내 attention·transformer 연산이 경량 모델에서 일부 오버헤드 발생 가능

***

## 5. 향후 연구에의 영향 및 고려사항  
- **향후 영향**  
  - FPN 대체를 위한 전역 피처 퓨전 아키텍처 연구 촉진  
  - 객체 검출·분할·인스턴스화 태스크 전반에 걸친 neck 모듈 혁신 방향 제시  
  - MIM 사전학습의 convnet 적용 가능성 확대  
- **고려사항**  
  - GD 구조의 연산 복잡도 최적화 및 경량화 연구  
  - 하드웨어별 연산 지원 현황과 맞춤형 모듈 설계  
  - 다양한 도메인(자율주행·의료 영상) 일반화 성능 검증 및 적응형 모듈 개발

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5c85ed15-aeef-493c-9bb8-1608d6344f66/2309.11331v5.pdf
