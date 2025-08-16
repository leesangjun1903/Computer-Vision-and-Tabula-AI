# YOLOv4: Optimal Speed and Accuracy of Object Detection

## 핵심 주장 및 주요 기여  
YOLOv4는 단일 GPU 환경에서 실시간 처리가 가능한 **고속·고정확도 객체 검출기**를 제안한다. 기존 실시간 검출기(YOLOv3 등)의 속도와 정확도를 동시에 개선하여, MS COCO 데이터셋에서 43.5% AP를 65 FPS( Tesla V100)로 달성한다.  
주요 기여는 다음과 같다.  
- **Bag of Specials (BoS)** 및 **Bag of Freebies (BoF)** 기법들의 효과적인 조합  
- Weighted Residual Connections(WRC), Cross-Stage-Partial connections(CSP), Cross mini-Batch Normalization(CmBN), Self-Adversarial Training(SAT), Mish activation 등의 채택  
- Mosaic, CutMix 등 데이터 증대 및 CIoU loss, DropBlock 정규화 도입  
- CSPDarknet53–SPP–PAN–YOLOv3 구조로 백본·넥·헤드 최적화  

***

## 1. 해결하고자 하는 문제  
- **실시간 검출 vs. 정확도**: 고정확도 모델은 느리고, 고속 모델은 정확도가 낮음  
- **단일 GPU 학습 한계**: 대용량 GPU 없이도 고성능 모델을 훈련·운용할 수 있어야 함  

## 2. 제안 방법  
### 2.1 네트워크 구조  
- Backbone: CSPDarknet53  
- Neck: SPP(Spatial Pyramid Pooling) + PAN(Path Aggregation Network)  
- Head: YOLOv3 one-stage anchor-based detector  

```
Input → CSPDarknet53 → SPP → PAN → YOLO head → NMS (DIoU-NMS)
```

### 2.2 핵심 기술  
1. **Bag of Specials (추론 성능 개선 기법)**  
   - Mish activation  
   - CSP connections, WRC (Weighted Residual Connections)  
   - SPP, SAM(Spatial Attention Module), PAN  
2. **Bag of Freebies (학습 전략 개선 기법)**  
   - Data augmentation: Mosaic, CutMix, Class label smoothing  
   - Self-Adversarial Training (SAT): 네트워크가 스스로 이미지를 변형해 공격 후 학습  
   - Cross mini-Batch Normalization (CmBN): 배치 내 미니배치 통계 공유  
   - DropBlock 정규화  
   - CIoU loss:  

$$
       \mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v,
     $$
     
  where $$\rho$$는 중심점 거리, $$c$$는 둘을 감싸는 최소 박스 대각선 길이, $$v$$는 종횡비 일치 항, $$\alpha$$는 가중치  

***

## 3. 성능 향상 및 한계  
- **성능**: MS COCO test-dev에서 AP50: 65.7%, AP: 43.5% (608×608)  
- **속도**: 65 FPS(Tesla V100), 96 FPS(Titan V)  
- **한계**:  
  - 복합 BoF/BoS 파이프라인의 복잡성  
  - 작은 객체 검출에서 여전히 개선 여지  
  - 하드웨어 환경 및 배치 크기에 따른 민감도  

***

## 4. 일반화 성능 향상 가능성  
- **Mosaic & MixUp**: 다양한 문맥 학습으로 과적합 완화  
- **SAT**: 적대적 예제를 통한 강건성 강화  
- **CmBN**: 배치 편차 감소로 도메인 변경에도 안정적 통계  
- **CIoU loss**: 박스 회귀의 안정성 및 수렴 속도 개선  
이들 기법은 다른 검출기나 분류·분할 모델에도 적용 가능하여 **모델의 일반화 성능**을 전반적으로 향상시킬 잠재력을 지닌다.

***

## 5. 향후 연구에 미치는 영향 및 고려사항  
- **영향**: 다양한 BoF/BoS 기법 조합을 실험·검증하는 **모범 사례**로 활용  
- **고려사항**:  
  1. **경량화**: 모바일·IoT 환경으로의 확장  
  2. **자율 학습**: SAT 기반 적대적 학습 심화  
  3. **소형 객체 검출**: 멀티스케일 피처 통합 개선  
  4. **자동 최적화**: NAS와 결합한 BoF/BoS 선택  
  5. **다중 도메인 적응**: CmBN 등 통계 공유 기법을 도메인 적응에 활용  

이와 같은 방향으로 연구를 확장하면, **실시간 고정확도 객체 검출**의 새로운 표준이 될 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8fbb51e4-dfb7-4756-af88-04a2300d9eb5/2004.10934v1.pdf
