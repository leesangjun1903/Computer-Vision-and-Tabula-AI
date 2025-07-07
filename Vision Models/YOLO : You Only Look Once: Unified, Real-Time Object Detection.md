# YOLO : You Only Look Once: Unified, Real-Time Object Detection | Object detection

## 핵심 주장 및 주요 기여  
You Only Look Once(YOLO)은 객체 검출 작업을 단일 회귀(regression) 문제로 재구성하여, 전체 영상을 한 번만 네트워크에 입력해 동시에 다수의 바운딩 박스와 클래스 확률을 예측하는 **단일 통합(object detection) 모델**을 제안한다.  
- **실시간 성능**: Titan X GPU에서 기본 YOLO는 45fps, Fast YOLO는 155fps를 달성.  
- **end-to-end 최적화**: 전통적 파이프라인(영역 제안, 분류, 후처리)을 폐지하고, 전체 네트워크를 검출 성능에 직접 최적화.  
- **글로벌 맥락 활용**: 영상 전체를 한 번에 보므로 배경 오류(false positives)가 감소.  
- **강한 일반화 능력**: 자연 영상에서 학습한 모델이 예술 작품 등 이질적 도메인으로 잘 전이됨.  

## 논문이 해결하고자 하는 문제  
전통 객체 검출 기법들은 분리된 단계(영역 제안, 특징 추출, 분류, 경계 상자 보정)를 갖추어 느리고, 단계별 최적화가 어려우며, 지역 정보만을 활용해 문맥 인식이 제한적이다.  

## 제안 방법  
1. **그리드 분할**  
   입력 영상을 S×S 그리드로 나누고, 각 셀에서 B개의 바운딩 박스와 클래스 확률 C개를 예측.  
2. **예측 텐서**  
   최종 출력은 크기 S×S×(B·5 + C) 텐서. PASCAL VOC 기준 S=7, B=2, C=20 → 7×7×30.  
3. **회귀 손실 함수**  

$$
   \begin{aligned}
   &\lambda_{\text{coord}}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathbf{1}^{\text{obj}}\_{ij}\bigl[(x_i - \hat x_i)^2 + (y_i - \hat y_i)^2\bigr] \\
   +&\lambda_{\text{coord}}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathbf{1}^{\text{obj}}\_{ij}\bigl[(\sqrt{w_i} - \sqrt{\hat w_i})^2 + (\sqrt{h_i} - \sqrt{\hat h_i})^2\bigr] \\
   +&\sum_{i=0}^{S^2}\sum_{j=0}^B \mathbf{1}^{\text{obj}}\_{ij}(C_i - \hat C_i)^2
   +\lambda_{\text{noobj}}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathbf{1}^{\text{noobj}}\_{ij}(C_i - \hat C_i)^2\\
   +&\sum_{i=0}^{S^2} \mathbf{1}^{\text{obj}}\_{i}\sum_{c\in\text{classes}}(p_i(c) - \hat p_i(c))^2
   \end{aligned}
$$  

   - λcoord=5, λnoobj=0.5  
   - 위치 좌표(x,y)와 크기(w,h)의 안정적 학습을 위해 제곱근 예측 사용  

4. **모델 구조**  
   - **24개 합성곱 + 2개 완전연결층**  
   - 1×1 reduction 레이어와 3×3 convolution 레이어 반복  
   - 입력 해상도: 사전학습(ImageNet) 시 224×224 → 검출 단계 448×448  

## 성능 향상 및 한계  
- **성능**  
  - PASCAL VOC 2007: 63.4% mAP(45fps), Fast YOLO 52.7% mAP(155fps)  
  - R-CNN 계열 대비: 배경 오류 13.6%→4.75% 감소, 단 위치 오류(calibration)는 증가[Fast R-CNN 대비]  
- **한계**  
  - **공간 제약**: 셀당 바운딩 박스 수 제한 → 인접 소형 객체 검출 어려움  
  - **위치 정밀도**: 작은 객체에서 localization 오류 상대적 증가  
  - **비정형 비율 일반화**: 학습 시 보지 못한 비율/구성 객체에 약함  

## 일반화 성능 향상 가능성  
- YOLO는 픽셀 수준 차이가 큰 도메인(회화, 만화 등)에도 **객체 형태·배치 정보**를 활용해 높은 전이성 보임.  
- **글로벌 맥락 학습**으로 배경과 객체 구분이 잘 되어, 도메인 간 격차가 적음.  
- 향후 연구로 **multi-scale feature fusion**이나 **anchor box 다변화**를 통해 소형 객체 및 밀집 객체 탐지 능력 강화 시 더욱 뛰어난 일반화 가능.  

## 향후 연구에 미치는 영향 및 고려사항  
- **단일 네트워크 탐지 패러다임 확산**: SSD, RetinaNet 등 후속 연구에 영감 제공.  
- **경량화 모델 연구**: 모바일/임베디드 적용 위한 추가 최적화 필요.  
- **손실 함수 개선**: IoU 기반 직접 최적화(예: IoU-loss) 등으로 작은 객체 localization 오류 보완.  
- **데이터 다양성**: 드론·자율주행 등 현실 데이터로 훈련해 특수 도메인 일반화 추가 검증 및 강화.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/27d4daba-8e10-4e8d-898a-55c7daa71ac0/1506.02640v5.pdf
