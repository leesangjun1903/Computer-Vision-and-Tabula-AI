# YOLOv3: An Incremental Improvement | Object detection

## 1. 핵심 주장과 주요 기여

YOLOv3 논문은 저자 Joseph Redmon이 직접 언급했듯이 "nothing super interesting, just a bunch of small changes that make it better"라는 철학 하에, 혁신적인 변화보다는 기존 YOLO 아키텍처의 점진적 개선에 중점을 두었습니다.[1]

### 주요 기여

**1) Multi-scale Detection 도입**
- FPN(Feature Pyramid Network)과 유사한 개념을 적용하여 3개의 서로 다른 스케일(13×13, 26×26, 52×52)에서 객체 탐지를 수행[2][3][4]
- 각 스케일마다 3개의 앵커 박스를 사용하여 총 9개의 앵커 박스 활용[5]

**2) 향상된 백본 네트워크 (Darknet-53)**
- 53개의 컨볼루션 레이어로 구성된 Darknet-53을 feature extractor로 도입[6][7]
- ResNet에서 영감을 받은 residual connections 추가로 깊은 네트워크 학습 안정화[4][8]
- ResNet-101보다 1.5배 빠르고, ResNet-152와 유사한 성능을 2배 빠른 속도로 달성[8][4]

**3) Multi-label Classification**
- Softmax 대신 독립적인 logistic classifier 사용으로 다중 라벨 분류 지원[9][4]
- Binary cross-entropy loss 적용으로 복합 객체 시나리오에서 성능 향상[5]

## 2. 해결하고자 하는 문제와 제안 방법

### 해결하고자 하는 문제

**1) 작은 객체 탐지 성능 부족**
- YOLOv2까지는 단일 스케일 예측으로 인한 작은 객체 탐지 한계[10][8]

**2) 정확한 객체 위치 추정의 어려움**
- 높은 IoU 임계값에서의 성능 저하 문제[10]

**3) 복합 라벨 데이터셋 처리 한계**
- Open Images Dataset과 같은 중복 라벨 데이터 처리 필요성[9]

### 제안 방법

**1) Bounding Box Prediction (수식 포함)**
YOLOv3는 YOLO9000과 동일한 bounding box 예측 방식을 사용합니다:

$$
bx = \sigma(tx) + cx
$$

$$
by = \sigma(ty) + cy  
$$

$$
bw = pw \cdot e^{tw}
$$

$$
bh = ph \cdot e^{th}
$$

여기서 $$(cx, cy)$$는 그리드 셀의 좌상단 오프셋, $$(pw, ph)$$는 앵커 박스의 너비와 높이, $$\sigma$$는 sigmoid 함수입니다.[1]

**2) Loss Function**
YOLOv3의 전체 손실 함수는 다음과 같이 구성됩니다:[11][12]

- **Objectness Loss**: Binary cross-entropy를 사용한 객체 존재 여부 판단
- **Box Regression Loss**: Sum of squared error를 사용한 바운딩 박스 좌표 회귀  
- **Classification Loss**: Binary cross-entropy를 사용한 다중 라벨 분류

**3) Multi-scale Feature Extraction**
Feature Pyramid 구조를 통해 서로 다른 해상도의 특징 맵을 융합:
- 높은 해상도 특징 맵에서 fine-grained 정보 추출
- 낮은 해상도 특징 맵에서 semantic 정보 추출
- Upsampling과 concatenation을 통한 특징 융합[7][4]

## 3. 모델 구조

### Darknet-53 Architecture
Darknet-53은 다음과 같은 구조적 특징을 가집니다:[4][6]

- **53개 컨볼루션 레이어**: 3×3과 1×1 필터를 번갈아 사용
- **Residual Connections**: Gradient vanishing 문제 해결
- **Strided Convolution**: Max pooling 대신 stride 2 컨볼루션으로 다운샘플링
- **Batch Normalization**: 모든 컨볼루션 레이어에 적용

### Multi-scale Detection Head
각 스케일에서 $$N \times N \times [3 \times (4 + 1 + 80)]$$ 텐서를 예측:
- **4**: 바운딩 박스 오프셋 (tx, ty, tw, th)
- **1**: Objectness score
- **80**: COCO 데이터셋 클래스 확률[5]

## 4. 성능 향상 및 한계

### 성능 향상

**정량적 성과**
- COCO 데이터셋에서 57.9% AP50 달성 (RetinaNet 57.5%와 유사)[1]
- 추론 속도: 320×320 해상도에서 22ms, 608×608에서 51ms[1]
- YOLOv2 대비 크게 향상된 mAP: 21.6% → 33.0%[8]

**기술적 개선사항**
- 작은 객체 탐지 성능 향상: Multi-scale detection으로 APS 성능 개선[1]
- 실시간 처리 능력 유지: SSD 대비 3배 빠른 속도[1]
- 다중 라벨 분류 지원: 복합 객체 시나리오 처리 능력 향상[9]

### 한계

**1) 높은 IoU 임계값에서의 성능 저하**
- COCO의 0.5~0.95 IoU 평균 mAP에서 RetinaNet 대비 낮은 성능[10][1]
- 정밀한 바운딩 박스 위치 추정에서의 한계[10]

**2) 계산 복잡도 증가**
- Darknet-53의 깊은 구조로 인한 메모리 및 연산 요구량 증가[13][14]
- 모바일 디바이스 배포 시 제약사항[15][14]

**3) 중간 크기 객체 탐지 한계**
- 작은 객체는 개선되었지만 중간 및 큰 객체에서는 상대적 성능 저하[1]

## 5. 일반화 성능 향상 가능성

### Multi-scale Detection의 일반화 효과

YOLOv3의 multi-scale detection은 다양한 크기의 객체에 대한 일반화 성능을 크게 향상시켰습니다:

**1) 스케일 불변성 향상**
- 3개의 서로 다른 해상도에서 예측함으로써 다양한 크기의 객체 처리 능력 향상[3][2]
- K-means clustering을 통한 앵커 박스 최적화로 데이터 분포에 적응[8]

**2) Feature Hierarchy 활용**
- 낮은 레벨 특징(fine-grained)과 높은 레벨 특징(semantic)의 효과적 결합[7][4]
- 다양한 추상화 수준의 정보 활용으로 일반화 능력 강화

**3) Multi-label Classification 지원**
- Softmax 제거로 상호 배타적이지 않은 클래스 관계 모델링[5]
- 복잡한 실제 환경에서의 적응성 향상

### 실제 응용 분야에서의 일반화 성능

연구 결과들은 YOLOv3의 우수한 일반화 성능을 보여줍니다:

- **의료 영상**: 치과 CBCT 이미지에서 98.6% mAP 달성[16]
- **농업**: 잡초 탐지에서 91.48% AP (monocot), 86.13% AP (dicot) 달성[17]
- **해양**: 수중 플라스틱 쓰레기 탐지에서 98.0% mAP 달성[18]
- **교통**: 차량 및 보행자 탐지에서 78% 정확도 달성[19]

## 6. 연구에 미치는 영향과 향후 고려사항

### 후속 연구에 미친 영향

**1) Architecture Evolution**
YOLOv3는 후속 YOLO 버전들의 기반이 되었습니다:
- **YOLOv4**: CSPDarknet, Mish activation 도입으로 성능 향상[20][21]
- **YOLOv5**: PyTorch 기반 구현으로 사용성 개선[21]
- **YOLOv8-v12**: Transformer 통합, attention 메커니즘 도입[22][21]

**2) Multi-scale Detection 표준화**
- FPN 기반 multi-scale detection이 객체 탐지 분야의 표준이 됨[2][3]
- 후속 모델들에서 지속적으로 채택되고 개선됨[22]

**3) Real-time Detection 발전**
- 실시간 객체 탐지의 실용성을 입증하여 산업 응용 확산[23][24]
- Edge computing 환경에서의 객체 탐지 연구 활성화[25]

### 향후 연구 시 고려사항

**1) 모델 경량화 연구**
- **Channel Pruning**: 불필요한 채널 제거로 모델 크기 83.6% 감소 가능[26]
- **Knowledge Distillation**: Teacher-Student 구조로 성능 유지하며 경량화[27]
- **Quantization**: FP32→FP16 변환으로 추가 압축 가능[26]

**2) 정확도 개선 방향**
- **Loss Function 개선**: IoU 기반 손실함수 (CIoU, DIoU) 도입[28]
- **Attention Mechanism**: CBAM 등 attention 모듈 통합[29]
- **Data Augmentation**: 다양한 증강 기법으로 일반화 성능 향상[30]

**3) 도메인 특화 최적화**
- **Small Object Detection**: 딜레이션 컨볼루션, multi-level fusion 적용[29]
- **Edge Deployment**: TensorRT, ONNX 최적화 고려[26]
- **Domain Adaptation**: 특정 도메인(의료, 농업 등)에 대한 전이학습 연구[10]

**4) 평가 지표 개선**
YOLOv3 저자들이 제기한 COCO 평가 지표의 한계를 고려하여:
- 실제 사용자가 중요시하는 평가 지표 개발
- 분류 정확도와 위치 정확도의 균형 있는 평가
- 도메인별 특화된 평가 방법론 연구[1]

YOLOv3는 점진적 개선을 통해 실시간 객체 탐지의 실용성을 크게 향상시켰으며, multi-scale detection과 deep backbone의 결합을 통해 후속 연구의 방향을 제시한 중요한 연구입니다. 특히 일반화 성능 향상과 다양한 응용 분야에서의 성공적인 적용을 통해 컴퓨터 비전 분야 발전에 significant한 기여를 하였습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f62d71e-c62b-4c29-be3c-252a575fcbb0/1804.02767v1.pdf
[2] https://ieeexplore.ieee.org/document/10837069/
[3] https://docs.ultralytics.com/models/yolov3/
[4] https://pyimagesearch.com/2022/05/09/an-incremental-improvement-with-darknet-53-and-multi-scale-predictions-yolov3/
[5] https://velog.io/@imfromk/CV-YOLOv3-An-Incremental-Improvement-review
[6] https://www.geeksforgeeks.org/computer-vision/darknet-53/
[7] https://herbwood.tistory.com/21
[8] https://arxiv.org/html/2406.10139v1
[9] https://www.ijircst.org/DOC/11-optimizing-real-time-object-detection-a-comparison-of-yolo-models.pdf
[10] https://arxiv.org/html/2508.02067v1
[11] https://wikidocs.net/167690
[12] https://stackoverflow.com/questions/55395205/what-is-the-loss-function-of-yolov3
[13] https://ieeexplore.ieee.org/document/9902919/
[14] https://www.nature.com/articles/s41598-022-15272-w
[15] https://ieeexplore.ieee.org/document/8793382/
[16] https://ieeexplore.ieee.org/document/9466037/
[17] https://www.mdpi.com/2072-4292/13/24/5182
[18] https://ieeexplore.ieee.org/document/9708245/
[19] https://ieeexplore.ieee.org/document/9558888/
[20] https://yolovx.com/evolution-of-yolo-a-timeline-of-versions-and-advancements-in-object-detection/
[21] https://blog.roboflow.com/guide-to-yolo-models/
[22] https://arxiv.org/html/2411.00201v2
[23] https://ieeexplore.ieee.org/document/11081174/
[24] https://iopscience.iop.org/article/10.1088/1742-6596/2866/1/012048
[25] https://ieeexplore.ieee.org/document/10743533/
[26] https://dl.acm.org/doi/10.1145/3705391.3705398
[27] https://link.springer.com/article/10.1007/s11554-022-01227-x
[28] https://learnopencv.com/yolo-loss-function-siou-focal-loss/
[29] https://arxiv.org/abs/2212.02809
[30] https://www.nature.com/articles/s41598-024-57799-0
[31] https://anapub.co.ke/journals/jmc/jmc_abstract/2023/jmc_volume_03_issue_03/jmc_volume3_issue3_9.html
[32] https://xlink.rsc.org/?DOI=D2AY01526A
[33] https://ieeexplore.ieee.org/document/10065732/
[34] https://ieeexplore.ieee.org/document/9421214/
[35] https://www.ewadirect.com/proceedings/ace/article/view/10282
[36] https://dl.acm.org/doi/10.1145/3397125.3397139
[37] https://arxiv.org/abs/2406.12395
[38] https://ieeexplore.ieee.org/document/10485961/
[39] https://onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12042
[40] https://www.tandfonline.com/doi/pdf/10.1080/21642583.2021.1901156?needAccess=true
[41] https://journals.sagepub.com/doi/pdf/10.1177/1729881420936062
[42] https://arxiv.org/pdf/2212.02809.pdf
[43] https://www.frontiersin.org/articles/10.3389/fnbot.2023.1092564/full
[44] https://pmc.ncbi.nlm.nih.gov/articles/PMC10984955/
[45] https://www.tandfonline.com/doi/pdf/10.1080/21642583.2020.1824132?needAccess=true
[46] https://www.mdpi.com/2072-4292/13/19/3908/pdf
[47] https://pmc.ncbi.nlm.nih.gov/articles/PMC9978332/
[48] https://arxiv.org/pdf/2302.07483.pdf
[49] https://www.youtube.com/watch?v=wYjkiI-Lm-8
[50] https://docsaid.org/en/papers/object-detection/yolov3/
[51] https://www.sciencedirect.com/science/article/abs/pii/S0960148122005079
[52] https://developer-lionhong.tistory.com/171
[53] https://ieeexplore.ieee.org/document/10420421/
[54] https://ieeexplore.ieee.org/document/9316169/
[55] https://dl.acm.org/doi/10.1145/3372278.3390710
[56] https://www.semanticscholar.org/paper/ffeb47a37d352fb02c42c9dbde6d64d8e6ed60f7
[57] https://arxiv.org/pdf/2305.12344.pdf
[58] https://www.mdpi.com/2072-4292/13/7/1311/pdf?version=1617951156
[59] http://arxiv.org/pdf/2307.10953.pdf
[60] https://drpress.org/ojs/index.php/ajst/article/download/241/189
[61] https://downloads.hindawi.com/journals/cin/2022/1759542.pdf
[62] http://arxiv.org/pdf/1804.02767.pdf
[63] https://www.mdpi.com/2072-4292/16/10/1774/pdf?version=1715873267
[64] https://ijai.iaescore.com/index.php/IJAI/article/download/21612/13558
[65] http://arxiv.org/pdf/2312.06458.pdf
[66] https://in.mathworks.com/help/vision/ug/getting-started-with-yolo-v3.html
[67] https://wikidocs.net/163613
[68] https://velog.io/@krec7748/Pytorch-Yolo-v3-%EA%B5%AC%ED%98%84
[69] https://kr.mathworks.com/help/vision/ug/object-detection-using-custom-training-loop.html
[70] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11878/2599401/Object-detection-using-improved-YOLOv3-tiny-based-on-pyramid-pooling/10.1117/12.2599401.full
[71] https://ieeexplore.ieee.org/document/9390981/
[72] https://www.techscience.com/cmc/v84n3/63198
[73] http://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/390
[74] https://www.ijraset.com/best-journal/social-distace-detection-with-the-help-of-deep-learning
[75] https://www.ijraset.com/fileserve.php?FID=33996
[76] https://arxiv.org/html/2411.00201
[77] https://pmc.ncbi.nlm.nih.gov/articles/PMC9448759/
[78] https://arxiv.org/abs/2209.12447
[79] https://pmc.ncbi.nlm.nih.gov/articles/PMC7180807/
[80] https://viso.ai/deep-learning/yolov3-overview/
[81] https://www.sciencedirect.com/science/article/pii/S2667241323000381
[82] https://www.labelvisor.com/history-of-yolo-from-yolov1-to-yolov10/
[83] https://www.sciencedirect.com/science/article/abs/pii/S1568494621007687
[84] https://so-development.org/from-yolo-to-sam-the-evolution-of-object-detection-and-segmentation/
