# Deep Residual Learning for Image Recognition | Optimization, Object Detection/Localization, Image classification

## 1. 핵심 주장 및 주요 기여  
딥 레지듀얼 러닝(Residual Learning) 프레임워크를 도입하여, 매우 깊은 신경망에서도 학습이 용이해지고 성능이 향상됨을 실험적으로 입증함.  
- **핵심 주장**: 층을 쌓아 만든 기존의 ‘plain’ 네트워크는 깊이가 증가할수록 학습 오류가 오히려 증가(“degradation 문제”)하나, 각 블록이 직접 함수 $$H(x)$$를 학습하는 대신 잔차 함수 $$F(x)=H(x)-x$$를 학습하고 출력에 입력 $$x$$를 더하는 구조를 도입하면(Residual Block), 훨씬 더 깊은 네트워크도 안정적으로 최적화되어 정확도가 크게 향상된다.  
- **주요 기여**  
  1. Residual Block 설계: 입력을 identity shortcut으로 건너뛰어 더하는 잔차 학습 구조(Fig.2) 제안.  
  2. 매우 깊은 네트워크(최대 152층)에서도 학습·테스트 오류가 지속 감소함을 ImageNet·CIFAR-10 실험으로 검증.  
  3. ILSVRC 2015 및 COCO 2015에서 classification, detection, localization, segmentation 1위 달성.  

## 2. 문제 정의·제안 방법·모델 구조·성능·한계  
### 해결하고자 하는 문제  
- **Degradation 문제**: 네트워크가 깊어질수록 train error가 오히려 증가하며 학습이 어려워짐(Fig.1)  

### 제안 방법  
- **Residual Mapping**  
  - 목표 함수 $$H(x)$$ 대신 잔차 함수 $$F(x)=H(x)-x$$를 학습하고, 출력으로 $$F(x)+x$$를 계산  
  - 수식  

$$
      y = F(x;\{W_i\}) + x
    $$  
  - 차원이 맞지 않을 때 Linear projection $$W_s x$$ 사용:  

$$
      y = F(x)+W_s x
    $$  
- **Shortcut 연결**: identity shortcut을 기본, 차원 변경 시 1×1 convolution projection 선택  

### 모델 구조  
- **Plain vs. Residual**: VGG·34-layer plain 대비 동일 복잡도의 34-layer ResNet 구조(Fig.3)  
- **Bottleneck 디자인**: 1×1→3×3→1×1 순의 세 층 블록으로 차원 축소·확장해 연산 절감(Fig.5)  
- **Depth**: ResNet-18/34/50/101/152 적용, CIFAR-10에선 최대 1202층까지 확장  

### 성능 향상  
- **ImageNet Classification**  
  - 34-layer plain: top-1 28.54% → ResNet-34: 25.03% (3.5%p 향상) [Table2]  
  - ResNet-152: top-5 4.49% (single), ensemble 3.57% (test)로 당시 최고  
- **CIFAR-10**  
  - Plain-56: 8.75% → ResNet-56: 6.97% (1.78%p 개선)  
  - ResNet-110: 6.43% (state-of-the-art 수준) [Table6]  
- **Object Detection/Localization**  
  - PASCAL VOC, COCO: VGG 대비 mAP@[.5,.95] 21.2→27.2 (COCO val) [Table8]  
  - ILSVRC DET/LOC: detection mAP 43.9→58.8, localization error 26.9→8.9%로 대폭 개선  

### 한계  
- **매우 깊은 네트워크 과적합**: CIFAR-10 1202층 실험에서 110층보다 테스트 성능↓  
- **추가 정규화 필요성**: dropout·maxout 같은 기법 결합 검토  
- **이론 해석 부족**: 왜 residual이 최적화에 유리한지 수학적 분석 미흡  

## 3. 일반화 성능 향상 가능성  
- **잔차 학습의 제너럴리티**: ImageNet뿐 아니라 작은 데이터셋(CIFAR-10), 다양한 과제(detection, segmentation)에서 일관된 성능 향상  
- **표현력 확장**: 깊이 증가에 따른 과적합 제어를 위해 BN과 identity shortcut 결합, 다양한 데이터 규모에 적응 가능  
- **전이 학습**: 사전학습된 ResNet이 다양한 downstream task에 강력한 backbone으로 활용됨  

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **후속 연구**:  
  - **DenseNet, ResNeXt 등 우회 연결 발전**: residual 개념을 확장해 다양한 연결 패턴 연구  
  - **이론적 분석**: 잔차 학습이 손실 함수 지형에 미치는 영향 수학적 규명  
- **고려할 점**:  
  - **네트워크 과적합 방지**: 깊이·폭 최적화, 정규화 기법 조합  
  - **경량화**: 모바일·임베디드 환경 위한 bottleneck·pruning·quantization  
  - **새로운 task 적용**: 비전 외 자연어·시계열 데이터에 residual 적용 가능성 탐색  

***
**핵심 시사점**: Residual Learning은 단순한 아이디어이지만, 네트워크 깊이를 극단적으로 확장해도 안정적 학습과 뛰어난 일반화를 보장하며, 이후 현대 딥러닝 모델 설계의 기반이 되었다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9c36852d-3950-4a9c-9039-4655dfaff7a5/1512.03385v1.pdf
