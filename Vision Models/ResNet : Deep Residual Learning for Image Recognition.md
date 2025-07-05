# ResNet : Deep Residual Learning for Image Recognition | Image classification
“Deep Residual Learning for Image Recognition”는 Kaiming He 등(2015)이 제안한 논문으로, **잔차 학습(residual learning)** 프레임워크를 도입하여 매우 깊은 신경망을 효율적으로 학습할 수 있게 한 연구이다[1].  

## 1. 연구 배경과 문제 제기  
기존 연구에서 네트워크를 깊게 쌓을수록 표현력과 정확도가 향상되지만, **깊이가 증가할수록 학습이 어려워지고 오히려 성능이 저하(degradation)되는 현상**이 관찰되었다. 이는 단순한 과적합(overfitting) 문제가 아니며, 학습 오류(train error)와 테스트 오류(test error) 모두 증가하는 현상이다[1].  

## 2. 잔차 학습(Residual Learning) 아이디어  
### 2.1 기본 개념  
- 기존 네트워크는 층(layer)마다 입력 $$x$$에 대해 목표 함수 $$\mathcal{H}(x)$$를 직접 학습한다.  
- 본 논문은 **잔차 함수** $$\mathcal{F}(x) = \mathcal{H}(x) - x$$를 학습하도록 설계하여, 출력은 $$\mathcal{F}(x) + x$$로 구성한다.  
- 이렇게 하면, 최적 함수가 항등 매핑(identity mapping)에 가까울 때 $$\mathcal{F}(x)$$를 0으로 쉽게 학습할 수 있어 최적화가 용이해진다[2].

### 2.2 잔차 블록(Residual Block) 구조  
```
Input x
   └─▶  ┌─ Conv(3×3) ─ BN ─ ReLU ─ Conv(3×3) ─ BN ┐
        │                                         │
        └─────────────── Identity ───────────-─-──┘
                  │
                  + (element-wise addition)
                  │
                 ReLU
                  ↓
                Output
```
- **Conv**: 합성곱 연산(Convolution)  
- **BN**: 배치 정규화(Batch Normalization)  
- **ReLU**: 활성화 함수  

이 블록이 쌓여 깊은 네트워크를 구성하며, 각 블록마다 입력을 직접 더하는 **스킵 연결(skip connection)** 을 통해 정보 흐름이 원활해진다[1][2].

## 3. 네트워크 아키텍처  
### 3.1 기본(Plain) 네트워크와 Residual 네트워크 비교  
| 깊이(depth) | Plain 네트워크 테스트 오류(%) | ResNet 테스트 오류(%) |
|------------|-------------------------------|----------------------|
| 34 layers  | 7.0                           | 5.6                  |
| 50 layers  | –                             | 4.9                  |
| 101 layers | –                             | 4.6                  |
| 152 layers | –                             | 4.5                  |

ResNet-152는 VGG보다 8배 깊지만, 오히려 복잡도는 낮고 테스트 오류가 4.49%로 크게 개선되었다[3].

### 3.2 Bottleneck 블록 (ResNet-50 이상)  
- **1×1** convolution으로 채널 수를 줄이고,  
- **3×3** convolution으로 특징을 추출하며,  
- 다시 **1×1** convolution으로 채널을 확장하여 계산 복잡도를 낮춘다.  
- 최종적으로 입력을 더해 잔차를 학습한다.  
이 구조를 통해 깊은 네트워크에서도 계산 비용을 절감하고 학습 안정성을 유지한다[4].

## 4. 주요 실험 결과  
- **ImageNet 2012**에서 ResNet-152 단일 모델로 Top-5 오류 4.49% 달성[3].  
- 동일 대회의 1위 기록(3.57% 오류, 앙상블 모델)로 ILSVRC 2015 챔피언 획득[1].  
- CIFAR-10에서 1000층 네트워크까지 학습이 가능함을 보였다.  
- COCO 객체 검출(Object Detection)에서 28% 상대 성능 향상 달성[1].

## 5. 결론 및 의의  
잔차 학습 프레임워크는 매우 깊은 네트워크 학습을 안정화하고, 최적화 문제를 효과적으로 해결하여 컴퓨터 비전 분야의 **딥러닝 모델 발전**에 큰 기여를 했다. 현재 다양한 후속 연구들이 ResNet 구조를 기반으로 발전 중이다.  

---

[1] K. He et al., "Deep Residual Learning for Image Recognition," arXiv:1512.03385, 2015.  
[3] K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016, pp.770–778.  
[4] S. Qiao, "ResNet bottleneck structure," 2023.  
[2] Papers With Code, "Residual Block Explained," 2018.

[1] https://arxiv.org/abs/1512.03385
[2] https://paperswithcode.com/method/residual-block
[3] https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[4] https://seanzqs.github.io/2023/06/05/resnet-bottleneck-structure/
[5] https://jxnjxn.tistory.com/22
[6] https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition
[7] https://cs231n.stanford.edu/reports/2016/pdfs/264_Report.pdf
[8] https://huggingface.co/learn/computer-vision-course/en/unit2/cnns/resnet
[9] https://arxiv.org/pdf/2004.04989.pdf
[10] https://phil-baek.tistory.com/entry/ResNet-Deep-Residual-Learning-for-Image-Recognition-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0

# Ref
- https://pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html
