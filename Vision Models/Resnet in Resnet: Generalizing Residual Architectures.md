# Resnet in Resnet: Generalizing Residual Architectures | Image classification

## 핵심 주장과 기여

**ResNet in ResNet (RiR)**은 기존 ResNet의 한계를 해결하기 위해 제안된 **이중 스트림(dual-stream) 아키텍처**입니다. 이 논문의 핵심 기여는 다음과 같습니다:[1][2]

- **일반화된 잔차 블록(Generalized Residual Block)** 설계로 ResNet과 표준 CNN을 통합
- **계산 오버헤드 없이** 구현 가능한 ResNet Init 초기화 방법론 제시  
- **CIFAR-100에서 새로운 SOTA** 달성 및 CIFAR-10에서 경쟁력 있는 성능 확보

## 해결하고자 하는 문제

### **기존 ResNet의 근본적 한계**

기존 ResNet은 여러 혁신적 장점에도 불구하고 구조적 제약을 가지고 있었습니다:[2]

**특징 표현의 혼재 문제**: 항등 연결(identity shortcut connections)로 인해 각 계층에서 서로 다른 수준의 특징 표현이 누적되어, 깊은 네트워크의 후반부에서는 초기 계층의 특징이 더 이상 유용하지 않을 수 있음[3][4]

**고정된 잔차 학습의 한계**: 잔차는 고정된 크기의 얕은 서브네트워크에서만 학습되어야 하는 제약이 있음에도 불구하고, 더 깊은 네트워크가 더 표현력이 풍부하다는 증거가 존재[2]

**정보 제거의 어려움**: 항등 가중치를 학습하기 어렵다는 가정과 마찬가지로, 특정 계층에서 표현으로부터 정보를 제거하기 위해 필요한 항등 가중치의 가법적 역원을 학습하는 것도 어려움[2]

## 제안하는 방법론

### **일반화된 잔차 블록 구조**

RiR의 핵심은 **이중 스트림 아키텍처**입니다. 각 일반화된 잔차 블록은 두 개의 병렬 스트림으로 구성됩니다:[2]

**잔차 스트림 (r)**: 원래 ResNet과 유사하게 항등 숏컷 연결을 포함하는 스트림

**순간 스트림 (t)**: 숏컷 연결 없이 표준 합성곱 계층으로 구성된 스트림

### **수학적 정의**

일반화된 잔차 블록의 동작은 다음 수식으로 표현됩니다:[2]

$$
r_{l+1} = \sigma(\text{conv}(r_l, W_{l,r \to r}) + \text{conv}(t_l, W_{l,t \to r}) + \text{shortcut}(r_l))
$$

$$
t_{l+1} = \sigma(\text{conv}(r_l, W_{l,r \to t}) + \text{conv}(t_l, W_{l,t \to t}))
$$

여기서:
- **같은 스트림 및 교차 스트림** 활성화가 합쳐진 후 배치 정규화와 ReLU 비선형성이 적용
- **순간 스트림**은 이전 상태의 정보를 비선형적으로 버릴 수 있는 능력 제공
- **잔차 스트림**은 기존 ResNet의 최적화 이점을 유지

### **ResNet Init 초기화**

RiR은 **ResNet Init**이라는 혁신적인 초기화 방법을 사용합니다. 이는 기존의 합성곱 계층을 수정된 초기화로 구현하여:[2]

- **단일 선형 연산**으로 일반화된 잔차 블록을 구현
- **추가적인 계산 비용이나 매개변수 없이** 효과를 달성
- **부분적인 항등 행렬**을 연결된 가중치 행렬에 추가하는 방식

## 모델 구조와 성능

### **아키텍처의 표현력**

일반화된 잔차 블록은 뛰어난 **표현 유연성**을 제공합니다:[2]

- **표준 CNN으로 동작**: 잔차 스트림을 0으로 학습
- **단일 계층 ResNet 블록으로 동작**: 순간 스트림을 0으로 학습  
- **기존 2-계층 ResNet 블록 포함**: 모든 중간 형태의 표현 가능

### **실험적 성능 향상**

**CIFAR-10 결과**:[2]
| 모델 | 정확도 (%) |
|------|-----------|
| ResNet (32 layers) | 92.49 |
| 18-layer + wide ResNet | 93.95 |
| **18-layer + wide RiR** | **94.99** |

**CIFAR-100 결과**:[2]
| 모델 | 정확도 (%) |
|------|-----------|
| 18-layer + wide ResNet | 76.58 |
| **18-layer + wide RiR** | **77.10** |

### **깊이별 성능 분석**

RiR은 **다양한 깊이에서 일관된 성능 향상**을 보여줍니다. 특히 ResNet이 깊어질수록 학습이 어려워지는 반면, RiR은 더 깊은 잔차를 효과적으로 학습할 수 있습니다.[2]

## 일반화 성능 향상 메커니즘

### **정보 흐름의 최적화**

RiR의 이중 스트림 구조는 **적응적 정보 처리**를 가능하게 합니다:[2]

- **잔차 스트림**: 중요한 정보의 직접적인 전파를 보장
- **순간 스트림**: 불필요한 정보의 선택적 제거를 담당
- **교차 연결**: 두 스트림 간의 정보 교환으로 표현력 극대화

### **시각화 분석 결과**

논문의 시각화 실험에서 **두 스트림 모두 정확도에 기여**하며, **처리 단계별로 잔차와 순간 스트림의 상대적 사용도가 변화**함을 확인했습니다. 이는 네트워크가 학습 과정에서 적응적으로 정보 처리 전략을 조정함을 의미합니다.[2]

### **그래디언트 흐름 개선**

기존 ResNet의 **그래디언트 소실 문제 해결**과 동시에, RiR은 **더 풍부한 그래디언트 경로**를 제공합니다. 이는 **vanishing gradient 문제를 완화**하면서도 **깊은 네트워크에서의 학습 안정성**을 향상시킵니다.[4][3]

## 한계점과 고려사항

### **구현의 복잡성**

RiR은 표면적으로는 간단해 보이지만, **L2 정규화와 같은 가중치 정규화가 존재**할 때 구현 방식에 따라 성능이 달라질 수 있습니다. ResNet Init 구현 시 정규화 적용 전에 부분 항등 행렬을 빼는 등의 세심한 처리가 필요합니다.[2]

### **하이퍼파라미터 최적화**

논문에서는 **잔차와 순간 스트림에 동일한 수의 필터**를 사용했지만, 이 하이퍼파라미터를 최적화하면 **추가적인 성능 향상**이 가능할 것으로 제시했습니다.[2]

## 미래 연구에 미치는 영향

### **아키텍처 설계의 새로운 패러다임**

RiR은 **단순한 항등 연결을 넘어서는** 잔차 학습의 가능성을 제시했습니다. 이는 후속 연구들에서 **다중 경로 아키텍처**와 **적응적 정보 처리** 메커니즘 설계에 영감을 제공했습니다.[5][6]

### **초기화 방법론의 발전**

**ResNet Init**과 같은 **구조-인식 초기화 방법**은 이후 neural architecture search와 자동화된 모델 설계 분야에서 중요한 참고점이 되었습니다.[7]

### **현대 아키텍처와의 연관성**

RiR의 이중 스트림 개념은 현재의 **Transformer 아키텍처**와 **attention 메커니즘**에서도 유사한 형태로 발견됩니다. **residual connections의 부작용**에 대한 최근 연구들도 RiR에서 제기한 문제의식과 맥을 같이 합니다.[6][5]

## 향후 연구 고려사항

### **대규모 데이터셋 검증**

RiR의 효과는 주로 **CIFAR 데이터셋**에서 검증되었으므로, **ImageNet과 같은 대규모 데이터셋**에서의 성능 검증이 필요합니다.

### **메모리 효율성 분석**

이중 스트림 구조가 **메모리 사용량과 추론 속도**에 미치는 영향에 대한 심화 분석이 요구됩니다.

### **다른 도메인으로의 확장**

**자연어 처리나 시계열 데이터** 등 다른 도메인에서 RiR의 일반화된 잔차 블록 개념이 어떻게 적용될 수 있는지 탐구할 필요가 있습니다.

RiR은 **계산 오버헤드 없이 성능을 향상**시키는 혁신적인 접근법으로서, 딥러닝 아키텍처 설계에 있어 **효율성과 표현력의 균형**을 추구하는 중요한 이정표를 제시했습니다.

[1] https://www.semanticscholar.org/paper/06a81b3b11f4f51a6b72f009841378547f85674c
[2] https://arxiv.org/abs/1603.08029
[3] https://www.mdpi.com/1099-4300/26/11/974
[4] https://ieeexplore.ieee.org/document/9686703/
[5] https://arxiv.org/html/2404.10947v1
[6] https://arxiv.org/html/2404.10947v4
[7] https://proceedings.neurips.cc/paper_files/paper/2022/file/7886b9bafe76c52fd568db10ff9772df-Paper-Conference.pdf
[8] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b998363e-5c9b-4e4c-9722-318ce32b2bf5/1603.08029v1.pdf
[9] https://www.mdpi.com/2227-7390/12/20/3227
[10] https://www.mdpi.com/2079-9292/13/9/1652
[11] https://link.springer.com/10.1007/s10278-025-01438-1
[12] https://ieeexplore.ieee.org/document/11011394/
[13] https://www.mdpi.com/2227-7390/13/9/1460
[14] https://www.emerald.com/insight/content/doi/10.1108/RIA-12-2024-0279/full/html
[15] https://www.ewadirect.com/proceedings/ace/article/view/24497
[16] https://arxiv.org/abs/2312.01431
[17] https://dl.acm.org/doi/10.1145/3512527.3531405
[18] https://arxiv.org/pdf/1603.08029v1.pdf
[19] http://arxiv.org/pdf/1611.08323.pdf
[20] http://arxiv.org/pdf/1811.04380.pdf
[21] https://arxiv.org/pdf/2401.09018.pdf
[22] https://arxiv.org/pdf/2103.07579.pdf
[23] http://arxiv.org/pdf/1812.04352.pdf
[24] https://arxiv.org/pdf/1811.00995.pdf
[25] https://arxiv.org/pdf/2403.12887.pdf
[26] http://arxiv.org/pdf/2212.05663.pdf
[27] https://arxiv.org/pdf/1505.00393.pdf
[28] https://openreview.net/pdf/lx9l4r36gU2OVPy8Cv9g.pdf
[29] https://openreview.net/pdf?id=rkmoiMbCb
[30] https://www.scitepress.org/Papers/2023/128004/128004.pdf
[31] http://d2l.ai/chapter_convolutional-modern/resnet.html
[32] https://blog.gopenai.com/understanding-resnet-a-thorough-exploration-of-convolutional-neural-networks-ab14b2568002
[33] https://www.jetir.org/papers/JETIR1906265.pdf
[34] https://arxiv.org/pdf/1603.08029.pdf
[35] https://en.wikipedia.org/wiki/Residual_neural_network
[36] https://www.sciencedirect.com/science/article/pii/S0957417423033626
[37] https://openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g
[38] https://nico-curti.github.io/PhDthesis/md/Chapter2/NeuralNetwork/Shortcut.html
[39] https://www.sciencedirect.com/science/article/pii/S2666521225000225
[40] https://www.tandfonline.com/doi/full/10.1080/1206212X.2025.2465727?src=exp-la
[41] https://www.nature.com/articles/s41598-024-63623-6
[42] https://www.tandfonline.com/doi/full/10.1080/01969722.2022.2151178
[43] https://www.semanticscholar.org/paper/cd958525291ee1ab856d23aa93cb95c86d87ccbe
[44] https://lv99.tistory.com/25
[45] http://ieeexplore.ieee.org/document/8315010/
[46] https://opg.optica.org/abstract.cfm?URI=ol-50-3-860
[47] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12451/2656831/Convolutional-neural-networks-with-constrained-shortcut-connections/10.1117/12.2656831.full
[48] https://www.ewadirect.com/proceedings/ace/article/view/10196
[49] https://ieeexplore.ieee.org/document/10249254/
[50] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13291/3033584/Implementation-of-FPGA-based-ResNet-accelerator-for-vehicle-detection/10.1117/12.3033584.full
[51] https://ieeexplore.ieee.org/document/11031994/
[52] https://ieeexplore.ieee.org/document/10170368/
[53] https://arxiv.org/pdf/2102.04159.pdf
[54] https://arxiv.org/html/2410.21564v2
[55] https://arxiv.org/pdf/1909.04653.pdf
[56] https://www.mdpi.com/2075-4418/13/20/3234/pdf?version=1697547349
[57] https://arxiv.org/pdf/1706.04964.pdf
[58] http://arxiv.org/pdf/2206.06929.pdf
[59] https://pmc.ncbi.nlm.nih.gov/articles/PMC10606037/
[60] https://arxiv.org/pdf/1902.06066.pdf
[61] https://arxiv.org/pdf/2101.00590.pdf
[62] https://pmc.ncbi.nlm.nih.gov/articles/PMC10442545/
[63] https://lswook.tistory.com/105
[64] https://pmc.ncbi.nlm.nih.gov/articles/PMC10073362/
[65] https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008533
[66] https://openreview.net/forum?id=1AxQpKmiTc
[67] https://imlim0813.tistory.com/35
[68] https://www.sciencedirect.com/science/article/pii/S0896627317301988
[69] https://sungbeomchoi.github.io/paper_implementation/2021-03-17-Resnet_Implementation
[70] https://ganghee-lee.tistory.com/41
[71] https://www.nature.com/articles/s41467-019-08840-8
[72] https://paperswithcode.com/sota/image-classification-on-cifar-10?p=cvt-introducing-convolutions-to-vision
[73] https://channelai.tistory.com/2
[74] https://elifesciences.org/articles/23871v1.pdf
[75] https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-%ED%8C%8C%EC%9D%B4%EC%8D%AC-ResNet%EC%9E%94%EC%B0%A8%EC%8B%A0%EA%B2%BD%EB%A7%9D%EC%9D%98-%EA%B0%9C%EB%85%90-22-CIFAR-10-%ED%99%9C%EC%9A%A9%ED%95%B4%EC%84%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B6%84%EB%A5%98%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84
[76] https://cumulu-s.tistory.com/33
[77] https://www.sciencedirect.com/science/article/pii/S0896627321006218
[78] https://stackoverflow.com/questions/61841938/how-can-i-improve-my-pytorch-implementation-of-resnet-for-cifar-10-classificatio
[79] https://wikidocs.net/165430
[80] https://www.tandfonline.com/doi/full/10.1080/09658211.2024.2408321
