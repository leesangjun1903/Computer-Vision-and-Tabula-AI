# Residual Connection 가이드

핵심 요약: Residual connection은 학습이 어려운 full mapping 대신 **잔차 F(x)=H(x)−x**를 학습하게 하여 깊은 네트워크의 최적화를 쉽게 만들고, 기울기 전달 경로를 짧게 유지해 **degradation 문제를 완화**합니다.[1][2][3]

## 왜 residual인가
- 깊이를 늘리면 train/test 오류가 같이 증가하는 degradation 문제가 발생합니다. 이는 단순 vanishing gradient와 구분되며, 매우 깊은 모델에서 최적화 자체가 난해해지는 현상입니다.[1]
- ResNet은 출력 H(x)를 직접 학습하지 않고, F(x)=H(x)−x를 학습해 H(x)=F(x)+x로 표현합니다. 이렇게 하면 identity mapping을 “우회로”로 주고 남은 잔차만 학습하므로 최적화가 쉬워집니다.[1]
- Identity skip과 더하기 이후 활성화(pre-activation)의 조합은 신호가 block 간 직접 전파되도록 만들어, forward/backward 흐름이 원활해지고 일반화도 개선됩니다.[2]

## 핵심 아이디어: identity mapping
- 스킵은 가중치가 없는 identity일 때 신호 소실 없이 전달됩니다. 따라서 최적해가 identity에 가까우면, 가중치가 0에 가까운 해로 쉽게 수렴할 수 있습니다.[2][1]
- He et al.는 “after-addition activation” 대신 “full pre-activation” 구조(BN–ReLU–Conv 순서)에서 신호 전파가 가장 원활하다는 것을 실험으로 보였습니다.[2]
- 이 설계로 1000+층 CIFAR ResNet과 200층 ImageNet ResNet이 안정적으로 학습되었고, 오류율이 크게 감소했습니다.[2]

## 다른 해석들: 앙상블, shattered gradient
- ResNet을 경로로 풀어 보면 각 블록마다 “통과 vs. 스킵” 두 선택이 존재해 2^n 경로의 조합이 생깁니다. 실험적으로 긴 경로의 기울기는 거의 기여하지 않고, 5~30층 수준의 짧은 경로가 대부분의 기울기를 운반합니다.[3][4]
- 이 특성은 “상대적으로 얕은 네트워크들의 앙상블처럼 행동”한다는 해석으로 이어지며, 학습 중 유효 경로 길이가 짧게 유지되어 소실을 피합니다.[5][3]
- 또 다른 관점은 “shattered gradients” 문제입니다. 일반 feedforward는 깊어질수록 기울기 상관이 지수적으로 붕괴해 white noise처럼 산산조각 나지만, skip을 쓰면 상관이 완만히 감소하여 안정적 최적화가 가능합니다.[6][7][8]

## 실전 가이드: 아키텍처 설계
- 규칙 1: 스킵은 가급적 identity. 채널이 달라질 때만 1×1 conv로 matching을 수행하세요.[1]
- 규칙 2: pre-activation(= BN–ReLU–Conv–BN–ReLU–Conv + identity add)을 기본값으로 하세요. 더 깊을수록 수렴이 쉽습니다.[2]
- 규칙 3: downsample 시 첫 conv에 stride를 주고, 스킵에는 1×1 stride로 맞추면 됩니다.[1]
- 규칙 4: 매우 깊을수록 shortcut에 불필요한 변형(conv+BN+ReLU)을 넣지 마세요. 정보 흐름을 방해합니다.[2]

## PyTorch 예제: Pre-activation BasicBlock
- ResNet-18/34 스타일의 3×3 basic block을 pre-activation으로 구현합니다. CIFAR/Medical X-ray 등 2D 이미지에 바로 적용 가능하며, identity 스킵을 기본으로 유지합니다.[1][2]
- 채널/해상도 변화가 있을 때만 1×1 projection을 사용합니다. 이는 He et al. 원저 설정과 일치합니다.[1]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # conv1는 pre-activation 후 수행
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # identity skip: 채널/stride 불일치 시 projection
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(out if self.shortcut is not None else x)
        # 주의: pre-activation에서는 projection에 활성/정규화 입력을 쓰는 변형도 사용됨
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += shortcut
        return out
```


## PyTorch 예제: Pre-activation Bottleneck
- ResNet-50/101/152 계열 bottleneck(1×1–3×3–1×1)도 pre-activation으로 구현합니다. 채널 압축/확장(expansion=4)을 사용합니다.[1][2]
- 대규모 데이터나 의료 영상의 고해상도 특징 추출에서 유리합니다. skip은 동일 원칙을 따릅니다.[1]

```python
class PreActBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(out if self.shortcut is not None else x)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out += shortcut
        return out
```


## ResNet Backbone 구현
- 아래는 CIFAR/일반 이미지용 ResNet을 구성하는 코드입니다. block과 layer 구성을 바꾸면 18/34/50/101로 손쉽게 스케일링 가능합니다.[1]
- 입력 stem은 간단하게 하였으며, ImageNet용이면 7×7 conv와 3×3 maxpool stem을 적용하세요.[1]

```python
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_width=64):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, base_width,   num_blocks, stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(base_width*8*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBasicBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet50(num_classes=1000):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=num_classes)
```


## 학습 팁과 실험 포인트
- 유효 경로 길이: 학습 초기에 짧은 경로가 주로 학습을 이끕니다. 너무 공격적으로 블록을 제거하면 정확도가 서서히 하락하지만, 일부 제거는 견딥니다(lesion study).[5][3]
- 초기화/정규화: pre-activation ResNet은 BN과 함께 안정적입니다. 매우 깊은 경우 even-odd 층 균형과 스킵 순수성 유지가 중요합니다.[2]
- 해석 관점 실험: 블록 제거, 모듈 재배열, 경로별 기여도(gradient concentration) 분석으로 앙상블 가설을 재현해 볼 수 있습니다.[3][5]

## 수식으로 보는 잔차 학습
- 목표는 $$H(x)$$ 대신 $$F(x)=H(x)-x$$를 학습하여 $$H(x)=F(x)+x$$로 표현하는 것입니다. 이때 identity skip이 최적해에 가깝다면, $$F(x)$$는 0에 가까운 함수를 학습하면 됩니다.[1]
- pre-activation에서는 블록 $$i$$의 입력과 출력 간 신호/기울기 전파가 직접적입니다. 이는 잔차 합 이후에 비선형을 두지 않기에 가능해집니다.[2]
- shattered gradients에서, 일반 네트워크의 기울기 상관은 깊이에 따라 $$\exp(-\alpha L)$$로 붕괴하지만, skip이 있는 네트워크는 완만히 감소합니다(준선형 스케일).[6]

## 연구 확장 아이디어
- Strict identity mapping과 gating: 스킵 경로를 거의 완전한 identity로 유지하거나, 스칼라 게이트로 잔차의 세기를 조절하는 변형을 탐구할 수 있습니다.[9][10]
- 생성 모델 맥락: 최근 연구는 잔차가 표현 학습에 미치는 영향과 깊이에 따른 identity 기여 감소 전략 등을 논의합니다. 과도한 identity 혼입이 추상화 발달을 방해할 수 있다는 시사도 있습니다.[11]
- 잡음 안정성: identity 연결은 입력 잡음에 대한 안정성을 높이는 방향으로 작용한다는 분석 결과도 있습니다. 도메인 잡음이 큰 의료 영상에서 유용할 수 있습니다.[12]

## 체크리스트
- 스킵은 가능하면 “있는 그대로” 두기. 1×1 projection은 채널/stride가 달라질 때만 사용.[1]
- pre-activation 블록을 기본값으로 채택. BN–ReLU–Conv 순서 유지.[2]
- 매우 깊을수록 shortcut을 방해하는 연산 제거. 합 이후 비활성화 피하기.[2]
- downsample 시 main과 skip 모두 stride 일치. 출력 채널은 block.expansion 반영.[1]

## 참고자료
- Deep Residual Learning for Image Recognition, CVPR 2016: 원조 ResNet, identity skip과 성능/안정성 보고.[1]
- Identity Mappings in Deep Residual Networks, ECCV 2016: pre-activation과 전파 해석, 1000+층 학습.[13][2]
- Residual Networks Behave Like Ensembles of Relatively Shallow Networks, NIPS 2016: 다중 경로와 짧은 경로의 기울기 지배.[4][3]
- The Shattered Gradients Problem, ICML 2017: skip이 기울기 상관 붕괴를 완화.[7][6]

이 글은 첨부된 사이트의 주요 내용을 바탕으로, 대학생 독자가 바로 구현과 실험을 진행할 수 있도록 구조화했습니다. 코드 블록은 연구에서 제안된 pre-activation 설계를 기본으로 하며, 블록 제거·경로 길이·초기화 등 실험 아이디어를 통해 학습 현상을 재현해 볼 수 있습니다.[6][3][2][1]

[1](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
[2](https://arxiv.org/abs/1603.05027)
[3](https://arxiv.org/abs/1605.06431)
[4](https://kjwilber.org/static/pdfs/resnets-ensembles.pdf)
[5](https://proceedings.neurips.cc/paper/2016/file/37bc2f75bf1bcfe8450a1a41c200364c-Reviews.html)
[6](https://arxiv.org/abs/1702.08591)
[7](http://proceedings.mlr.press/v70/balduzzi17b/balduzzi17b.pdf)
[8](https://dl.acm.org/doi/10.5555/3305381.3305417)
[9](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2742.pdf)
[10](https://home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf)
[11](http://arxiv.org/pdf/2404.10947.pdf)
[12](https://users.cs.duke.edu/~tomasi/papers/yu/yuIcml19.pdf)
[13](https://arxiv.org/pdf/1603.05027.pdf)
[14](https://channelai.tistory.com/2)
[15](https://journals.lww.com/10.1097/ICL.0000000000000695)
[16](https://www.semanticscholar.org/paper/5779812a8ab116e6283a9b48755620be392987e4)
[17](http://photonics.pl/PLP/index.php/letters/article/view/13-8)
[18](https://www.semanticscholar.org/paper/f577a1d3059b76dd6e39a1a7f188dffbd5708163)
[19](https://arxiv.org/pdf/2412.14695.pdf)
[20](https://www.aclweb.org/anthology/D17-1191.pdf)
[21](http://arxiv.org/pdf/2502.16003.pdf)
[22](http://arxiv.org/pdf/1701.02362.pdf)
[23](https://arxiv.org/pdf/2110.11464.pdf)
[24](http://arxiv.org/pdf/1807.08920.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC9988239/)
[26](https://arxiv.org/pdf/1809.08959.pdf)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC5973945/)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC1484494/)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC6370471/)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230348/)
[31](http://arxiv.org/pdf/1803.10362.pdf)
[32](https://pmc.ncbi.nlm.nih.gov/articles/PMC8127140/)
[33](https://mole-starseeker.tistory.com/12)
[34](https://www.semanticscholar.org/paper/The-Shattered-Gradients-Problem:-If-resnets-are-the-Balduzzi-Frean/e79fa48078e9c1794cb67bfb1aab9557f263820f)
[35](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Wu2016ARXIV.pdf)
[36](https://sooho-kim.tistory.com/144)
[37](https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)
[38](https://pdfs.semanticscholar.org/2a24/b68ef180c0c8742bd494a55fb6f68864efed.pdf)
[39](https://openreview.net/pdf?id=HkpYwMZRb)
[40](https://www.scribd.com/document/334150809/2016-12-07-NIPS-Poster-Resnets-Behave-Like-Ensembles)
