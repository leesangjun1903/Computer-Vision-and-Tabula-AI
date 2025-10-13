# Energy-based Out-of-distribution Detection | 2020 · 1933회 인용, OOD Detection

## 1. 핵심 주장과 주요 기여

이 논문의 핵심 주장은 기존의 **softmax confidence score 대신 energy score를 사용하면 OOD(Out-of-Distribution) 탐지 성능을 크게 향상시킬 수 있다**는 것입니다. 주요 기여는 다음과 같습니다:[1]

**주요 기여**:
- **통합된 energy 기반 OOD 탐지 프레임워크** 제안: 사전 훈련된 분류기에서 재훈련 없이 사용 가능한 scoring function과 모델 fine-tuning을 위한 trainable cost function 두 가지 방식 모두 지원[1]
- **이론적 우수성 증명**: Energy score가 입력 데이터의 확률 밀도와 이론적으로 정렬되어 있어 softmax confidence score보다 우수함을 수학적으로 입증[1]
- **실증적 성능 향상**: CIFAR-10 기준 WideResNet에서 energy score 사용 시 softmax 대비 평균 FPR@95TPR을 18.03% 개선[1]

## 2. 해결하고자 하는 문제와 제안 방법

**해결하려는 문제**:
신경망이 **훈련 데이터와 다른 분포의 입력(OOD)에 대해 과도하게 높은 confidence를 보이는 문제**입니다. 기존 softmax confidence score는 OOD 데이터에 대해서도 임의로 높은 값을 출력하여 안전한 AI 시스템 구축을 어렵게 만듭니다.[1]

**제안하는 방법**:

### Energy Score 정의
Energy score는 다음과 같이 정의됩니다:

$$E(x; f) = -T \log \sum_{i=1}^{K} e^{f_i(x)/T}$$

여기서 $$f(x)$$는 neural classifier의 logit 출력, $$T$$는 temperature parameter입니다.[1]

### Energy-bounded Learning Objective
Fine-tuning을 위한 학습 목표는 다음과 같습니다:

$$\min \sum_{(x,y) \in D_{train}^{in}} -\log F_y(x) + L_{energy}$$

여기서 에너지 정규화 손실은:

$$L_{energy} = \sum_{(x_{in},y) \in D_{train}^{in}} \max(0, E(x_{in}) - m_{in})^2 + \sum_{x_{out} \in D_{train}^{out}} \max(0, m_{out} - E(x_{out}))^2$$

이 손실함수는 **두 개의 squared hinge loss 항**을 사용하여 in-distribution 데이터에는 낮은 에너지를, OOD 데이터에는 높은 에너지를 할당하도록 에너지 표면을 명시적으로 형성합니다.[1]

## 3. 모델 구조

**기본 구조**:
- **기존 discriminative 분류 모델 활용**: WideResNet 등 표준 CNN 아키텍처 사용[1]
- **Parameter-free 추론**: 기존 모델에서 재훈련 없이 energy score 직접 계산 가능[1]
- **Fine-tuning 옵션**: 보조 OOD 데이터와 energy-bounded learning으로 성능 추가 향상[1]

**모델의 두 가지 사용 방식**:
1. **추론 시점 사용**: 사전 훈련된 모델에서 hyperparameter 조정 없이 바로 적용
2. **훈련 시점 사용**: Energy-bounded learning으로 모델 fine-tuning하여 더 나은 성능 달성[1]

## 4. 성능 향상 및 한계

**성능 향상**:
- **CIFAR-10**: Energy score가 softmax 대비 평균 FPR@95TPR 18.03% 개선[1]
- **CIFAR-100**: Energy fine-tuning이 Outlier Exposure 대비 10.55% 개선[1]
- **다양한 OOD 데이터셋**: 6개 벤치마크에서 일관된 성능 향상 확인[1]
- **분류 정확도 유지**: Energy fine-tuning 후에도 in-distribution 분류 성능은 거의 동일하게 유지(CIFAR-10에서 4.87% vs 5.16%)[1]

**한계점**:
논문에서 명시적으로 언급된 주요 한계는 다음과 같습니다:
- **Temperature scaling 민감성**: 높은 temperature 값 사용 시 예측이 더욱 균등 분포화되어 energy score의 구별력이 감소[1]
- **Margin parameter 최적화**: Energy fine-tuning에서 $$m_{in}$$이 너무 작으면 최적화 어려움과 성능 저하 발생[1]

## 5. 일반화 성능 향상 가능성

**이론적 근거**:
Energy score는 **입력 데이터의 확률 밀도와 이론적으로 정렬**되어 있어 더 나은 일반화 성능을 제공합니다. 반면 softmax confidence score는 최대 logit 값으로 shifted된 biased scoring function으로, 밀도와 정렬되지 않아 OOD 탐지에 부적합합니다.[1]

**실증적 증거**:
- **부드러운 분포 형성**: Energy score는 사전 훈련된 네트워크에서도 in-distribution과 OOD 데이터에 대해 자연스럽게 부드러운 분포를 형성[1]
- **강건한 성능**: 다양한 OOD 테스트 데이터셋에서 일관된 성능 향상으로 일반화 능력 입증[1]
- **생성 모델 대비 우수성**: JEM 등 하이브리드 생성 모델보다도 우수한 성능 달성[1]

## 6. 향후 연구에 미치는 영향과 고려사항

**향후 연구 영향**:
- **확장 가능성**: 논문은 이미지 분류를 넘어 **active learning 등 다른 머신러닝 태스크**로의 확장을 제안[1]
- **에너지 기반 관점**: OOD 불확실성 추정에 대한 **더 넓은 에너지 기반 관점**으로의 관심 확대 기대[1]
- **안전한 AI 시스템**: 자율주행차, 의료진단 등 **안전 중요 애플리케이션**에서의 신뢰할 수 있는 불확실성 추정 도구 제공[1]

**연구 시 고려할 점**:
1. **계산 효율성**: Energy score는 logsumexp 연산으로 간단하게 계산 가능하여 실용적[1]
2. **하이퍼파라미터 설정**: Fine-tuning 시 margin parameter들($$m_{in}$$, $$m_{out}$$)의 적절한 선택 필요[1]
3. **보조 OOD 데이터**: 더 나은 성능을 위해서는 적절한 보조 outlier 데이터 확보 중요[1]
4. **도메인 적용**: 각 도메인별 특성에 맞는 energy threshold 설정 및 검증 필요[1]

이 논문은 **실용적이면서도 이론적으로 견고한 OOD 탐지 방법**을 제시하여, 안전한 AI 시스템 구축을 위한 중요한 기초를 마련했다고 평가됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e4c8bb1f-29ac-4b83-905d-a570210b61cf/2010.03759v4.pdf)
