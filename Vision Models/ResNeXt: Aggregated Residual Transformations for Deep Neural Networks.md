# ResNeXt: Aggregated Residual Transformations for Deep Neural Networks | Image classification

## 핵심 주장과 주요 기여

ResNeXt 논문은 딥러닝 네트워크 아키텍처에서 **"cardinality"(기수)**라는 새로운 차원을 도입하여 모델 성능을 향상시킬 수 있다는 핵심 주장을 제시한다. 기존의 depth(깊이)와 width(너비) 차원에 더해, transformation의 개수를 나타내는 cardinality가 모델의 표현력 향상에 핵심적인 역할을 한다는 것이다.[1]

주요 기여는 다음과 같다:
- 동일한 계산 복잡도를 유지하면서도 더 나은 성능을 달성하는 간단하고 모듈화된 아키텍처 제안[1]
- VGG/ResNet의 블록 반복 전략과 Inception의 split-transform-merge 전략을 결합한 새로운 접근법[1]
- ImageNet-1K에서 ResNet-101/152, Inception-v3, Inception-ResNet-v2를 능가하는 성능 달성[1]

## 해결하고자 하는 문제

### 문제 정의
기존의 네트워크 설계는 하이퍼파라미터(width, filter sizes, strides 등)의 수가 증가함에 따라 점점 더 어려워지고 있었다. 특히 Inception 모델들은 우수한 성능을 보였지만, 각 transformation마다 맞춤형 설계가 필요하고 새로운 데이터셋/태스크에 적용하기 어려운 문제가 있었다.[1]

### 제안 방법 및 수식

ResNeXt는 **Aggregated Transformations**라는 개념을 도입한다. 핵심 수식은 다음과 같다:

**기본 변환 함수:**

$$F(x) = \sum_{i=1}^{C} T_i(x)$$

여기서 $$C$$는 cardinality(변환의 개수), $$T_i(x)$$는 개별 변환 함수이다.[1]

**잔차 연결과 결합:**

$$y = x + \sum_{i=1}^{C} T_i(x)$$

여기서 $$y$$는 출력이고, 잔차 연결을 통해 입력 $$x$$가 더해진다.[1]

### 모델 구조

ResNeXt는 세 가지 동등한 형태로 구현할 수 있다:[1]

1. **Split-Transform-Merge 형태**: 입력을 여러 경로로 분할하고 각각 변환한 후 합산
2. **Early Concatenation 형태**: Inception-ResNet과 유사하지만 모든 경로가 동일한 topology를 가짐
3. **Grouped Convolution 형태**: AlexNet의 grouped convolution을 활용한 효율적 구현

핵심 설계 원칙:
- 동일한 spatial map 크기를 생성하는 블록들은 같은 하이퍼파라미터 공유
- Spatial map이 절반으로 줄어들 때마다 width를 2배로 증가[1]

## 성능 향상

### ImageNet-1K 결과
- ResNeXt-50 (32×4d): ResNet-50 대비 1.7% 향상 (22.2% vs 23.9%)
- ResNeXt-101 (32×4d): ResNet-101 대비 0.8% 향상 (21.2% vs 22.0%)
- ResNeXt-101은 ResNet-200보다 우수한 성능을 50% 복잡도로 달성[1]

### Cardinality vs Depth/Width 비교
동일한 계산 복잡도(~15 billion FLOPs)에서:
- ResNet-200: 21.7% 에러율
- Wider ResNet-101: 21.3% 에러율
- ResNeXt-101 (64×4d): 20.4% 에러율[1]

이는 cardinality 증가가 depth나 width 증가보다 더 효과적임을 보여준다.

### 다른 데이터셋에서의 성능
- **ImageNet-5K**: ResNeXt-50이 ResNet-50 대비 3.2% 향상[1]
- **CIFAR-10/100**: 3.58%/17.31% 에러율로 당시 최고 성능 달성[1]
- **COCO Detection**: Faster R-CNN에서 AP@0.5 기준 2.1% 향상[1]

## 일반화 성능 향상

### 강력한 표현력
ResNeXt는 단순히 정규화 효과가 아닌 **더 강한 표현력(stronger representations)**을 제공한다. 이는 잔차 연결이 있든 없든 일관되게 더 나은 성능을 보이는 것으로 확인되었다:[1]
- ResNeXt-50 (잔차 연결 제거): 26.1% 에러율
- ResNet-50 (잔차 연결 제거): 31.2% 에러율[1]

### 다양한 태스크에서의 일반화
ResNeXt는 다양한 시각적 인식 태스크에서 일관된 성능 향상을 보여준다:
- 분류 (ImageNet-1K, ImageNet-5K, CIFAR)
- 객체 탐지 (COCO)
- 후속 연구에서 인스턴스 분할 (Mask R-CNN)에도 활용[1]

### 설계 단순성과 확장성
- Inception 모델들보다 훨씬 단순한 설계
- 새로운 데이터셋/태스크에 쉽게 적응 가능
- 하이퍼파라미터 선택의 자유도 감소로 과적합 위험 축소[1]

## 한계

### 계산 효율성
- Grouped convolution 구현이 당시 병렬화에 최적화되지 않음
- ResNeXt-101 훈련 시간이 ResNet-101 대비 약 36% 증가 (0.95s vs 0.70s per mini-batch)[1]

### 제한된 transformation 형태
- 모든 변환이 동일한 topology를 가져야 함
- 더 다양한 transformation 조합의 가능성은 탐구되지 않음[1]

## 향후 연구에 미치는 영향

### 긍정적 영향
1. **Cardinality 개념의 확산**: 네트워크 설계에서 새로운 차원으로 cardinality가 널리 인식됨
2. **Grouped Convolution의 재조명**: 단순한 엔지니어링 타협책에서 성능 향상 도구로 재평가
3. **효율적 아키텍처 설계**: 복잡도를 증가시키지 않고도 성능을 향상시키는 방법론 제시
4. **후속 연구의 기반**: MobileNet, ShuffleNet 등의 효율적 네트워크 아키텍처 발전에 기여

### 향후 연구 시 고려사항

**기술적 측면:**
- 더 효율적인 grouped convolution 구현 필요
- 다양한 transformation topology 조합 탐구
- 하드웨어 최적화를 고려한 cardinality 설정

**이론적 측면:**
- Cardinality와 다른 하이퍼파라미터 간의 상호작용 분석
- 최적 cardinality 결정을 위한 이론적 가이드라인 개발
- 다양한 태스크별 cardinality 효과 분석

**실용적 측면:**
- 메모리 제약 환경에서의 cardinality 활용 전략
- 전이 학습 시나리오에서의 ResNeXt 최적화
- AutoML과 결합한 자동 cardinality 튜닝 방법론

ResNeXt는 네트워크 아키텍처 설계에서 "더 깊게 또는 더 넓게"를 넘어서는 "더 많은 변환으로"라는 새로운 패러다임을 제시했으며, 이는 현재까지도 효율적인 딥러닝 모델 설계의 핵심 원칙 중 하나로 자리잡고 있다.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/06abc67b-0803-4989-808a-1856aaa74db7/1611.05431v2.pdf
