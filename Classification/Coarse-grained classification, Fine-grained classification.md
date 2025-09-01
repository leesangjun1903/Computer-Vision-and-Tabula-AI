# Coarse-grained classification, Fine-grained classification

컴퓨터 비전과 머신러닝 분야에서 **Coarse-grained Classification**과 **Fine-grained Classification**은 분류 작업의 세밀함 정도를 나타내는 중요한 개념입니다.

## Coarse-grained Classification (거친 단위 분류)
**Coarse-grained Classification**은 큰 범주나 상위 단계에서 객체를 분류하는 방식입니다. 여기서 "Coarse-grained"는 "결이 거친", "조잡한"이라는 뜻으로, 상대적으로 큰 단위의 분류를 의미합니다.[1][2][3]

### 특징과 예시
- **큰 범주의 분류**: CIFAR-10, CIFAR-100, MNIST, ImageNet과 같은 데이터셋에서 볼 수 있는 일반적인 분류 작업[1][3]
- **뚜렷한 차이**: 서로 다른 클래스 간의 차이가 상대적으로 명확함
- **일반적인 객체 인식**: 동물, 자동차, 비행기, 개, 고양이 등의 기본 범주를 구분[4][5]

**대표적인 예시:**
- ImageNet의 1000개 클래스를 127개의 상위 범주로 그룹화[4]
- CIFAR-100의 100개 세부 클래스를 20개의 상위 클래스로 분류[5][4]
  - 예: "물고기" 상위 클래스 안에 수족관 물고기, 가자미, 가오리, 상어, 송어 등이 포함[5]

## Fine-grained Classification (세밀한 단위 분류)
**Fine-grained Classification**은 같은 상위 범주 내에서도 더욱 세밀하고 구체적인 하위 범주를 구분하는 분류 방식입니다. "Fine-grained"는 "세밀한", "정교한"이라는 뜻으로, 매우 유사한 객체들 사이의 미묘한 차이를 구분합니다.[6][1][7]

### 특징과 도전점**주요 특징:**
- **높은 클래스 내 변동성**: 같은 카테고리 내에서도 자세, 시점, 나이 등에 따른 큰 변화[7][8]
- **미묘한 클래스 간 차이**: 서로 다른 카테고리 간에도 매우 유사한 외관을 보임[7]
- **전문 지식 요구**: 라벨링에 전문적인 지식과 상당한 주석 작업이 필요[7]

**도전적인 이유:**
- 시각적으로 매우 유사한 하위 카테고리들 사이의 정확한 구분이 필요
- 배경 정보가 주의를 분산시킬 수 있어 주의집중(attention) 메커니즘이 중요[6]
- 부분 기반 특징 학습과 세밀한 특징 추출이 핵심[5]

### 대표적인 데이터셋과 응용**주요 데이터셋:**
- **Stanford Dogs**: 120종의 개 품종 분류, 20,000여 장의 이미지[9][8]
- **CUB-200-2011**: 200종의 북미 조류 종 분류[10][9]
- **FGVC Aircraft**: 항공기 모델 세부 분류
- **NABirds**: 400여 종의 북미 조류 분류[7]**실제 응용 분야:**
- 의료 진단 및 질병 분류[7]
- 생물 다양성 모니터링[7]
- 전자상거래에서의 제품 세부 분류[11]
- 음식 종류의 정확한 구분

## 주요 차이점 비교
| 구분 | Coarse-grained | Fine-grained |
|------|----------------|--------------|
| **분류 범위** | 큰 범주, 상위 클래스 | 세부 범주, 하위 클래스 |
| **클래스 간 차이** | 명확하고 뚜렷함 | 미묘하고 세밀함 |
| **데이터셋 예시** | CIFAR-10, ImageNet | Stanford Dogs, CUB-200 |
| **라벨링 난이도** | 상대적으로 용이 | 전문 지식 필요 |
| **모델 복잡도** | 상대적으로 단순 | 복잡한 특징 학습 필요 |
| **성능 평가** | 일반적으로 높은 정확도 | 더 도전적이고 낮은 정확도 |## 계층적 관계와 발전 방향두 분류 방식은 상호 배타적이지 않으며, 실제로는 **계층적 분류(Hierarchical Classification)** 구조를 형성합니다. 예를 들어:[12][11]

```
동물 (Coarse) → 개 (Coarse) → 골든 리트리버 (Fine-grained)
```

**최근 연구 동향:**
- **Coarse-to-Fine 학습**: 거친 라벨로 학습한 후 세밀한 분류로 확장[13][14][4]
- **주의집중 메커니즘**: 중요한 부분에 집중하는 어텐션 기법 개발[6][7]
- **계층적 손실 함수**: 계층 구조를 고려한 새로운 손실 함수 설계[11][5][12]

## 실용적 고려사항
**모델 선택 기준:**
- **데이터 가용성**: Fine-grained는 더 많은 라벨링된 데이터 필요
- **응용 목적**: 정확한 종 분류가 필요한지, 일반적인 범주 분류면 충분한지
- **계산 자원**: Fine-grained 분류는 더 복잡한 모델과 더 많은 계산 자원 요구
- **도메인 전문성**: Fine-grained 분류는 해당 분야의 전문 지식 필요

이러한 두 분류 방식은 각각의 장단점과 적용 분야를 가지고 있으며, 실제 문제 해결에서는 요구사항에 따라 적절한 접근 방식을 선택하거나 두 방식을 결합한 계층적 접근법을 사용하는 것이 중요합니다.

[1](https://node-softwaredeveloper.tistory.com/26)
[2](https://chaelin0722.github.io/concept/fine_coarse-grained/)
[3](https://light-tree.tistory.com/215)
[4](https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Grafit_Learning_Fine-Grained_Image_Representations_With_Coarse_Labels_ICCV_2021_paper.pdf)
[5](https://web.pkusz.edu.cn/adsp/files/2019/12/%E9%99%86%E8%B6%85%E8%B1%AA2019MMM.pdf)
[6](http://papers.neurips.cc/paper/7344-maximum-entropy-fine-grained-classification.pdf)
[7](https://arxiv.org/html/2412.19606v1)
[8](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b5e3beb791cc17cdaf131d5cca6ceb796226d832)
[9](https://www.sciopen.com/article_pdf/1397395713164926977.pdf)
[10](https://openaccess.thecvf.com/content/CVPR2024W/VDU/supplemental/Pal_Improving_Noisy_Fine-Grained_CVPRW_2024_supplemental.pdf)
[11](https://aclanthology.org/2020.ecnlp-1.10.pdf)
[12](https://www.geeksforgeeks.org/machine-learning/hierarchical-classification/)
[13](https://aclanthology.org/2021.emnlp-main.46/)
[14](https://arxiv.org/html/2406.11070v1)
[15](https://arxiv.org/abs/2507.16531)
[16](https://pubs.acs.org/doi/10.1021/acscentsci.8b00913)
[17](https://arxiv.org/html/2407.00018v1)
[18](https://pubs.acs.org/doi/10.1021/acsomega.0c05321)
[19](https://paperswithcode.com/task/fine-grained-image-classification/codeless)
[20](https://www.cecam.org/workshop-details/machine-learning-how-to-coarse-grain-26)
[21](https://www.sciencedirect.com/science/article/pii/S2666827023000105)

Coarse-grained classification 은 Cifar10, Cifar100, MNIST 등의 데이터셋을 사용해 classification 하는 것이 Coarse-grained classification 의 예시입니다.  
"Coarse-grained classification"은 데이터나 정보를 크고 일반적인 범주로 나누는 과정을 의미합니다. 이 과정은 다양한 분야에서 사용될 수 있으며, 주로 다음과 같은 특징이 있습니다:

일반화: 세부 사항보다는 주요 카테고리와 특성에 중점을 둡니다.

효율성: 데이터 처리 속도가 빠르며, 시스템의 복잡성을 줄일 수 있습니다.

응용 분야: 텍스트 분류, 이미지 분류 등 여러 분야에서 활용됩니다.



Fine-grained classification 은 Coarse-grained classification 보다 더 세밀하게 classification 을 한다고 이해할 수 있습니다. Stanford dogs 가 가장 유명한 Fine-grained classification dataset 입니다.  
Fine-grained classification 은 Coarse-grained classification 보다 상대적으로 비슷한 특징을 가진 classs 들을 분류하는 것이라고 이해할 수 있습니다.  

세밀한 분류(fine-grained classification)란, 주로 머신 러닝 및 컴퓨터 비전 분야에서 사용되는 용어로, 고유한 특징을 바탕으로 서로 유사한 개체들 사이에서 더 세부적으로 구별하는 과정을 말합니다. 예를 들어, 개의 품종을 구별할 때, 같은 종류의 개 중에서도 털 색깔, 크기, 얼굴 형태 등을 통해 더 정확하게 분류하는 것입니다.

세밀한 분류의 주요 요소는 다음과 같습니다:

특징 추출: 데이터에서 유의미한 특징을 추출하여 구별 기준을 마련합니다.

데이터 셋: 다양한 사례로 구성된 학습 데이터를 활용하여 모델의 정확도를 향상시킵니다.

알고리즘 선택: 분류 작업에 적합한 머신 러닝 알고리즘을 선택합니다.

이와 같은 세밀한 분류는 이미지 인식, 자연어 처리 등 여러 분야에서 중요한 역할을 하고 있으며, 보다 정밀한 분석과 결과를 제공합니다. 

# Reference
https://light-tree.tistory.com/215
