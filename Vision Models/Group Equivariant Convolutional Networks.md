# Group Equivariant Convolutional Networks

**핵심 요약:**  
Group Equivariant Convolutional Networks(G-CNN)은 전통적 CNN이 갖는 공간적 대칭(translation)만이 아니라 회전(rotation)과 반전(reflection) 등의 더 넓은 이산 대칭군을 네트워크 전층에 걸쳐 보존함으로써 파라미터 수를 늘리지 않고도 표현력을 크게 확장하고 일반화 성능을 크게 향상시키는 모델이다.[1]

## 1. 논문이 해결하고자 하는 문제
전통적 CNN은 공간 이동(translation) 대칭에는 회귀(equivariance)를 보장하지만 회전·반전 등 일반적인 이차원 격자의 대칭에는 그렇지 못하다. 이로 인해 데이터에 회전·반전 변화가 많을 때 학습 샘플 효율성과 일반화 성능이 저하된다.[1]

## 2. 제안하는 방법
G-CNN은 선택한 대칭군 $$G$$의 작용을 네트워크 전층에 걸쳐 보존하도록 설계된 새로운 합성곱 연산인 **G-convolution**을 도입한다.  
- 입력 함수 $$f: \mathbb{Z}^2 \to \mathbb{R}^K$$와 필터 $$\psi: \mathbb{Z}^2 \to \mathbb{R}^K$$에 대해,  
  첫 레이어 G-correlation:  

```math
    [f \star \psi](g) \;=\; \sum_{y\in \mathbb{Z}^2}\sum_k f_k(y)\,\psi_k(g^{-1}y),
``` 
  
  이후 레이어(함수 공간 $$\mathbb{Z}^2\to G$$)에서는  

```math
    [f \star \psi](g) \;=\; \sum_{h\in G}\sum_k f_k(h)\,\psi_k(g^{-1}h)
``` 
  
  로 일반화된다.[1]
- 모든 비선형층(pointwise nonlinearity), 배치 정규화, 잔차 블록, 풀링 연산 또한 $$G$$-equivariance를 만족하도록 설계되어 네트워크 전층이 대칭 작용을 보존한다.

## 3. 모델 구조
표준 CNN 구조에서 각 합성곱 레이어를 G-convolution으로 교체하며,  
- $$p4$$ 그룹(90° 회전군), $$p4m$$ 그룹(회전+반전군) 등을 사용  
- 필터 개수는 파라미터 수 유지를 위해 $$\sqrt{|G|}$$만큼 줄임  
- 마지막에는 그룹 풀링(group pooling)을 적용하여 불변성(invariance)을 얻거나, 중간층에는 풀링을 피해 표현력을 유지  

## 4. 성능 향상
- **Rotated MNIST:** 기존 최고 3.98% 오류율 대비 $$p4$$-G-CNN은 2.28% 오류율 달성.[1]
- **CIFAR-10:** 표준 All-CNN-C 9.44%→ $$p4m$$ -CNN 7.59%, ResNet44 9.45%→$$p4m$$-ResNet44 6.46%까지 개선.[1]

## 5. 한계 및 고려 사항
- **불연속 이산군 한정:** $$G$$가 이산 이면 적용 가능하나 연속 대칭군에는 직접 확장 어려움  
- **군 크기 문제:** 대칭군이 지나치게 크면 모든 변환을 열거(enumeration)하기 어려움  
- **데이터 적합도:** CIFAR-10처럼 물체가 일반적 회전 대칭을 갖지 않는 경우에도 성능이 향상되나, 실제 대칭 분포에 따라 최적의 군 선택이 필요  

## 6. 일반화 성능 향상 관점
- 네트워크가 학습한 특징이 대칭군 작용 아래 보존(equivariance)되므로, 데이터 증강 없이도 회전·반전 분포에 강건한 표현을 학습  
- **표본 복잡도(sample complexity)** 감소로 적은 학습 샘플로도 높은 일반화 성능 확보 가능  
- 깊은 층에서도 수학적 구조(structured representation)가 유지되어 고차원 특징 공간에서도 대칭 정보 보존  

## 7. 향후 연구에 미치는 영향 및 고려점
G-CNN은 대칭군 이론과 딥러닝을 결합한 구조적 표현 학습의 사례로,  
- **3D 공간군, 육각 격자(hexagonal lattice)** 등 다양한 군으로의 확장 가능  
- **연속군 대칭(continuous symmetry)** 근사화 연구 필요  
- **대칭군 선택 및 축소(disentangling)** 기법 연구를 통해 과도한 군 크기 문제 완화  
- **Transfer learning**에서 사전 학습된 G-CNN을 다른 도메인으로 활용 시 일반화 이점  

이러한 방향은 딥러닝 모델의 **추상적 유사성 추론** 능력을 높이며, 복잡한 대칭·구조 정보를 효율적으로 활용하는 후속 연구의 기반이 될 것이다.  

 1602.07576v3.pdf 파일 내용.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/42ee3c64-3187-4723-a8e5-17c15c85da72/1602.07576v3.pdf)
