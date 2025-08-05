# Identity Mappings in Deep Residual Networks | Image classification

## 핵심 주장 및 주요 기여  
**Identity Mappings in Deep Residual Networks** 논문은 **스킵 연결(skip connection)** 과 **추가 후 활성화 함수(post-activation)** 의 역할을 수리적으로 분석하여, 정보가 네트워크 전반에 “간결한(identity) 경로”로 전달될 때 학습이 용이해지고 일반화 성능이 개선된다는 점을 증명한다.  
1. **정형화된 전·후방 전파 식 도출**:  
   - Residual Unit을  

$$
       y_l = h(x_l) + F(x_l, W_l),\quad x_{l+1} = f(y_l)
     $$  
     
에서 $$h(x_l)=x_l$$, $$f(y_l)=y_l$$일 때,  

$$
       x_L = x_l + \sum_{i=l}^{L-1} F(x_i,W_i),\quad
       \frac{\partial E}{\partial x_l} 
       = \frac{\partial E}{\partial x_L}\bigl(1 + \sum_{i=l}^{L-1}\tfrac{\partial F}{\partial x_l}\bigr).
     $$  
     
이를 통해 스킵 연결이 정보 흐름에 미치는 이점을 수식으로 명확히 제시했다.  
2. **Pre-activation Residual Unit 제안**:  
   - 활성화(BN, ReLU)를 합산 후(post-activation)가 아니라, 합산 전(pre-activation)에 배치하여 $$f(y_l)=y_l$$이 되도록 재설계.  
   - 이 구조는 수렴 속도를 크게 높이고 과적합을 줄여, 1001-layer CIFAR-10에서 에러율을 7.61%→4.62%로 개선했다.  
3. **광범위한 실험적 검증**:  
   - CIFAR-10/100 및 ImageNet에서 다양한 스킵 연결(스케일링·게이팅·1×1 conv·dropout)과 활성화 순서 대안 실험을 통해, 정교한 설계가 최적화 난이도와 일반화에 미치는 영향을 체계적으로 평가했다.

## 문제 정의  
- **딥 네트워크 최적화의 어려움**: 층 수가 늘어날수록 경사 소실(vanishing gradient) 및 정보 왜곡 문제가 심화되어 학습이 불안정하고 정확도 포화 혹은 감소 현상이 발생.  
- **기존 Residual Unit 한계**: 스킵 연결은 있지만, 합산 후 ReLU(followed-by-ReLU) 구조로 인해 음수 신호가 잘려나가며 정교한 아이덴티티 통로가 깨짐.

## 제안 방법  
1. **수학적 분석**  
   - 전방 전파:  

$$
       x_{l+1} = x_l + F(x_l,W_l) 
       \;\Rightarrow\;
       x_L = x_l + \sum_{i=l}^{L-1}F(x_i,W_i)
     $$

- 후방 전파:  

$$
       \frac{\partial E}{\partial x_l}
       = \frac{\partial E}{\partial x_L}
         \Bigl(1 + \sum_{i=l}^{L-1}\frac{\partial F}{\partial x_l}\Bigr).
     $$  
   
   - 스케일링·게이팅 등 비정형 스킵 연결 시 계수 곱이 지수적으로 작아지거나 커져 학습이 어려워짐을 실험적으로 확인.  
2. **Pre-activation Residual Unit**  
   - 구성: **BN → ReLU → weight layer** ×2 → addition → (identity)  
   - 수식:  

$$
       x_{l+1} = x_l + F(\,\hat f(x_l),W_l),\quad \hat f(x)=\text{BN}\circ\text{ReLU}(x).
     $$  
   - 장점: 스킵 경로에 어떠한 활성화·정규화 연산도 개입하지 않아 정보가 순수하게 전달되고, 모든 weight 입력에 BN을 적용해 정규화 효과 극대화.  

## 모델 구조  
- **Residual Unit 옵션**  
  - Original: weight→BN→ReLU→weight→BN + skip → ReLU  
  - Pre-activation: BN→ReLU→weight→BN→ReLU→weight + skip → identity  
- **네트워크 깊이**  
  - CIFAR: 110-, 164-, 1001-layer ResNet  
  - ImageNet: 152-, 200-layer ResNet  

## 성능 향상  
| 데이터셋   | 네트워크        | Original 에러율 | Pre-act 에러율   |
|-----------|----------------|---------------|-----------------|
| CIFAR-10  | ResNet-110     | 6.61%         | 6.37%           |
| CIFAR-10  | ResNet-164     | 5.93%         | 5.46%           |
| CIFAR-10  | ResNet-1001    | 7.61%         | **4.62%**       |
| CIFAR-100 | ResNet-164     | 25.16%        | 24.33%          |
| CIFAR-100 | ResNet-1001    | 27.82%        | **22.71%**      |
| ImageNet  | ResNet-200     | 21.8%↑        | **20.7%**       |  

- **일반화 개선**: Pre-activation 모델은 수렴 시 훈련 손실이 다소 높으나, 테스트 오차가 낮아 과적합이 줄어듦을 확인.  
- **최적화 용이**: 1000층 규모에서도 수렴 곡선이 빠르게 하강하며 학습 안정화.

## 한계  
- **계산 비용 증가**: 깊이에 비례해 학습 시간·메모리 요구량 증가(예: ImageNet ResNet-200에 8GPU 3주 소요).  
- **스페셜 케이스 처리**: 네트워크 첫·마지막 Residual Unit만 별도 pre-activation 적용 설계가 필요.  
- **구조적 확장성**: 향후 다양한 블록 형태(다중 분기, 비전 트랜스포머 등)로의 일반화 검증 필요.

## 일반화 성능 향상 관점  
- **BN 정규화 효과 강화**: 모든 weight 입력 직전에 BN을 적용해 내부 공변량 변화(internal covariate shift) 감소, 과적합 억제.  
- **클린 스킵 경로**: 역전파 시 지수적 감소·증폭 없이 잔차만 누적되므로 깊은 층에서 정보 손실 최소화.  
- **실험적 증거**: CIFAR-100, ImageNet의 테스트 오차 하락 및 훈련-테스트 오류 간 격차 축소.

## 향후 연구에 미치는 영향 및 고려 사항  
- **초고층 네트워크 설계**: 1000층 이상의 모델 학습 가능성 열어둠.  
- **활성화 순서 일반화**: 다양한 활성화 함수·정규화 기법과 pre-activation 조합 탐색 필요.  
- **경량화와 효율화**: 연산 복잡도 감소, 메모리 최적화를 고려한 효율적 스킵 연결 구조 연구.  
- **다양한 도메인 적용**: 객체 검출, 분할, 비전-언어 모델 등으로 pre-activation 블록 확장 검증.  

위 논문은 **잔차 학습(residual learning)** 의 핵심 설계 원칙을 수학적으로 뒷받침하고, 이후 대부분의 딥러닝 모델이 따르는 **pre-activation** 구조를 제시했다는 점에서 딥러닝 아키텍처 연구에 지대한 영향을 미쳤다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f024f3b2-aec2-48fc-9ff6-c309c674ea0d/1603.05027v3.pdf
