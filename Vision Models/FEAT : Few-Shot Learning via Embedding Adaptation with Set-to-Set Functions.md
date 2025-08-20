# FEAT : Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions | Image classification

## 1. 핵심 주장 및 주요 기여  
이 논문은 **기존의 task-agnostic 임베딩**이 새로운 클래스 간의 분별력을 최적으로 반영하지 못한다는 한계를 지적하고, **Transformer 기반의 set-to-set 함수**를 통해 지원(지원셋) 인스턴스 임베딩을 **task-specific**하게 변환함으로써 성능을 크게 향상시킨다.  
주요 기여는 다음과 같다:  
- **모델 기반 임베딩 적응(MBA)** 프레임워크 제안: 지원셋 전체를 입력으로 받아 임베딩을 co-adaptation하는 **set-to-set 함수**를 도입.  
- **다양한 함수 비교·분석**: Bi-LSTM, DeepSets, GCN, Transformer 변형을 실험해, **Transformer(FEAT)**가 가장 효과적임을 확인.  
- 표준 5-way 1-shot/5-shot 및 교차 도메인, 전이 학습(transductive), generalized few-shot, low-shot 등 확장 설정에서 **일관된 성능 개선**을 달성.  

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계  

### 2.1 해결 과제  
Few-Shot Learning(FSL)은 소수 샘플로 미지(UNSEEN) 클래스 분류기를 만드는 문제이다.  
기존 메타-러닝이나 프로토타입 네트워크는 “보편적” 임베딩 $$E(x)$$을 학습해 거리 기반 분류를 수행하나,  
– 다른 UNSEEN 클래스 조합에 대해 **분별적 특징**이 달라지므로  
– 단일 임베딩 공간이 모든 downstream 분류 작업에 최적이 아니다.  

### 2.2 제안 방법: Embedding Adaptation  
지원셋 임베딩 $$\{\phi_x\}$$을 **set-to-set 함수** $$T$$로 변환해 task-specific 임베딩 $$\{\psi_x\}$$를 얻는다.  
수식으로:  

```math
\{\psi_x\}_{x\in X_\text{train}} \;=\; T\bigl(\{\phi_x=E(x)\}_{x\in X_\text{train}}\bigr),
```

$$
\hat y_\text{test} \;=\; \arg\max_{n}\;\exp\Bigl(\gamma\cdot \mathrm{sim}(\phi_{x_\text{test}},\,\psi_{c_n})\Bigr),
\quad
\psi_{c_n} = \frac1M\sum_{y_i=n}\psi_{x_i}.
$$  

여기서 $$\mathrm{sim}$$은 cosine 유사도 또는 음의 유클리드 거리, $$\gamma$$는 온도 스케일.  

추가로 **contrastive loss**를 도입하여 같은 클래스 중심과 거리를 좁히고 타 클래스 중심과 확장시킨다.  

### 2.3 모델 구조  
– **Backbone**: ConvNet / ResNet-12 / WideResNet-28-10  
– **Set-to-Set 함수 후보**:  
  – Bi-LSTM (순서 의존, 성능 저조)  
  – DeepSets (합·최댓값 집계)  
  – GCN (유사도 그래프 전파)  
  – **Transformer (FEAT)**: self-attention으로 permutation-invariant, 풍부한 co-adaptation 지원  
– **FEAT 구성**:  
  1. 지원셋 임베딩 $$\phi$$ 계산  
  2. 단일-레이어·단일-헤드 self-attention 적용  
  3. dropout+layer-norm residual 추가  
  4. 클래스별 프로토타입 계산 및 분류  

### 2.4 성능 향상  
| 데이터셋 | 방법 | Backbone | 1-Shot 5-Way (%) | 5-Shot 5-Way (%) |
|:-------:|:----:|:--------:|:----------------:|:----------------:|
| MiniImageNet | ProtoNet | ConvNet | 52.6 → | 71.3 → |
|              | **FEAT** | ConvNet | **55.2** | **71.6** |
| TieredImageNet | ProtoNet | ResNet-12 | 68.2 → | 84.0 → |
|                | **FEAT** | ResNet-12 | **70.8** | **84.8** |
| OfficeHome (C→R) | ProtoNet | ConvNet | 29.5 → | – |
|                  | **FEAT** | ConvNet | **30.9** | – |

– **교차 도메인**(Clipart→Real): +1.4%↑  
– **Transductive FSL**: unlabeled test 포함 self-attention에 활용 시 +2.4%↑  
– **Generalized FSL**: SEEN+UNSEEN 통합 예측 성능 +4.8%↑  

### 2.5 한계  
- **Transformer 깊이·헤드 수**를 늘려도 성능 추가 향상 미미, 과적합 위험  
- 복잡도 증가: 지원셋 크기 $$N\times M$$에 따라 self-attention 계산량이 $$\mathcal{O}((NM)^2)$$  
- **Negative 예**(노이즈) 상황에서 adaptation이 오히려 분류 어려움 초래 가능  

***

## 3. 일반화 성능 향상 관점 강조  
FEAT의 **self-attention**은 지원셋 내 모든 인스턴스 관계를 모델링하여,  
- **Domain Shift**에 강인한 특징 강조  
- **Way 수 변화**(interpolation/extrapolation)에도 일관된 성능 유지  
- **Unlabeled Test** 활용 시 semi-supervised 형태로 추가 성능 개선  

이는 **task-specific** 임베딩 공간을 학습함으로써, 다양한 실제 시나리오(교차 도메인·범용 분류·대규모 low-shot)에서도 높은 일반화력을 보장한다.

***

## 4. 향후 연구 영향 및 고려 사항  
- **모델 확장**: 보다 효율적 self-attention (sparse/linearized)으로 대규모 지원셋 처리  
- **정교한 regularization**: 깊고 multi-head Transformer의 과적합 방지 기법 연구  
- **Meta-contrastive** 학습: 클래스 간 관계를 더욱 명확히 반영하는 loss 설계  
- **Cross-modal FEAT**: 텍스트·오디오 등 비전 외 도메인에도 set-to-set adaptation 적용  

FEAT는 few-shot 영역뿐 아니라 **any-to-any** 소수 샘플 학습, **유연한 task adaptation** 연구에 중요한 기반을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0ab5e2f0-5ad5-4d0b-b750-a6cddec4fffc/1812.03664v6.pdf
