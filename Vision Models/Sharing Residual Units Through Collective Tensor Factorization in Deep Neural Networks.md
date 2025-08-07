# Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks | Image classification

## 주요 주장 및 기여  
본 논문은 **Residual Unit**의 다양한 변형들을 ‘일반화된 블록 텐서 분해(Generalized Block Term Decomposition, GBTD)’ 관점에서 통합적으로 설명하고, 이를 바탕으로 **Collective Residual Unit(CRU)** 아키텍처를 제안한다. CRU는 서로 다른 Residual Unit들 사이에서 **첫 두 계층의 파라미터를 공유**함으로써 파라미터 효율을 획기적으로 높이며, ResNet-200 급 성능을 ResNet-50 모델 크기로 구현한다.

## 1. 해결하고자 하는 문제  
기존 Residual Network는 깊어질수록 성능이 향상되나,  
- 모델 크기 증가 대비 성능 향상 폭이 점점 작아지는 **파라미터 비효율**  
- 동일 구조의 여러 Residual Unit 간 **중복된 정보**  

이 두 가지 한계를 극복하는 것이 목표이다.

## 2. 제안 방법  
### 2.1. 일반화된 블록 텐서 분해(GBTD)  
- 고차원 텐서 연산자 $$X^*\in \mathbb{R}^{d_1\times d_2\times d_3\times d_4}$$를  

```math
X^* = \sum_{r=1}^R G_r \times_{3}^{\sigma} A^{(3)}_r\times_{4}^{\sigma} A^{(4)}_r
```
  
형태로 분해  
- 여기서
```math
G\_r\in\mathbb{R}^{d\_1\times d\_2\times d^*\_3\times d^*_4} ,
``` 

$$A^{(3)}_r\in\mathbb{R}^{d_3\times d^*_3}$$,  

$$A^{(4)}_r\in\mathbb{R}^{d_4\times d^*_4}$$  

- $$\times_n^{\sigma}$$는 비선형 활성화 $$\sigma$$를 포함한 mode- $$n$$ 곱

### 2.2. 기존 Residual 변형의 통합  
- $$R=1$$인 경우: **vanilla ResNet**, **Wide ResNet**  
- $$R>1$$인 경우: **ResNeXt** (cardinality $$=R$$)  
- 모든 변형을 **GBTD** 관점에서 설명 가능

### 2.3. Collective Residual Unit (CRU)  
- 깊은 네트워크에서 동일 구조의 Residual Unit $$L$$개를 쌓을 때, 네트워크 전체의 4차원 커널들을 4번째 모드(출력 채널)로 연결한 뒤  

```math
    X^+ = [X^*\_1,\ldots,X^*_L]\quad ∈ \mathbb{R}^{d_1×d_2×d_3×(L·d_4)}
``` 

- 이를 GBTD로 분해하여 **첫 두 계층**의 인자 $$A^{(3)}_r$$, $$G_r$$를 **모든 유닛에 공유**  
- 마지막 $$1×1$$ 계층 $$A^{(4)}_r$$만 유닛별로 따로 학습  
- 비선형성 보전 및 BatchNorm 문제 해결을 위해 3번째 계층 뒤에 추가 $$1×1$$ 컨벌루션 삽입  

## 3. 모델 구조  
CRU Unit (비선형 포함)  
```
Input
  ──▶ Shared 1×1 conv (채널 축소; A(3))
    ──▶ Shared 3×3 group conv (G)
      ──▶ Shared 1×1 conv (채널 복원)
        ──▶ 유닛별 1×1 conv (A(4)) + BatchNorm + Activation
          ──▶ Shortcut 합산
            ──▶ Output
```
- **Shared**: 여러 유닛 간 파라미터 공유  
- **Unshared**: 각 유닛별 독립 학습

## 4. 성능 향상  
### 4.1. ImageNet-1k  
- ResNet-50 대비  
  - CRU-Net-56 (32×4d @ conv3/4) ⇒ Top-1 21.9% (ResNet-50: 23.9%)  
- 모델 크기 유지(≈25.5M parameters)  
- ResNet-200 성능(21.7%)을 ResNet-50 크기로 달성  

### 4.2. Places365-Standard  
- ResNet-152 대비  
  - CRU-Net-116 (32×4d) ⇒ Top-1 56.60% (ResNet-152: 54.74%)  
- 모델 크기 감소(163MB vs. 226MB)  

## 5. 한계 및 모델의 일반화  
- **연산 비용 증가**: CuDNN의 group convolution 비지원으로 실제 속도 느림  
- **공유 범위 조정 필요**: 과도한 공유 시 각 유닛의 특수화 능력 저하 가능  
- **Over-fitting**: 지나치게 깊거나 큰 CRU-Net은 학습 데이터에 과적합

CRU는 **파라미터 효율**을 극대화하나, **실제 추론 속도**와 **공유 강도 최적화**를 위해 추가 연구가 필요하다.

## 6. 향후 연구에의 영향 및 고려 사항  
- **공유 설계의 일반화**: 동일 아이디어를 Transformer, RNN 등 다른 구조에 확대 적용  
- **효율적 그룹 컨벌루션 구현**: 하드웨어·라이브러리 차원에서 그룹 연산 최적화  
- **동적 공유 메커니즘**: 데이터·태스크에 따라 공유 강도를 조절하는 적응형 공유 기법  
- **일반화 성능 강화**: 공유 및 비공유 계층 비율, 위치 탐색을 통한 최적화로 과적합 완화  

이 논문은 **콘볼루션 신경망의 파라미터 중복** 문제를 **텐서 분해** 관점에서 접근함으로써, 향후 네트워크 효율화 및 공유 기반 아키텍처 연구에 중요한 이정표를 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f8e99da5-2bff-4cd1-9937-781bd6b863aa/1703.02180v2.pdf
