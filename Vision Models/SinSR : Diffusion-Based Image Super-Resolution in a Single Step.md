# SinSR: Diffusion-Based Image Super-Resolution in a Single Step

# Abstract
확산 모델을 기반으로 한 초해상도(SR) 방법은 유망한 결과를 보이지만, 필요한 추론 단계의 수가 많기 때문에 실용적인 적용이 방해받고 있습니다.  
최근의 방법은 초기 상태에서 열화된 이미지를 활용하여 마르코프 체인을 단축합니다.  
그럼에도 불구하고 이러한 솔루션은 열화 과정의 정확한 공식화에 의존하거나 여전히 비교적 긴 생성 경로(예: 15회 반복)를 필요로 합니다.  
추론 속도를 향상시키기 위해 단일 단계 SR 생성을 달성하기 위한 간단하지만 효과적인 방법인 SinSR을 제안합니다.  
구체적으로, 먼저 확산 기반 SR을 가속화하기 위한 가장 최근의 최첨단(SOTA) 방법에서 결정론적 샘플링 프로세스를 도출합니다.  
이를 통해 입력된 무작위 노이즈와 생성된 고해상도 이미지 간의 매핑을 훈련 중에 감소되고 허용 가능한 수의 추론 단계로 얻을 수 있습니다.  
우리는 이 deterministic 매핑을 단 한 번의 추론 단계 내에서 SR을 수행하는 student model로 증류할 수 있음(큰 모델(Teacher Network)로부터 증류한 지식을 작은 모델(Student Network)로 transfer하는 일련의 과정, https://baeseongsu.github.io/posts/knowledge-distillation/) 을 보여줍니다.  
또한 증류 과정에서 실제 이미지를 동시에 활용하여 Student Network의 성능이 Teacher Network의 feature 다양체에만 국한되지 않도록 하는 새로운 novel consistency-preserving loss을 제안하여 추가적인 성능 향상을 가져올 수 있습니다.  
종합 데이터 및 실제 데이터 세트에 대해 수행된 광범위한 실험은 제안된 방법이 단 한 번의 샘플링 단계에서 이전 SOTA 방법과 교사 모델에 비해 비슷하거나 심지어 우수한 성능을 달성하여 최대 10배의 놀라운 추론 속도 향상을 달성할 수 있음을 보여줍니다.

# Introduction
이미지 초해상도(SR)는 주어진 저해상도(LR) 대응물로부터 고해상도 이미지를 재구성하는 것을 목표로 합니다[44].  
최근 복잡한 분포를 모델링하는 효과를 보여준 Diffusion 모델은 널리 채택되고 있으며, 특히 지각적인 품질 측면에서 SR 작업에서 놀라운 성능을 입증했습니다.  

특히, 현재 Diffusion 모델을 사용하기 위한 전략은 Diffusion 모델의 denoiser 에서 LR 이미지를 입력에 연결하는 것[31, 32](LDM, SR3)과 사전 훈련된 확산 모델의 inverse process(DDRM)를 조정하는 두 가지 스트림으로 크게 분류할 수 있습니다[4, 5, 13](Conditional Diffusion Models).  
유망한 결과를 얻었음에도 불구하고 두 전략 모두 계산 비용 문제에 직면합니다.  
특히, 이러한 조건부 확산 모델의 초기 상태는 LR 이미지의 사전 지식을 사용하지 않은 순수한 가우시안 노이즈입니다.  
즉, 정해진 열화방식으로만 좋은 성능이 나온다는 것이죠.   
따라서 만족스러운 성능을 달성하려면 상당한 수의 추론 단계가 필요하며, 이는 확산 기반 SR 기술의 실제 적용을 크게 방해합니다.  

확산 모델의 샘플링 효율을 향상시키기 위한 노력이 기울여져 다양한 기술이 제안되었습니다[21, 27, 36](Improved DDPM, DDIM).  
그러나 높은 충실도를 유지하는 것이 중요한 Low-level Vision 영역에서는 이러한 기술이 성능을 희생하여 가속을 달성하기 때문에 종종 부족한 면이 있습니다.  
최근에는 초기 확산 상태의 신호 대 잡음비를 개선하여 마르코프 체인을 단축하는 데 중점을 둔 이미지 복원 작업에서 확산 프로세스를 재구성하는 혁신적인 기술이 등장했습니다.  
예를 들어, [42] (Exposurediffusion)는 입력 노이즈 이미지로 노이즈 제거 확산 프로세스를 시작하고, [45] (ResShift)는 초기 단계를 LR 이미지와 무작위 노이즈의 조합으로 모델링합니다.  
그럼에도 불구하고 이러한 가장 최근의 작업에서도 [42, 45]는 한계가 지속됩니다.  
예를 들어, [42]는 단 세 번의 추론 단계 내에서 유망한 결과를 보여주지만 이미지 저하 프로세스의 명확한 공식화가 필요합니다.  
또한 [45]는 여전히 15개의 추론 단계가 필요하며 추론 단계 수를 더 줄이면 눈에 띄는 아티팩트로 성능이 저하됩니다.

이러한 문제를 해결하기 위해 그림 1 및 그림 2와 같이 확산 모델의 다양성과 지각 품질을 손상시키지 않고 한 번의 샘플링 단계에서만 고해상도(HR) 이미지를 생성할 수 있는 새로운 접근 방식을 소개합니다.  

![image](https://github.com/user-attachments/assets/8bcd4191-a99e-43d9-b498-dd6a558d59f8)

![image](https://github.com/user-attachments/assets/3abfdbb1-caff-45eb-9978-8fb3d15c1ed4)

특히, 우리는 Teacher Diffusion 모델에서 입력된 무작위 노이즈와 생성된 HR 이미지 사이의 잘 페어링된 양방향 결정론적 매핑을 직접 학습할 것을 제안합니다.  
잘 일치하는 훈련 데이터의 생성을 가속화하기 위해, 먼저 원래의 확률적 공식에서 확산 기반 SR을 가속화하기 위해 설계된 최신 작업[45]에서 deterministic sampling strategy을 도출합니다.  
또한, 실제 이미지를 활용하기 위해 novel consistency-preserving loss를 새롭게 제안하여 실제(GT) 이미지와 예측된 초기 상태에서 생성된 이미지 간의 오차를 최소화하여 생성된 HR 이미지의 지각 품질을 더욱 향상시킵니다.  
실험 결과는 우리의 방법이 SOTA 방법 및 Teacher Diffusion 모델[45]과 비교했을 때 비슷하거나 훨씬 더 나은 성능을 달성하는 동시에 추론 단계 수를 15단계에서 1단계로 크게 줄여 추론 속도를 최대 10배까지 향상시킨다는 것을 보여줍니다.

주요 기여 사항은 다음과 같이 요약됩니다:  
• 우리는 확산 기반 SR 모델을 처음으로 비슷하거나 더 우수한 성능을 가진 단일 추론 단계로 가속화합니다. 생성 프로세스의 마르코프 체인을 단축하는 대신, deterministic generation function를 Student Network에 직접 증류하는 간단하면서도 효과적인 접근 방식을 제안합니다. 

• 훈련을 더욱 강화하기 위해 SR 작업을 가속화하여 잘 일치하는 훈련된 쌍을 효율적으로 생성할 수 있도록 하는 최근 SOTA 방법[45]에서 deterministic sampling strategy을 도출합니다. 

• 우리는 훈련 중에 실제 이미지를 활용하여 Student Model이 Teacher Diffusion Model의 결정론적 매핑에만 집중하지 못하도록 하여 더 나은 성능으로 이어질 수 있는 새로운 novel consistency-preserving loss를 제안합니다. 

• 합친 생성된 데이터 세트와 실제 데이터 세트에 대한 광범위한 실험은 우리가 제안한 방법이 SOTA 방법 및 Teacher diffusion 모델과 비슷하거나 심지어 우수한 성능을 달성하는 동시에 추론 단계의 수를 15단계에서 1단계로 크게 줄일 수 있음을 보여줍니다.

# Related Work
## Image Super-Resolution
딥 러닝의 부상으로 딥 러닝 기반 기술은 점차 SR 작업의 주류가 되었습니다[8, 44] (SRCNN, SR survey).  
초기 작업의 일반적인 접근 방식 중 하나는 쌍을 이룬 학습 데이터[1, 2, 15, 43] (VDSR, 를 사용하여 회귀 모델을 훈련하는 것입니다.  
사후 분포(조건부 분포)에 대한 기대치는 잘 모델링될 수 있지만 지나치게 매끄러운(over-smooth) 문제[16, 26, 33]로 인해 어려움을 겪을 수밖에 없습니다.  
생성된 HR 이미지의 지각 품질을 개선하기 위해 생성 기반 SR 모델은 autoregressive-based 모델[6, 25, 28, 29]과 같이 점점 더 많은 관심을 받고 있습니다.  
상당한 개선이 이루어지지만 자동 회귀 모델의 계산 비용은 일반적으로 큽니다.  
그 후, normalizing flows[22, 41]은 효율적인 추론 프로세스에서 좋은 지각 품질을 갖는 것으로 입증된 반면, 네트워크 설계는 가역성과 계산 용이성의 요구 사항에 의해 제한됩니다.  
또한 GAN 기반 방법은 지각 품질 측면에서도 큰 성공을 거두었습니다[9, 12, 16, 26, 33] (Conditional PixelCNN, EnhanceNet).  
그러나 GAN 기반 방법의 훈련은 일반적으로 불안정합니다. 최근 Diffusion 기반 모델은 SR[4, 5, 13, 31, 32]에서 널리 연구되고 있습니다.  
Diffusion 기반 SR 방법은 크게 두 가지 범주로 요약할 수 있으며, 이는 LR 이미지를 노이즈 제거기의 입력과 연결하고 사전 학습된 Diffusion 모델[4, 5, 13]의 Reverse 프로세스를 수정합니다.  
유망한 결과가 달성되지만 많은 수의 추론 단계에 의존하기 때문에 Diffusion 기반 모델의 적용을 크게 방해합니다.

## Acceleration of Diffusion Models
최근 Diffusion 모델의 가속화는 점점 더 많은 관심을 받고 있습니다.  
일반 확산 모델[21, 27, 36, 37]에 대해 몇 가지 알고리즘이 제안되었으며 이미지 생성에 매우 효과적인 것으로 입증되었습니다.  
그중 하나의 직관적인 전략은 확산 모델을 학생 모델로 증류하는 것입니다.  
그러나 추론 프로세스의 일반 미분 방정식(ODE)을 풀기 위한 막대한 학습량으로 인해 대규모 데이터 세트에서 이 체계의 매력이 떨어집니다[23].  
학습량를 완화하기 위해 일반적으로 progressive distillation strategies이 채택된다[24, 34].  
한편, 증류를 통해 Teacher Diffusion 모델의 동작을 단순히 시뮬레이션하는 대신 반복적인 방식으로 더 나은 추론 경로가 탐색된다[19, 20].  
progressive distillation는 학습량을 효과적으로 감소시키는 반면, 오류가 동시에 누적되어 SR에서 명백한 성능 손실로 이어집니다.  
가장 최근에는 이미지 복원 작업을 대상으로 일부 작업은 degradation process[42] 또는 초기 state[45]의 사전 정의된 분포를 사용하여 확산 프로세스를 재구성하여 생성 프로세스의 마르코프 체인을 단축하고 low level vision 작업에 DDIM[36]을 직접 적용하는 것보다 더 나은 성능을 제공합니다.  
그러나 degradation에 대한 명확한 공식이 필요하거나 여전히 상대적으로 많은 추론 단계가 필요합니다.

# Motivation
## Preliminary
LR 이미지 y와 해당 HR 이미지 $x_0$가 주어졌을 때, 기존 확산 기반 SR 방법은 일반적으로 초기 상태 $$x_T \sim \mathcal{N}(0, I)$$ 인 순방향 프로세스가 일반적으로 $$q\left(x_t \mid x_{t-1}\right)=\mathcal{N}\left(x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t I\right)$$ 으로 정의되는 마르코프 체인을 통해 조건부 분포 $q(x_0|y)$를 모델링하는 것을 목표로 합니다.  
Diffusion 모델의 역할은 입력 도메인(표준 가우시안 노이즈)을 HR 이미지에 넣어주어 조건화된 LR 이미지 도메인으로 전달하는 것으로 간주할 수 있습니다.
$x_T$와 $x_0$의 매칭 관계는 알 수 없기 때문에 일반적으로 $x_T$와 $x_0$ 사이의 알려지지 않은 매핑을 학습/추론하려면 반복적인 방식을 통한 확산 모델[10, 20, 32]이 필요합니다.
우리의 방법은 조건부 분포 $q(x_0|y)$를 효과적으로 캡처하고 LR 이미지 $y$ 가 주어지면 $$x_T \text { 과 } \hat{x} 0$$ 사이의 결정론적 매핑을 설정하는 SR 모델을 사용하면 그림 3과 같이 $$f_{\hat{\theta}}$$ 로 표시된 다른 네트워크를 사용하여 추론 프로세스를 단일 단계로 간소화하여 $\hat{x}_0$ 과 $x_T$ 사이의 대응을 학습할 수 있다는 아이디어에 근거합니다.

![image](https://github.com/user-attachments/assets/38d1c76e-4f97-4e1e-99cc-9becb4d1b65b)

$\hat{x}_0$ : 생성된 고화질 이미지

## Distillation for diffusion SR models: less is more.
Student Network에 대한 $\hat{x}_0$ 와 $x_T$ 간의 매핑을 증류하는 개념은 이전에 탐색된 적이 있지만[20] SR에 적용하면 몇 가지 문제가 발생합니다:  
- 이전 모델의 많은 추론 단계로 인해 학습량이 한 단계 증류에 의해 상당해집니다(예: LDM[31]은 추론을 위해 DDIM[36]을 사용한 후에도 여전히 100단계가 필요하며, 이는 Student Model의 학습 데이터로 고품질 쌍($\hat{x}_0$, $x_T$, $y$)들을 생성합니다.  
- 성능 저하는 반복을 포함하는 보다 복잡한 증류 전략의 도입에 기인합니다.  
예를 들어, 학습량을 줄이기 위해 iterative distillation strategy[34]을 채택하여 학습 중 추론 단계의 수를 점진적으로 줄입니다.  
그러나 생성 작업에서 만족스러운 결과를 얻었음에도 불구하고 SR 작업이 이미지 품질에 상대적으로 더 민감하기 때문에 누적되는 오류는 SR 결과의 fidelity에 상당한 영향을 미칩니다.  
앞서 언급한 두 가지 과제를 해결하기 위해, 우리는 다음 관찰을 기반으로 Diffusion SR 프로세스를 간단하지만 효과적인 방법으로 single step으로 증류할 것을 제안합니다.  
관찰에 대한 자세한 내용은 5.3항에서 확인할 수 있습니다.

우리는 100개의 DDIM 단계에서 LDM[31]과 15단계로 비슷한 성능을 달성하는 확산 기반 SR[45]을 가속화하는 가장 최근의 SOTA 방법이 $x_T$ 와 $x_0$ 사이에 결정론적 매핑을 가지고 있음을 보여줍니다.  
또한 추론 단계의 수가 크게 감소하고 결정론적 매핑의 존재로 인해 그림 6 및 표 4와 같이 단일 단계 증류를 학습할 수 있습니다.  

![image](https://github.com/user-attachments/assets/3a9b919b-8278-41da-9209-c2e4bf7f48d1)

![image](https://github.com/user-attachments/assets/60c169eb-7f80-4c53-a653-7e0373f68451)

- $x_T$와 $\hat{x}_0$ 간의 매핑을 학습하는 것은 표 5와 같은 다양한 노이즈 레벨에서 $x_t$를 노이즈 제거하는 것보다 더 쉬운 것으로 밝혀졌습니다.  
따라서 반복 증류에 의한 누적 오류를 피할 수 있도록 $x_T$와 $\hat{x}_0$ 간의 매핑을 직접 학습하는 것이 가능합니다.

![image](https://github.com/user-attachments/assets/2d49f028-f064-4694-948b-4dde5ef35431)

- 누적된 오류로 인해 보다 정교한 증류 전략(반복 기반)은 표 6과 같은 설정 개선에 기여하지 않습니다.

![image](https://github.com/user-attachments/assets/8340d82e-800c-43f1-810c-3eddb538f2ad)

다음 섹션의 구성은 다음과 같습니다:
우리는 먼저 추론 과정이 원래 확률적인 ResShift[45]가 Sec 4.1에서 재학습 없이 결정론적 모델로 변환될 수 있음을 입증한 다음 Sec 4.2에서 제안된 consistency preserving distillation를 유지합니다.

# Methodology
## Deterministic Sampling
ResShift[45]와 LDM[31]의 핵심 차이점은 초기 상태 $x_T$의 공식화입니다.  
구체적으로, ResShift[45]에서 LR 이미지 y의 정보는 다음과 같이 확산 단계 $x_t$에 통합됩니다.

$$
q\left(x_t \mid x_0, y\right)=\mathcal{N}\left(x_t ; x_0+\eta_t\left(y-x_0\right), \kappa^2 \eta_t \mathbf{I}\right)
$$

여기서 $\eta_t$는 시간 단계 t에 따라 단조롭게 증가하는 일련의 하이퍼 파라미터로, $\eta_T \rightarrow 1$ 및 $\eta_0 \rightarrow 0$을 따릅니다.  
따라서 확산 과정의 역은 다음과 같이 LR 이미지 y의 풍부한 정보를 가진 초기 상태에서 시작하여 $$x_T=y+\kappa \sqrt{\eta_T} \epsilon$$ 를 따르고, 여기서 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 입니다. 

주어진 이미지 y에서 HR 이미지 x를 생성하려면 [45]의 원래 역 과정은 다음과 같습니다:

$$
p_\theta\left(x_{t-1} \mid x_t, y\right)=\mathcal{N}\left(x_{t-1} \mid \mu_\theta\left(x_t, y, t\right), \kappa^2 \frac{\eta_{t-1}}{\eta_t} \alpha_t \mathbf{I}\right)
$$

여기서 $\mu_\theta\left(x_t, y, t\right)$는 심층 네트워크에 의해 reparameterize(예측)됩니다.  
위 방정식에서 볼 수 있듯이 초기 상태 $x_T=y+\kappa \sqrt{\eta_T} \epsilon$이 주어졌을 때 생성된 이미지는 $p_\theta\left(x_{t-1} \mid x_t, y\right)$에서 샘플링하는 동안 무작위 노이즈가 존재하기 때문에 확률적입니다.  
DDIM 샘플링[36]에서 영감을 받아, 우리는 사전 훈련된 모델에 직접 채택할 수 있도록 marginal distribution(주변확률분포) $q\left(x_t \mid x_0, y\right)$를 변경하지 않고 유지하는 non-Markovian reverse 프로세스 $q\left(x_t \mid x_0, y\right)$가 존재한다는 것을 발견했습니다.  
재구성된 결정론적 역과정은 다음과 같습니다:

$$
q\left(x_{t-1} \mid x_t, x_0, y\right)=\delta\left(k_t x_0+m_t x_t+j_t y\right)
$$

여기서 $δ$는 단위 임펄스이며, $k_t$, $m_t$, $j_t$는 다음과 같습니다:

$$
\left\lbrace\begin{array}{c}
m_t=\sqrt{\frac{\eta_{t-1}}{\eta_t}} \\
j_t=\eta_{t-1}-\sqrt{\eta_{t-1} \eta_t} \\
k_t=1-\eta_{t-1}+\sqrt{\eta_{t-1} \eta_t}-\sqrt{\frac{\eta_{t-1}}{\eta_t}}
\end{array} 
\right.
$$

파생에 대한 자세한 내용은 보충 자료에서 확인할 수 있습니다.  
결과적으로 추론을 위해 y를 조건으로 하는 역 프로세스는 다음과 같이 재구성됩니다:

```math
\begin{aligned}
x_{t-1} & =k_t \hat{x}_0+m_t x_t+j_t y \\
& =k_t f_\theta\left(x_t, y, t\right)+m_t x_t+j_t y
\end{aligned}
```

여기서 $f_\theta\left(x_t, y, t\right)$ 는 사전 훈련된 ResShift[45] 모델에서 예측된 HR 이미지입니다.  
$x_{t-1}$ 의 재구성된 프로세스에서 샘플링함으로써 $x_T$(또는 $ε$)와 $\hat{x} 0$ 사이의 결정론적 매핑을 얻을 수 있으며 $F_\theta\left(x_T, y\right)$ 로 표시됩니다.

## Consistency Preserving Distillation
### Vanilla distillation.
우리는 무작위 초기화 상태 $x_T$ 와 그 결정론적 출력 $F_\theta\left(x_T, y\right)$ 사이의 결정론적 매핑 $F_\theta$ 을 Teacher Diffusion 모델에서 학습하기 위해 Student network $f_\theta$ 을 활용할 것을 제안합니다.  
바닐라 증류 손실은 다음과 같이 정의됩니다:

```math
\mathcal{L}_{\text {distill }}=L_{M S E}\left(f_{\hat{\theta}}\left(x_T, y, T\right), F_\theta\left(x_T, y\right)\right)
```

여기서 $f_{\hat{\theta}}\left(x_T, y, t\right)$ 는 단 한 단계만으로 HR 이미지를 직접 예측하는 학생 네트워크이며, $F_\theta$ 는 $\theta$ 로 매개변수화된 사전 훈련된 네트워크를 사용하여 반복적인 방식을 통해 4.1절의 ResShift[45]의 제안된 결정론적 추론 프로세스를 나타냅니다.  
우리는 ${L}_{\text {distill }}$ 의 증류 손실로만 훈련된 학생 모델이 결과 표에서 "증류 전용"으로 표시된 것처럼 이미 단 한 번의 추론 단계에서 유망한 결과를 달성하는 것을 관찰합니다.

### Regularization by the ground-truth image.
앞서 언급한 vanilla distillation strategy의 한계는 훈련 중에 GT 이미지가 활용되지 않아 Student 모델의 상위 성능 한계가 제한된다는 것입니다.  
Student의 성능을 더욱 향상시키기 위해 학습된 HR 이미지 반전을 통합하여 실제 이미지에서 추가 정규화를 제공하는 새로운 전략을 제안합니다.  
vanilla distillation strategy 외에도 Student network는 훈련 중에 다음 손실을 최소화하여 역 매핑을 동시에 학습합니다,

```math
\mathcal{L}_{\text {inverse }}=L_{M S E}\left(f_{\hat{\theta}}\left(F_\theta\left(x_T, y\right), y, 0\right), x_T\right)
```

여기서 $f_{\hat{\theta}}$의 마지막 매개 변수는 방정식 6의 T에서 0으로 설정되며, 이는 모델이 $\hat{x}_0$ 대신 역을 예측하고 있음을 나타냅니다.  
그런 다음 실제 이미지 $x_0$ 을 사용하여 다음과 같이 예측된 역 $\hat{x}_T$ 가 주어지면 출력 SR 이미지를 정규화할 수 있습니다,

```math
\begin{aligned}
\hat{x}T = \text{detach}(f{\hat{\theta}}(x_0, y, 0)) \\
\mathcal{L}_{gt} = L_{MSE}(f_{\hat{\theta}}(\hat{x}_T, y, T), x_0),
\end{aligned}
```

여기서 $L_{gt}$ 는 제안된 consistency preserving loss입니다.  
$f_{\hat{\theta}}$ 을 재사용하여 $f_{\hat{\theta}}(\hat{x}T, y, T)$ 와 $f_{\hat{\theta}}(\hat{x}_T, y, 0)$ 을 동시에 학습함으로써 Teacher의 $\theta$로부터 Student Model의 매개 변수 $\hat{\theta}$ 을 초기화하여 훈련 속도를 높일 수 있습니다.

### The overall training objective.
학생 네트워크는 앞서 언급한 세 가지 Loss를 동시에 최소화하도록 훈련됩니다,

```math
\hat{\theta} = \arg \min_{\hat{\theta}} \mathbb{E}{y,x_0,x_T} [\mathcal{L}_{distill} + \mathcal{L}_{inverse} + \mathcal{L}_{gt}],
```

여기서 Loss는 각각 방정식 6, 7, 8에 정의되어 있습니다.  
각 Loss 항에 동일한 가중치를 부여하고 후속 연구는 보충 자료에 나와 있습니다.  
제안된 방법의 전체적인 내용은 알고리즘 1과 그림 4에 요약되어 있습니다.  

![image](https://github.com/user-attachments/assets/3d264a6f-d0f5-45c7-8aad-75f2db89f1c8)

![image](https://github.com/user-attachments/assets/8bf74e58-7aec-4ed7-bf48-ea103848faa8)

# Experiment
## Experimental setup
### Training Details.
공정한 비교를 위해 [45]에서와 동일한 실험 설정과 백본 설계를 따릅니다.  
특히, 가장 큰 차이점은 [45]에서 50만 번을 처음부터 훈련하는 대신 30만 번의 반복에 대해 모델을 미세 조정했다는 점입니다.  
우리는 학생 모델이 빠르게 수렴하여 각 반복마다 짝을 이루는 훈련 데이터를 얻기 위해 ODE를 푸는 데 추가 시간이 필요하더라도 [45] 이후 모델을 처음부터 재훈련하는 것보다 전체 훈련 시간이 훨씬 짧다는 것을 발견했습니다.  
우리는 RealESRGAN[40]에서 열화 모델을 채택한 ResShift[45]와 동일한 파이프라인을 따라 ImageNet[7]의 훈련 세트에서 모델을 훈련합니다.

### Compared methods. 
우리는 우리의 방법을 RealSR-JPEG[11], ESRGAN[39], BSRGAN[46], SwinIR[17], RealESRGAN[40], DASR[18], LDM[31] 및 ResShift[45]를 포함한 여러 대표적인 SR 모델과 비교합니다.  
포괄적인 비교를 위해 샘플링 단계 수가 감소한 확산 기반 모델 LDM[31] 및 ResShift[45]의 성능을 추가로 평가합니다.  
또한 제안된 방법을 표 6의 생성 프로세스를 단일 단계로 압축할 수 있는 SOTA 방법인 RequiredFlow[20]와 비교합니다.

![image](https://github.com/user-attachments/assets/5527864f-dc37-4fd1-8a1f-609ca4a1087d)

### Metrics. 
참조 이미지가 포함된 합성 테스트 데이터 세트에서 제안된 방법을 평가하기 위해 PSNR, SSIM 및 LPIPS[47]를 활용하여 충실도 성능을 측정합니다.  
또한, 대규모 데이터 세트(Lion400M[35])와 MUSIQ[14]에서 사전 학습된 CLIP 모델[30]을 활용하는 CLIPIQA[38]라는 두 가지 최근의 SOTA 비참조 지표를 사용하여 모든 이미지의 사실성을 정당화합니다.

## Experimental Results.
### Evaluation on real-world datasets. 
RealSR[3] 및 RealSet65[45]는 보이지 않는 실제 데이터에서 모델의 일반화 능력을 평가하기 위해 채택되었습니다.  
특히 RealSR[3]에는 서로 다른 시나리오에서 두 대의 카메라로 캡처한 100개의 실제 이미지가 있습니다.  
또한 RealSet65[45]에는 널리 사용되는 데이터 세트와 인터넷에서 수집된 총 65개의 LR 이미지가 포함되어 있습니다.  
이 두 데이터 세트에 대한 결과는 표 1에 보고되어 있습니다.  

![image](https://github.com/user-attachments/assets/78fa983b-4d83-4b4f-b0ac-3190efbafa46)

표에 나와 있듯이, 추론 단계가 하나뿐인 제안된 방법은 우리가 사용한 교사 모델을 큰 차이로 능가할 수 있습니다.  
또한 최신 메트릭 CLIPIQA의 경우 제안된 방법이 모든 경쟁업체 중에서 가장 우수한 성능을 기록합니다.  
일부 시각적 비교는 그림 5에 나와 있으며, 제안된 방법은 단 한 단계만 사용하여 유망한 결과를 달성합니다.

![image](https://github.com/user-attachments/assets/2d4d8261-652f-43db-9b9c-f5827dbfd230)

### Evaluation on synthetic datasets. 
우리는 [45]의 설정에 따라 합성 데이터 세트 ImageNet-Test에서 다양한 방법의 성능을 추가로 평가합니다.  
구체적으로, 먼저 ImageNet[7]의 검증 세트에서 3000개의 고해상도 이미지를 무작위로 선택합니다.  
해당 LR 이미지는 [45]에 제공된 스크립트를 사용하여 얻습니다.  
표 2와 같이 추론 단계를 15단계에서 1단계로 줄이면 PSNR과 SSIM이 약간 감소하는 반면, 제안된 방법은 SSIM보다 최근의 전체 참조 이미지 품질 평가(IQA) 지표인 LPIPS로 측정한 최고의 지각 품질을 달성합니다.  

![image](https://github.com/user-attachments/assets/809c40b3-b1ae-4cb1-80bd-eec7367256ae)

또한 제안된 방법은 가장 최근의 SOTA 지표 CLIPIQA[38]에서 측정된 모든 방법 중 가장 우수한 성능을 달성하여 제안된 1단계 모델이 지각 품질 측면에서 15단계의 추론 단계를 가진 교사 모델과 동등하거나 심지어 약간 더 나은 것으로 나타났습니다.

### Evaluation of the efficiency. 
우리는 SOTA 접근 방식과 비교하여 제안된 방법의 계산 효율성을 평가합니다.  
표 3에 나타난 바와 같이, 제안된 방법은 단 한 번의 추론 단계만으로 우수한 성능을 보여주며, 이미 LDM[31]에 비해 추론 시간을 크게 단축한 채택된 교사 모델인 ResShift[45]를 능가합니다.  
표 3에 제시된 모든 방법은 잠재 공간에서 실행되며 VQ-VAE의 계산 비용이 계산된다는 점에 유의할 필요가 있습니다.

![image](https://github.com/user-attachments/assets/a813a155-3318-4849-88a8-fe5f62b70746)

## Analysis
### How important is the deterministic sampling? 
우리는 [45]에서 제안된 deterministic sampling 과 기본 stochastic sampling strategy($x_T$, ${\hat{F}_{\theta}}(x_T, y)$) 에서 생성된 쌍으로 훈련된 모델의 성능을 평가합니다.  
생성된 샘플 
```math 
x \sim F_\theta(x_T, y)
```
의 무작위성으로 인해 무작위 노이즈 $ε$가 주어지면 예측은 조건부 분포에 대한 기대입니다.  
그림 6의 비교는 훈련된 w/o 결정론적 교사 모델의 결과가 흐릿한 세부 사항을 나타내는지 추가로 검증합니다.  

![image](https://github.com/user-attachments/assets/83ba07e0-f305-4b30-bf54-e1e937f66096)

또한 표 4에서 볼 수 있듯이 제안된 결정론적 샘플링을 [45]의 기본 샘플링으로 대체할 때 상당한 성능 저하가 있으며, 이는 제안된 결정론적 샘플링의 효과와 필요성을 보여줍니다.

![image](https://github.com/user-attachments/assets/c840edaa-bde4-4e33-9d6e-7a3857db5550)

### Why does a single-step distillation work? 
이전 연구에 따르면 생성 프로세스의 비인과적 특성으로 인해 일반적으로 xT와 x0 간의 매핑을 직접 학습하는 것이 어렵다는 것이 일반적으로 밝혀졌습니다[19].  
그러나 우리의 경험적 발견에 따르면 SR 작업에서 xT와 x0 간의 매칭은 확산 모델과 마찬가지로 서로 다른 노이즈 수준에서 노이즈를 제거하는 것보다 상대적으로 학습하기 쉽습니다.  
특히, 학생 네트워크 $\hat{f_{\theta}}$ 의 용량은 한 단계만 사용하여 ODE 프로세스 $F_{\theta}$ 을 효과적으로 캡처하기에 충분합니다.  
우리의 가정을 검증하기 위해 우리는 서로 다른 전략으로 훈련된 소규모 모델의 성능을 평가한다.  
구체적으로, 한 모델은 실험 설정에 따라 훈련되지만 매개 변수의 수는 118.6M에서 24.3M으로 감소한다.  
다른 모델은 표준 크기 교사 확산 모델에서 $x_T$ 와 $\hat{x_0}$ 간의 매핑 관계를 직접 학습하면서 앞서 언급한 소규모 모델과 동일한 백본을 사용한다.  
이 두 소규모 모델 간의 비교는 표 5에 보고되어 있다.  

![image](https://github.com/user-attachments/assets/4d3e98e7-fcc1-4404-b7ab-2f6c52d6a1a5)


결과에서 알 수 있듯이, 서로 다른 노이즈 수준에서 노이즈를 제거하도록 훈련된 모델은 서로 간의 결정론적 매핑을 직접 학습하는 모델에 비해 심각한 성능 저하를 겪습니다.  
이는 결정론적 매핑을 직접 학습하는 것이 상대적으로 쉽다는 우리의 가정을 강력하게 뒷받침한다.

### Is a more sophisticated distillation strategy necessary?

### Learned inversion.

### Training overhead.

# Conclusion
이 연구에서는 확산 기반 SR 모델을 단일 추론 단계로 가속화하기 위한 새로운 전략을 제안합니다.  
특히, 입력 노이즈와 생성된 고해상도 이미지 사이의 결정론적 매핑을 학습하기 위해 1단계 양방향 증류가 제안되며, 그 반대의 경우도 도출된 결정론적 샘플링을 사용하는 교사 확산 모델에서 학습할 수 있습니다.  
한편, 증류 중에 새로운 일관성 보존 손실이 동시에 최적화되어 학생 모델이 사전 학습된 교사 확산 모델의 정보를 사용할 뿐만 아니라 실제 이미지에서 직접 학습할 수 있습니다.  
실험 결과는 제안된 방법이 단 한 단계 만에 교사 모델보다 동등하거나 더 나은 성능을 달성할 수 있음을 보여줍니다.
