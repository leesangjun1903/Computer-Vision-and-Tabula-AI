# CAMixerSR: Only Details Need More “Attention” | Super resolution

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
- 서로 다른 복잡도의 이미지 영역에는 **다양한 연산 복잡도**가 요구되므로, 단일 토큰 믹서(self-attention 또는 convolution) 사용은 비효율적이다.  
- 복잡한 영역에는 **변형 가능 윈도우 self-attention**을, 단순 영역에는 **가벼운 convolution**을 선택적으로 적용하는 **Content-Aware Mixer (CAMixer)** 구조를 제안한다.  

**주요 기여**  
1. **Content-Aware Mixer (CAMixer)**: 간단한 패치에는 convolution, 복잡한 패치에는 self-attention을 적용하여 연산량 절감과 표현력 향상을 동시에 달성.  
2. **강력한 predictor 모듈**: 로컬·글로벌·위치 인코딩을 입력으로 offsets, 윈도우 분류 마스크, 공간·채널 attention을 예측하여 CAMixer의 분류 정확도 및 표현 능력을 개선.  
3. **CAMixerSR**: CAMixer를 SwinIR-light 기반으로 쌓아 만든 SR 네트워크로, 경량 SR·2K–8K 대용량 SR·Omnidirectional SR에서 **최고 수준의 PSNR–FLOPs 트레이드오프**를 달성.

## 2. 해결 문제, 제안 방법, 모델 구조, 성능·한계

### 2.1 해결하고자 하는 문제  
- **대형(2K–8K) 이미지 SR**와 **경량 SR** 모두에서 전체 이미지에 동일한 연산을 적용하면, 단순 영역의 과잉 연산과 복잡 영역의 표현력 부족 문제가 공존.  
- 기존의 ClassSR, ARM 같은 콘텐츠 기반 라우팅은 **분류 성능 부족**·**수용 영역 제한** 문제를 가짐.

### 2.2 제안 방법 (수식 포함)  
- 입력 특징 $$X\in\mathbb{R}^{C\times H\times W}$$에서 값 $$V=f_{\text{PWConv}}(X)$$ 계산  
- **Predictor**:  

$$
    F = f_{\text{head}}(C_l,\,C_g,\,C_w),\quad
    \Delta p = r\cdot f_{\text{offsets}}(F),\quad
    m = \hat F W_{\text{mask}},\quad
    A_s = f_{\text{sa}}(F),\quad
    A_c = f_{\text{ca}}(F)
$$
   
  $$\Delta p$$로 윈도우 왜곡, $$m$$으로 hard/simple 패치 분류, $$A_s,A_c$$로 convolution 보강.  
- **Attention branch** (복잡 영역, 비율 $$\gamma$$):  
  1) $$\tilde X = \phi(X, \Delta p)$$로 변형된 특징 생성  
  2) $$\tilde Q, \tilde K$$ 생성 후  
  
$$
    V_{\text{hard}} = \mathrm{softmax}\bigl(\tfrac{\tilde Q\tilde K^T}{\sqrt d}\bigr)\,V_{\text{hard}}
$$  
- **Convolution branch** (단순 영역):  

$$
    V_{\text{conv}} = \mathrm{DWConv}(V_{\text{attn}})\cdot A_c + V_{\text{attn}}
$$  
- **출력**  

$$
    V_{\text{out}} = f_{\text{PWConv}}(V_{\text{conv}})
$$  
- **학습 손실**: $$\ell_1$$ SR 손실 + $$\ell_{\mathrm{ratio}}$$ hard-token 비율 제어 MSE 손실.

### 2.3 모델 구조  
- SwinIR-light를 기반으로 **20개 CAMixer+FFN 블록**, 채널 수 60, 윈도우 크기 $$16\times16$$.  
- γ=1.0 원형 모델, γ=0.5 절반 self-attention 모델, 실험에 따라 γ 조정.

### 2.4 성능 향상  
- **대형 입력 SR**(2K–8K): CAMixerSR-Base(765K 파라미터, 1.96G FLOPs)가 RCAN(15.6M, 32.6G)과 동등 PSNR 유지[8K:33.81dB vs.33.76dB]하며, FLOPs 17× 절감.  
- **경량 SR**(×4 업스케일): CAMixerSR(765K, 53.8G) PSNR 32.51dB로 SOTA(2위 SwinIR-light 32.44dB) 달성.  
- **ODI SR**: CAMixerSR(1.32M) SUN360에서 EDSR 대비 +0.26dB 개선.

### 2.5 한계  
- Predictor 분류 오차에 따른 self-attention 비율 제어 오차(γ′≈γ이지만 완전 일치 않음).  
- 큰 윈도우($$32\times32$$)에서 partition 성능 저하 관찰.  
- ClassSR과 결합 시 제한된 수용 영역 문제로 성능 격차 존재.

## 3. 모델 일반화 성능 향상 가능성 집중 논의  
- **다양한 입력 조건**(로컬·글로벌·위치) 결합으로 Predictor 정확도 향상 → SR뿐 아니라 다른 비전 과제(복합 영역 강조)에도 적용 가능.  
- **γ 동적 제어**: 학습 중 γref 조정 통해 연산량-성능 트레이드오프를 유연하게 관리.  
- **Deformable offsets**: 윈도우 attention을 복잡 영역 텍스처에 적응시켜, 단순 SR을 넘어 **객체 검출·분할** 등의 공간적 민감 작업에도 확장 기대.  
- **모듈형 구조**: CAMixer를 기존 토큰 믹서 기반 모델(SwinIR, ViT 등)에 삽입 가능, 다양한 백본과 호환되어 범용성 우수.

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **효율적 토큰 믹싱**: 복잡도 기반 토큰 스파시피케이션 연구 활성화  
- **다중 조건 Predictor 설계**: 로컬·글로벌·위치 외 추가 정보(ex. 왜곡 맵, depth) 통합 가능성 탐색  
- **Dynamic γ 스케줄링**: 영상 복잡도 예측에 따른 실시간 연산량 조절 연구  
- **대형 윈도우 Limit 극복**: 윈도우 크기·모양 최적화, 또는 계층적 윈도우 설계 검토  
- **응용 확대**: SR 외 객체 검출·세그멘테이션·영상 합성 등에서 CAMixer 기반 경량·고성능 모델 개발 고려

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/904aba87-f337-4c7b-a5e7-083761469079/2402.19289v2.pdf

# Abs
CAMixerSR은 이미지 초해상도를 위한 content aware mixer로, 다양한 SR 작업에서 우수한 성능을 보입니다.  
CAMixer는 복잡한 영역에는 self-attention를, 간단한 영역에는 convolution을 할당하여 모델 가속 및 토큰 믹싱 전략을 통합합니다.  
CAMixerSR은 가벼운 SR, 대형 입력 SR 및 전방향 이미지 SR에 대해 우수한 성능을 보이며 SOTA 품질-계산 교환을 달성합니다.  

대규모 이미지(2K-8K) 초해상도(SR)에 대한 수요가 빠르게 증가하고 있는 상황을 충족시키기 위해 기존 방법은 두 가지 독립적인 트랙을 따릅니다:  
1) content-aware routing을 통해 기존 네트워크를 가속화하고  
2) token mixer refining를 통해 더 나은 초해상도 네트워크를 설계합니다.  

이럼에도 불구하고 품질과 복잡성 균형의 추가적인 개선을 제한하는 피할 수 없는 결함(예: 유연하지 않은 경로 또는 비차별적 처리)에 직면합니다.
이러한 단점을 지우기 위해 간단한 컨텍스트에 대해서는 컨볼루션을 할당하고 희소한 텍스처에 대해서는 추가적인 변형 가능한 window-attention를 할당하는 content-aware mixer(CAMIXer)를 제안하여 이러한 체계를 통합합니다.
특히, CAMixer는 학습 가능한 예측기를 사용하여 windows warping을 위한 오프셋, 윈도우를 분류하기 위한 마스크, dynamic property를 가진 convolutional attentions를 포함한 여러 bootstraps을 생성하며, 이는 보다 유용한 텍스처를 스스로 포함하도록 attention를 조절하고 컨볼루션의 표현 기능을 향상시킵니다.
이 모델은 예측기의 정확도를 향상시키기 위해 global classification loss을 도입합니다. 단순히 CAMixer를 적층하여 대규모 이미지 SR, lightweight SR 및 모든 방향을 가지는 이미지 SR에서 우수한 성능을 달성하는 CAMixerSR을 얻습니다.

# Introduction
신경망에 대한 최근 연구는 이미지 초해상도(SR) 품질을 크게 향상시켰습니다[22, 34, 43].  
그러나 기존 방법은 시각적으로 만족스러운 고해상도(HR) 이미지를 생성하지만 특히 2K-8K 대상의 경우 실제 사용에서 철저한 계산을 거칩니다.   
고비용를 완화하기 위해 실제 초해상도 적용을 위해 많은 accelerating frameworks[4, 19]와 lightweight networks[14, 32]가 도입되었습니다.  
그러나 이러한 접근 방식은 협력 없이 완전히 독립적입니다.  
첫 번째 전략인 accelerating frameworks[11, 19, 39]는 이미지 영역마다 서로 다른 네트워크 복잡성이 필요하다는 관찰을 기반으로 하며, 이는 다양한 모델의 (콘텐츠 인식 관점)content-aware routing 관점에서 문제를 해결합니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.02.24.png)

그림 1의 중간 이미지에 표시된 것처럼, 큰 이미지를 고정된 패치들로 분해하고 extra classification network를 통해 네트워크에 패치를 할당했습니다.  
ARM[4]은 효율성을 향상시키기 위해 LUT 기반 classifier 및 파라미터 공유 설계 방식을 도입하여 전략을 더욱 발전시켰습니다.  
이러한 전략은 모든 신경망에 일반적이지만 피할 수 없는 두 가지 결함이 남아 있습니다.  
하나는 분류가 제대로 되지 않고 유연하지 못한 partition입니다.  
그림 1은 간단한 모델에 부적절하게 전송된 세부 정보가 거의 없는 창을 표시합니다. (복잡한 이미지 부분과 단순한 이미지 부분이 제대로 분류되지 않는다는 듯)  
다른 하나는 제한된 receptive fields입니다.  
표 2에서 볼 수 있듯이 패치로 이미지를 자르는 것은 receptive fields를 제한하므로 성능에 영향을 미칩니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.12.55.png)

두 번째 전략인 lightweight model design[7, 8, 17, 44]는 제한된 계층 내에서 더 강력한 feature 표현 기능, 즉 이미지를 재구성하기 위해 더 많은 intra-information를 사용할 수 있도록 신경 연산자(self-attention or convolution)와 중심 구조를 개선하는 데 중점을 둡니다.  
예를 들어, NGswin[5]은 self-attention를 위해 N-Gram을 활용하여 계산을 줄이고 receptive field를 확장했습니다.  
IMDN[14]은 효율적인 블록 설계를 위해 information multi-distillation를 도입했습니다.  
이러한 lightweight method은 720p/1080p 이미지에서 인상적인 효율성에 도달했지만, 더 큰 이미지(2K-8K)에서는 사용법이 거의 검토되지 않습니다.  
또한 이러한 접근 방식은 서로 다른 콘텐츠를 식별하고 처리할 수 없습니다.  

먼저 위의 전략을 통합한 이 논문은 서로 다른 feature 영역이 다양한 수준의 token-mixer(토큰이 섞인) 복잡성을 요구한다는 도출된 관찰을 기반으로 합니다.  
표 1에서 볼 수 있듯이 간단한 컨볼루션(Conv)은 간단한 패치에 대해 훨씬 더 복잡한 convolution + self-attention(SA + Conv)로 유사한 성능을 발휘할 수 있습니다.  
따라서 콘텐츠에 따라 서로 다른 복잡성을 가진 토큰 믹서의 루트를 정하는 content-aware mixer(CAMIXer)를 제안합니다.  
그림 1에서 볼 수 있듯이 당사의 CAMixer는 복잡한 window에는 복잡한 self attention(SA)를 사용하고 일반 윈도우에는 간단한 convolution을 사용합니다.  
또한 ClassSR의 한계를 해결하기 위해 보다 정교한 예측기를 소개합니다.  
이 예측기는 여러 조건을 활용하여 추가적인 가치 있는 정보를 생성하여 partition의 정확도를 향상시키고 표현을 개선하여 CAMixer를 향상시킵니다.  
CAMixer를 기반으로 초해상도 작업을 위한 CAMixerSR을 구성합니다.  
CAMixer의 성능을 완전히 검토하기 위해 lightweight SR, 대용량 이미지(2K-8K) SR 및 모든 방향의 이미지 SR에서 실험을 수행합니다.  
그림 2는 CAMixerSR이 lightweight SR과 accelerating framework를 모두 크게 발전시키는 것을 보여줍니다.  

![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.02.36.png)

다음과 같이 요약됩니다:
- 우리는 convolution and self-attention를 통합한 Content-Aware Mixer(CAMIXer)를 제안하며, 이는 컨볼루션에 간단한 영역을 할당하고 self-attention에 복잡한 영역을 할당하여 추론 계산을 적응적으로 제어할 수 있습니다. 
- 우리는 convolution 적용 후의 갈라진 형태, mask 및 간단한 공간/채널 attention를 생성하기 위한 강력한 예측기를 제안하며, 이는 더 적은 계산으로 넓은 상관 관계를 포착하도록 CAMixer를 변조합니다.
- CAMixer를 기반으로 lightweight SR, 대용량 이미지 SR, 모든 방향의 이미지 SR의 세 가지 까다로운 초해상도 작업에서 최첨단 품질의 계산 절충점을 보여주는 CAMixerSR을 구축합니다.

# Related Work
## Accelerating framework for SR

## Lightweight SR

# Method
## Content-Aware Mixing
![](https://github.com/leesangjun1903/Computer-Tomograpy-reconstruction/blob/main/image/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202024-08-14%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2011.13.08.png)

## Network Architecture

## Training Loss

# Reference
- https://linnk.ai/insight/%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC/camixersr-content-aware-mixer-for-image-super-resolution-3DtqCD_s/
