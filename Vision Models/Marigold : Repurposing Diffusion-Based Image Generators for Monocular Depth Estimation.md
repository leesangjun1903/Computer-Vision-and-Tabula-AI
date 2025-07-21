# Marigold : Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation | Depth estimation

**핵심 주장 및 주요 기여**  
이 논문은 대규모 비전-언어 데이터를 이용해 학습된 생성형 확산 모델이 보유한 “풍부한 시각적 사전 지식”을 단안 깊이 추정에 전이할 수 있음을 보인다. Stable Diffusion의 잠재 공간과 U-Net 구조를 최소 변경만으로 활용해, 오직 합성 RGB-D 데이터로만 수 시간의 미세 조정(fine-tuning)을 거쳐 실제 실세계 데이터셋에 **제로샷**으로 우수한 성능을 달성하는 Marigold 모델을 제안한다.

1. **기여 1**: Stable Diffusion 기반 LDM(Latent Diffusion Model)을 깊이 추정기로 전환하는 효율적 미세 조정 프로토콜  
2. **기여 2**: 순전 합성 데이터만으로 미세 조정해도 실세계의 다양한 실내·외 장면에서 최첨단(SoTA) 제로샷 성능 달성  
3. **기여 3**: 다중 해상도 잡음(multi-resolution noise)과 애니일링(annealed scheduling) 적용으로 미세 조정 효율 및 정확도 개선  

## 1. 문제 정의  
단안 깊이 추정은 단일 RGB 이미지만으로 픽셀별 깊이 값을 예측하는데, 이는 기하학적으로 불완전한(inherently ill-posed) 문제이다. 기존에는 대규모 RGB-D 데이터셋(실세계 실내·도로·웹 사진 등)으로 수백만~천만 샘플을 학습하여 일반화 성능을 확보해 왔으나, 여전히 낯선 도메인에 취약하다.

## 2. 제안 방법  

### 2.1 모델 구조 및 수식  
- **잠재 공간 사용**: Stable Diffusion의 VAE를 동 зам결(frozen)하여 RGB와 깊이 맵 d를 잠재 코드 z(x), z(d)로 인코딩  
- **U-Net 수정**: 입력 채널을 2배로 늘려 $ϵ_θ(z(d)_t, z(x), t)$ 형태로 결합 입력. 첫 레이어 가중치 복제 후 ½ 스케일 조정하여 사전 학습 구조 유지  
- **확산 학습 목표**:  
  
$$
    \mathcal{L} = \mathbb{E}_{d,ϵ,t}\big\|ϵ - ϵ_θ(d_t,\,x,\,t)\big\|^2_2
  $$  

- **깊이 정규화(affine-invariant)**: 각 맵의 2%·98% 백분위를 기준으로  
 

```math
\tilde d = \left (\frac{d - d_{2\%}}{d_{98\%} - d_{2\%}} \right) - 0.5\Bigr)\times2
```

전역 스케일·오프셋 불변(depth up to scale and shift)  

### 2.2 잡음 스케줄  
- **Multi-resolution noise**: 여러 해상도 잡음을 합성해 잡음 다양성 증대  
- **Annealed scheduling**: t→0로 갈수록 고해상도 Gaussian noise로 선형 보간  

### 2.3 학습 및 추론 프로토콜  
- **학습 데이터**: Hypersim(54K), Virtual KITTI(20K) 합성 전용  
- **미세 조정**: RTX-4090 단일 GPU, 2.5일, 배치 32, Adam lr=3e-5, 18K 이터레이션  
- **추론**: DDIM 50 스텝, 앙상블 N=10 반복 예측 후 픽셀별 중위수 취하고, 스케일·오프셋 정렬  
- **평가**: AbsRel, δ1(1.25) 기준으로 5개 실세계 전이테스트셋( NYUv2, KITTI, ETH3D, ScanNet, DIODE ) 제로샷 비교  

## 3. 성능 및 한계  

### 3.1 성능 향상  
- **Tab.1 주요 결과**:  
  - NYUv2 AbsRel 5.5% (최고), δ1 96.4%  
  - KITTI AbsRel 9.9% (2위 근소), δ1 91.6%  
  - ETH3D AbsRel 6.5%, δ1 96.0%  
  - 전 영역 평균 순위 1.4위 확보 (앙상블)[표1]  

- **제로샷 전이**만으로 기존 MiDaS, DPT, HDN, Omnidata 대비 20% 이상 성능 개선  
- **잡음·앙상블·스텝 수**에 대한 철저한 분석(ablation)으로 효율적 파라미터 설정  

### 3.2 한계  
- **추론 속도 저하**: feed-forward 모델 대비 느린 확산 반복(50 스텝 + 앙상블)  
- **앙상블 필요성**: 일관된 예측 위해 N>10 권장, 연산 증가  
- **원격 장면**: 아주 먼(depth>80m) 영역은 학습 데이터 한계로 다소 불안정  

## 4. 일반화 성능 향상 핵심 요인  
1. **풍부한 사전 지식 활용**: 인터넷 규모 이미지로 학습된 Stable Diffusion의 잠재 공간이 다양한 도메인(scene prior)을 보유  
2. **합성 데이터 전용 미세 조정**: 센서 노이즈·결측 없는 합성 깊이를 이용해 “노이즈 없는” 그라디언트 업데이트  
3. **Affine-invariance**: 스케일·오프셋 정규화로 미지 카메라 설정에도 강건  

→ 이 세 요소의 결합이 “제로샷으로 실제 도메인”에 일반화되는 결정적 요인  

## 5. 향후 영향 및 주의 사항  

**영향**  
- 확산 기반 생성 모델을 **다양한 구조화 예측**(depth, normal, optical flow 등)으로 전환하는 연구 가속  
- “Foundation Vision Models”의 잠재 공간 활용 가능성 확대: VAE 잠재 공간을 다양한 인식 과제로 미세 조정  

**고려할 점**  
- **추론 최적화**: distillation·프롬프트 엔지니어링으로 스텝 수 대폭 축소  
- **실세계 메트릭 추정**: 카메라 내부 파라미터 입력 없이 절대 깊이(metric depth) 복원 방안  
- **합성–실세계 도메인 간 격차** 완화: 고품질 합성 다양성 추가 또는 도메인 적응  
- **일관성 강화**: 생성적 특성으로 인한 샘플 간 분산 감소 기법  

Marigold는 **합성 전용 학습**과 **기존 확산 모델의 잠재 공간 전이**를 통해 단안 깊이 추정에서 새로운 일반화 기준을 제시했으며, 빠르게 확산형 생성 모델을 다목적 시각 인식으로 확장하는 발판이 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3ace8455-90ca-4bec-b4cf-f06756a6bf23/2312.02145v2.pdf
