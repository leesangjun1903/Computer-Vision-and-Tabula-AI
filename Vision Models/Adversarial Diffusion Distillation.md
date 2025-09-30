# ADD : Adversarial Diffusion Distillation | 2023 · 561회 인용, Image generation

**핵심 주장 및 주요 기여**  
Adversarial Diffusion Distillation(ADD)는 대형 사전학습 이미지 확산 모델을 **1–4단계** 만에 고품질 이미지를 생성하도록 증류하는 새로운 학습 기법이다. 스코어(distillation) 기반 지식 증류와 **적대적 손실( adversarial loss)**을 결합하여 단일 단계(1 step)에서도 기존의 GAN·Latent Consistency Models를 능가하며, 4단계에서는 SDXL급 성능을 달성한다. 이를 통해 **실시간(single-step) 이미지 합성**을 가능케 하는 최초의 방법을 제시했다.[1]

***

## 1. 해결 과제  
- **확산 모델의 반복적 샘플링**: 수십에서 수백 단계의 샘플링으로 고품질 이미지를 얻지만, 실시간 응용에는 부적합.  
- **GAN의 품질 한계**: 단일 단계 생성을 제공하나 대규모 데이터셋·복합 텍스트 조합에서는 확산 모델 대비 화질 및 구성력 부족.

***

## 2. 제안 방법  
ADD는 세 네트워크를 활용한다(그림 생략).

1. **ADD-Student(θ)**: 사전학습된 U-Net 확산 모델로 초기화  
2. **Discriminator(ϕ)**: Vision Transformer 기반 특징망 F와 다수의 판별기 헤드 Dϕ,k  
3. **Teacher Diffusion Model(ψ)**: 고정된 사전학습 확산 모델  

### 학습 손실  

$$
L \;=\; L^{G}_{\mathrm{adv}}\bigl(\hat{x}_\theta(x_s,s),\,\phi\bigr)\;+\;\lambda\,L_{\mathrm{distill}}\bigl(\hat{x}_\theta(x_s,s),\,\psi\bigr)
$$  

- **적대적 손실 $$L^{G}_{\mathrm{adv}}$$**: Student가 생성한 샘플 $$\hat{x}_\theta$$이 실제 이미지 $$x_0$$처럼 보이도록 Discriminator를 속임.  
- **증류 손실 $$L_{\mathrm{distill}}$$**: Teacher의 denoising 예측 $$\hat{x}_\psi$$를 재구성 목표로 하여 Student가 Teacher의 구조적 지식을 학습하도록 유도.  

학습 과정에서 노이즈 레벨 $$s$$를 균일하게 선택하며, $$\tau_n=1000$$을 포함한 N=4개의 학생 타임스텝만 활용해 효율성을 극대화한다.

***

## 3. 모델 구조  
- **Generator**: U-Net 아키텍처(사전학습 확산 모델 그대로 활용)  
- **Discriminator**: Frozen ViT 특징망 F + 경량 판별기 헤드 Dϕ,k, projection conditioning 적용 가능  
- **Teacher**: 동일 혹은 대형 확산 모델(SDXL)으로 고정  

이 구조는 **반복적 개선(iterative refinement)**을 유지하면서도 단일 단계를 가능케 한다.

***

## 4. 성능 향상  
- **1단계 샘플링**: 기존 GAN 및 LCM(1–4단계) 대비 선호도·ELO 점수 우위  
- **4단계 샘플링**: Teacher인 SDXL-Base(50단계) 대비 인간 평가에서 더 높은 선호도 획득  
- **자동 지표**: COCO zero-shot FID5k 및 CLIP 스코어 비교에서 모든 증류·빠른 샘플러 방법 제압  

*Inference 속도 대비 품질 개선*을 시각화한 결과, ADD-XL(1/4 step)은 SDXL(50 step) 및 기타 모델들보다 탁월한 trade-off를 보인다.

***

## 5. 한계  
- **샘플 다양성 저하**: 적대적 손실로 현실성·텍스처 개선 시 다소 diversity 감소 관찰.  
- **대규모 연산 비용**: 사전학습 모델 활용에도 학생 네트워크 학습 시 GPU·메모리 부담 존재.  
- **범용성**: 텍스트-이미지 외 다른 모달리티(비디오·3D) 적용 시 추가 연구 필요.

***

## 6. 일반화 성능 개선 관점  
- **강화된 샘플 현실성**: 적대적 손실이 오버스무딩 문제를 완화, 세밀한 디테일 학습 촉진.  
- **Teacher 지식 전이**: 거대한 확산 모델의 compositionality·구조적 이해를 학생이 고단계 없이 습득.  
- **반복적 미세 조정**: 1→2→4단계 과정에서 점진적 세부 개선이 가능해 다양한 환경·프롬프트에 적응 능력 향상.  

이로써 **새로운 도메인**(의료·과학 시각화 등)에서 빠른 샘플링과 높은 fidelity를 동시에 달성할 잠재력 보유.

***

## 7. 영향 및 향후 연구 고려사항  
**영향**:  
- **실시간 응용**: AR/VR, 대화형 AI, 모바일·엣지 환경에서 대형 확산 모델 활용 장벽 해소.  
- **모델 증류 연구**: adversarial+distillation 하이브리드 학습 패러다임을 확산 모델 전반으로 확장 유도.  

**향후 고려사항**:  
- **다양성 복원**: diversity와 fidelity 균형을 맞추는 새로운 손실 설계  
- **다중 모달 증류**: 텍스트·오디오·3D 등 다른 데이터 유형에 대한 ADD 확장  
- **경량화 및 효율화**: 메모리·연산 부담 감소를 위한 저전력 디바이스 대응  

이러한 방향성은 **ADD가 제시한 단일 단계 증류** 기술을 더욱 발전시켜, 차세대 실시간 생성 AI의 기반을 마련할 것이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2592aae1-50bc-49ee-9642-53616b6b6a4d/2311.17042v1.pdf)
