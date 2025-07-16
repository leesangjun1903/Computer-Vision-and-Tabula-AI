# EDM : Elucidating the Design Space of Diffusion-Based Generative Models | Image generation

## 1. 핵심 주장 및 주요 기여  
“Elucidating the Design Space of Diffusion-Based Generative Models”는 **확산 기반 생성 모델**을 구성하는 핵심 요소를 모듈화된 설계 공간으로 명료하게 분리하고, 각 구성 요소를 독립적으로 개선함으로써 품질(FID) 및 샘플링 속도를 획기적으로 향상시켰음을 주장한다.  
주요 기여:  
- 확산 모델을 훈련, 샘플링, 네트워크 전처리(프리컨디셔닝) 3단계로 나누어 설계 공간을 명시적으로 정리  
- 샘플링: 2차 Runge–Kutta(Heun) 방법, 최적 시간 스케줄 σ(t)=t, 다항식 스텝 크기 조정, stochasticity 조절로 8×~300× 샘플링 속도 개선  
- 훈련: 입력·출력 스케일링 및 스킵 연결을 통한 프리컨디셔닝, 로그-정규 분포 기반 노이즈 레벨 샘플링, 균등 화된 손실 가중치 λ(σ) 도입  
- 데이터 증강(비누설): 기하학적 변환을 조건 입력으로 제공해 과적합 억제, FID 1.79→1.36(Conditional CIFAR-10), 1.97(Unconditional CIFAR-10), ImageNet-64 FID 2.07→1.36 기록  

## 2. 문제 정의 및 제안 기법  
### 2.1 해결하고자 하는 문제  
- 기존 확산 모델 문헌은 이론·수식에 치중해 구성 요소 간의 의존성을 지나치게 복잡하게 얽어 놓음  
- 샘플링 속도가 느리고, 훈련 중 노이즈 레벨 선택·네트워크 전처리 최적화가 미흡  

### 2.2 수식 기반 제안 방법  
1. **확산 확률 흐름 ODE**  
   
$$
   dx = -{\dot\sigma(t)}{\sigma(t)}\nabla_x\log p(x;\,\sigma(t))\,dt
$$  

2. **2차 Heun 샘플러** (알고리즘 요약)  

$$
   \begin{aligned}
   & d_i = \Bigl(\tfrac{\dot\sigma(t_i)}{\sigma(t_i)}+\tfrac{\dot s(t_i)}{s(t_i)}\Bigr)x_i - \tfrac{\dot\sigma(t_i)s(t_i)}{\sigma(t_i)}D_\theta\bigl(\tfrac{x_i}{s(t_i)};\sigma(t_i)\bigr),\\
   & x_{i+1} = x_i + h_i\,d_i + \tfrac{h_i}{2}\bigl(d_{i+1}-d_i\bigr),
   \end{aligned}
   $$

   여기서 $$s(t)=1,\;\sigma(t)=t$$, $$h_i=t_{i+1}-t_i$$.  

3. **스케줄링**  
   
$$
   \sigma_i = \bigl[\sigma_{\max}^{1/\rho} + \tfrac{i}{N-1}\bigl(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\bigr)\bigr]^\rho,\quad \rho=7
  $$  

4. **스코어 네트워크 전처리**  

$$
   D_\theta(x;\sigma)=c_{\rm skip}(\sigma)\,x + c_{\rm out}(\sigma)\,F_\theta\bigl(c_{\rm in}(\sigma)\,x;\,c_{\rm noise}(\sigma)\bigr)
$$

$$
     c_{\rm skip}=\tfrac{\sigma_{\rm data}^2}{\sigma^2+\sigma_{\rm data}^2},\quad
     c_{\rm out}=\tfrac{\sigma\,\sigma_{\rm data}}{\sqrt{\sigma^2+\sigma_{\rm data}^2}},\quad
     c_{\rm in}=\tfrac{1}{\sqrt{\sigma^2+\sigma_{\rm data}^2}}
$$  

5. **훈련**  
   - 노이즈 레벨 분포 $$\ln\sigma\sim\mathcal{N}(-1.2,\;1.2^2)$$ (로그-정규)  
   - 손실 가중치 $$\lambda(\sigma)=\tfrac{\sigma^2+\sigma_{\rm data}^2}{(\sigma\,\sigma_{\rm data})^2}$$  

### 2.3 모델 구조  
- 기존 DDPM++(VP), NCSN++(VE), iDDPM(Conditional ImageNet-64) 아키텍처 그대로 사용  
- 입력·출력 스케일링, 스킵 연결 등만 교체  
- 샘플러 독립성 확인: 훈련 구조에 관계없이 샘플러만 바꿔도 전 모델 FID 대폭 개선  

### 2.4 성능 향상  
- **CIFAR-10 32×32**  
  - Class-conditional: FID 1.85→1.79(35 NFE, determin.)  
  - Unconditional: FID 2.10→1.97(35 NFE)  
- **ImageNet-64**  
  - Pre-trained: FID 2.07→1.55(1023 NFE, stochastic)  
  - 재훈련: FID 1.48→1.36(511 NFE)  
- **속도**: V100 한 장으로 CIFAR-10 35 NFE 기준 초당 26.3장  

### 2.5 한계  
- **하이퍼파라미터 민감도**: $$\rho,\,\sigma_{\min},\,S_{\rm churn},\,S_{\rm noise}$$ 등 튜닝 필요  
- **고해상도 확장**: 64×64 이상의 해상도에선 각 파라미터 재조정 필요  
- **스토캐스틱-훈련 상호작용**: stochastic 샘플링과 훈련 목적 함수 간 상호작용 미해명  

## 3. 일반화 성능 향상 가능성  
- **비누설(Non-leaky) 데이터 증강**: 기하학적 변환을 조건 입력으로 제공함으로써 과적합 억제, 소규모 데이터셋에서도 네트워크 일반화 개선  
- **노이즈 레벨 재분포**: 로그-정규 분포로 훈련 노이즈 레벨 샘플링 → 중간 σ에서 손실 집중 → 모델이 잡음 분포 전체가 아닌 주요 영역 학습 → 새로운 데이터 상황에도 견고  
- **균등화된 손실 가중치**: $$\lambda(\sigma)$$로 σ별 기울기 크기 균일화 → 학습 안정성 및 일반화 촉진  

## 4. 향후 연구 영향 및 고려사항  
- **모듈화된 설계 공간**: 각 구성요소(샘플러, 프리컨디셔닝, 훈련 스케줄)를 독립 연구 가능  
- **고해상도·다양한 도메인 확장**: 제외된 요소(초고해상도, 텍스트 조건화, 잠재 공간 확산)와 결합 연구  
- **자동 하이퍼파라미터 튜닝**: 샘플러·훈련 파라미터 자동 최적화 기법 개발  
- **스토캐스틱-훈련 관계 연구**: stochastic 샘플링이 학습 dynamics에 미치는 영향 심층 분석 필요  

---  
**주요 시사점**: “EDM” 프레임워크는 확산 모델 혁신을 위한 **종합적·모듈화된 분석** 틀을 제공하며, 속도·품질·일반화 측면에서 새로운 연구 방향을 제시한다. 러닝과 샘플링 단계를 명확히 분리함으로써 다양한 도메인·해상도에 걸친 확산 모델의 **맞춤형 최적화**가 가능해질 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3c814455-7ae3-42f4-8ee4-92d450f2eab4/2206.00364v2.pdf
