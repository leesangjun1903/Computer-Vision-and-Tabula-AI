# NCSN : Generative Modeling by Estimating Gradients of the Data Distribution | Image generation, Image inpainting

**Yang Song & Stefano Ermon (NeurIPS 2019; arXiv:1907.05600v3)**  

**핵심 주장 및 주요 기여**  
이 논문은 **데이터 분포의 로그밀도 기울기(스코어)** 를 직접 추정하고, 이를 이용해 **Langevin 동역학** 으로 샘플링하는 새로운 생성 모델을 제안한다.  
1. 저차원 매니폴드 상에 놓인 실제 데이터에 대해 스코어가 미정의되는 문제와, 희소 영역에서의 부정확한 스코어 추정 문제를  
   **다중 수준의 가우시안 잡음** 을 가해 해소  
2. 서로 다른 노이즈 크기에 조건화된 **Noise Conditional Score Network (NCSN)** 를 학습하여 모든 노이즈 수준의 스코어를 단일 네트워크로 추정  
3. 노이즈 세기를 점진적으로 감소시키며 Langevin 동역학을 수행하는 **Annealed Langevin Dynamics** 를 통해 고품질 샘플 생성  
4. GAN 대비 안정적 학습, 명확한 학습 목적 함수, 샘플링 단계 분리라는 구조적 장점  

## 1. 문제 정의  
- 데이터 분포 $$p_{\mathrm{data}}(x)$$ 의 스코어 $$\nabla_x \log p_{\mathrm{data}}(x)$$ 를 학습  
- 전통적 스코어 매칭은  

$$
    \min_\theta \;\tfrac12 \,\mathbb{E}\_{p_{\mathrm{data}}}\big[\|\!s_\theta(x)-\nabla_x\log p_{\mathrm{data}}(x)\|^2\big]
$$
  
  를 사용하나,  
  - **매니폴드 가설**: 실제 데이터가 저차원 매니폴드에 존재 → 스코어 미정의  
  - **저밀도 영역**: 데이터 샘플 부족 → 스코어 부정확 → Langevin 샘플링 느린 수렴 및 모드 불균형  

## 2. 제안 기법  

### 2.1 Noise Conditional Score Networks (NCSN)  
- 노이즈 수준 $$\{\sigma_i\}_{i=1}^L$$ 의 가우시안 섞음 분포  

$$
    q_{\sigma_i}(x) = \int p_{\mathrm{data}}(t)\,\mathcal{N}(x\mid t,\sigma_i^2 I)\,dt
$$

- 조건부 스코어 네트워크 $$s_\theta(x,\sigma)$$ 학습:  

$$
    \min_\theta \frac1L \sum_{i=1}^L \lambda(\sigma_i)\,\mathbb{E}\_{p_{\mathrm{data}}(x)}\mathbb{E}\_{\tilde x\sim\mathcal{N}(x,\sigma_i^2I)}
    \Big\|\!s_\theta(\tilde x,\sigma_i)+\frac{\tilde x-x}{\sigma_i^2}\Big\|^2
$$
  
  - 가중치 $$\lambda(\sigma)=\sigma^2$$ 로, 모든 노이즈 수준에서 손실 스케일 균일화  

### 2.2 Annealed Langevin Dynamics  
- 샘플 $$\tilde x_0\sim\text{Uniform noise}$$ 초기화  
- 단계별로 노이즈 $$\sigma_1>\dots>\sigma_L$$ 에 대해  

$$
    \tilde x_t \leftarrow \tilde x_{t-1} + \frac{\alpha_i}{2}\,s_\theta(\tilde x_{t-1},\sigma_i)
    +\sqrt{\alpha_i}\,z_t,\quad z_t\!\sim\!\mathcal{N}(0,I)
$$

$$
    \alpha_i = \epsilon\,\frac{\sigma_i^2}{\sigma_L^2},\;\;T\text{ steps}
$$

- 큰 노이즈에서 시작해 점차 감소시키며 표본 품질과 모드 커버리지 모두 확보  

## 3. 모델 구조  
- **U-Net 스타일 RefineNet** + **dilated convolution** + **(수정된) 조건부 인스턴스 정규화(CondInstanceNorm++)**  
- 입력: 이미지 $$x$$ 와 노이즈 레벨 $$\sigma$$  
- 출력: $$x$$ 와 동일 차원의 스코어 벡터 필드  

## 4. 성능 및 한계  

| 데이터셋      | Inception Score | FID Score | 비교 모델 대비                                    |
|--------------|-----------------|-----------|----------------------------------------------------|
| CIFAR-10 (uncond.) | **8.87**           | 25.32     | 종전 최고 GAN (SNGAN) 8.22 ≤ IS < 8.80; FID ~21.7–36.4 |
| MNIST, CelebA   | 시각적 품질 우수 (정량비교 어려움) | –         | 기존 likelihood/GAN 모델과 비견 가능                |

- **장점**:  
  - 비대립학습(Non-adversarial) → 안정적  
  - 훈련 중 MCMC 불필요 → 효율적  
  - 명확한 우선순위적 학습 목표  
- **한계**:  
  - 샘플링 단계 $$T$$ 높일수록 비용 증가  
  - 고해상도·고차원 데이터로의 확장 연구 필요  
  - 스코어 네트워크 구조·하이퍼파라미터 민감도  

## 5. 일반화 성능 향상 가능성  
- **다중 노이즈 조건화** 로 저밀도 영역 학습 신호 강화 → 과적합 완화  
- **Annealing**: 점진적 도메인 이동(domain bridging)으로 모드 간 이동성 보장  
- **비대립목표**: 노이즈 크기마다 균등 학습 → 특정 모드 편향 방지  
- 잠재공간 대신 **스코어 공간** 학습으로, 데이터 분포 전반에 걸친 표현력 향상  

## 6. 향후 영향 및 고려사항  
- **에너지기반 모델(Energy-Based Models)**: 스코어 추정-샘플링 프레임워크 일반화  
- **고해상도 영상·시계열·3D** 등 다양 영역 적용 전망  
- **스코어의 불확실성 정량화**: 베이지안·분포 추정 기법 결합 가능  
- **샘플링 최적화**: 하이퍼파라미터·스텝수 절감 위한 적응형 스케줄 연구  
- **실시간 생성**: 효율적 요약 및 근사 알고리즘 설계  

위 접근은 **비대립적, 목적 함수 명료성, 매니폴드 및 저밀도 문제 해결** 이라는 점에서 생성 모델 연구에 새로운 방향을 제시하며, 향후 다양한 도메인으로 확장될 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2c8dc2b1-f850-4cb1-a85b-953e9e919c50/1907.05600v3.pdf
