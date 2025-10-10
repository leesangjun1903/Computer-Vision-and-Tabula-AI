# GDP : Generative Diffusion Prior for Unified Image Restoration and Enhancement | 2023 · 320회 인용, Image restoration

**핵심 요약**  
Generative Diffusion Prior(GDP)은 사전 학습된 비지도 확산 모델(DDPM)을 활용해 선형 역문제(super-resolution, deblurring, inpainting, colorization)부터 비선형·블라인드(low-light enhancement, HDR recovery) 문제까지 단일 프레임워크로 처리하며, 복원 모델 파라미터를 역전파로 추정하고 계층적 가이던스 및 패치 기반 기법으로 임의 해상도 이미지를 복원함으로써 다양한 벤치마크에서 기존 비지도 방식 대비 일관성·지각 품질을 크게 향상시킨다.[1]

## 1. 핵심 주장 및 주요 기여  
GDP는 다음 네 가지 주요 기여를 제안한다 :[1]
-  단일 비지도 DDPM을 영상 복원·향상에 범용 프라이어로 활용하는 최초 프레임워크  
-  블라인드 복원을 위해 열화 모델 파라미터를 랜덤 초기화 후 확산 과정에서 최적화  
-  계층적 가이던스와 패치 기반 방법으로 임의 해상도 이미지 복원 지원  
-  모든 복원 작업(super-resolution, deblurring, inpainting, colorization, low-light enhancement, HDR recovery)에 일관된 샘플링 프로토콜 적용  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제  
기존 방법은 알려진 선형 열화 모델에 의존하고 감독 학습이 필요하여 복합·실제 열화 상황에 적용이 제한된다. 반면 GDP는  
- 선형 역문제: $$y = D(x)$$ 형태의 알려진 합성 열화를 역전파  
- 비선형·블라인드 문제: $$y = f(x) + M$$ 형태의 조명 스케일·마스크 파라미터를 동시 추정하며 복원  

### 2.2 제안하는 방법  
GDP는 사전 학습된 DDPM의 역확산 샘플링에 열화 연산 및 관측 이미지 $$y$$에 대한 가이던스(gradient guidance)를 삽입한다. 핵심 식은 다음과 같다 :[1]

$$
\log p(y \mid x_t) \approx -s\,L(D(x_t),y) - \lambda\,Q(x_t)
\quad\Longrightarrow\quad
\nabla_{x_t}\log p(y\mid x_t)=-s\,\nabla_{x_t}L(D(x_t),y)-\lambda\,\nabla_{x_t}Q(x_t)
$$  

이를 통해 기존 unconditional transition $$p_\theta(x_{t-1} \mid x_t)=\mathcal{N}(x_{t-1};\mu,\Sigma)$$의 평균을 $$\mu+\Sigma\,\nabla_{x_t}\log p(y\mid x_t)$$로 조정해 조건부 역과정을 근사한다.  
- GDP- $$x_t$$: 노이즈 이미지 $$x_t$$에 직접 가이던스  
- GDP- $$\tilde x_0$$: DDPM이 예측한 중간 복원 $$\tilde x_0$$에 가이던스하여 더 안정된 복원  

계층적 가이던스 및 패치 기반 방법(Alg.6)을 통해 임의 크기 이미지를 처리하며, HDR recovery는 다중 LDR 입력을 각각 최적화 가능한 열화 파라미터로 동시 가이딩한다.[1]

### 2.3 성능 향상  
GDP- $$\tilde x_0$$는 ImageNet 기반 선형 복원 벤치마크에서 FID와 Consistency 지표에서 최고 성능을 기록하며 비지도 방법을 크게 앞선다.[1]
비선형 사례인 low-light enhancement와 HDR recovery에서도 zero-shot 조건에서 기존 최상위 모델 대비 PSNR·SSIM·FID·PI·LOE 전 지표를 우수하게 달성한다.[1]

### 2.4 한계  
GDP는 매 단계 가이던스를 적용하고 파라미터를 최적화하므로 샘플링 시간이 길어 실시간·모바일 적용 시 제약이 있다. 또한 최적 가이던스 스케일을 데이터 분포별로 수작업 조정해야 한다는 불편이 존재한다.[1]

## 3. 모델의 일반화 성능 향상  
GDP는 ImageNet 사전학습 모델만으로 CelebA, LSUN, USC-SIPI와 같은 분포가 다른 데이터셋에서도 뛰어난 복원 품질을 보였으며, 다양한 해상도와 복합 열화 유형(다중 선형·비선형)에 대해 재학습 없이도 강건한 성능을 유지한다. 이는 확산 프라이어가 데이터 분포에 의존적이지 않고 이미지 통계 전반을 포괄적으로 학습했기 때문이다.[1]

## 4. 향후 연구 영향 및 고려 사항  
GDP는 단일 프리트레인 디퓨전 모델로 복원·향상 작업을 통합하며, 비지도 복원 분야 패러다임 전환을 제시한다. 앞으로 고려할 사항은 다음과 같다:  
- **가속화 기법 연구**: DDIM, distillation, 적응적 노이즈 스케줄로 샘플링 속도 개선  
- **자동 스케일 선택**: 메타러닝으로 가이던스 스케일 자동 최적화  
- **3D 데이터 및 LiDAR**: 3D 복원, 포인트 클라우드·LiDAR 열화 복원으로 확장 가능성  
- **자기지도 학습 융합**: GDP 구조를 기반으로 한 self-supervised 사전학습 전략 도입으로 성능·일반화 강화  

이상으로 GDP는 확산 모델을 활용한 복원·향상 연구에 새로운 방향을 제시하며, 후속 연구에서 속도·자동화·다양한 데이터 유형으로의 확장이 핵심 과제로 남는다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c419923d-8e95-4c4e-b0b9-ca095851f240/2304.01247v1-abcugdoem.pdf)
