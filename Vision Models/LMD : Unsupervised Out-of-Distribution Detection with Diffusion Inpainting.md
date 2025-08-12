# LMD : Unsupervised Out-of-Distribution Detection with Diffusion Inpainting | Image inpainting
# 핵심 요약

**주장**: 본 논문은 레이블 없는(in-domain) 데이터만을 활용하여 확산 모델(diffusion model)의 **inpainting** 능력을 이용한 **비지도형 Out-of-Distribution (OOD) 검출** 기법인 **Lift, Map, Detect (LMD)**를 제안한다.  
**기여**:  
- 확산 모델이 학습한 도메인 내 이미지 **manifold**로의 매핑(mapping) 특성을 활용해 OOD 샘플을 분리  
- 이미지 마스킹(masking)→확산 기반 inpainting→원본-재구성 영상 간 거리 계산의 반복적 과정으로 검출 강인성 강화  
- 다양한 데이터셋(CIFAR, MNIST, CelebA-HQ 등)에서 최첨단 성능 및 일관된 우수성 입증  

***

## 1. 문제 정의

- **Unsupervised OOD Detection**: 라벨 없이 주어진 in-domain 데이터 분포 $$D$$만을 학습하여, 테스트 샘플 $$x$$가 $$D$$로부터 생성되었는지 여부를 판단.  
- 기존 방법들은 사전 분포 모형화(likelihood), 재구성 오류(autoencoder), 특징 기반 거리(Mahalanobis) 등 다양한 접근을 사용하나, 확산 모델의 도메인 매핑 능력은 미활용 상태.

***

## 2. 제안 방법: Lift, Map, Detect (LMD)

1) **Lift (마스킹)**  
   - 입력 이미지 $$x_{\text{orig}}$$를 N×N 격자 기반의 **체커보드 마스크** $$M$$로 가려, 이미지가 원래 manifold에서 벗어나도록 함.  
2) **Map (Inpainting)**  
   - in-domain 데이터로 학습된 확산 모델 $$\theta_{\text{in}}$$을 사용해, 마스크 영역을 inpainting으로 채워 넣음.  
   - 이때 생성된 재구성 이미지 $$x_{\text{inp}}$$가 in-domain manifold로 매핑됨.  
3) **Detect (거리 측정)**  
   - 원본 $$x_{\text{orig}}$$과 재구성 $$x_{\text{inp}}$$ 사이의 **LPIPS** 거리 $$d(x_{\text{orig}}, x_{\text{inp}})$$를 계산.  
   - **r**회 반복하여 거리들의 중앙값 $$\mathrm{median}(d_1, \dots, d_r)$$ 을 최종 OOD 점수로 사용.  

수식으로 정리하면, 각 반복 $$i$$:  

$$
d_i = \mathrm{LPIPS}\bigl(x_{\mathrm{orig}},\; \mathrm{Inpaint}(x_{\mathrm{orig}}, M_i; \theta_{\mathrm{in}})\bigr)
$$

최종 OOD 점수:

$$
s(x) = \mathrm{median}\bigl(d_1, d_2, \dots, d_r\bigr)
$$

***

## 3. 모델 구조 및 구현

- 확산 모델: DDPM 계열(backbone: U-Net)  
- 마스킹: 기본은 **8×8 alternating checkerboard** (두 번의 invert로 전체 영역 커버)  
- 반복 횟수 $$r=10$$, 거리 측정은 LPIPS  
- 학습: CIFAR10/100, SVHN, MNIST 계열은 자체 학습, CelebA-HQ는 FFHQ 사전학습 체크포인트 사용  

***

## 4. 성능 및 비교

- **ROC-AUC** 기준, 12개 데이터셋 쌍 평균 0.907로 경쟁 기법(Pretrained, Likelihood Regret 등) 대비 우수  
- CIFAR100 vs. SVHN에서는 10%p 이상의 큰 폭 성능 향상  
- 고해상도(CelebA-HQ vs. ImageNet)에서도 0.991의 탁월한 분리도 확보  
- 마스킹 크기, 거리 측정(MSE, SSIM, SimCLRv2) 등 다양한 설정에서 안정적  

***

## 5. 한계 및 일반화 가능성

- **속도**: 확산 모델의 반복 샘플링→inpainting으로 인해 실시간 적용에는 부적합  
- **데이터 다양성 의존**: in-domain manifold가 복잡·다양할수록 작은 마스크에서도 정확 재구성이 어려워질 수 있음  
- **일반화**:  
  - 다양한 데이터 도메인에 대해 일관되게 우수한 ROC-AUC를 보였으나, 매우 낮은 해상도나 특수 영역(의료 영상 등)에서는 추가 검증 필요  
  - **일반화 성능 향상**을 위해 더 빠른 샘플링 기법(Nichol & Dhariwal, Watson et al.) 또는 마스크 패턴 최적화 연구가 유망  

***

## 6. 향후 연구 방향 및 고려 사항

- **고속화**: 샘플링 스텝 축소, pseudo numerical methods, fast samplers 통합  
- **마스크 자동화**: 최적 마스크 패턴·크기 학습을 통한 적응적 lifting  
- **다양한 거리 측정**: SimCLRv2 등 사전학습 특징 기반 거리 함수를 상황별 동적 선택  
- **도메인 확장**: 비정형 영상(의료, 위성)·시계열·텍스트 등 멀티모달 OOD 검출에 LMD 원리 적용  

LMD는 확산 모델의 핵심 특성인 도메인 매핑 능력을 OOD 검출에 최초로 전환시킨 접근으로, 차세대 비지도형 검출 기법의 새로운 기준을 제시한다. 앞으로의 연구에서는 속도, 적응성, 범용성 강화를 통해 실용적 OOD 솔루션으로 발전시킬 수 있을 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/48d2162b-711e-435b-bd12-eabe238e21a1/2302.10326v2.pdf
