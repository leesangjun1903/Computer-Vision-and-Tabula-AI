# HLLHGAN : To learn image super-resolution, use a GAN to learn how to do image degradation first | Super resolution

# 핵심 요약  
**“To learn image super-resolution, use a GAN to learn how to do image degradation first”** 논문의 핵심 주장은 실제 저해상도(real-world LR) 이미지의 다양한 열화(degradation) 과정을 명시적으로 모델링하는 대신, High-to-Low GAN을 통해 이러한 열화를 학습하고, 이를 Low-to-High GAN 학습에 활용함으로써 실제 환경에서의 초해상도 성능을 크게 향상시킬 수 있다는 것이다[1].  

주요 기여  
- 실제 저화질 얼굴 이미지의 복잡한 열화 과정을 **비지도(unpaired) 방식**으로 학습하는 High-to-Low GAN 제안  
- High-to-Low GAN이 생성한 페어(열화된 LR ↔ 원본 HR)를 이용해 Low-to-High GAN을 **지도(paired) 학습**하는 2단계 파이프라인 구축  
- GAN 손실을 주도적으로 활용하고 L2 픽셀 손실을 보조적으로 배치하여, 잡음 제거 및 세부 디테일 복원을 동시에 달성  

# 문제 정의  
기존 초해상도 연구는 HR 이미지를 단순한 블리니어 다운샘플링 또는 블러 후 다운샘플링하여 LR 이미지를 생성한 뒤 학습하였다. 그러나 실제 환경의 LR 이미지에는 모션 블러, 압축 아티팩트, 노이즈, 조명 변화 등 복합적인 열화 요소가 존재하며, 기존 방식으로는 이에 대응할 수 없어 복원 품질이 저하된다[1].  

# 제안 방법  

## 1단계: High-to-Low GAN 학습  
- **목적**: 실제 LR 이미지의 열화 분포를 학습  
- **데이터**:  
  - HR 데이터셋 (CelebA, AFLW, LS3D-W, VGGFace2 합산 약 183K장)  
  - 실제 LR 데이터셋 (Widerface에서 50K장, 그중 3K장 테스트용 분리)  
- **네트워크 구조**:  
  - Generator: HR 이미지 + 노이즈 벡터 $$z$$ 입력, 12개의 ResNet 블록, 인코더-디코더 형태로 $$64\!\times\!64\to4\!\times\!4\to16\!\times\!16$$  
  - Discriminator: 6개의 ResNet 블록, 입력 $$16\!\times\!16$$  
- **손실 함수**:  

$$
    \ell = \alpha \ell_{\text{pixel}} + \beta \ell_{\text{GAN}}
  $$ 
  
  - GAN 손실: Spectral Normalization GAN의 힌지 손실  
  - 픽셀 손실: HR 이미지 풀링 후 L2 거리  
  - 하이퍼파라미터: $$\alpha=1, \beta=0.05$$[1]  

## 2단계: Low-to-High GAN 학습  
- **목적**: High-to-Low GAN이 생성한 현실적 LR 이미지를 이용해 고품질 초해상도 학습  
- **데이터**: High-to-Low 출력 LR ↔ 대응 HR  
- **네트워크 구조**:  
  - Generator: 17개의 ResNet 블록, 스킵 연결, $$16\!\times\!16\to64\!\times\!64$$ 업샘플링  
  - Discriminator: High-to-Low와 유사, 추가 풀링  
- **손실 함수**: 동일한 형태의 GAN+픽셀 손실, GAN 손실이 메인 역할(잡음 제거), 픽셀 손실이 내용 보존[1]  

# 모델 구조  
| 단계 | Generator 구조 | Discriminator 구조 | 입력 해상도 → 출력 해상도 |
|:-----|:--------------:|:------------------:|:----------------------------:|
| High-to-Low | ResNet 12 blocks, noise concat | ResNet 6 blocks | 64×64 → 16×16 |
| Low-to-High | ResNet 17 blocks, skip connections | ResNet 6+ blocks | 16×16 → 64×64 |  

# 성능 향상  
- **FID**: 제안 기법 FID=14.89로, SRGAN(104.8), CycleGAN(19.0) 등 대비 우수[1]  
- **PSNR (합성 평가)**: LS3D-W에서 PSNR≈19.3 dB로, 순수 paired 학습 대비 약간 낮으나 실제 LR 복원에서 질적 우위 확보[1]  
- **정량·정성 평가**: 실제 저화질 얼굴에서 디테일 복원, 노이즈 제거 및 자연스러운 질감 재현에서 탁월  

# 한계 및 실패 사례  
- 입력이 얼굴 형태를 벗어나거나 극단적 블러·포즈·폐색(occlusion)에서는 복원이 왜곡되거나 실패율 약 10% 발견[1]  
- HR 데이터의 극단 포즈는 대부분 합성 데이터로, 실제 분포와 차이가 존재  
- 잡음 모델이 Widerface 데이터에 종속적 → 다양한 도메인 일반화 한계  

# 일반화 성능 향상 가능성  
- **다양한 객체 범주 적용**: 얼굴 외 차량·자연물 등, unpaired LR·HR 데이터만 확보 시 파이프라인 확장 가능  
- **도메인 분산 강화**: High-to-Low에 여러 LR 도메인(압축, 저조도, 센서 노이즈) 학습 추가로 일반화 제고  
- **적응형 노이즈 입력**: 노이즈 벡터 $$z$$를 고정 분포가 아닌 입력 이미지 특성에 맞춰 조건화하여 열화 표현 다양성 극대화  

# 향후 연구에 미치는 영향 및 고려 사항  
- **열화 모델 학습 패러다임**: 복합 열화 현실적 학습→초해상도 학습 적용 사례 확산  
- **Unpaired 방식 활용**: 대응 이미지 페어 수집 어려운 분야(위성, 의료 영상 등)에 적용 가능성  
- **도메인 적응 연구 연계**: High-to-Low 출력과 실제 LR 도메인 간 도메인 갭 해소 기술 필요  
- **노이즈·아티팩트 해석**: GAN 기반 열화 과정의 해석 가능성 및 안전성 평가 고려  

[1] Adrian Bulat et al., “Learning image super-resolution via learning image degradation,” arXiv:1807.11458v1.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9d16aa11-236d-4035-a1b9-ade185eb9b5e/1807.11458v1.pdf
