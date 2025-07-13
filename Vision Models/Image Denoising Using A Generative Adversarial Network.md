# Image Denoising Using A Generative Adversarial Network | Image denoising

**핵심 주장**  
이 논문은 GAN 기반의 딥러닝 모델을 활용하여 Monte Carlo 경로 추적(path tracing)으로 생성된 노이즈 이미지를 실시간에 가까운 속도로 고품질의 포토리얼리즘 이미지로 복원할 수 있음을 보인다[1].  

**주요 기여**  
- 적은 샘플 수(1‒8 샘플)로 렌더링된 노이즈 이미지를 GAN을 통해 1초 미만에 고품질 이미지로 복원  
- ResNet 기반의 심층 생성기(generator)와 판별기(discriminator) 구조 설계  
- 픽셀 손실(pixel loss), 특징 손실(feature loss), 부드러움 손실(smooth loss), 적대 손실(adversarial loss)을 결합한 정제된(perceptual) 손실 함수 제안  
- 도메인 외(사진, CT 스캔) 일반화 성능 실험 및 성공적 복원 사례 제시  

# 문제 정의 및 제안 기법  

## 해결하고자 하는 문제  
- **고품질 경로 추적의 계산 비용**: Monte Carlo 방식으로 1픽셀당 수천 ~ 수만 레이(ray)를 쏘는 기존 방법은 8~16시간/프레임 소요  
- **노이즈 제거**: 샘플 수를 극소화하면 빠르지만 심한 노이즈가 발생하며, 이를 고속으로 제거하는 방법 필요[1].  

## 제안 방법  

1. **네트워크 아키텍처**  
   - Generator:  
     - 앞단: 3개의 Conv+BatchNorm+LReLU  
     - 중간: 3개의 ResNet 블록(각 2× Conv+BatchNorm+LReLU + 스킵 연결)  
     - 후단: 3개의 Sub-pixel Conv (deconvolution) + BatchNorm+LReLU, 최종 Sigmoid  
     - 출력 크기: 64×64 → 256×256  
   - Discriminator:  
     - 5개의 Conv(4×4, stride 2) + BatchNorm+LReLU  
     - 2개의 Conv(4×4, stride 1) + BatchNorm+LReLU  
     - 마지막 Sigmoid로 진위 확률 판별  

2. **정제된 손실 함수**
 
$$
     L = \lambda_a L_{adv} + \lambda_p L_{pixel} + \lambda_f L_{feature} + \lambda_s L_{smooth}
$$  
   - $$L_{adv}$$: GAN 적대 손실  
   - $$L_{pixel}$$: 생성 이미지와 GT 간 픽셀별 MSE  
   - $$L_{feature}$$: VGG16 Conv2 레이어 피처 맵 간 MSE  
   - $$L_{smooth}$$: 인접 픽셀 간 차이로 잔여 아티팩트 억제  
   - 하이퍼파라미터: $$\lambda_a{=}0.5,\;\lambda_p{=}1.0,\;\lambda_f{=}1.0,\;\lambda_s{=}10^{-4}$$[1].  

3. **학습 및 성능**  
   - 데이터: Pixar 프레임 40장 + Gaussian noise(5단계, σ 다양) → 학습 1,000장, 테스트 40장  
   - 하드웨어: AWS p2.xlarge GPU, 배치 7, 10,000 iterations  
   - 결과:  
     - 경로 추적 8–16시간 → 네트워크 1초 미만  
     - 도메인 외 사진·CT 스캔에도 의미 있는 노이즈 제거 성능[1].  

# 모델 구조 및 성능 향상 분석  

| 구성 요소       | 세부 내용                                                                                        |
|--------------|-------------------------------------------------------------------------------------------------|
| Generator    | Conv3→ResBlock×3(스킵 연결)→Sub-pixel Conv3; 입력 64²→출력 256²[1]                                         |
| Discriminator| Conv(4×4, stride2)×5→Conv(4×4, stride1)×2→Sigmoid; 입력 256²→진위 확률 출력[1]                             |
| 손실 함수      | $$L_{adv}$$, $$L_{pixel}$$, $$L_{feature}$$, $$L_{smooth}$$ 조합                                    |
| 시간 단축     | 8–16시간 → <1초                                                                                 |
| 일반화        | 학습 도메인 외 사진, CT 스캔, 영상 프레임에서 노이즈 제거 성공                                        |

# 모델의 일반화 및 한계  

- **일반화 성능**:  
  - 학습에 사용되지 않은 자연광 사진, CT 스캔에도 양호한 복원 성능[1]  
  - GAN의 판별 손실이 실제 분포를 학습하여 도메인 불일치에도 어느 정도 대응  

- **한계**:  
  - 학습 데이터 부족: 40장 소규모 → 다양한 장르·노이즈 유형 학습 필요  
  - 구조 고도화 미흡: Residual 블록 3개 → 블록 수 증가 시 성능↑ 예상하나 학습 비용 증가  
  - 노이즈 종류: Gaussian에 한정 → Monte Carlo 노이즈, 움직임 블러, 그림자, 전역 조명 등 과제  

# 향후 연구 영향 및 고려 사항  

- **연구 영향**:  
  - 실시간 경로 추적 실현 가능성 제시 → 게임 그래픽, 의료 시각화, VR/AR 렌더링 가속  
  - GAN 기반 이미지 처리 분야에서 정제 손실 함수 활용 확장  

- **고려 사항**:  
  - 대규모·다양 노이즈 데이터셋 구축: Monte Carlo, 자연 노이즈, 구조적 노이즈 포함  
  - 네트워크 심화: Residual 블록·어텐션 메커니즘 도입으로 복원 품질 개선  
  - 추가 입력 정보: 깊이 맵, 노말 맵 등 3D 정보 활용하여 의미적 보완  
  - 평가 지표 확장: PSNR/SSIM 외 인지적 품질 평가 및 사용성 테스트 병행  


[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/136ceca7-6972-4a27-8516-111ba31d517c/Image_Denoising_Using_A_Generative_Adversarial_Network.pdf
