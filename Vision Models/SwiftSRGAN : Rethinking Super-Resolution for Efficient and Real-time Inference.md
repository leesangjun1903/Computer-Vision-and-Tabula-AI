# SwiftSRGAN : Rethinking Super-Resolution for Efficient and Real-time Inference | Super-resolution

**핵심 주장**  
SwiftSRGAN은 기존 SRGAN(및 ESRGAN) 대비  
1. 모델 크기를 1/8로 줄이면서  
2. 실시간(60FPS 이상) 추론이 가능한 초경량 GAN 기반 초해상화 모델을 제안한다.  

**주요 기여**  
- Depth-wise Separable Convolution을 전 레이어에 도입하여 파라미터 수와 연산량을 대폭 절감  
- 채널 결합용 1×1 Point-wise Convolution과 분리된 Depth-wise Convolution 구조 채택  
- VGG19 기반 퍼셉추얼 손실 대신 MobileNetV2 기반 MobileNet Loss를 적용하여 학습·추론 효율 개선  
- 표준 벤치마크(Set5, Set14)에서 기존 SRGAN·ESRGAN과 유사한 PSNR/SSIM 성능 유지  
- 270p→1080p 업샘플링 시 5.6ms/frame, 540p→4K 업샘플링 시 16.2ms/frame 실현  

***

## 1. 해결하고자 하는 문제  
초고해상화 GAN은 일반적으로 고성능 GPU를 필요로 하며, 추론 속도가 느려 모바일·임베디드 실시간 애플리케이션에 부적합하다.  
- **목표**: 저사양 GPU·모바일 환경에서도 초해상화 결과의 질을 크게 손상시키지 않고 실시간 처리 가능하도록 경량화  

***

## 2. 제안하는 방법  

### 2.1 모델 구조  
- **Generator**  
  - 입력: 3×256×256 LR 이미지 → 출력: 3×1024×1024 SR 이미지  
  - 16개의 Residual Block (각 Block: Depth-wise Conv → 배치 정규화 → PReLU)  
  - 두 번의 업샘플 블록(Depth-wise Separable Conv + PixelShuffle×2 + PReLU)  
  - 최종 9×9, 출력 채널 3, stride=1 Depth-wise Separable Conv  
- **Discriminator**  
  - 8개의 Depth-wise Separable Conv 블록(첫 블록 배치 정규화 없음)  
  - Adaptive Average Pooling → flatten → 1024-뉴런 FC → real/fake 판별  

### 2.2 손실 함수  
총 손실 $$ \ell_{SR} $$ 은 **Content Loss**와 **Adversarial Loss**의 가중합으로 정의된다:  

$$
\ell_{SR} = \ell_{content} + 10^{-3}\,\ell_{adv}
$$  

1) **Content Loss (MobileNet Loss)**  

$$
\ell_{content} = \frac{1}{W_i H_i} \sum_{x=1}^{W_i}\sum_{y=1}^{H_i}\bigl\|\phi_i(I_{HR})\_{x,y} - \phi_i(G_\theta(I_{LR}))_{x,y}\bigr\|^2
$$  

– $$ \phi_i $$: MobileNetV2 16번째 블록 활성화 출력  

2) **Adversarial Loss (Generator)**  

$$
\ell_{adv} = -\sum_{n=1}^N \log D_\theta\bigl(G_\theta(I_{LR}^{(n)})\bigr)
$$  

***

## 3. 성능 향상 및 한계  

| 항목          | SRGAN      | ESRGAN     | SwiftSRGAN (제안) |
|---------------|------------|------------|-------------------|
| PSNR (Set5)   | 29.40      | 32.70      | 25.13             |
| SSIM (Set5)   | 0.8501     | 0.9011     | 0.7940            |
| PSNR (Set14)  | 26.02      | 28.70      | 23.29             |
| SSIM (Set14)  | 0.7397     | 0.7917     | 0.7012            |
| 추론 속도     | 812ms/frame| 974ms/frame| 5.605ms/frame     |
| (270p→1080p)  |            |            |                   |

- **장점**:  
  - 초경량화(모델 크기 1/8)로 메모리·연산량 절감  
  - 실시간 스트리밍·모바일 적용 가능  
- **한계**:  
  - PSNR·SSIM 절대값은 기존보다 낮아, 화질 극대화가 목적일 때는 부적합  
  - 훈련 데이터셋 규모(DIV2K+Flickr2K, 3,669장)가 ImageNet 기반 방법보다 작음  

***

## 4. 일반화 성능 향상 가능성  
- **데이터 규모 확장**: ImageNet 등 대규모 데이터셋으로 사전훈련을 실시하면 일반화 성능 개선 기대  
- **다중 스케일 손실**: 여러 레벨의 MobileNet 블록 활성화를 결합한 멀티스케일 편차 손실 적용  
- **도메인 적응**: 의료·위성 등 특화 도메인에 맞춘 파인튜닝으로 자연·비자연 영상 모두에 범용화  
- **Self-ensembling**: 추론 단계에서 여러 크기로 입력을 재랜더링해 앙상블 처리  

***

## 5. 향후 연구 방향 및 고려 사항  
- **더 풍부한 손실 설계**: 주파수 영역 손실(예: FES Loss)이나 인물·텍스트 특화 지각 손실 도입  
- **하드웨어 최적화**: 모바일 NPU·FPGA·ASIC에 맞춘 커널 최적화 및 양자화 기법 적용  
- **비지도·약지도 학습**: 레이블 없는 저해상도 영상으로도 학습 가능한 손실 함수를 개발  
- **안전성·공정성**: SR 과정 중 생성 왜곡 및 개인정보 노출 가능성 최소화  

SwiftSRGAN은 “경량·실시간”이라는 새로운 설계 목표를 제시하여, 모바일·임베디드 환경의 초해상화 연구에 중요한 이정표가 될 것이다. 앞으로 다양한 데이터·도메인으로 확장 적용하며, 성능과 효율의 균형을 더욱 개선하는 연구가 필요하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/00a6ec10-1e69-4c35-bf89-a9c1cc952fb9/2111.14320v1.pdf
