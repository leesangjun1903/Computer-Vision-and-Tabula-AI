# SwinIR: Image Restoration Using Swin Transformer | Image reconstruction, Super resolution, Image denoising

**SwinIR**(Swin Transformer 기반 이미지 복원 모델)은 기존 CNN 기반 복원 방법의 한계를 극복하고, 적은 파라미터로 더 뛰어난 성능을 내는 범용 이미지 복원 시스템을 제안한다.  
1. **핵심 주장**  
   - *콘텐츠 기반 가변 컨볼루션*과 *장거리 의존성 모델링*을 Swin Transformer의 지역-이동 윈도우(self-attention) 구조로 통합해, 이미지 복원에서 CNN 대비 우수한 성능을 달성한다.  
2. **주요 기여**  
   - Residual Swin Transformer Block(RSTB) 설계: Swin Transformer Layer와 3×3 컨볼루션, 잔차 연결을 결합한 모듈로, 공간적 불변성과 장거리 상관관계를 동시에 학습.  
   - 단일 아키텍처로 SR(클래식·경량·실세계), 노이즈 제거(흑백·컬러), JPEG 압축 아티팩트 감소 등 **6개 태스크**에서 SOTA 성능 달성(파라미터 11.5M 수준)[1].  
   - **파라미터 효율성**: IPT(115.5M) 대비 10분의 1 이하 파라미터로 유사·우수 성능.  

# 문제 정의  
- **기존 한계**  
  1. CNN의 고정 커널은 이미지 내용에 무관한 연산 → 세밀한 지역별 복원 한계  
  2. 지역처리(local receptive field)로 인해 *장거리 픽셀 상호작용* 미흡  
- **해결 목표**:  
  - 콘텐츠 의존적 필터링 및 윈도우 간 shifted self-attention으로 장·단기 의존성 모두 포착  

# 제안 방법

## 모델 구조 개요  
SwinIR는 세 모듈로 구성된다[1]:  
1. **Shallow Feature Extraction**: 3×3 Conv → 저주파 정보 보존  
2. **Deep Feature Extraction**: K개의 RSTB 쌓음 → 고주파 복원 집중  
3. **Reconstruction**: (F₀ + F_DF) 결합 후 upsampling or Conv → 최종 복원  

## Residual Swin Transformer Block(RSTB)  
- 입력 $$F_{i,0}$$ → L개의 Swin Transformer Layer(지역 M×M self-attention + shifted 윈도우) 순차 처리 → 컨볼루션 → 잔차 연결  
- 수식:

$$
    F_{i,j} = \mathrm{STL}\_{i,j}(F_{i,j-1}),\quad
    F_{i,\mathrm{out}} = \mathrm{Conv}(F_{i,L}) + F_{i,0}
$$  

## 손실 함수  
- SR(클래식·경량): $$L_1$$ 픽셀 손실  
- 실세계 SR: 픽셀 + GAN + 지각적(perceptual) 손실  
- 노이즈/아티팩트 제거: Charbonnier 손실

$$
    L = \sqrt{\lVert I_\mathrm{out}-I_\mathrm{gt}\rVert^2 + \epsilon^2}
$$

# 성능 향상  
- **클래식 SR×4**: DIV2K+Flickr2K 학습 시 타 메소드 대비 최대 +0.20dB[1].  
- **경량 SR×2**: 0.53dB↑, 파라미터·연산량 동급[1].  
- **JPEG q=10–40**: PSNR +0.1~0.2dB, 파라 11.5M vs DRUNet 32.7M[1].  
- **노이즈 제거(σ=50)**: Urban100 기준 +0.3dB 우위[1].  

# 한계 및 일반화  
- **한계**:  
  - 복잡도: SwinIR 중·대형 모델은 연산량(1.1s@1k²) 여전히 높음.  
  - 실세계 SR에서 복원 왜곡(border artifact) 및 세밀한 사실감 부족 가능성  
- **일반화 성능 향상 가능성**:  
  - SwinIR 구조는 *shifted 윈도우*로 다양한 해상도·잡음 패턴에 적응 가능 → 다양한 복원 태스크 간 **전이 학습** 용이  
  - RSTB 내 컨볼루션+잔차 연결이 CNN의 inductive bias 유지 → 제한적 데이터 환경에서도 안정적 학습 및 **빠른 수렴** 확인[1]  

# 향후 연구 및 고려사항  
- **멀티 태스크 확장**: Dehazing·Deblurring·Deraining 등 추가 복원 과제에 SwinIR 적용  
- **효율화**:  
  - 윈도우 크기·레이어 수 자동 최적화  
  - 경량화: 동적 윈도우·양자화 기법 도입  
- **실세계 데이터**: 합성 저해상도·잡음 모델 외, **도메인 격차** 줄이기 위한 실제 촬영 데이터 확보 및 도메인 적응 연구 필요  
- **융합 기술**: GAN·Diffusion 모델과 결합해 시각적 사실감 및 질감 재현력 강화  

**참고**  
[1] Liang et al., “SwinIR: Image Restoration Using Swin Transformer,” arXiv:2108.10257, 2021.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/543257f3-8167-4217-8c96-5f10636fadd7/2108.10257v1.pdf
