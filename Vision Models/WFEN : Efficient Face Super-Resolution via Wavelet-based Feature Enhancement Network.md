# WFEN : Efficient Face Super-Resolution via Wavelet-based Feature Enhancement Network | Super resolution

## 1. 핵심 주장 및 주요 기여  
이 논문은 **파형 변환(wavelet transform)** 을 이용해 저해상도 얼굴 영상의 특징 손실을 최소화하면서 효율적으로 고해상도 얼굴 영상을 복원하는 새로운 네트워크(WFEN)를 제안한다.  
- **Wavelet Feature Downsample (WFD)**: 인코더 단계에서 입력 특징을 저·고주파 성분으로 분리하여 손실 없는 다운샘플링 수행  
- **Wavelet Feature Upgrade (WFU)**: 디코더 단계에서 인코더의 다중 스케일 정보를 주파수 성분별로 통합해 업샘플링  
- **Full-domain Transformer (FDT)**: 저주파 특징을 로컬·리저널·글로벌 수준에서 동시에 추출하도록 설계된 셀프 어텐션 구조  
- 이로써 종합 성능(PSNR, SSIM, LPIPS, VIF, ID 정확도)과 모델 규모·연산량·추론 속도 간 최적의 균형을 달성  

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- 기존 FSR(face super-resolution)에서는 인코더의 다운샘플링(스트라이드 컨볼루션·평균 풀링·보간)으로 인해 **고주파(에지·텍스처) 정보가 소실**되거나 aliasing이 발생  
- Transformer 기반 방법들은 로컬·글로벌 특징을 분리 추출하지만, **리저널(눈·코·입 등) 정보** 포착이 미흡  

### 2.2 제안 모델 구조  
인코더–병목(bottleneck)–디코더 구조  
1) Shallow feature extraction  
2) **Wavelet Feature Downsample (WFD)**  
   - 입력 $$F_s\in\mathbb{R}^{H\times W\times C}$$에 대해 DWT 실행
   
$$
       \{A_{LL}, H_{LH}, V_{HL}, D_{HH}\} = \mathrm{WT}(F_s)
$$  

   - 저주파 성분 $$A_{LL}$$은 FDT 처리, 고주파 성분 $$\{H,V,D\}$$는 잔차블록(residual block) 처리  
   - 두 경로 출력 융합 후 크기 절반으로 다운샘플링  
3) **Full-domain Transformer (FDT)**  
   - **Regional Self-Attention (RSA)**: 작은 윈도우 나눔→로컬·리저널 특징 포착→윈도우 간 shift로 연결성 확보  
   - **Global Self-Attention (GSA)**: 채널 분할 multi-head 어텐션으로 글로벌 상관관계 모델링→head shuffle로 정보 교환  
4) **Wavelet Feature Upgrade (WFU)**  
   - 디코더에서 상위 스케일 특징에 DWT 적용→저주파와 디코더 출력 결합, 고주파 보강 후 IWT로 복원  
5) Residual 연결 및 1×1 컨볼루션을 통한 최종 HR 얼굴 이미지 생성  

### 2.3 수식 요약  
- WFD:

$$
    \{A_{LL},H_{LH},V_{HL},D_{HH}\} = \mathrm{WT}(F_s),\quad
    F_{\text{low}}=T(A_{LL}),\;F_{\text{high}}=R(\{H,V,D\})
$$  

- WFU (스케일 상향):
 
$$
    \{A_{LL},H_{LH},V_{HL},D_{HH}\} = \mathrm{WT}(F_s),\quad
    F'\_s = \mathrm{IWT}\bigl(\text{concat}(A_{LL},F_{s+1}),\{H,V,D\}\bigr)
$$  

- RSA/GSA:

$$
    \mathrm{Attention}(Q,K,V)=V\,\mathrm{ReLU}\Bigl(\tfrac{QK^\top}{\alpha}\Bigr)
  \quad\text{or}\quad
    V\,\mathrm{ReLU}\Bigl(\tfrac{QK^\top}{\beta}\Bigr)
$$

## 3. 성능 향상 및 한계  
- **PSNR 28.04 dB, SSIM 0.8032** (CelebA)로 기존 최고 Restormer-M(27.94 dB) 대비 우위, 모델 크기 6.8 M·FLOPs 7.5 G·추론 33.9 ms로 효율성도 확보  
- **SCface 실감 시나리오**에서도 ID 일치율 0.725~0.742로 최상위  
- **한계**:  
  - 얼굴 외 일반 물체 SR로 확장성 불명  
  - 복합 조명·표정 변화·비정면(비프론트얼) 상황 성능 분석 부족  
  - DWT 기반 모듈이 특정 해상도 배율(×8) 이상에서만 검증  

## 4. 모델 일반화 및 향후 연구 고려사항  
- **일반화 가능성**  
  - 본 논문도 여러 FSR 네트워크에 WFD+WFU를 결합하여 평균 PSNR+0.1 dB 이상 향상시켜 도입 이식성 입증  
  - 잔차블록과 Transformer 모듈을 다른 도메인 SR에 적용해볼 여지  
- **향후 연구 시 고려할 점**  
  1. **다양한 해상도 배율 및 비정면 얼굴**에 대한 DWT 모듈의 적응성 연구  
  2. **비얼굴 SR**(자연·의료 영상 등)으로 일반화하여 저·고주파 정보 분리의 가치를 검증  
  3. **학습된 파형 기저**를 end-to-end로 학습하거나 대체 변환(e.g., 푸리에)과 비교 분석  
  4. **실시간 적용**을 위한 하드웨어 최적화(모델 경량화·양자화) 및 동적 배율 지원  
  5. **표정·조명·포즈 변화**에 강건한 평가 지표·데이터셋 확충  

---
**결론:** WFEN은 DWT 기반 다운·업샘플링과 전 영역(Self-, Regional-, Global) 어텐션을 결합해 얼굴 SR의 **성능과 효율성**을 동시에 개선했으며, 다양한 SR 모델에 모듈 형태로 확장 적용 가능해 후속 연구에 유용한 설계 원칙을 제시했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/016e5057-1d68-4752-863b-07fbbb734e3f/2407.19768v2.pdf
