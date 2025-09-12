# BRDNet : Image Denoising Using Deep CNN with Batch Renormalization | Image denoising

## 1. 핵심 주장 및 주요 기여 (간결 요약)
“Image denoising using deep CNN with batch renormalization” 논문은 **배치 정규화(batch normalization)의 소형 배치 한계**를 극복하기 위해 **배치 리노멀라이제이션(batch renormalization)**을 도입하고, **잔차 학습(residual learning)** 및 **팽창 합성곱(dilated convolution)** 기법을 결합한 **BRDNet** 구조를 제안한다. 이를 통해 기존의 심층 CNN 기반 모델 대비 **학습 안정성과 성능 포화 해소**, **연산 비용 절감**, **소형 배치에서도 높은 일반화 성능**을 달성함을 보인다.[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제
전통적인 심층 CNN 기반 이미지 디노이징은  
- **네트워크이 깊어질수록** 학습 시 **그래디언트 소실/폭발** 문제가 발생  
- **배치 정규화(BN)** 는 소형 미니배치에서는 통계 추정 오류로 **성능 저하**  
- **깊이 증가** 시 **연산 비용 증가** 및 **학습 수렴 포화** 문제  
이 논문은 위 문제를 통합적으로 해결하고자 한다.[1]

### 2.2 제안 방법 및 수식
- **배치 리노멀라이제이션(BRN)**  
  - BN의 배치 통계(μ, σ) 대신 **개별 샘플 통계**를 부분 보정하여 사용  
  - 보정 계수 $$r = \frac{\sigma_B}{\sigma}$$, $$d = \frac{\mu_B - \mu}{\sigma}$$를 제한하여,  

$$
      \hat{x}_i = \frac{x_i - \mu_B}{\sigma_B} \cdot r + d,\quad
      y_i = \gamma \hat{x}_i + \beta
    $$  
    
형태로 정규화 수행.[1]

- **잔차 학습(Residual Learning)**  
  - $$f(y;\theta)$$를 노이즈 맵으로 예측하고,  

$$
      \mathcal{L}(\theta) = \frac{1}{2N}\sum_{j=1}^N \|f(y_j;\theta) - (y_j - x_j)\|^2
    $$  
    
  으로 학습. 최종 복원은 $$x = y - f(y;\theta)$$.[1]

- **팽창 합성곱(Dilated Convolution)**  
  - 필터 사이 공백을 두어 **수용 영역(receptive field)**을 비례 확대해 깊이 증가 없이 맥락 정보 획득  
  - 예: 팽창 계수 2, 14개 레이어에 적용 시 사실상 30레이어 수준 효과.[1]

### 2.3 모델 구조
BRDNet은 **두 개의 서브네트워크**(각각 17개 레이어) 병렬 결합 구조로,  
1. **Upper Network**: Conv+BRN+ReLU ×16 → Conv  
2. **Lower Network**: Conv+BRN+ReLU → Dilated Conv ×7 → Conv+BRN+ReLU → Dilated Conv ×6 → Conv  
  
두 경로 출력 채널을 Concatenate 후 Conv → 잔차 학습으로 노이즈 제거.[1]

### 2.4 성능 향상
- **소형/대형 노이즈(σ=15–75)** 모두에서 BM3D, DnCNN, FFDNet 대비 최고 PSNR 달성 (e.g., BSD68 σ=25에서 +0.72 dB).[1]
- **소형 배치(size=20)** 환경에서도 BN 대비 +0.14 dB 향상, 소형 GPU에서도 안정적 수렴 확인.[1]
- **연산 효율성**: 1024×1024 이미지 GPU 처리 0.788 s, DnCNN(0.410 s) 대비 약간 느리지만 두 DnCNN 대비 파라미터 및 FLOPs 동일하면서 일반화 우위.[1]

### 2.5 한계
- 복잡한 **Real-world 노이즈**(저조도, 블러)에는 추가적인 **신뢰도** 필요  
- 두 서브네트워크 병렬 구조로 메모리 사용량 증가  
- **Dilated convolution**의 격자 아티팩트(grid artifact) 가능성 미완전 검증

## 3. 일반화 성능 향상 관점
BRDNet은 다음 요소로 일반화 성능을 개선한다.  
- **BRN**: 배치 통계 추정 오차 감소로 **비균질·소형 배치**에서도 안정적 정규화  
- **Residual Learning**: 깊이에 독립적 손실 함수 설계로 **학습 안정성** 확보  
- **Dilated Conv**: 다양한 수용 영역 정보 통합으로 **공간적 문맥** 활용  
- **Width 증가**: 메모리·연산 대비 특징 다양성 강화로 **과적합 억제**

이로써 **다양한 노이즈 분포** 및 **제한적 학습 자원** 환경에서 **일반화** 우수성을 갖춘다.[1]

## 4. 향후 연구 영향 및 고려 사항
- **배치 정규화 대안** 연구에 브랜치: BRN 외 **Layer/Group Normalization**과 비교 가능성  
- **실제 촬영 노이즈**(저조도·모션 블러·압축 노이즈) 처리 위한 **Prior 통합** 연구  
- **경량화 모델**: 모바일·임베디드 기기 적용 위해 **채널 프루닝**·지식 증류 결합  
- **격자 아티팩트** 완화 및 **다중 스케일 팽창** 조합으로 더욱 강건한 **수용 영역 설계**  
- **데이터 불균형** 상황에서 BRN 통계 안정성 검증 및 **온라인 학습** 시나리오 고려

이 논문은 **소형 배치, 제한된 자원 환경**에서도 **안정적이고 강건한 디노이징** 가능성을 입증하며, 배치 정규화 대체 기법 연구와 **실제 임상·산업 적용**을 위한 후속 연구의 초석이 될 것이다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/149aeb44-3daa-4b48-a99a-5df8339b1273/1-s2.0-S0893608019302394-main.pdf)
