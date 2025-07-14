# NAFNet: Simple Baselines for Image Restoration | Image restoration, Image denoising

## 1. 핵심 주장 및 주요 기여  
이 논문은 **복잡한 비선형 활성화 함수와 지나치게 정교한 모듈 없이**, 매우 단순한 구성 요소만으로도 이미지 복원 분야의 최신 성능을 경신할 수 있음을 보인다.  
- **Simple Baseline**: 단일 단계 U-Net 구조에 LayerNorm, 3×3 깊이별(depthwise) 합성곱, GELU, 채널 어텐션(SE Block)만을 사용하여 기존 SOTA를 초과 달성.  
- **NAFNet (Nonlinear Activation Free Network)**: Simple Baseline에서 GELU와 SE 모듈의 비선형성을 각각 **SimpleGate** (두 채널 분할 후 곱셈)과 **Simplified Channel Attention** (전역 평균풀링 후 1×1 합성곱)으로 바꾸어 ‘비선형 활성화 함수 전무’ 네트워크를 제안.  
- **성능**: GoPro 데이터셋에서 33.69 dB PSNR, SIDD에서 40.30 dB PSNR 기록하며 종전 대비 0.38 dB, 0.28 dB 상승, 계산량은 8.4%–50% 수준으로 크게 절감[1][1].  

## 2. 해결 문제, 제안 방법, 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
기존 SOTA 이미지 복원(노이즈 제거·블러 제거) 모델들은  
- **Inter-block**(다단계 네트워크·멀티스케일 퓨전) 및  
- **Intra-block**(복잡한 어텐션·GELU·GLU 등)  
복잡도를 지나치게 높여 실험·비교·재현이 어려움.  

### 2.2 제안 방법  
1. **Simple Baseline 구성**[Fig. 3c][1]:  
   - 단일 U-Net 구조  
   - 블록(Block): LayerNorm → 3×3 깊이별합성곱 → GELU → 1×1 점합성곱 → 채널 어텐션(SE) → 잔차 연결  
2. **NAFNet**:  
   - **SimpleGate**: 채널을 절반으로 분할한 두 텐서를 요소별 곱( $$\mathrm{SG}(X)=X_1 \odot X_2$$ )으로 대체[Eq 4][1].  
   - **Simplified Channel Attention (SCA)**: SE 모듈의 MLP와 Sigmoid 제거, 전역 평균풀링 뒤 1×1 합성곱만 수행[Eq 7][1].  
   - 네트워크 전반에서 ReLU/GELU/Sigmoid 등 모든 비선형 활성화를 제거.  

### 2.3 모델 구조  
```
Input → [Encoder] → Bottleneck → [Decoder] → Output
```
- **블록 반복**: 기본 36개  
- **Skip Connection**: 인코더↔디코더에서 원소별 덧셈  
- **다운/업샘플링**: 2×2 stride-2 합성곱, PixelShuffle  
- **블록 내부**:  
  - PlainNet: Conv–ReLU–Conv  
  - Baseline: + LayerNorm + GELU + SE  
  - NAFNet: + LayerNorm + SimpleGate + SCA  

### 2.4 성능 향상  
- **연산량 대비 PSNR**:  
  | 모델       | GoPro PSNR | SIDD PSNR | MACs (G)   |
  |-----------|-----------|----------|-----------|
  | Restormer | 32.92 dB  | 40.02 dB | 140       |
  | Baseline  | 33.40 dB  | 40.30 dB | 84        |
  | NAFNet    | **33.69 dB** | **40.30 dB** | **65**     |

- **속도/메모리**: 단일 2080Ti에서 U-Net 대비 1.09× 가속, 메모리 오버헤드 미미[Table 2][1].  

### 2.5 한계  
- **초기 학습 불안정**: PlainNet은 높은 학습률 시 발산. LayerNorm으로 해결했으나, 초기 설정 민감도 존재.  
- **비선형성 제거의 일반성**: JPEG 아티팩트·RAW 영상·저조도 등 다양한 환경에서 검증했으나, 극단적 왜곡 상황에선 성능 한계 가능성.  
- **모델 크기 확장성**: 블록 수 증가 시 성능 포화(36→72 블록) 및 지연 증가, 최적 블록 수 탐색 필요[Table 3][1].  

## 3. 모델의 일반화 성능 향상 가능성  
- **GLU 관점**: SimpleGate는 채널 간 상호작용으로 비선형성을 네이티브하게 확보. 이로써 네트워크가 **특정 활성화 함수에 과도하게 의존하지 않아**, 다양한 복원 문제에 **보다 견고하게 일반화** 가능함.  
- **경량화된 SCA**: SE 블록 대신 SCA로 전역 컨텍스트만 유지하므로, 역전파 시 어텐션 스케일 조정 없이도 **다양한 노이즈·블러 패턴**에 적응력 향상 기대.  
- **하이퍼파라미터 공유**: Baseline과 NAFNet이 동일 MACs 구조를 공유하므로, **사전학습된 가중치 전이학습** 시 모델 간 호환성 및 재학습 비용 절감.  

## 4. 향후 연구 영향 및 고려 사항  
- **디자인 간소화 장려**: 복잡한 어텐션·활성화 없이도 SOTA 달성 가능성을 보였으므로, 향후 복원 모델 설계 시 **불필요한 모듈 배제**가 권장됨.  
- **새로운 게이팅 메커니즘 탐색**: SimpleGate 외에도 **다양한 게이트 구조**(채널별·공간별) 실험을 통해 일반화 성능 추가 개선 여지.  
- **이론적 해석 강화**: 비선형 활성화 제거 시에도 높은 성능이 유지되는 **이유와 한계**를 수학적·정보 이론적 관점에서 분석 필요.  
- **더 넓은 도메인 적용**: 자율주행·의료영상 등 **도메인 특화 복원** 과제에서 NAFNet의 효용성과 튜닝 전략 연구.  

---  
1. Chen et al., “Simple Baselines for Image Restoration,” arXiv:2204.04676v4 [cs.CV] (Aug 1, 2022)[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3abe02a5-1620-4c58-9ec9-bd4414f1188d/2204.04676v4.pdf
