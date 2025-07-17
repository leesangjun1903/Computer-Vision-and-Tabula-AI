# BCRN : A Very Lightweight Image Super-Resolution Network | Super resolution, Lightweight

**주요 주장 및 기여**  
이 논문은 **매우 적은 파라미터**로도 기존 경량(super-lightweight) 단일 영상 초해상도(single-image super-resolution, SR) 모델과 대등하거나 우수한 성능을 달성하는 새로운 네트워크 **BCRN**(Blueprint-ConvNeXt Residual Network)을 제안한다.  
1. **초경량화**: BSConv(blueprint separable convolution)과 ConvNeXt residual 구조를 결합한 잔차 블록(BCB)을 통해 모델 파라미터를 289K 수준으로 획기적으로 감소.  
2. **단순 연결 전략**: 복잡한 채널 분할이나 재귀 구조 대신, 6개의 BCB 블록을 단순히 스택한 뒤 Dense 연결으로 다중 계층 피처를 집계.  
3. **이중 주의(attention) 모듈**: *Enhanced Spatial Attention*(ESA)와 *Contrast-aware Channel Attention*(CCA)를 동시 적용하여 중요한 공간·채널 정보를 효과적으로 강조.  
4. **성능**: Set5×4 기준 PSNR 32.23 dB, SSIM 0.8951로, 파라미터·연산량 대비 최상위 성능 달성.  

## 1. 해결하고자 하는 문제  
- 딥러닝 기반 SR 모델들은 대규모 파라미터와 연산량으로 모바일·임베디드 기기에 적용 어려움.  
- 기존 경량 SR 기법(그룹 컨볼루션, 재귀 구조, 채널 분할 등)은 파라미터 감소에 기여했지만,  
  -  복잡한 구조로 구현·추론 부담  
  -  여전히 과도한 연산 발생  
  
**목표**: 간단한 구조·주의 메커니즘 적용을 통해 “초경량” 파라미터 수와 연산량을 유지하면서 SR 성능 저해 없이 개선.

## 2. 제안 방법

### 2.1 네트워크 구성  
전체 모델은 다음 네 단계로 구성된다 (Fig. 2 참조).  
1. **Shallow feature extraction**: 3×3 BSConv로 입력 ILR → 고차원 특징 F₀  
2. **Deep feature extraction**: Fₖ = BCB(Fₖ₋₁), k = 1…6  
3. **Multi-layer feature aggregation**:  

$$F_\text{fused} = H_\text{fusion}(\text{Concat}(F_1, …, F_6))$$  

4. **Reconstruction**:  

$$I_{SR} = H_\text{rec}(F_\text{fused} + F_0)$$  
   
   $H_{rec}$ 은 BSConv + sub-pixel convolution

### 2.2 잔차 블록 BCB  
- **BSConv**(3×3) → **ConvNeXt residual** (그룹 컨볼루션 3×3, inverse bottleneck, GELU)  
- 이어서 ESA, CCA 모듈 순으로 적용  
- 수식:  

$$
    F_{\text{out1}} = \mathrm{BSConv}(F_{\text{in}}),\quad
    F_{\text{out2}} = \mathrm{ConvNeXt}(F_{\text{out1}}),
    \quad F_{\text{out}} = \mathrm{CCA}(\mathrm{ESA}(F_{\text{out2}}))
$$

### 2.3 주의 모듈  
- **ESA**: 1×1 컨볼루션으로 차원 축소 → 스트라이드 컨볼루션·풀링으로 공간 축소 → 업샘플→ attention mask  
- **CCA**: 채널별 평균+표준편차를 결합해 대비 정보 생성 → 1×1 컨볼루션 두 단계로 채널 차원 변환 → sigmoid

## 3. 성능 향상 및 한계

| 구성 | 파라미터 (K) | Multi-Adds (G) | Set5 PSNR(dB) | Urban100 PSNR(dB) |
|:----:|:----------:|:-------------:|:--------------:|:-----------------:|
| BCRN-woESA | 264 | 15 | 32.06 | 25.91 |
| BCRN-woCCA | 285 | 16 | 32.11 | 25.96 |
| **BCRN** | **289** | **16** | **32.16** | **26.04** |

- ESA 제거 시 PSNR 약 –0.10 dB, CCA 제거 시 –0.05 dB [1 Table 1].  
- GELU 활성화가 ReLU 대비 최대 +0.12 dB 기여 [1 Table 2].  
- 6개 BCB 블록이 최적 절충점, 7개 이상 시 이득 미미 [1 Table 3].  
- 타 경량 모델 대비 파라미터 절반 이하, 추론시간 40.8 ms로 가장 빠름[1 Table 4].

**한계**  
- GPU 병렬 연산에 최적화되지 않은 딥 컨볼루션 구조로 인해 단일 블록당 추론시간 일부 증가.  
- 채널·공간 주의 모듈이 경량이지만, 모바일 SoC 특성에 따라 메모리 접근 비용 발생 가능.

## 4. 일반화 성능 향상 가능성  
- **Dense 연결 기반 단순 집계**: 다양한 피처 스케일을 효과적 으로 융합하나, 복잡한 변형(deformable) 또는 non-local attention 적용 시 일반화 강화 가능.  
- **BSConv + ConvNeXt 블록**: 다른 비전 작업(object detection, denoising)에 전이 학습으로 활용 시 추가 경량화 달성 여지.  
- **ESA/CCA 모듈**: 적은 파라미터로 유의미한 정보 강조하므로, 다양한 SR 도메인(의료영상, 위성영상)에 적용해도 과적합 억제 기대.

## 5. 향후 연구 방향 및 유의점  
1. **경량화 기법 조합**: BSConv 외에도 구조적 희소화(sparsity), 지식 증류 결합 검토.  
2. **하드웨어 최적화**: 모바일·엣지 디바이스의 메모리·대역폭 제약 고려한 컴파일러·커널 최적화 필요.  
3. **비지도·자기지도 학습**: 레이블 없는 데이터 활용해 SR 일반화 성능 강화.  
4. **다중 degradations**: 실제 저해상도 영상 복원을 위한 노이즈·블러 복합 환경 대응 연구.

---  
**참고문헌**  
논문 내 표·실험 데이터[1][Tables 1–5].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9fbbfb6a-7eb0-4a14-8f5a-c109d3fb0510/s41598-024-64724-y.pdf
