# MIRNet : Learning Enriched Features for Fast Image Restoration and Enhancement | Image deblurring, Image enhancement, Super resolution

**핵심 주장:**  
MIRNet은 고해상도 공간 정보를 유지하면서 멀티스케일 문맥 정보를 효과적으로 융합해, 이미지 복원 및 향상(task-agnostic)에서 최첨단 성능을 달성하는 **경량‧고속‧고성능** 모델이다.

**주요 기여:**  
1. **고해상도-저해상도 병렬 구조**: 하나의 메인 브랜치는 입력 해상도를 그대로 유지해 세밀한 공간 정보(텍스처·에지)를 보존하고, 여러 저해상도 브랜치는 넓은 문맥을 인코딩한다.  
2. **Selective Kernel Feature Fusion (SKFF)**: 멀티해상도 스트림의 출력을 어텐션 기반으로 융합해, 각 위치에서 최적의 리셉티브 필드를 동적으로 선택한다.  
3. **Residual Contextual Block (RCB)**: 그룹 컨볼루션과 글로벌 컨텍스트 모듈을 결합해, 덜 유용한 특징은 억제하고 핵심 정보를 강조하는 블록을 설계했다.  
4. **Progressive Learning Regime**: 학습 초기에는 작은 패치, 후기로 갈수록 큰 패치로 훈련해 학습 속도와 최종 성능을 동시에 개선하는 커리큘럼 방식을 도입했다.

# 상세 설명

## 1. 해결하고자 하는 문제  
- 기존 CNN 기반 복원 모델은  
  - **Full-resolution**: 공간적 세부는 잘 보존하나 문맥 범위가 제한됨,  
  - **Encoder–decoder**: 문맥은 넓게 인코딩하나 중간 다운샘플링으로 세부 정보 손실 발생  
- **목표**: 두 특성(고해상도 정밀도 vs. 문맥 정보) 모두 충족하는 단일 네트워크 설계

## 2. 제안 방법

### 2.1 전체 파이프라인  
입력 $$I\in \mathbb{R}^{H\times W\times3}$$ → 저수준 컨볼루션 → $$N$$개의 Recursive Residual Group (RRG) → 최종 컨볼루션 → 잔차 영상 $$R$$ → $$\hat I = I + R$$로 복원  
손실: Charbonnier Loss  

```math
\mathcal{L}(\hat{I}, I^*) = \sum_{x} \sqrt{(\hat{I}(x) - I^*(x))^2 + \epsilon^2} \quad (\epsilon = 10^{-3})
```

### 2.2 Multi-Scale Residual Block (MRB)  
- **Parallel Streams**: 해상도 $$1,\tfrac12,\tfrac14$$ 에서 채널 $$80, 120, 180$$  
- 각 스트림간 정보를 교환·융합  

#### 2.2.1 Selective Kernel Feature Fusion (SKFF)  
- **Fuse**: 두 스트림 특징 $$\mathbf L_1,\mathbf L_2$$를 합산 → GAP → 디멘션 축소/확장 컨볼루션 → 어텐션 벡터 $$\mathbf v_1,\mathbf v_2$$  
- **Select**: 소프트맥스로 $$\mathbf s_1,\mathbf s_2$$ 계산 → $$\mathbf U = s_1\odot L_1 + s_2\odot L_2$$로 융합  

#### 2.2.2 Residual Contextual Block (RCB)  
- **Group Convs**: 입력 $$\mathbf F_b$$에 3×3 그룹 컨볼루션(그룹 수 $$g=2$$)×2  
- **Context Modeling**: 1×1 Conv → 소프트맥스 → reshape→ 글로벌 디스크립터  
- **Feature Transform**: 1×1 Conv×2로 채널 의존성 인코딩  
- **Fusion**: 원본 $$\mathbf F_b$$와 덧셈  

### 2.3 Progressive Learning Regime  
- 학습 단계별 패치 크기: 128 → 144 → 192 → 224  
- 초기 작은 패치로 빠르게 수렴, 이후 큰 패치로 세부 학습  

## 3. 모델 구조 개요  

| 모듈 | 구성 요소 | 세부 사항 |
|------|-----------|-----------|
| Low-level Conv | 3×3, 채널 C | 입력 영상 특징 맵 생성 |
| RRG (×4) | MRB(×2) | MRB: 3 해상도 스트림 & SKFF, RCB |
| Final Conv | 3×3, 채널 3 | 잔차 영상 출력 |
| Loss | Charbonnier | $$\epsilon=10^{-3}$$ |

## 4. 성능 향상 및 한계

### 4.1 성능  
- **Defocus Deblurring (DPDD)**: PSNR +0.81 dB 개선[Table 2]  
- **Denoising (SIDD/DND)**: PSNR +0.32 dB, +0.11 dB 개선[Table 3]  
- **Super-resolution (RealSR)**: ×2,+0.48 dB; ×3,+0.73 dB; ×4,+0.24 dB 개선[Table 4]  
- **Enhancement (LoL/FiveK)**: PSNR +3.44 dB; +0.93 dB 개선[Tables 5–6]  

### 4.2 일반화 성능  
- SIDD 학습→DND 평가 시도, 학습 데이터 불일치에도 SOTA 달성(PSNR 39.86 dB) → **강력한 일반화 능력**  
- 멀티스케일 구조+어텐션 융합이 다양한 잡음·블러 유형에 적응

### 4.3 한계  
- 모델 크기 및 연산량(동시 처리 스트림) 존재  
- 동영상 복원 등 시공간 연속성 처리 불포함  
- 매우 극단적 저조도·고노이즈 상황에서 추가 튜닝 필요

## 5. 향후 연구에 대한 영향 및 고려 사항

1. **멀티스케일 어텐션 융합**: 다양한 복원·향상 분야에 일반화 가능하며, 시퀀스 데이터(비디오)나 3D 볼류메트릭으로 확장 여지  
2. **경량화 전략**: RCB의 그룹 컨볼루션·SKFF 선택적 융합은 모바일·엣지 디바이스 적용 시 유용  
3. **학습 스케줄링**: Progressive Learning은 커리큘럼 러닝 관점에서 다른 저수준 비전 과제에도 이식 가능  
4. **한계 보완**: 블라인드 복원(미지 저품질 유형) 및 실시간 처리 파이프라인 통합을 위한 추가 최적화 필요  

이 논문은 **고해상도 디테일 보존**과 **넓은 문맥 인코딩**을 통합하는 새로운 네트워크 설계 패러다임을 제시함으로써, 이후 저수준 영상 처리 연구에 강력한 기반을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5d47bd0a-fd29-4d58-9421-89c244a9defc/2205.01649v1.pdf
