# TerDiT: Ternary Diffusion Models with Transformers

## 1. 핵심 주장 및 주요 기여  
TerDiT는 **초저비트(1.58-bit) 양자화**를 활용해 대규모 Diffusion Transformer(DiT) 모델을 **처음부터** 훈련하고, **효율적인 배포**까지 가능케 하는 첫 번째 프레임워크이다.  
- **핵심 주장**: Extremely low-bit(ternary) QAT(Quantization-Aware Training)을 통해 600M~4.2B 규모 DiT 모델이 full-precision 모델과 동등한 이미지 생성 품질을 유지하면서 메모리·체크포인트 크기를 크게 절감할 수 있다.  
- **주요 기여**:  
  1. DiT 전용 **Weight-Only QAT** 기법 제안 및 1.58-bit ternary 양자화 함수 설계.  
  2. adaLN 모듈 내에 **RMSNorm**을 삽입하여 저비트 양자화로 인한 활성치 폭발 문제 해결.  
  3. 600M→4.2B 파라미터, 256²→512² 해상도 DiT 모델 훈련 및 2-bit CUDA 커널 기반 배포로 메모리 8×, 체크포인트 10× 절감.  

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계  
### 2.1 해결하고자 하는 문제  
- **대규모 DiT 배포의 비효율성**: 파라미터 수(수십억) 때문에 GPU 메모리 요구량 및 저장 크기가 과도.  
- **저비트 양자화 적용의 난제**: 기존 PTQ(post-training quantization)는 2-bit 이하에서 품질 저하가 심각, QAT 연구도 U-Net 계열에만 국한.  

### 2.2 제안 방법  
1) **Ternary Quantization 함수**  

$$
     \gamma=\frac{1}{mn}\sum_{i,j}|W_{ij}|,\quad \tilde W=\mathrm{Clamp}(\mathrm{round}(W/\gamma+\epsilon), -1,1)
   $$  

$$
     W_Q=\alpha\cdot\tilde W,\quad\alpha\in\mathbb{R}^+
   $$  
   
   – full-precision 가중치 $$W$$를 평균 절댓값 $$\gamma$$로 정규화 후 $$\{-1,0,1\}$$로 클램프, 학습 가능한 스케일 $$\alpha$$ 적용.  

2) **Quantization-Aware Training**  
   – Forward: ternary 가중치 $$\to$$ 백워드: STE(straight-through estimator)로 full-precision 파라미터에 gradient 전파.  

3) **RMSNorm-보강 adaLN**  
   – 원본 adaLN(module for conditional scale/shift) 뒤에 RMSNorm 적용:  

$$
       \mathrm{adaLN}(c)=\mathrm{RMSNorm}(\mathrm{MLP}(\mathrm{SiLU}(c)))
     $$  
  
   – 저비트 양자화 시 MLP 출력의 activation 폭발 억제, 훈련 안정성·수렴속도 대폭 개선[Fig. 4][Fig. 7][Fig. 8].

### 2.3 모델 구조  

```
Input → Patchify → {… Ternary DiT Block (MHSA＋SwiGLU＋RMS-adaLN)}×N → Linear Decoder → Output Noise
```

- 각 블록: Transformer 기반 latent 패치 처리  
- adaLN(Zero) 모듈 내 MLP 뒤에 RMSNorm 추가  

### 2.4 성능 향상  
- **ImageNet 256², cfg=1.5**  
  -  TerDiT-4.2B-G: FID 2.42 vs. Large-DiT-4.2B-G 2.10 (소폭 저하)／체크포인트 1.1 GB vs. 16 GB[Tab. I][Tab. II]  
- **512²**: TerDiT-4.2B-G FID 2.81 vs. Large-DiT-4.2B-G 2.52  
- **Deployment**:  
  – 메모리 사용 1.92 GB vs. 17 GB, Inference GPU 97 s vs. 83 s(4.2B)  
- **저비트 성능**: 2-bit PTQ(Q-DiT, Q-Diffusion) 및 2-bit QAT(EfficientDM) 모두 실패. TerDiT만 정상 작동.  

### 2.5 한계  
- **추가 추출 오버헤드**: 2-bit → pack/unpack 비용으로 inference 지연  
- **훈련 시간 증가**: RMSNorm 삽입 후에도 full-precision 대비 더 긴 학습 필요  
- **하드웨어 지원 부재**: ternary DiT 전용 가속기·소프트웨어 미흡  

## 3. 일반화 성능 향상 가능성  
- **활성치 정규화**(RMSNorm)로 저비트 distortion 억제 → 다양한 조건(task)에서 안정적 수렴 기대.  
- **스케일 업**(600M→4.2B) 시 양자화 성능 저하폭 감소 → 더 큰 모델일수록 일반화 능력 강화.  
- **cfg 강건성**: cfg=10까지도 생성 왜곡 경미[Fig. 12] → 컨디셔닝 강도 변화에도 일반화 견고.  

## 4. 향후 영향 및 고려 사항  
- **영향**:  
  1. 대규모 diffusion transformer의 경량화·효율 배포 가능성 제시  
  2. ultra-low-bit QAT 및 구조적 정규화(RMSNorm) 조합의 중요성 강조  
  3. 후속 연구에서 **하드웨어–소프트웨어 통합 가속** 연구 자극  
- **고려점**:  
  - 추가 **하드웨어 커널 최적화**(특히 unpack 병목 해소)  
  - **학습 비용 및 탄소발자국** 저감을 위한 효율적 QAT 스케줄링  
  - 다양한 **도메인·조건부** 확장성 검증 및 **다중 모달** 일반화 성능 평가

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c0bb7d23-3786-412c-bc95-c0e0a99d9b12/2405.14854v2.pdf
