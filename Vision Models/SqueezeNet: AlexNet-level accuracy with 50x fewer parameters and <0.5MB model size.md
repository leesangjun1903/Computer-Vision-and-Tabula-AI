# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size | Image classification

**핵심 주장:** SqueezeNet은 AlexNet 수준의 ImageNet 이미지 분류 정확도를 유지하면서 파라미터 수를 50배 줄이고 모델 크기를 &lt;0.5 MB로 압축한 경량 CNN 아키텍처이다.  
**주요 기여:**  
- 3×3 필터를 1×1 필터로 대체하고, 소위 “Fire 모듈”을 도입하여 입력 채널 수를 줄임으로써 파라미터 수를 급감시킴.  
- 지연된 다운샘플링(late downsampling)을 적용해 더 큰 활성화 맵을 유지하여 정확도 손실을 최소화.  
- 모델 압축(Deep Compression) 적용 시 최종 모델 크기를 0.47 MB로 축소하며, 압축 전·후 모두 AlexNet과 동등한 정확도를 달성.  

# 문제 정의 및 제안 방법

## 해결하고자 하는 문제  
- 기존 CNN은 정확도 향상에 초점을 맞추지만, 같은 정확도를 달성하면서 모델 크기와 파라미터 수를 획기적으로 줄이는 연구가 부족함.  
- 대규모 모델은 분산 학습 및 온디바이스 배포(예: 자율주행차, FPGA)에 비효율적임.

## 설계 전략  
1. 3×3 필터를 1×1 필터로 대체 (파라미터 수 9배 절감)  
2. 3×3 필터의 입력 채널 수 감소 (`squeeze` 단계 도입)  
3. 네트워크 말단으로 다운샘플링을 지연시켜 큰 활성화 맵 유지  

## Fire 모듈 구조  
- **Squeeze 레이어:** 1×1 필터만 갖는 합성곱, 출력 채널 수 = s₁×₁  
- **Expand 레이어:** 1×1 필터 e₁×₁개 + 3×3 필터 e₃×₃개  
- 하이퍼파라미터: s₁×₁, e₁×₁, e₃×₃  
- squeeze ratio $$SR = \tfrac{s_{1\times1}}{e_{1\times1}+e_{3\times3}} $$  

## SqueezeNet 전체 아키텍처  
- Conv1 → Fire2–Fire9 (총 8개) → Conv10 → 글로벌 평균 풀링 → Softmax  
- Max-pooling(Str=2) 적용 위치: Conv1, Fire4, Fire8, Conv10 직후 (지연된 다운샘플링)  
- ReLU, Dropout(0.5), Zero-padding(3×3 필터 전) 등 표준 기법 적용  

## 수식 요약  
- Fire 모듈 파라미터 수  

$$
  \underbrace{s_{1\times1} \times 1 \times 1}\_{\text{squeeze}} +
    \underbrace{e_{1\times1} \times 1 \times 1 + e_{3\times3} \times 3 \times 3}_{\text{expand}}
  $$

- squeeze ratio $$ SR $$ 조절로 파라미터 및 모델 크기 조절  

# 성능 향상 및 한계

| 모델                  | 파라미터 수 | 모델 크기    | Top-1 정확도 | Top-5 정확도 |
|-----------------------|-------------|--------------|--------------|--------------|
| AlexNet (기준)        | 60M         | 240 MB       | 57.2%        | 80.3%        |
| SqueezeNet            | 1.3M        | 4.8 MB       | 57.5%        | 80.3%        |
| SqueezeNet + 압축(6bit)| 1.3M       | 0.47 MB      | 57.5%        | 80.3%        |

- **파라미터 50×↓, 크기 510×↓**: AlexNet 대비 동일 또는 소폭 향상된 정확도 달성.  
- **압축 적합성**: Dense-Sparse-Dense, Deep Compression 등과 호환되어 추가 성능 향상 및 크기 절감 가능.  
- **한계**:  
  - 디자인 공간 탐색 시 SR, 3×3 비율, bypass 등 메타파라미터 튜닝 필요.  
  - 작은 모델이 과적합에 더 민감할 수 있어, 일반화 성능을 위해 추가 규제(예: Dropout, 데이터 증강) 필요.  

# 일반화 성능 향상 가능성

- **지연된 다운샘플링**: 큰 활성화 맵이 더 풍부한 특징 학습에 기여, 전이학습 시 성능 유지력 강화.  
- **Bottleneck 경로 회피**: residual bypass 연결(Simple/Complex) 추가 시 학습 안정성 및 일반화 성능 상승 (Top-1 +2.9%p).  
- **경량화+정규화 시너지**: Fire 모듈 구조 자체가 파라미터 수를 억제하며, Dropout·배치정규화와 결합 시 과적합 완화 효과.  

# 향후 연구 영향 및 고려 사항

- **확장성**: SqueezeNet 구조를 기반으로 더 깊거나 넓은 네트워크 설계 시 파라미터-정확도 절충 탐색 가이드라인 제공.  
- **온디바이스 AI**: IoT, 모바일, FPGA 등 자원제한 환경에서 CNN 배포 표준으로 자리매김 가능.  
- **자동화된 설계 탐색**: Bayesian 최적화·진화 알고리즘과 결합해 Fire 모듈 메타파라미터를 자동 튜닝하여 성능 극대화 연구.  
- **정량적 일반화 평가**: 소규모 모델에 대한 다양한 정규화 기법 및 데이터셋 편향·노이즈 영향 연구 필요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/955b2fc1-19e8-4ccf-bbd5-5a56e4633b3a/1602.07360v4.pdf
