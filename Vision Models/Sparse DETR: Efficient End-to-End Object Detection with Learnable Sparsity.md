# Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity | Object detection

## 1. 핵심 주장 및 주요 기여  
Sparse DETR은 기존 DETR 계열 모델의 **인코더 계산 병목**을 해소하고, **더 가볍지만 동등 또는 더 나은 성능**을 내는 엔드-투-엔드 객체 검출기를 제안한다.  
- **쿼리·키 양방향 희소화**  
  - 인코더에서는 전체 토큰 중 중요한 ρ%만 업데이트(쿼리 희소화)  
  - 디코더에서는 Deformable DETR의 키 희소화 유지  
- **학습 가능한 중요도 예측**  
  - 디코더의 크로스 어텐션 맵(DAM)을 바이너리화해 인코더 토큰의 중요도를 예측  
  - Objectness Score 대비 디코더 관점에서 최적화  
- **엔코더 보조 손실(auxiliary loss)**  
  - 선택된 인코더 토큰에만 헝가리안 매칭 손실 적용 → 심층화된 인코더 안정적 학습  

이로써, COCO 기준으로 Swin-T 백본에서 10% 희소화만으로 Deformable DETR 대비 연산량 38% 절감, FPS 42% 향상, AP 48.2를 달성하였다.  

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- **DETR**: NMS 제거, end-to-end 학습 가능하나 멀티스케일 처리시 인코더 토큰 N≈20× 증가 → 연산량 급증  
- **Deformable DETR**: 키 희소화로 O(NK)로 개선하나, 멀티스케일 인코더 토큰 수가 여전히 병목  

### 2.2 제안 방법  
1. **인코더 토큰 희소화 (Query Sparsification)**  
   - 입력 벡터 $$x_{\text{feat}}\in\mathbb{R}^{N\times D}$$에 스코어링 네트워크 $$g: \mathbb{R}^D\!\to\!\mathbb{R}$$ 적용  
   - 상위 ρ% 토큰 $$\Omega^s_\rho$$만 Deformable Self-Attn. 및 FFN 업데이트, 나머지는 identity  
   - 계산 복잡도 $$O(SK)$$로 감소 ($$S=ρN\ll N$$, $$K\ll N$$)  
2. **Decoder Cross-Attention Map (DAM) 기반 스코어링**  
   - 디코더의 각 레이어에서 객체 쿼리↔인코더 토큰 어텐션 가중치 누적 → DAM  
   - DAM 상위 ρ% 바이너리화 후, BCE 손실로 스코어링 네트워크 학습  
   - 식:  

$$
       \mathcal{L}\_{\text{DAM}} = -\frac1N \sum_{i=1}^N \mathrm{BCE}\bigl(g(x_{\text{feat},i}),\,\mathrm{DAM}_{\text{bin},i}\bigr)
     $$

3. **엔코더 보조 손실 & Top-k 디코더 쿼리**  
   - 선택된 인코더 토큰에만 헝가리안 매칭 기반 auxiliary detection head 부착  
   - 최종 인코더 출력에서 objectness 기준 Top-k 토큰을 디코더 쿼리로 사용  

### 2.3 모델 구조  
```
Backbone → Flatten → Scoring Net → Top-ρ% Token 선택 → Sparse Encoder (Def. Attn + FFN)
    ↳ Encoder Auxiliary Heads (선택된 토큰만)
Sparse Encoder Output → Objectness Head → Top-k Query 선택 → Deformable Decoder → Prediction
```

## 3. 성능 향상 및 한계  
| 백본 | 희소화 비율 ρ | AP (val2017) | FLOPs 감소 | FPS 향상 |
|------|--------------|--------------|------------|----------|
| Swin-T | 100% (Deformable+DC5+) | 48.0 | – | 15.4 |
| Swin-T | 10% (Sparse DETR) | 48.2 | –38% | +38% |

- **효율성**: 인코더 연산량 최대 82% 감소  
- **성능**: ρ≥30% 에서 비희소 기반 대비 동등 또는 상회  
- **안정성**: 엔코더 레이어 심층화 시 vanishing gradient 문제 해소  

한계  
- **초기 학습 안정성**: DAM 기반 스코어러는 초기 학습 단계 DAM 품질에 의존  
- **동적 희소화**: 훈련 시 ρ 고정 → 추론 시 ρ 변화 시 성능 저하 소폭 발생(≤0.5 AP)  

## 4. 일반화 성능 향상 가능성  
- **엔코더 auxiliary loss**: 중간층에 직접 gradient 제공 → 더 깊은 인코더에서도 안정적 학습  
- **DAM 스코어링**: 디코더 관점의 중요 토큰 선택 → 다양한 도메인에도 디코더 토큰 중요도 산출 가능  
- **토크나이저 확장**: ViT 계열 다른 모델(Swin-B, 대형)에서도 유사 또는 더 큰 효율·성능 이득 확인  
- **추론 시 스케일 조정**: ρ 조절만으로 계산량 대비 성능 균형 조절 가능 → 다양한 하드웨어 환경 적응  

## 5. 향후 연구 방향 및 고려 사항  
- **동적 희소비율 학습**: 훈련 단계에서 ρ를 다양하게 샘플링해 추론 시 유연성 강화  
- **DAM 품질 개선**: 초기 학습 불안정 해소를 위한 warm-up 스케줄 또는 스무딩 기법 적용  
- **백본·엔코더 공동 최적화**: Self-supervised 백본 초기화(예: SCRL)와 희소 인코더 연계 최적화  
- **다중 태스크 확장**: 분할, 인스턴스 세그멘테이션 등 다른 비전 태스크로의 일반화 검증  

Sparse DETR은 **인코더 토큰의 학습 가능한 스파스화**로 객체 검출의 **효율성과 성능**을 동시에 달성하며, 향후 다양한 트랜스포머 기반 비전 모델의 경량화·가속화 연구에 중요한 토대를 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cf13673e-709d-4afd-a5bc-07ce7895a6ac/2111.14330v2.pdf
