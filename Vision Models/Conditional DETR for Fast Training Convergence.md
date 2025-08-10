# Conditional DETR for Fast Training Convergence | Object detection

## 핵심 주장 및 주요 기여  
**Conditional DETR**는 DETR의 느린 수렴 문제를 한층 개선하여, 전역적(dense) cross-attention 구조를 유지하면서도 학습 속도를 크게 향상시킨다.  
1. **Conditional Spatial Query 도입**: Decoder의 이전 레이어 출력 임베딩과 참조점(reference point)을 결합해 각 query마다 *조건부 공간 임베딩*(conditional spatial query)을 생성.  
2. **Cross-Attention 분리 설계**: Query–Key dot-product를 공간(𝑝_q⊤𝑝_k)과 내용(𝑐_q⊤𝑐_k) 주의로 분리하여, 공간 주의가 객체의 사각형 극단(extremity) 및 내부 영역에 집중하도록 유도.  
3. **학습 속도 대폭 향상**: ResNet-50/101 기반 모델에서 6.7×, 고해상도 DC5 백본에서는 10× 빠른 수렴을 달성.  

## 해결하고자 하는 문제  
- **Slow Training Convergence**: 원본 DETR은 전역적 self/cross-attention에 의존해 500 epoch 이상 훈련해야 안정적 성능 달성  
- **공간 정보 약화**: 기존 DETR의 object query가 spatial+content 역할을 동시에 수행하며, 제한된 학습 단계에서는 정확한 공간 집중이 어렵고 content embeddings에 과도한 의존 발생  

## 제안 방법  
### 1) Conditional Spatial Query 생성  
- 참조점 𝑠를 sigmoid로 정규화한 후 sinusoidal positional embedding 𝑝_s = sinusoidal(sigmoid(𝑠)) 생성  
- Decoder 임베딩 𝑓를 FFN을 통해 변환해 대각 행렬 Λ_q(= λ_q)로 학습  
- 공간 쿼리:  

$$
p_q = \Lambda_q \,\odot\, p_s
$$  
  
### 2) Multi-Head Conditional Cross-Attention  
- Query = [𝑐_q; 𝑝_q], Key = [𝑐_k; 𝑝_k] 형태로 concatenation  
- Attention weight = content attention + spatial attention  

$$
\text{score} = c_q^⊤ c_k + p_q^⊤ p_k
$$  

- 각 헤드는 객체의 사각형 네 극단 및 내부 영역에 분산 집중  

### 3) 전체 모델 구조  
- Backbone (ResNet / DC5-ResNet) → Transformer Encoder → Transformer Decoder (6 layers)  
- 각 Decoder 레이어에 Self-Attention → Conditional Cross-Attention → FFN  
- Box 예측:  

$$
b = \mathrm{sigmoid}(\mathrm{FFN}(f) + [s^\top, 0]^\top)
$$  

- Classification: $$e = \mathrm{FFN}(f)$$  
- Reference point 𝑠는 논문 실험상 “학습 파라미터” 또는 “object query 기반 예측” 두 가지 모두 유사한 성능  

## 성능 향상  
- ResNet-50: 50 epoch에서 AP 34.9 → 40.9 (≈+6.0)  
- DC5-ResNet-50: 50 epoch에서 AP 36.7 → 43.8 (≈+7.1)  
- ResNet-101, DC5-ResNet-101에서도 6–10× 빠른 수렴 및 동등 이상의 성능  
- 단일-스케일 DETR 확장판(Deformable DETR-SS, UP-DETR) 대비 동등 혹은 우수  

## 한계 및 일반화 성능  
- **고해상도·다중 스케일 미지원**: Deformable DETR 등과 달리 멀티스케일 encoder 적용되지 않아 크기가 극단적으로 다양한 객체에 대한 일반화 여지는 추가 연구 필요  
- **대각 행렬 제한**: λ_q를 대각 행렬로 학습했지만, 블록/풀 행렬과도 유사 성능. 더 유연한 공간 변환이 일반화 성능 향상에 기여할지 불확실  
- **다른 도메인 적용성**: 현재 COCO 객체 검출에만 검증. 인간 자세 추정, 선 분할(line detection) 등 다른 시각 과제 일반화 검증 필요  

## 향후 연구 영향 및 고려 사항  
- **공간–내용 분리 메커니즘 확장**: 언어, 음성, 포즈 추정 등 다양한 transformer 기반 과제에 conditional attention 적용 가능성  
- **멀티스케일·고해상도 결합**: Conditional cross-attention에 deformable/multi-scale encoder 결합 시 속도와 성능 동시 개선  
- **참조점 학습 방식 심화**: 더욱 유연한 참조점 예측(FPN 레벨별, object 크기별)으로 작은 객체 일반화 성능 강화  
- **비전–언어 융합 과제 적용**: 공간 집중이 중요한 VLP(vision-language pretraining) 과제에서 BERT-유사 positional query로 확장 연구  


[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eeb187c6-d35c-4fa6-a5b3-cdc97eea5988/2108.06152v3.pdf
