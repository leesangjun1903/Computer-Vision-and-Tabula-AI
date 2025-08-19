# D²ETR: Decoder-Only DETR with Computationally Efficient Cross-Scale Attention | Object detection

**핵심 주장**  
D²ETR(Decoder-Only DETR)은 기존 DETR 계열의 엔코더-디코더 구조에서 엔코더를 제거하고, Backbone 단계에서 **효율적인 다(多)스케일(feature) 융합**을 수행함으로써  
- **계산량 크게 감소**  
- **검출 성능 유지 또는 향상**  
를 동시에 달성할 수 있음을 보인다.

**주요 기여**  
1. **Computationally Efficient Cross-Scale Attention (CECA)**  
   - Transformer Backbone 내에 단일 스케일을 쿼리로, 이전 스케일을 키·밸류로 사용하는 희소적(self-sparse) 크로스스케일 융합 모듈 도입  
   - 계산 복잡도를 $$O(S h w P^2 C)$$로 제한하여, 기존 엔코더 기반의 $$O(4^S S h w P^2 C)$$보다 크게 절감  
2. **Decoder-Only 구조**  
   - Backbone 출력을 디코더에 직접 투입하여 엔코더 단계를 완전 제거  
   - 단일 스케일( Vanilla D²ETR) 및 다중 스케일( Deformable D²ETR) 디코더 모두 적용 가능  
3. **위치 인지(Location-Aware) 및 토큰 라벨링(Token Labeling) 보조 손실**  
   - 예측 박스와 실경계의 IoU, 중심 거리를 예측해 분류 점수 보정  
   - 비전 트랜스포머계 토큰 수준의 분류 손실로 Backbone 표현력 강화  

***

## 1. 해결하고자 하는 문제

- **DETR 계열 한계**  
  1. **엔코더의 높은 계산 비용**: 4개 스케일(feature) 모두에서 $$O(H^2W^2C)$$ 수준의 셀프-어텐션 수행  
  2. **수렴 속도 및 스몰 오브젝트 성능 한계**: 복잡한 앵커·NMS 제거로 학습이 느리고 작은 객체 검출이 어려움  

- **목표**:  
  엔코더를 제거해 단순화하면서도, Backbone 단계에서 충분한 **스케일별 상호작용**을 보장하여 성능 저하 없이 효율성만 개선

***

## 2. 제안 방법

### 2.1 CECA 수식 모델링

각 스테이지 $$i$$에서  
- **Transformer stage**:  

$$
    x_i = H_i(x_{i-1})
  $$

- **Fusion stage (CECA)**:  

```math
    x^*_i = \mathrm{SA}(x_q, x_k, x_v),
    \quad x_q = x_i,\quad x_k = x_v = [x^*_1, \dots, x^*_{i-1},\, x_i]
```
  
  여기서 $$\mathrm{SA}$$는 멀티헤드 어텐션, $$[\,]$$는 채널·공간 concat.  
- **복잡도 절감**:  
  키·밸류에만 풀링(pool)$$+1\times1$$conv 적용해, 쿼리 수를 하나의 스케일로 제한  

$$
    O(\mathrm{CECA})=O(S\,h\,w\,P^2\,C)\ll O(4^S\,S\,h\,w\,P^2\,C).
  $$

### 2.2 전체 모델 구조

```plaintext
Input Image
    ↓
Pyramid Vision Transformer Backbone
    ↳ 각 스케일별 Transformer stage + CECA fusing stage (병렬)
    ↓
Fine-fused Multi-Scale Feature Maps
    ↓
Transformer Decoder (6-layer)
    ↓
Classification & Regression Heads
```

- **Vanilla D²ETR**: 최종 1개 스케일(feature)만 디코더에 투입  
- **Deformable D²ETR**: 멀티스케일 Deformable Attention 디코더

### 2.3 보조 손실

1. **Location-Aware Loss**  

$$
   L_{\text{awr}} = \frac{1}{B}\sum_{i=1}^B\Bigl[\mathrm{BCE}(\widehat{\mathrm{IoU}},\,\mathrm{IoU}(b_i,\hat b_i)) + \mathrm{BCE}(\widehat{\mathrm{CTR}},\,\mathrm{CTR}(b_i,\hat b_i))\Bigr]
   $$
   
   분류 점수에 $$\mathrm{IoU}^\alpha\times\mathrm{CTR}^\beta$$를 곱해 저품질 박스 억제  
2. **Token Labeling Loss**  
   각 픽셀 영역에 soft 마스크 라벨링 후, Focal loss로 Backbone 출력을 보조 학습  

$$
   L_{\text{token}} = \frac{1}{B}\sum_{i=1}^B\sum_{j}\sum_{p,q} \mathrm{Focal}\bigl(\mathrm{FFN}(x_j[p,q]),\,t_j[p,q]\bigr)
   $$

***

## 3. 성능 향상 및 한계

### 3.1 성능 요약 (COCO val)

| 모델                      | Epoch | GFLOPs | AP   | APS  | APM  | APL  |
|--------------------------|-------|--------|------|------|------|------|
| DETR-R50 (500ep)         | 500   | 86     | 42.0 | 20.5 | 45.8 | 61.1 |
| Deformable DETR-PVT2     | 50    | 163    | 48.3 | 30.5 | 51.6 | 63.8 |
| **D²ETR-PVT2**           | 50    | 82     | 43.2 | 22.0 | 48.5 | 62.4 |
| **Deformable D²ETR-PVT2**| 50    | 93     | 50.0 | 31.7 | 53.4 | 66.7 |

- **학습 속도**: 50 에폭으로 원본 DETR 대비 10× 빠른 수렴  
- **계산 효율**: Deformable D²ETR-PVT2는 93 GFLOPs로 Deformable DETR-PVT2(163 GFLOPs) 대비 43% 절감  
- **스몰 오브젝트 성능**: APS 31.7 → 30.5 (기존) 대비 개선  

### 3.2 한계

- **Backbone 의존성**: Transformer 백본(PVT, Swin) 조건에서만 CECA 효율 극대화  
- **추론 잠재 지연**: CECA 병렬화 최적화 필요, 경량화 하드웨어 배포 과제  
- **토큰 라벨링 일반화**: Vanilla D²ETR에선 효용 제한적, 디코더 초기화 방식과 결합 연구 필요  

***

## 4. 일반화 성능 향상 가능성

- **다양한 Vision Transformer 호환**: Swin, PVT 외에 ViT 계열 백본에 CECA 적용 가능  
- **다양한 다운스트림 태스크**: 세그멘테이션·비디오 프레임 검출 등, CECA 융합으로 고해상도 정보 유지  
- **사전학습 임베딩 활용**: CECA 포함 백본 사전학습으로 풍부한 크로스스케일 표현 획득  

***

## 5. 연구적·실제적 적용 시 고려 사항

- **병렬화 최적화**: fusing stage와 Transformer stage 동시 실행으로 추론 효율 개선  
- **하이퍼파라미터 튜닝**: $$\alpha,\beta$$ (IoU/Center 가중치) 및 Pool 크기(P) 조정  
- **토큰 라벨링 스킴**: 다양한 마스크 해상도·Loss 조합으로 Backbone 표현력 추가 강화  
- **경량화 버전 탐색**: 모바일·엣지 디바이스 실시간 검출용 더 경량화된 CECA 설계  

***

## 6. 향후 연구에 미치는 영향

- **엔코더 폐지 가능성 시사**: 백본 내 효율적 융합만으로 디텍터 단순화·고성능 달성  
- **크로스스케일 어텐션 모듈 연구 확대**: CECA 변형을 통한 다양한 스케일 간 정보 교환 구조 연계  
- **Transformer 백본 발전 가속**: CECA 가 결합된 사전학습 모델이 downstream 태스크 전반의 성능을 제고  

**결론**: D²ETR은 엔코더 의존 없이도 **크로스스케일 정보 융합**만으로 DETR 계열의 계산 효율성과 검출 성능을 동시에 잡는 새로운 방향을 제시하며, 추후 다양한 비전 태스크의 경량화·고성능화 연구에 중요한 기반을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/87fb69e8-2e3c-4953-88d2-1f4a07f7648a/2203.00860v1.pdf
