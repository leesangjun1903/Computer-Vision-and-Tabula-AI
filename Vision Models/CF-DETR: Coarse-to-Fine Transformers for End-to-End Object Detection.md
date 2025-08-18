# CF-DETR: Coarse-to-Fine Transformers for End-to-End Object Detection | Object detection

## 핵심 주장 및 주요 기여  
**CF-DETR**는 기존 DETR 기반 엔드투엔드 객체 검출기의 한계를 보완하기 위해 **Coarse-to-Fine(Coarse→Fine) 디코더 구조**를 도입함으로써, 글로벌 컨텍스트와 로컬 멀티스케일 특징을 효과적으로 융합하여 **소형 객체 검출 성능** 및 **학습 수렴 속도**를 대폭 향상시킨다.

주요 기여  
1. **Coarse-to-Fine 디코더 레이어**: 전통적 Transformer 디코더(‘Coarse 레이어’)와 RoI 기반 지역 특성 정제 모듈(‘Fine 레이어’)을 결합하여 객체 쿼리의 위치 예측 정확도를 높임.  
2. **Adaptive Scale Fusion (ASF) 모듈**: 각 객체 쿼리에 따라 다중 스케일 RoI 특징을 동적으로 융합, 소형 객체의 세밀한 공간 정보를 보강.  
3. **Local Cross-Attention (LCA) 모듈**: 융합된 RoI 특징과 객체 쿼리 간에 깊이별 국소 크로스 어텐션을 수행해 경계 및 형태 정보를 정교하게 추출.  
4. **Transformer Enhanced FPN (TEF) 모듈**: Transformer 인코더에서 추출한 고수준 비국소 특성을 기존 FPN 구조로 전파하여 피라미드 수준 전반의 표현력을 강화.

***

## 1. 해결하고자 하는 문제  
- **소형 객체 검출 정확도 저하**: DETR은 낮은 IoU(0.1–0.4) 구간에서 우수하나, IoU ≥0.5 구간에서 박스 위치 정밀도가 부족하여 APs(AP for small) 성능이 낮음[표 참조].  
- **느린 수렴 속도**: 전역적 크로스어텐션만으로는 학습 초반 label-assignment가 불안정해 수렴에 많은 epochs가 필요.

***

## 2. 제안 방법 자세히 설명  

### 2.1 모델 구조 개요  
  
```text
Backbone → C2–C5 특징맵  
    ↓  
Transformer Encoder (6-layer) → E5  
    ↓  
TEF 모듈 → {E2, E3, E4, E5}  (FPN 유사)  
    ↓  
N× Coarse-to-Fine 디코더 레이어  
    ├─ Coarse 레이어: 전통적 Transformer 디코더  
    └─ Fine 레이어: ASF → LCA → 박스 정제  
    ↓  
클래스·박스 헤드 → 최종 예측
```

### 2.2 수식 및 세부 모듈

1) **ASF 모듈**  
   - 다중 스케일 RoI 특징 $$\{f_i^l \in \mathbb{R}^{c\times h\times w}\}_{l=1}^L$$을 채널 방향으로 연결:  

$$
       f_i = \mathrm{Concat}(f_i^1,\dots,f_i^L)\in \mathbb{R}^{Lc\times h\times w}
     $$
  
   - 객체 쿼리 $$o_i\in\mathbb{R}^c$$로부터 컨볼루션 가중치 생성 및 depthwise convolution 적용 → 채널별 중요도 반영  
   - 1×1 컨볼루션으로 $$f_i'\in\mathbb{R}^{c\times h\times w}$$ 획득  

2) **LCA 모듈**  
   - 융합 특징 $$f_i'$$에 3×3 depthwise conv → key $$f_i^K\in\mathbb{R}^{c\times h\times w}$$  
   - 쿼리 $$o_i$$를 동일 공간 크기로 확장 → $$o_i^Q\in\mathbb{R}^{c\times h\times w}$$  
   - 어텐션 맵 계산:  

$$
       A_i = (o_i^Q,f_i^K)\,W_1\,W_2
     $$  
     
  ($$W_1\in\mathbb{R}^{2c\times 2c/r},\,W_2\in\mathbb{R}^{2c/r\times (c\,k'\,k')}$$)  
   
   - 국소 어텐션 가중합:  

$$
       f_i^O(h',w') = \sum_{u=1}^{k'}\sum_{v=1}^{k'} A_{i,u,v,h',w'}\odot f_i^V(h',w')_{u,v}
     $$  
   
  - $$f_i^O$$를 flatten 후 FC → refined query  

3) **TEF 모듈**  
   - Transformer 인코더 출력 $$E_5$$을 C4에 업샘플+합산 → E4  
   - 반복하여 다중 스케일 피라미드를 구성  

4) **Set Prediction Loss**  

$$
     \mathcal{L}\_{det} = \lambda_{cls}\,L_{cls} + \lambda_{L1}(L_{c}^{L1}+L_{f}^{L1}) + \lambda_{giou}(L_{c}^{giou}+L_{f}^{giou})
   $$
   
   - $$L_{cls}$$: Coarse 레이어 분류용 Focal Loss  
   - $$L^{L1}, L^{giou}$$: Coarse/Fine 레이어 박스 회귀 손실  

***

## 3. 성능 향상 및 한계  

### 3.1 성능 향상  
- **AP (R50, 36 epochs)**:  
  - DETR-R50: 42.0 → CF-DETR-R50: 46.5  
  - APs: 20.5 → 28.4 (소형 객체 7.9pt 상승)  
- **수렴 속도**: 500→36 epochs로 단축, 1× 학습률 스케줄에서도 빠른 수렴 관찰  
- **다양한 백본**: ResNeXt-101+DCN TTA 적용 시 AP 53.0  

### 3.2 한계  
- **계산 비용 증가**: Fine 레이어 내 RoI Align·ASF·LCA 연산으로 inference 속도 감소 (약 18 FPS)  
- **디코더 깊이 과다 시 과적합**: 12개 레이어 이상에서 성능 오히려 하락  
- **모듈 복잡도**: ASF·LCA 하이퍼파라미터(r, k, 레벨 수) 민감  

***

## 4. 모델 일반화 성능 향상 가능성  
- **다중 객체 환경**: CF 구조가 여러 객체 간 국소·전역 정보 구별에 기여해 **밀집 객체 장면**에서 일반화 우수  
- **도메인 적응**: ASF 모듈의 동적 융합이 도메인별 스케일 분포 차이 완화에 유리  
- **높은 해상도 입력**: TEF를 통해 고해상도 저수준 피처 강화, 다양한 해상도에서 견고한 검출 가능  

***

## 5. 향후 연구 영향 및 고려 사항  
- **복합 모듈 경량화**: Fine 레이어 가속화·Pruning 연구를 통해 실시간 적용성 확보  
- **다른 Transformer 변형 결합**: Conditional DETR, Deformable DETR 등과의 융합으로 성능·효율 균형 탐색  
- **신뢰도 추정 연계**: RoI 기반 정제 정보로 불확실도 예측 및 후처리 최소화  
- **자율 주행·의료 영상** 등 실제 응용 도메인에서 **소형·밀집 객체** 검출 성능 검증 및 도메인 특화 모듈 설계  

위와 같이 CF-DETR은 전역·국소 정보의 유기적 결합을 통해 검출 성능과 수렴 속도를 동시 개선하며, 향후 다양한 Transformer 검출기 연구에 영감을 제공할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0856eef2-11c6-4790-8228-0caf04ad5ea7/19893-Article-Text-23906-1-2-20220628.pdf
