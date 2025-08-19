# REGO-DETR : Recurrent Glimpse-based Decoder for Detection with Transformer | Object detection

## 1. 핵심 주장 및 주요 기여 요약  
Recurrent Glimpse-based Decoder (REGO)는 DETR(Detection with Transformer)의 전역 어텐션이 요구하는 **장시간 학습 문제**를 **RoI 기반의 반복적 관심 영역 정제**로 해결한다.  
- **핵심 아이디어**: DETR의 예측 결과를 RoI로 활용해 점진적으로 관심 영역(glimpse)을 좁혀가며 디코더를 반복 적용  
- **주요 기여**:  
  1. **RoI 기반 반복 정제 모듈**을 도입해 DETR 학습을 13× 단축(500→36 epochs)  
  2. 다양한 DETR 변형(원조 DETR, Deformable DETR)에 **플러그 앤 플레이** 적용으로 AP 최대 7% 상대 향상  
  3. **Coarse-to-fine glimpse** 단계 설계로 초기에는 넓은 문맥, 후반에는 정밀 지역 정보 활용  

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
원조 DETR은 전역 어텐션 학습이 어렵고, MS COCO 기준 500 epochs까지 필요해 학습 비용이 과다하다.  

### 2.2 REGO 구조 개요  
- **다단계 반복 처리**: i단계에서 이전 단계 예측된 박스 $$\mathrm{O}_{\text{box}}^{(i-1)}$$를 기반으로 RoI 생성  
- **Glimpse 특징 추출**:  

$$
    V^{(i)} = \mathrm{RoIAlign}\bigl(X,\; \alpha^{(i)} \times \mathrm{O}_{\text{box}}^{(i-1)}\bigr)
  $$  
  
$\alpha^{(i)}$: 단계별 확대 비율, $$i$$번째에 $$\alpha(i)=i$$  

- **Glimpse-based Decoder**:  

$$
    H_{\text{g}}^{(i)} = \mathrm{Attention}\bigl(V^{(i)},H_{\text{dec}}^{(i-1)}\bigr)
  $$

$$
    H_{\text{dec}}^{(i)} = [H_{\text{g}}^{(i)},\,H_{\text{dec}}^{(i-1)}]
  $$

$$
    \mathrm{O}\_{\text{cls/box}}^{(i)} = F_{\text{cls/box}}\bigl(H_{\text{dec}}^{(i)}\bigr)
  $$  

### 2.3 모델 구조  
1. **백본**(ResNet-50/101, X101) → 특성 맵 $$X$$  
2. 원조 DETR 디코더를 초기 단계로 활용($$H_{\text{dec}}^{(0)},O^{(0)}$$)  
3. **REGO Glimpse Decoder** 2개 인코더 없이 2개 디코더 레이어 연속 적용  
4. **Auxiliary loss** 통해 중간 출력 지도  

### 2.4 성능 향상  
- **Deformable DETR+REGO**:  
  - 50 epochs: AP 43.8→45.9 (+2.1)  
  - 36 epochs: AP 42.7→44.8 (+2.1)  
- **원조 DETR+REGO**(R50, 50 epochs): AP 39.3→42.3 (+3.0)  
- **X101 백본**: AP 47.7→49.1 (+1.4)  
- **학습 속도**: 50→36 epochs로 28% 단축, 원조 DETR 대비 94% 학습 기간 절감  
- **추가 GFLOPs**: 약 17 GFLOPs(기준 모델 대비 +10%)  

### 2.5 한계  
- **추가 학습 비용**: REGO 모듈 자체 학습 부담(17 GFLOPs)  
- **환경 비용**: 여전히 다수 GPU 일수 필요  
- **단계 과다 시 소폭 개선**: 3단계 이상부터 성능 향상 정체  

## 3. 일반화 성능 향상 가능성  
- **Coarse-to-fine RoI** 전략은 객체 크기·문맥 다양성에 강건  
- **반복적 디코더**로 각 단계에서 지역적 세부와 전역 정보를 융합해 다양한 데이터셋으로 확장 가능  
- **Auxiliary loss**와 **레이어노름** 적용으로 중간 표현 안정화, 분산된 도메인에도 적응력 기대  
- **인퍼런스 시 모듈 제거**해도 성능 ~1 AP 유지, 배포 효율성  

## 4. 향후 연구 영향 및 고려할 점  
- **효율적 학습**: RoI 기반 반복 정제가 Transformer 전반 학습 가속 방안으로 확산  
- **경량화**: Glimpse 단계 축소/모듈 압축 연구로 **추론 경량화**  
- **다중 도메인**: 소수 박스·극단적 스케일 객체 검출에서 확장성 평가 필요  
- **환경적 지속가능성**: REGO의 추가 연산 대비 성능 이득–환경 비용 균형 최적화  

—  
REGO는 **Transformer 검출기 훈련 비용 문제**를 RoI 기반 관심 영역 정제로 극복하며, 학습 효율성과 검출 성능을 모두 끌어올린 혁신적 방법으로 향후 다양한 비전 태스크에 응용될 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/aeef0e1f-0061-4320-91de-33c7a8e849b4/2112.04632v2.pdf
