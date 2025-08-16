# YOLOv10: Real-Time End-to-End Object Detection

**주요 주장 및 기여 요약**  
YOLOv10은 전통적인 CNN 기반 YOLO 계열의 한계를 극복하여, 후처리 단계인 NMS(Non-Maximum Suppression)를 제거하고 모델 구조를 효율·정확도 관점에서 전방위적으로 재설계함으로써, 실시간 종단 간(end-to-end) 객체 탐지의 새로운 기준을 제시한다.  

1. **NMS-free 일관 이중 할당(Consistent Dual Assignments)**  
   - 학습 시 one-to-many(다대다)와 one-to-one(일대일) 라벨 할당을 동시에 수행.  
   - 추론 시 one-to-one 브랜치만 사용하여 NMS 제거, 지연(latency) 대폭 감소.  
   - 일관 매칭 지표 $$m = s \cdot p^{\alpha} \cdot \mathrm{IoU}^\beta$$를 both 브랜치에 동일하게 적용해 두 브랜치 간 감독(supervision) 격차를 최소화.  

2. **전방위 효율-정확도 구동 모델 설계(Holistic Efficiency-Accuracy Driven Design)**  
   - **경량 분류 헤드**: 분류 헤드 연산량을 2.5× 경감, 정확도 손실 최소화.  
   - **공간-채널 분리 다운샘플링**: 채널 확장(Pointwise Conv)과 해상도 축소(Depthwise Conv) 분리로 연산 절감.  
   - **랭크-가이드 블록 설계**: 각 스테이지의 고윳값 기반 중복성 분석을 통해, 중복성 큰 스테이지에 Compact Inverted Block(CIB) 배치로 효율 극대화.  
   - **대형 커널 및 부분적 자기주목(PSA) 모듈**:  
     - 작은 스케일 모델에만 7×7 대형 Depthwise Conv 적용해 수용 영역 확장.  
     - Stage4 이후에만 PSA 도입, 일부 채널만 Multi-Head Self-Attention으로 처리해 전역 모델링 강화.  

3. **최신 벤치마크 성능**  
   - COCO val에서 S 모델 기준 AP 46.3%, 전방향(end-to-end) 추론 지연 2.49ms로 RT-DETR-R18 대비 1.8× 빠름.  
   - 모델 크기 전 범위(N/S/M/B/L/X)에서 YOLOv8 대비 AP +0.3~1.4%, 파라미터 28~57%↓, FLOPs 23~38%↓, 추론 지연 37~70%↓ 기록.  

4. **한계**  
   - 소형 모델(N/S)는 여전히 one-to-many+NMS 학습 대비 0.5~1.0% AP 손실 존재.  
   - 대규모 데이터셋(Objects365 등) 사전학습 미실시로 일반화 성능 잠재력 미검증.  

***

## 1. 해결 과제  
기존 YOLO 계열은 다수의 양성 샘플을 선택하는 one-to-many 할당으로 우수한 성능을 내지만, 이를 NMS로 후처리해야 해 종단 간 배포에 비효율적이며 하이퍼파라미터에 민감하다. 반면 DETR 계열은 end-to-end이나 CNN 기반에 비해 전방향 연산 효율이 낮다.  

## 2. 제안 방법  
### 2.1. 일관 이중 할당(Consistent Dual Assignments)  
– **Dual Label Assignments**:  
  -  one-to-many 헤드: 풍부한 감독 신호 제공  
  -  one-to-one 헤드: 추론 시 NMS 불필요  
– **Consistent Matching Metric**:

$$
    m(\alpha, \beta) = s \cdot p^{\alpha} \cdot \mathrm{IoU}(\hat b, b)^{\beta}
  $$  
  
  두 브랜치에 $$\alpha_{\text{o2o}}=\alpha_{\text{o2m}}, \beta_{\text{o2o}}=\beta_{\text{o2m}}$$ 적용해 최고 양성 샘플 일치 보장.  

### 2.2. 모델 구조 최적화  
– **경량 분류 헤드**: 3×3 depthwise ×2 + 1×1 conv 조합.  
– **공간-채널 분리 다운샘플링**: 1×1 → 채널, 3×3 depthwise → 해상도.  
– **랭크-가이드 블록 배치**:  
  1. 각 스테이지 마지막 conv의 특이값 분석, 중복성 높은 스테이지 순 정렬  
  2. 순서대로 CIB로 교체, AP 유지 시 채택  
– **대형 커널 & PSA**:  
  7×7 depthwise conv(소형 모델), 구조 재매개변수화; Stage4 이후 PSA(채널 절반에만 MHSA+FFN).  

## 3. 성능 및 일반화 가능성  
– **성능 향상**: COCO val AP↑, 파라미터·FLOPs·추론 지연↓ 전 범위 모델 스케일에서 확인.  
– **일반화 성능**:  
  -  Dual assignments로 양쪽 브랜치의 일관된 학습 지도 → feature discriminability 강화  
  -  PSA의 전역 컨텍스트 학습, 대형 커널의 수용 영역 확장 → 드물거나 복잡한 객체에도 견고  
  -  한계: 소형 모델 일반화 격차 존재하므로, 대규모 사전학습과 다양한 도메인 검증 필요  

## 4. 향후 연구 방향 및 고려점  
– **사전학습**: 대규모 외부 데이터셋(pre-train)으로 일반화 성능 극대화  
– **감독 격차 해소**: 소형 모델의 one-to-many 대비 성능 손실 최소화 기법 탐색  
– **경량화 vs. 정확도**의 더 세밀한 균형: CIB, PSA 등 구조 조합 최적화  
– **실제 배포 환경**: 다양한 하드웨어(CPU·임베디드)에서 end-to-end 효율성 평가  
– **공정성·안전성 고려**: 객체 탐지 민감 분야(의료·자율주행)에서 잠재적 오탐·윤리성 검증

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/bb0ef2b3-eae6-413a-a2e3-564e79b6c79d/2405.14458v2.pdf
