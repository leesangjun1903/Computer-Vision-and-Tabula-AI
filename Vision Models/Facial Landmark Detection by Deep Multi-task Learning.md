# Facial Landmark Detection by Deep Multi-task Learning | Facial Landmark Detection

# 핵심 요약

**“Facial Landmark Detection by Deep Multi-task Learning”** 논문은 얼굴 랜드마크 검출을 단일 문제로 보지 않고, 머리 자세(pose) 추정, 성별 분류, 안경 착용 여부, 미소 여부 등 **이종(heterogeneous) 보조 과제(auxiliary tasks)** 와 공동 학습함으로써 검출의 **견고성**과 **일반화 성능**을 크게 향상시킨다.  
주요 기여는 다음과 같다:
1. **Tasks-Constrained Deep Convolutional Network (TCDCN)** 제안  
   - 하나의 CNN 내부에서 여러 과제를 동시에 학습하도록 설계  
   - 보조 과제의 손실을 주 랜드마크 검출 손실과 결합한 다중 손실 함수(식(3))  
   - 보조 과제별 중요도 λₐ를 도입해 자동 최적화  
2. **Task-wise Early Stopping** 도입  
   - 보조 과제별 학습 곡선을 모니터링하여 과적합 직전에 학습 중단(식(5))  
   - 메인 과제 성능 저하 없이 학습 안정화 및 수렴 가속  
3. **경량화된 단일 CNN**  
   - 4개의 합성곱 계층 + 풀링 계층 + 완전연결 계층 구조 (입력 40×40)  
   - Sun et al.의 23-단계 cascaded CNN 대비 연산량 7배 절감, CPU 17ms 처리  
4. **실험적 우수성**  
   - AFLW, AFW, COFW 데이터셋에서 기존 RCPR, SDM, ESR, CDM, Luxand SDK, cascaded CNN 등을 모두 능가  
   - 프로필 및 부분적 폐색(occlusion)에 특히 강함  

# 상세 설명

## 1. 문제 정의 및 목표  
얼굴 랜드마크 검출은 *큰 머리 회전*, *표정 변화*, *안경·마스크·머리카락* 등에 의해 여전히 불안정하다.  
논문은 검출(main task)과 보조 과제(auxiliary tasks)를 **공유된 특징 공간**에서 **동시 학습**하여 보조 과제로부터 얻은 **상호 제약**(pose→inter-ocular distance, smile→입 주변 근육 활성화 등)을 랜드마크 검출에 활용하고자 한다.

## 2. 제안 방법 및 수식  
– **공유 특징 학습**  
  입력 이미지 $$x_0$$를 합성곱·풀링·비선형 활성화(절대 tanh) 계층을 거쳐 공유 표현 $$x_l$$로 맵핑(식(4)).  
– **다중 손실 함수** (식(3))  

$$
\min_{W_r,\{W_a\}}
\frac{1}{2N}\sum_i \|y^r_i - W_r^T x_i\|^2
-\sum_{a\in A}\frac{\lambda_a}{N}\sum_i y^a_i\log p(y^a_i|x_i;W_a)
+\frac{T}{2}\|W\|^2
$$

  - $$W_r$$: 랜드마크 위치(회귀) 가중치  
  - $$W_a$$: 각 보조 과제(교차엔트로피 분류) 가중치  
  - $$\lambda_a$$: 과제 중요도 계수(학습 중 최적화)  
– **Task-wise Early Stopping** (식(5))  
  각 보조 과제의 훈련/검증 오차 추세와 일반화 오차 비율을 기준으로 과적합 직전에 학습 중단.  

## 3. 모델 구조  
- 입력: 40×40 그레이스케일 얼굴  
- 합성곱 계층 1–4, 각 계층 뒤 풀링, 마지막 완전연결(fc100)  
- fc100을 공유해 하나는 5개 랜드마크(10차원) 회귀, 나머지는 pose(5클래스), gender, glasses, smile(각 2클래스) 분류  


## 4. 성능 향상  
- **AFLW**: 전체 failure rate 25%→15% 감소; 프로필(face ±60°) 개선폭 최대 20%↑  
- **AFW**: 주요 랜드마크 검출 오차 모두 RCPR·SDM·ESR·CDM 대비 1–3%p 개선  
- **COFW**: RCPR 초기화로 사용 시 occlusion 영역 랜드마크 오차 10% 이상 추가 감소  
- **연산 효율**: CPU 17ms, GPU 1.5ms (cascaded CNN 대비 7× 빠름)

## 5. 한계  
- **보조 과제 라벨링 비용**: 다수의 attribute annotation 필요  
- **이종 과제의 확장성**: 과제 추가 시 λₐ 최적화 및 early stopping 임계값 튜닝 필요  
- **밀집 랜드마크**(68점 등)에 대한 적용 미검증

# 일반화 성능 향상 관점  
공유 표현 학습과 task-wise early stopping이 결합돼 보조 과제(overfitting)로 인한 잡음 없이 **핵심 표현**만 메인 과제에 전달된다.  
이는 엄격한 정규화(regulation)와 *representation transfer*를 통해 **다양한 얼굴 변형**(pose·표정·폐색·저해상도)에 견고한 특징을 학습하도록 유도한다.

# 향후 연구 및 고려사항  
1. **라벨 효율적 학습**: 반/비감독 보조 과제(silver-label, self-supervised) 통합으로 Annotation 비용 절감  
2. **연속·밀집 랜드마크**: iBUG 300-W 등 고밀도 주석에 대한 TCDCN 확장성 평가  
3. **과제 관계 학습**: λₐ 간 상호 공분산 구조 학습으로 과제 중요도 자동 발굴  
4. **경량화·모바일 적용**: MobileNet·Quantization 도입으로 실시간 모바일 얼굴 분석 강화  
5. **도메인 적응**: illumination·카메라·인종 차이 등 **도메인 변화**에 대한 적응 기법 접목  

이 논문은 **멀티태스크 딥 러닝**이 얼굴 분석 전반에 미치는 영향을 입증했으며, 보조 과제의 적절한 통합·제어를 통한 **일반화 성능 강화** 전략으로 후속 연구의 방향성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/78dca638-22a5-47d9-9fba-161318d224ed/eccv_2014_deepfacealign.pdf
