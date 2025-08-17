# UPerNet : Unified Perceptual Parsing for Scene Understanding | Semantic segementation
# 핵심 주장 및 주요 기여 요약

**Unified Perceptual Parsing (UPP)** 논문의 핵심 주장은 한 이미지에서 **장면(scene), 객체(object), 객체 부위(part), 재질(material), 질감(texture)** 등 여러 수준의 시각 개념을 **단일 네트워크**로 동시에 인식할 수 있다는 것이다.[1]
주요 기여는 다음과 같다:
- **새로운 과제 정의**: 다양한 시각 개념을 한 번에 파싱하는 UPP 제안.  
- **다중 레벨 멀티태스크 프레임워크(UPerNet)**: 이종(annotation) 데이터셋을 통합하여 학습하는 구조 설계.  
- **효율적 학습 전략**: 매 iteration마다 데이터 소스를 무작위 선택하여 해당 태스크 경로만 업데이트함으로써 이질적 레이블 간의 간섭 최소화.  
- **풍부한 시각 지식 발견**: 학습된 모델을 통해 장면-객체, 객체-재질, 재질-질감 등의 구성적 관계를 자동 추출.

# 문제 정의 및 제안 기법

## 해결하고자 하는 문제  
기존의 비전 시스템은 객체 검출, 장면 분류, 재질·질감 인식을 **독립적으로** 연구해 왔다. 반면 인간은 한 번의 시각적 인식으로 다양한 수준의 시각 정보를 동시에 이해한다. 이를 컴퓨터 비전에도 재현하기 위해, **다양한 수준의 레이블(annotation)이 혼재된** 이종 데이터셋에서 **단일 네트워크로 여러 개념을 동시에 파싱**하는 과제(UPP)를 정의하였다.[1]

## 제안하는 방법: UPerNet  
UPerNet은 ResNet₍b₁₎을 백본으로 하고, Feature Pyramid Network(FPN)와 Pyramid Pooling Module(PPM)을 결합한 구조이다.[1]

- ResNet의 각 스테이지별 출력 $$\{C_2, C_3, C_4, C_5\} $$를 FPN을 통해 상향(top-down)·측면(lateral) 연결로 합성하여 $$\{P_2, P_3, P_4, P_5\} $$를 얻음.
- PPM을 $$C_5$$ 직후에 적용해 전역(context) 정보를 강화 후 FPN에 입력.
- 각 개념 수준별 헤드를 다음과 같이 분리 배치:
  - Scene: $$P_5$$의 전역 평균 풀링(Global Average Pooling) → 선형 분류기.
  - Object/Part: $$\mathrm{concat}(P_2,P_3,P_4,P_5)$$ → conv → 픽셀 분류.
  - Material: $$P_2$$ 위에 픽셀 분류 헤드.
  - Texture: $$C_2$$ 위에 별도 conv 블록, **백본에 역전파 차단** 후 픽셀 수준 분류, 작은 해상도(~64×64)로 학습.  

학습 시 매 스텝마다 임의로 데이터 소스를 선택하여 해당 태스크의 분기(branch)만 업데이트하며, 손실 함수는  

$$
\mathcal{L} = \sum_{t\in\{\mathrm{scene,obj,part,mat,tex}\}} \alpha_t\,\mathcal{L}_t
$$  

형태로 구성한다. 미표기(‘unlabeled’) 픽셀 영역은 obejct/material 손실에서 제외하고, part 학습에는 background를 별도 클래스로 포함하여 픽셀 IoU-bg를 계산한다.[1]

# 모델 구조

```plaintext
Input Image
    ↓  ResNet Backbone (C2–C5)
    ↓  PPM on C5
    ↓  FPN Top-Down + Lateral → P2–P5
       ├─ Scene Head: GAP(P5) → FC
       ├─ Object/Part Head: Concat(P2–P5) → Conv → Pixel-wise Softmax
       ├─ Material Head: P2 → Conv → Pixel-wise Softmax
       └─ Texture Head: C2 → Conv Block (역전파 차단) → Pixel-wise Softmax
```

# 성능 향상 및 한계

| 태스크      | 단일 PSPNet(Res-50) mIoU/P.A.(%) | UPerNet mIoU/P.A.(%) |
|-------------|----------------------------------|----------------------|
| Object (ADE20K) | 41.68 / 80.04                 | 41.22 / 79.98        |
| Object (+FPN)   | 34.28 / 76.35                 | 40.13 / 79.61 (PPM)  |
| Broden+ UPP 전체| –                              | mIoU_obj 23.36 / P.A.77.09mIoU_part(bg)28.75 / P.A.46.92Top-1_scene 70.87%mIoU_mat 54.19 / P.A.84.45T-1_tex 35.10% |

- **효율성**: PSPNet 대비 학습 시간 63%로 단축.[1]
- **통합 성능**: 이질 데이터셋을 함께 학습해도 각 태스크 간 간섭이 미미하며, 오히려 객체 정보가 재질 예측에 도움을 줌.  
- **한계**:  
  - Texture 분류 정확도(35.1%)가 낮아, 자연 이미지와 DTD 간 도메인 차이 극복 필요.  
  - Part IoU-bg 성능이 객체 IoU 대비 낮아, 부위 균형 학습 전략 보완 여지.

# 모델의 일반화 성능 향상 가능성

- **Feature Hierarchy 활용**: 다중 레벨 FPN이 저수준(low-level)·고수준(high-level) 특징 모두 포착하므로, 새로운 태스크 추가 시도에 유리.  
- **이질 데이터 어댑테이션**: Texture처럼 도메인 차가 큰 데이터에 대해 별도 분기(frozen backbone, 소해상도)로 대응한 전략은, 이후 도메인 적응(domain adaptation)이나 메타 러닝 프레임워크로 확장 가능.  
- **Dynamic Routing**: 스텝별 랜덤 소스 샘플링 대신, 난이도·관련도 기반 샘플링 비율 최적화나 task-aware 라우팅으로 일반화 강화 여지.  
- **Self-supervised Pretraining**: UPP와 유사한 멀티 레벨 표현 학습을 사전학습으로 활용하면, 라벨이 부족한 환경에서도 강건한 일반화 성능 확보 기대.

# 향후 연구 영향 및 고려 사항

이 논문은 **단일 프레임워크**로 시멘틱 분할·분류·재질 인식·질감 예측을 통합함으로써, 향후 비전 시스템이 **장면 이해(scene understanding)** 을 넘어 **상식 지식(grammar of visual world)** 을 자동 발견하도록 이끈다.  
1. **지식 베이스 구축**: 학습된 모델로부터 장면-객체-재질-질감 등의 구성 관계를 그래프 형태로 수집·추론하는 연구 확대.  
2. **새로운 태스크 통합**: 행동 예측, 물체 관계 인식 같은 고차원 비전 과제와 결합하여 전방위 성능 검증.  
3. **도메인 적응**: DTD처럼 라벨 유형이 다른 데이터 간 도메인 격차 해소 기법 개발.  
4. **효율적 멀티태스크 학습**: 태스크 수가 늘어날수록 발생하는 학습 간섭을 최소화하는 동적 가중치 조정 및 경로 분리 전략 탐색.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c786ef20-1c0c-41bf-af77-8ecad8d005c7/1807.10221v1.pdf
