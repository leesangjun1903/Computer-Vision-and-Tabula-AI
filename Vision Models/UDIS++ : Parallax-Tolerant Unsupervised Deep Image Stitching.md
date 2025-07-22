# UDIS++ : Parallax-Tolerant Unsupervised Deep Image Stitching | Image stitching

## 1. 핵심 주장 및 주요 기여  
본 논문은 전통적 기하학 기반 스티칭과 기존의 딥러닝 기반 스티칭 방식이 각각 겪는  
– 기하학적 특징이 부족한 장면(저해상도·저텍스처)에서의 실패  
– 큰 시차(parallax) 처리 불능  
문제를 동시에 해결하는 **Parallax-Tolerant Unsupervised Deep Image Stitching (UDIS++)** 기법을 제안한다[1].

주요 기여  
1. 글로벌 호모그래피와 지역 Thin-Plate Spline(TPS)을 통합한 **강건하고 유연한 워핑** 기법 제안  
2. 파라랄렉스 영역에서의 블러 없이 불연속을 최소화하는 **Unsupervised Seam-Driven Composition** 마스크 학습  
3. 라벨 없는 상황에서도 **반복적(iterative) 워프 적응 전략**을 통해 타 데이터셋·타 해상도에 대한 일반화 성능 강화  

## 2. 해결 과제 및 제안 방법 상세  
### 2.1 해결하고자 하는 문제  
- 전통 방식: SIFT, LSD, 에지 등 복잡한 기하학적 특징에 의존 → 텍스처 부족 시 정합 실패, 계산량 폭증  
- 기존 딥러닝 무감독 방식(UDIS): 호모그래피만으로는 큰 시차 정합 불가, 재구성 기반 복합 시 파라랄렉스 영역에 블러 발생  

### 2.2 제안 워핑 모델  
- **호모그래피**(global)와 **TPS**(local)의 파라미터를 통합  
- 이미지 겹침(overlap) 영역 정합과 비겹침 영역 왜곡(distortion) 보존을 동시에 최적화  
  - Alignment loss:  

  $$
      L_{\text{align}} = \lambda\|I_r\odot\phi(1,H) - \phi(I_t,H)\|_1 + \ldots
    $$  

   - Distortion loss: 인접 제어점(mesh) 간격 왜곡 억제(ℓ_intra), 비겹침 영역 구조 보존(ℓ_inter)[1]  

### 2.3 네트워크 구조  
1. ResNet50 인코더로 1/16, 1/8 해상도 피처맵 추출  
2. Contextual Correlation Layer → 회귀 네트워크로 4-pt 호모그래피 예측 → 제어점 초기 이동값 생성  
3. 다시 1/8 해상도에서 두 번째 회귀 네트워크로 TPS 제어점 잔차 예측 → 최종 warp 산출  

### 2.4 Seam-Driven Composition  
- Warped 이미지 두 장을 별도 인코더(공유 가중치)로 추출 후 피처 차이를 디코더로 복원하여 soft composition mask 생성  
- **Boundary term**: 겹침 영역 경계 픽셀은 어느 한 이미지로부터만 합성하도록 유도 → seam의 양 끝점 고정  
- **Smoothness term**: color difference map과 stitched image gradient를 활용해 seam 경로의 부드러움 유지  

### 2.5 반복적 워프 적응 전략  
- 사전학습된 모델을 새로운 데이터셋·해상도 단일 샘플에도 무감독 손실  

$$
    L\_{\text{adaption}} = \|I_r\odot\phi(1,TPS) - \phi(I_t,TPS)\|_1
  $$  

- 손실 감소 또는 최대 반복 횟수 달성 시 중단  

## 3. 성능 향상 및 한계  
### 3.1 성능 향상  
- **Warp 정합 품질**: UDIS-D 데이터셋에서 PSNR 25.43, SSIM 0.838로 기존 최고 수준 갱신[1]  
- **Generalization**: cross-dataset·cross-resolution 반복 적응 후 정합 오차 크게 감소  
- **속도**: GPU 활용 시 워핑 0.2s 미만, 합성 0.14s 미만으로 실시간성 확보  

### 3.2 한계  
- TPS 제어점 수증가 시 메모리·연산 비용 상승  
- 반복 적응 단계 추가로 추론 지연 발생 가능  
- 매우 극단적 조명 변화·비전적 도전 사례에 대한 세부적 오류 분석 필요  

## 4. 일반화 성능 향상 관점  
반복적 워프 적응(Iterative Warp Adaption)은 단일 샘플만으로도 현장 데이터에 모델을 빠르게 맞출 수 있어,  
– 드론·자율주행 등 다양한 해상도·장치에 **라벨 없이** 적용 가능  
– 추론 단계에서 경량화된 재학습으로 **도메인 편차** 문제 최소화  

## 5. 향후 연구 영향 및 고려 사항  
- **혼합 도메인 스티칭**: 의료·산업·야외 이미지 등 이질적 데이터셋 융합 워핑 연구 확장  
- **경량화**: TPS 제어점·네트워크 구조 축소를 통한 모바일·엣지 디바이스 최적화  
- **삼각형 메시 vs TPS**: TPS 대비 병렬 효율 분석 기반 새로운 warp 모델 탐색  
- **Seam 마스크 해석 가능성**: soft mask 경계를 통해 seam 위치 해석·수정 인터페이스 연구  

이로써 UDIS++는 **라벨 없는 환경**에서 **시차 대응**과 **일반화**를 동시에 달성, 딥러닝 기반 스티칭의 새로운 전기를 마련할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c385c2da-13d5-40b6-8c98-9665238b602b/2302.08207v2.pdf
