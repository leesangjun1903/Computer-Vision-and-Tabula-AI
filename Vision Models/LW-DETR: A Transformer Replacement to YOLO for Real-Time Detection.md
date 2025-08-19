# LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection | Object detection

## 1. 핵심 주장 및 주요 기여  
LW-DETR은 경량화된 DETR(Detection Transformer) 구조로, 기존의 YOLO 계열 실시간 검출기를 뛰어넘는 속도와 정확도를 동시에 달성한다.  
- **단순 구조**: plain ViT 인코더 + 얕은 DETR 디코더 + C2f 프로젝터  
- **연산 효율화**: 전역(Global) 어텐션 일부를 윈도우(Window) 어텐션으로 대체, 윈도우-메이저(feature map 재배열 제거)  
- **학습 효율화**: IoU 인지 분류 손실, 그룹 DETR, 대규모 Objects365 사전학습  
- **성능**: COCO val2017에서 YOLO-NAS·YOLOv8·RTMDet 대비 모든 스케일(tiny~xlarge)에서 mAP↑, FPS↑  

## 2. 해결 과제 및 제안 방법  
### 2.1 해결하고자 하는 문제  
실시간 검출에서 YOLO 계열의 CNN 기반 방법들이 지배적이나, DETR 기반 트랜스포머는 연산량 과다 및 느린 수렴 속도로 실시간 적용이 어렵다.

### 2.2 제안 방법  
1) **모델 구조**  
   - ViT 인코더: 입력 패치를 192∼768 차원으로 임베딩 후 6~10개 레이어에서 전역/윈도우 어텐션을 교대로 수행  
   - 멀티레벨 특징 집계: 중간·최종 인코더 출력 결합  
   - C2f 프로젝터: 단일·다중 스케일(1/8, 1/32) 특징 생성  
   - DETR 디코더: 3개 레이어, deformable cross-attn, 100~300 객체 쿼리  

2) **수식: IoU 인지 분류 손실**  

$$\ell_{\text{cls}} = \sum_{i\in\text{pos}}\text{BCE}(s_i,\,t_i) + \sum_{j\in\text{neg}}s_j^2\text{BCE}(s_j,0),$$  

여기서 $$t=s^\alpha u^{1-\alpha}$$, $$u$$는 예측박스와 GT IoU, $$\alpha=0.25$$.  

3) **효율화 기법**  
   - 윈도우 어텐션: 연산량 제곱에서 선형으로 감소  
   - 윈도우-메이저 배열: row-major→window-major 전환 비용 제거  
   - 그룹 DETR: 13개 그룹 병렬 디코더로 학습 가속  

### 2.3 성능 향상  
- Objects365 사전학습 후 COCO mAP +8.7↑ (small)  
- LW-DETR-small: 48.0 mAP, 340+ FPS  
- LW-DETR-large: 56.1 mAP, 113 FPS (YOLO-NAS 대비 +3.8 mAP, 1.4× 속도)  
- 전반적: 모든 스케일에서 기존 실시간 검출기 대비 mAP·FPS 우위  

### 2.4 한계  
- 개방형 검출(open-world detection)·다중 작업(포즈, 3D) 적용 미검증  
- 사전학습 의존도가 높아, 작은 데이터셋 단일 학습만으로는 성능 저하 가능  
- 추후 NAS·긴 학습 스케줄·증류 기법 결합 필요  

## 3. 일반화 성능 향상 관점  
- **사전학습 이득**: Objects365로부터 평균 +5.5 mAP 상승, transformer 특유의 대규모 데이터 친화성 입증  
- **교차 도메인**: UVO(80개 COCO 외 클래스 포함)에서 class-agnostic mAP +1.3↑, AR@100 +4.1↑  
- **다중 도메인 파인튜닝**: Roboflow100 7개 도메인에서 YOLOv8·RTMDet 대비 AP50 평균 +2.4↑  
- → 경량 DETR 구조가 다양한 도메인·데이터 규모에 강인함  

## 4. 연구적 함의 및 고려사항  
- **영향**: 트랜스포머 기반 실시간 검출 가능성 제시, ViT 단순 구조의 실용성 강조  
- **향후 연구 시**  
  - 개방형·비지도 검출로 확장  
  - NAS·토큰 스파스화·증류 기법 통합  
  - 추가 데이터 증강·긴 학습 스케줄 효과 분석  
  - 효율적 메모리 배열 기법 일반화  

LW-DETR는 트랜스포머가 실시간 검출에서도 CNN을 능가할 수 있음을 보여준 첫 사례로, 후속 연구의 토대를 마련한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8dc62a2e-23fe-47d1-8245-1a2f0429b02f/2406.03459v1.pdf
