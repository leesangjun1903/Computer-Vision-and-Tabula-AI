# WB-DETR: Transformer-Based Detector without Backbone | Object detection

## 1. 핵심 주장 및 주요 기여
WB-DETR는 **CNN 백본 없이 순수 transformer** 만으로 객체 검출을 수행할 수 있음을 보인 최초의 모델이다.  
주요 기여는 다음과 같다.  
- 입력 이미지를 직접 분할·벡터화(tokenization)하여 **CNN 피처 추출 단계 제거**  
- 인접 토큰 간 정보 통합뿐 아니라 각 토큰 내부의 국소 정보를 강화하는 **LIE-T2T 모듈** 제안  
- DETR 대비 **파라미터 수 절반**, 추론 속도 8 FPS 향상, AP 유사 성능 달성  

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
기존 DETR  모델은 CNN 백본에 의존해 국소적 피처를 먼저 추출한 뒤 transformer로 전역 문맥을 학습한다.  
- CNN 백본 설계 복잡도  
- 앵커, NMS 등의 휴리스틱 제거 한계  

### 2.2 모델 구조
WB-DETR는 **Encoder–Decoder** 구조만으로 구성된다.

1. **Image → Tokens**  
   - 입력 이미지 $$x \in \mathbb{R}^{H\times W \times C}$$를 패치 크기 $$p\times p$$, 스트라이드 $$s$$로 잘라 $$l = \frac{H W}{s^2}$$개의 패치 생성  
   - 각 패치 $$x_p \in \mathbb{R}^{p^2 C}$$를 **선형 투영**하여 차원 $$d$$의 토큰 $$T_0 \in \mathbb{R}^{l\times d}$$ 생성  

2. **LIE-T2T Encoder**  
   - 기존 T2T 모듈에 **채널 주의(attention)** 연산을 추가하여 토큰 내부의 국소 정보를 강화  
   - 각 레이어별 연산:  

$$
     \begin{aligned}
     T &= \mathrm{Unfold}\bigl(\mathrm{Reshape}(T_i)\bigr),\\
     S &= \sigma\bigl(W_2\,\mathrm{ReLU}(W_1\,T)\bigr),\\
     T_{i+1} &= W_3\,(T \odot S),
     \end{aligned}
     $$  
     
  여기서 Reshape/Unfold는 토큰↔피처맵 변환, $$W_{1,2,3}$$는 FC 파라미터, $$\sigma$$는 sigmoid, $$\odot$$는 요소별 곱  

3. **Transformer Decoder**  
   - learnable object queries를 이용해 N개 예측을 병렬 디코딩  
   - 각 디코더 출력은 FFN을 거쳐 \{클래스, 바운딩 박스\} 예측  

4. **Loss**  
   - Hungarian matching 기반 세트 예측 loss (cross-entropy + $$L_1$$+IoU)  

## 3. 성능 향상 및 한계

| 모델                | Params | FLOPs | FPS  | AP   | AP₅₀ | AP₇₅ |  
|--------------------|:------:|:-----:|:----:|:----:|:-----:|:-----:|  
| DETR               | 41M    | 86G   | 28   | 42.0 | 62.4  | 44.2  |  
| **WB-DETR (2-12)** | **24M**| 98G   | **36**| 41.8 | 63.2  | 44.8  |  

- **파라미터 절반**, 8 FPS 증가, AP 거의 동일  
- LIE-T2T 층 수 변화에 따라 AP📈: 0층 30.7 → 2층 40.2 → 3층 40.3 (작은 객체 성능 대폭 개선)  
- 토큰 패치·스트라이드 설정(예: 16×16, 스트라이드 8) 중요  

**한계**  
- 고해상도나 극소 객체에서 transformer-only 토큰링의 국소성 한계  
- 학습 효율화를 위한 CNN 사전학습 부재로 수렴 속도 느림  
- 연산량(FLOPs)은 여전히 높음  

## 4. 일반화 성능 향상 가능성
- **토큰화 방식 개선**: 패치 크기·스트라이드를 데이터 분포에 적응적으로 조정하거나 가변 패치 크기 도입  
- **자기 지도 학습**: DINO, MAE형 사전학습으로 이미지 도메인 적응성 및 일반화력 강화  
- **경량화 및 증강**: LIE-T2T 외 추가 경량 모듈(예: 지역적 합성곱) 결합으로 작은 객체와 도메인 갭 해소  

## 5. 향후 연구 방향 및 고려 사항
- CNN 없이 완전 순수 transformer 설계의 **실용성 검증**: 다양한 데이터셋·도메인(위성, 의료영상) 적용 실험  
- **사전학습 전략**: 검출 전용 transformer 사전학습 기법 개발  
- **계층적 토큰 구조**: 다중 해상도 토큰 처리로 국소·전역 정보 균형  
- **효율화 연구**: 연산량 경감, 하드웨어 최적화, 모바일·엣지 환경 적용  

WB-DETR는 객체 검출 패러다임을 CNN-splicing에서 순수 transformer로 전환 가능함을 입증했으며, 향후 backbone-free 비전 모델 연구에 중요한 토대가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4cedc5fb-f6e8-4e17-9147-210d0fc28027/Liu_WB-DETR_Transformer-Based_Detector_Without_Backbone_ICCV_2021_paper.pdf
