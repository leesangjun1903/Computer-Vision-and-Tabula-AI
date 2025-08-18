# Dynamic DETR: End-to-End Object Detection with Dynamic Attention | Object detection

## 1. 핵심 주장과 주요 기여 요약
Dynamic DETR는 DETR의 두 가지 주요 한계인  
(1) **작은 물체에 대한 낮은 검출 성능**  
(2) **느린 학습 수렴(convergence)**  
을 동시에 해결하는 **동적 어텐션(dynamic attention)** 프레임워크를 제안한다.[1]
주요 기여는 다음과 같다.
- **Dynamic Encoder**: convolution 기반의 동적 인코더로 다중 해상도(feature pyramid)의 스케일, 공간, 채널 중요도를 동적으로 학습하여 풍부한 표현력을 확보  
- **Dynamic Decoder**: ROI 기반 동적 디코더로 cross-attention을 coarse-to-fine 방식으로 수행하여 학습 난이도를 크게 낮추고 학습 에포크 수를 14× 단축  
- **최신 성능 달성**: ResNet-50 백본, 표준 1× 학습 스케줄(12 에포크)에서 42.9 mAP를 달성하여 종래 기법 대비 3.6 mAP 향상[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하려는 문제
- Transformer 인코더의 **쿼드러틱(self-)어텐션 복잡도**로 인해 feature map 해상도 제약  
- Transformer 디코더의 cross-attention이 전역(global) feature map을 한 번에 학습하여 **수렴 속도 저하**  

### 2.2 제안 방법 및 수식
1) **Dynamic Encoder**  
   - 다중 해상도 입력 $$P = \{P_1, \dots, P_k\}$$에 대해  
   - 이웃 스케일 간 2D deformable convolution으로 공간 어텐션:  

$$
       P_i^+ = \{\mathrm{Up}(\mathrm{DefConv}(P_{i-1}, s_i)),\;\mathrm{DefConv}(P_i, s_i),\;\mathrm{Down}(\mathrm{DefConv}(P_{i+1}, s_i))\}
     $$  
   
   - scale attention via Squeeze-and-Excitation:

$$
       w_i = \mathrm{SE}(P_i^+)
     $$  
   
   - representation attention via DyReLU:  

$$
       \mathrm{DyReLU}(x_c) = \max(a_{1,c}x_c + b_{1,c},\,a_{2,c}x_c + b_{2,c}),\quad
       \{a,b\} = \Delta(x_c)
     $$  
   
   - 최종 multi-scale self-attention:  

$$
       \mathrm{MultiScaleSelfAttn}(P)
       = \mathrm{Concat}_{i=1}^k\bigl(\mathrm{DyReLU}(w_i\,P_i^+)\bigr)
     $$  

2) **Dynamic Decoder**  
   - object query $$Q$$에 self-attention 수행:  

$$
       Q^* = \mathrm{MultiHeadSelfAttn}(Q,Q,Q)
     $$  
   
   - box encoding $$B$$로부터 RoI pooling된 region feature $$F$$; $$Q^*$$로부터 동적 필터 $$W_Q$$ 생성:  

$$
       W_Q = \mathrm{FC}(Q^*),\quad
       QF = \mathrm{Conv}_{1\times1}(F,W_Q)
     $$  
   
   - $$QF$$로부터 물체 임베딩, 박스 좌표, 클래스 예측 생성:  

$$
       \hat Q = \mathrm{FFN}(QF),\quad
       \hat B = \mathrm{ReLU}(\mathrm{LN}(\mathrm{FC}(\hat Q))),\quad
       \hat C = \mathrm{Softmax}(\mathrm{FC}(\hat Q))
     $$  
   
   - 여러 디코더 스테이지를 거치며 coarse-to-fine refinement 수행  

### 2.3 모델 구조
- **Backbone**: ResNet/FPN (5 scales)  
- **Dynamic Encoder**: 6-layer convolution-based self-attention stack  
- **Dynamic Decoder**: 6-layer ROI-based dynamic attention stack, 각 레이어에 4개의 self-attention 헤드, 300 object queries  
- **Optimizer**: AdamW, 학습률 1e-4, 1×(12 epoch) 및 3×(36 epoch) 스케줄  

### 2.4 성능 향상
- **COCO val** (1×): DETR 대비 +27.4 mAP, Deformable DETR 대비 +5.7 mAP  
- **COCO test-dev** (1×): 42.9 mAP로 종래 BorderDet(41.4 mAP) 대비 +1.5 mAP  
- **수렴 속도**: 50 epoch 내 수렴(약 4× 빠름)[1]

### 2.5 한계
- **추가 연산**: Dynamic 모듈 도입으로 인코더·디코더 파라미터 및 연산량 증가  
- **복잡도**: 구조 이해 및 구현 난이도 상승  
- **실시간 적용**: 높은 연산 비용으로 모바일·임베디드 환경 적용은 미검증  

## 3. 일반화 성능 향상 관점
Dynamic Attention이 다양한 스케일·공간·채널 정보에 적응적으로 작용함으로써,  
- 훈련 데이터에 없는 **크기·비율·조명 변화**에 보다 로버스트하게 대응  
- Coarse-to-fine 박스 리파인먼트로 **오버피팅 감소**  
- 다양한 도메인(예: 자율주행, 의료 영상)으로의 전이 학습(transfer learning) 성능 향상 기대[1]

## 4. 향후 연구에 미치는 영향 및 고려 사항
- **영향**: Vision Transformer 기반 detection 연구에 동적 attention 통합 전략의 새 방향 제시  
- **고려점**:  
  - **효율화**: 동적 모듈 경량화 및 지연 연산(lazy computation) 도입  
  - **다중 태스크 확장**: 세분화(segmentation), 비디오 객체 탐지로의 확장  
  - **전이 학습**: 소량 라벨 데이터 상황에서의 일반화 실험  
  - **하드웨어 최적화**: FPGA/Edge GPU 상 성능·전력 최적화  

***

 X. Dai et al., “Dynamic DETR: End-to-End Object Detection with Dynamic Attention,” ICCV 2021.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/a1fd3672-bae0-4fec-8c0e-78b94fe6f1ad/Dai_Dynamic_DETR_End-to-End_Object_Detection_With_Dynamic_Attention_ICCV_2021_paper.pdf
