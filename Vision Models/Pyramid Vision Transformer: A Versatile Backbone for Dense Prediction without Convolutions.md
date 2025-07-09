# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions | Image classification

**핵심 주장**  
Pyramid Vision Transformer(PVT)는 *완전한* Transformer 기반의 백본으로, CNN 없이 이미지 분류(CLS)뿐 아니라 객체 검출(DET), 인스턴스·시맨틱 분할(SEG) 등의 픽셀 단위 촘촘한 예측(dense prediction) 과제에 범용(backbone)으로 사용할 수 있음을 보인다.  

**주요 기여**  
1. **문제 정의**  
   - 기존 Vision Transformer(ViT)는 단일 해상도(columnar) 출력만 내어 객체 검출·분할에 부적합하고, 고해상도(input patch 크기 4×4 픽셀 불가능) 처리 시 계산·메모리 비용 과다.  
2. **제안 모델(PVT)**  
   - *Pyramid 구조*: CNN의 특징 피라미드(FPN)처럼 입력 이미지를 4×4 → 2×2 → 2×2 → 2×2 픽셀 패치로 점진적 축소(shrinking)하여 4단계(stage)의 멀티스케일 피처맵 \{F₁(1/4), F₂(1/8), F₃(1/16), F₄(1/32)\} 생성.  
   - *Spatial-Reduction Attention (SRA)*: 일반 Multi-Head Attention(MHA)의 Key·Value를 Ri² 폴딩(folding) + projection하여 $$O((H W)²)$$ → $$O((H W)²/R_i²)$$ 연산으로 감소.  
   
   $$\text{SRA}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_j)W^O,\quad \mathrm{head}_j=\mathrm{Attention}(QW_j^Q,\;\mathrm{SR}(K)W_j^K,\;\mathrm{SR}(V)W_j^V)$$  

   $$\mathrm{SR}(x)=\mathrm{Norm}(\mathrm{Reshape}(x,R_i)W^S)$$  

3. **모델 구조**  
   - 4단계 Transformer encoder. 각 단계 i는 패치 크기 $$P_i$$, 출력 채널 $$C_i$$, 층수 $$L_i$$, SRA 축소비 $$R_i$$, 헤드수 $$N_i$$, FFN 확장비 $$E_i$$로 정의.  
   - 네 가지 규모(PVT-Tiny/S/M/L)로 깊이 및 채널 수 조정, 파라미터 수가 ResNet18/50/101/152와 유사.  
4. **성능 향상**  
   - COCO 객체 검출(RetinaNet 기준): ResNet50(36.3 AP) 대비 PVT-Small 40.4 AP (+4.1) 기록.  
   - COCO 인스턴스 분할(Mask R-CNN 기준): ResNet50 34.4 APm → PVT-Tiny 35.1 APm (+0.7).  
   - ADE20K 시맨틱 분할(Semantic FPN 기준): ResNet50 36.7 mIoU → PVT-Small 39.8 mIoU (+3.1).  
   - 순수 Transformer 검출(DETR 결합): ResNet50-DETR 32.3 AP → PVT-Small-DETR 34.7 AP (+2.4).  
5. **한계 및 계산 비용**  
   - 입력 해상도가 커질수록 GFLOPs 증가 폭이 ResNet보다 크고, 800px 기준 PVT-Small의 추론 지연(ms)이 다소 높음.  
   - 자잘한 CNN용 모듈(SE, Dilated Conv 등) 미적용.  

## 모델의 일반화 성능 향상 가능성

- **글로벌 어텐션의 장점**: SRA로 경량화된 전역 의존성 학습은 다양한 스케일·도메인에 걸쳐 강건한 표현 학습을 지원.  
- **피처 피라미드 통합**: 멀티스케일 피처 제공으로 DET, SEG head 설계 단순화 및 전이 학습 시 안정적인 공간 정보 유지.  
- **사전학습 효과**: ImageNet 사전학습 시 빠른 수렴·높은 성능을 보이며, 다양한 입력 크기·데이터셋에 걸쳐 적용 가능성 높음.  

## 향후 연구에 미치는 영향 및 고려 사항

- **완전 Transformer 파이프라인 확대**: PVT를 백본으로 OCR, 3D 포인트 클라우드, 의료 영상 등 타 분야 밀도 예측 모델 연구에 활용 가능.  
- **경량화·효율화**: 더 낮은 복잡도의 SRA 변형 또는 동적 축소비 적용으로 실시간 임베디드 비전에도 적용.  
- **혼합 모달리티**: 비전·언어 멀티모달 태스크에 PVT 백본 활용 시, 멀티스케일 특징과 어텐션 융합 효과 기대.  
- **CNN 고유 모듈과의 융합**: SE, Dilated Conv, NAS 등 CNN 특화 기술 접목으로 성능·효율 추가 향상 여부 검토 필요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/9e12b371-81f1-4b40-b8fa-b07f10d035b0/2102.12122v2.pdf
