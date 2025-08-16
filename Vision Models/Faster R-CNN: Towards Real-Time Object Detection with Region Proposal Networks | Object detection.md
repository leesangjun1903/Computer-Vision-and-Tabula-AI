# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | Object detection

**핵심 주장:**  
Faster R-CNN은 기존 물체 탐지 시스템의 병목인 **영역 제안(region proposal)** 단계를 딥네트워크 기반의 Region Proposal Network(RPN)으로 통합하여, 제안 단계의 계산 비용을 사실상 ‘거의 무료’로 만들고, 전체 탐지 파이프라인을 단일 네트워크로 합쳤다.  

**주요 기여:**  
- RPN을 통해 입력 이미지의 컨볼루션 특징맵으로부터 직접 고품질의 물체 후보영역을 예측  
- RPN과 Fast R-CNN 객체 검출기를 **공유된 컨볼루션 계층**으로 학습시켜 제안+검출을 하나의 통합망(unified network)으로 구현  
- 앵커(anchor) 메커니즘(3종의 스케일×3종의 종횡비)을 도입, 다중 크기·비율 영역을 단일 스케일 특징맵에서 효율적으로 처리  
- PASCAL VOC 및 MS COCO에서 기존 방법 대비 유사하거나 더 우수한 정확도(mAP)를 유지하면서, VGG-16 기준 약 5fps(최소 17fps) 실시간 성능 달성  

# 1. 해결 과제  
기존의 사전 영역 제안 기법(Selective Search, EdgeBoxes 등)은 CPU 기반으로 0.2–2s 수준의 비용이 필요해, GPU 가속된 Fast R-CNN보다 느려 전체 시스템의 병목이 됨[1].  
따라서, “제안 단계”를 **딥러닝으로 학습**하고, 검출망과 **특징 공유**하여 계산 중복을 제거하는 것이 목표이다.

# 2. 제안 방법

## 2.1 Region Proposal Network (RPN)  
- **입력:** 이미지 처리 후 마지막 공유 컨볼루션 특징맵  
- **슬라이딩 윈도우:** $$n\times n$$ (논문에서는 $$3\times3$$)  
- **출력 채널 수:**  
  - 박스 분류: $$2k$$ (물체/배경)  
  - 박스 회귀: $$4k$$ (각 앵커별 바운딩박스 조정)  
  - 기본 앵커 수 $$k=9$$ (3 scales $$\times$$ 3 ratios)  
- **손실 함수:**

$$
L({p_i}, {t_i}) = \frac{1}{N_{\text{cls}}} \sum_{i} L_{\text{cls}}(p_i, p_i^* ) + \lambda \frac{1}{N_{\text{reg}}} \sum_{i} p_i^* L_{\text{reg}}(t_i, t_i^*)
$$

  - $$p_i$$: i번째 앵커의 물체 확률, $$p_i^*$$: 정답 레이블(1/0)  
  - $$t_i$$, $$t_i^*$$: 회귀된/정답 박스 파라미터($$x,y,w,h$$)  
  - $$L_{cls}$$: 이진 소프트맥스, $$L_{reg}$$: smooth $$L_1$$  
  - $$\lambda=10$$, $$N_{cls}=256$$, $$N_{reg}\approx2400$$로 정규화  

### 앵커(anchor)  
- 중심이 슬라이딩 위치에 고정된 **기준 박스**  
- 3가지 크기($$128^2,256^2,512^2$$)와 3가지 종횡비($$1:1,1:2,2:1$$)  
- 앵커별로 회귀(regression) 및 분류(classification) 레이어가 weight sharing 없이 독립 학습  

## 2.2 통합 학습된 Faster R-CNN  
4단계 교대로 학습(Alternating training)  
1. RPN 훈련(이미지넷 사전학습된 컨볼루션 초기화)  
2. RPN이 생성한 지역으로 Fast R-CNN 훈련  
3. 2단계의 컨볼루션 고정 후 RPN 미세조정 → 공유 컨볼루션 확보  
4. 3단계 컨볼루션 고정 후 Fast R-CNN 미세조정 → 완전 통합망 완성  

# 3. 성능 향상 및 한계

| 데이터셋         | 방법                   | #제안 | mAP (%)           |
|------------------|------------------------|-------|-------------------|
| PASCAL VOC2007   | Selective Search+Fast R-CNN (ZF) | 2000  | 58.7              |
|                  | RPN+Fast R-CNN (ZF, 공유)       | 300   | **59.9**          |
|                  | RPN+Fast R-CNN (VGG, 공유)      | 300   | **69.9**          |
| MS COCO test-dev | Fast R-CNN (SS)                 | 2000  | 39.3 (@0.5 IoU)   |
|                  | Faster R-CNN (RPN)              | 300   | **42.1** @0.5; 21.5@[.5,.95] |

- **계산 속도:**  
  - VGG-16 기준 SS+Fast R-CNN 전체 1.83s → Faster R-CNN 0.198s (≈5fps) [Table 5]  
  - ZF 모델 사용 시 ≈17fps  
- **한계:**  
  - 단일 스케일(짧은 변 $$s=600$$) 설정으로 작은 물체 검출 한계  
  - 앵커 배치가 고정되어 물체 크기·비율 분포가 다른 데이터에서는 재튜닝 필요  
  - Approximate joint training은 박스 좌표에 대한 그라디언트 무시  

# 4. 일반화 성능 향상 가능성

- **데이터 규모 확대:** MS COCO(80개 클래스) 학습 → PASCAL VOC 미세조정만으로 mAP 78.8% 달성[Table 12]  
- **네트워크 깊이 확장:** ResNet-101 적용 시 COCO val mAP@[.5,.95] 21.2→27.2%까지 크게 상승  
- **하이퍼파라미터 민감도 낮음:** 앵커 비율·스케일, $$\lambda$$ 값 변화에 mAP 변화는 1–2% 수준[Table 8,9]  
- **공유 학습 프레임워크:** 제안+검출망 간에 특징 공유가 가능해 다양한 도메인(3D, 인스턴스 분할 등)으로 확장 용이[13–16]  

# 5. 향후 연구에 미치는 영향 및 고려사항

- **앵커리스(anchor-free) 제안 기법 발전:** RPN 한계를 보완할 수 있는 키포인트나 센터 기반 방법 연구  
- **다중 스케일·해상도 학습:** 단일 스케일의 제안 정확도 제약 극복을 위한 FPN(Feature Pyramid Networks) 등  
- **정밀도 향상을 위한 비약식 학습:** 제안 좌표에 대한 완전한 그라디언트 적용(예: RoI warping)  
- **도메인 적응 및 소규모 데이터:** 전이학습과 약지도 학습으로 드문 객체 카테고리 일반화  
- **경량화 및 실시간성 강화:** 모바일·임베디드 환경을 위한 경량 백본, 앵커 최적화  

Faster R-CNN은 **통합된 region proposal과 detection** 체계를 제시하며, 이후 모든 객체 탐지 연구의 표준이 되었고, **FPN, Mask R-CNN, RetinaNet, anchor-free** 방법론 등으로 발전하는 기반을 마련했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3eea7b72-68b4-4042-b066-bd9b7bb22685/1506.01497v3.pdf
