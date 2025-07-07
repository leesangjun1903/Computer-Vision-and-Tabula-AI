# R-CNN : Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation | Object detection, Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
이 논문은 **CNN 기반의 고용량 특징 표현**을 지역 제안(region proposals)과 결합한 단순·확장 가능한 객체 검출 알고리즘 R-CNN을 제안한다.  
- PASCAL VOC 2012에서 **mAP 53.3%**를 달성해 이전 대비 30% 상대 성능 개선[1].  
- ILSVRC2013 200-class 검출에서 **mAP 31.4%**로 최상위 성능 달성[1].  
- 소량의 도메인 특화 데이터에 대해 **감독된 사전학습 → 미세조정(fine-tuning)** 패러다임을 확립.  

## 2. 문제 정의, 제안 방법, 모델 구조 및 수식  
### 2.1 해결 과제  
- 기존 HOG/SIFT 기반 검출은 mAP 정체.  
- CNN 분류 성능(ILSVRC) → 검출 성능으로 일반화 불확실.  
- 학습용 검출 데이터 부족 시 대용량 CNN 학습 어려움.  

### 2.2 R-CNN 알고리즘 개요[1]
1. 입력 이미지에서 약 2,000개의 **Selective Search** region proposals 생성.  
2. 각 제안 윈도우를 227×227 크기로 **affine warp**(문맥 패딩 p=16)  
3. Caffe 구현된 **Krizhevsky CNN**(5 conv + 2 fc)로 특징 φ(P) 추출.  
4. 클래스별 **linear SVM**으로 스코어링 후 NMS 수행.  

### 2.3 수식: 바운딩 박스 회귀[1]
제안 상자 P=(Px, Py, Pw, Ph) → 예측 박스 Ĝ:  

$$
\begin{aligned}
\hat G_x &= P_w\,d_x(P)+P_x,\\
\hat G_y &= P_h\,d_y(P)+P_y,\\
\hat G_w &= P_w\exp\bigl(d_w(P)\bigr),\quad
\hat G_h = P_h\exp\bigl(d_h(P)\bigr),
\end{aligned}
$$  

여기서 

$$ d\_* (P) = w\_*^\top \phi_5(P) $$ 

는 pool5 피처 기반 선형 회귀, 목표값 $$t_x=(G_x-P_x)/P_w$$ 등으로 정의하며 ridge 회귀로 학습[1].  

### 2.4 네트워크 구조  
- **T-Net**: AlexNet 유사(5 conv, 2 fc; fc6/fc7=4096)  
- **O-Net**: VGG-16 유사(13 conv, 3 fc) 사용 시 VOC2007 mAP 58.5→66.0%로 대폭 향상[1].  

## 3. 성능 향상 및 한계  
### 3.1 성능 향상  
- HOG-DPM 대비 VOC2007 mAP 33.7→54.2%[1].  
- 사전학습 → fine-tuning: mAP +8%p, 바운딩 박스 회귀 +3–4%p 개선.  
- O-Net 활용 시 추가 +7.8%p VOC2007.  

### 3.2 한계  
- **위치 정확도 오류** 다수(Region proposals 한계)[1].  
- 추론 속도 느림(Selective Search+warp 비용).  
- 제안 영역 종속성, 작은 객체 검출 취약.  

## 4. 일반화 성능 향상 가능성  
- 범용 사전학습 + 도메인 특화 미세조정 패러다임은 다양한 시각 인식에 적용 가능[1].  
- Conv 레이어(pool5) 특징이 ImageNet→PASCAL 간에 높은 일반화력 보임.  
- Datasets 간 사전학습 시 **도메인 적응** 및 **하드 예제 마이닝**으로 더 강인한 검출 모델 확보 여지.  

## 5. 향후 연구 영향 및 고려 사항  
- **사전학습→미세조정**은 영상 분할·추적·3D 인식 등 데이터 부족 영역에 핵심 전략.  
- Region proposal 단계 대체(End-to-end 예: Faster R-CNN) 개발 촉진.  
- **효율성 개선**: 공유 합성곱 연산, 경량 아키텍처(모바일·엣지) 필요.  
- **다중 스케일·작은 객체** 대응을 위한 제안 기법 고도화 및 학습 데이터 확대 고려.  

[1] Rich feature hierarchies for accurate object detection and semantic segmentation (Ross Girshick et al., arXiv:1311.2524v5)

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/2cfda68d-4ab7-449a-8ae1-8dd996822903/1311.2524v5.pdf
