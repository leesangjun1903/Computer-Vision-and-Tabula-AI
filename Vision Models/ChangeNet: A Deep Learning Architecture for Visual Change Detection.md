# ChangeNet: A Deep Learning Architecture for Visual Change Detection | Change detection, Semantic segmentation

**핵심 주장 및 주요 기여**  
ChangeNet은 드론 촬영 이미지 쌍(reference vs. test) 간의 **구조적 변화(구조물 삽입·삭제·변형)** 를 픽셀 단위로 감지하고, 동시에 변화 객체를 **의미론적 분류**(예: 차량, 표지판, 쓰레기통 등)까지 수행하는 **완전 통합형 딥러닝 프레임워크**이다.  
1. 병렬으로 구동되는 **시암 네트워크**(ResNet-50 기반)로 두 이미지를 동일한 특성 공간에 임베딩.  
2. 다중 레이어(feature hierarchy) 수준에서 특징을 추출·결합해, 변화의 **공간적 위치**와 **객체 카테고리**를 동시에 예측.  
3. VL-CMU-CD, TSUNAMI, GSV 세 가지 복잡도·촬영 조건이 다른 데이터셋에서 **최신 기법 대비 모든 지표에서 우수**한 성능을 입증(98.3% 픽셀 정확도, 88.9% AUC 등).

***

## 1. 문제 정의  
- **배경 변화 노이즈**(조명·계절·시점 변화)와 **실제 구조 변화**(물체 삽입·삭제·변형)를 구분하는 고차원 시맨틱 추론이 필요.  
- 기존 기법(픽셀 차분, 배경 모델링, 슈퍼픽셀 기반 등)은 조명·해상도 변화에는 강하나, 뷰 앵글·스케일·객체 이동 변화에 취약.

## 2. 제안 방법  
### 2.1 모델 구조  
- 입력: 시험 이미지 $$I_{test}$$와 기준 이미지 $$I_{ref}$$ (크기 $$224\times224\times3$$)  
- **병렬 시암 네트워크**  
  - 두 서브넷워크는 **가중치 공유**(feature extractor)는 ResNet-50 프리트레인 모델  
  - 레이어 $$\ell_1(7\times7),\,\ell_2(14\times14),\,\ell_3(28\times28)$$ 단계마다 출력 특징 맵 $$\mathbf{f}\_{test}^\ell,\mathbf{f}\_{ref}^\ell$$ 을 획득  
- **다중 스케일 특징 결합**  
  - 각 레이어 별로 upsampling(bilinear)→공간 차원 $$224\times224$$로 복원  
  - $$[\mathbf{f}\_{test}^\ell\,,\mathbf{f}\_{ref}^\ell]$$ 연결(concatenation) → $$1\times1$$ 컨볼루션으로 $$N$$ 클래스 대응 채널 수로 차원 축소  
  - $$\ell=1,2,3$$의 결과를 합산한 뒤 softmax 분류  
- 출력: $$224\times224\times N$$ 차원 변화 지도(change map)  
  - $$N$$은 변화 종류(10개 객체 + 배경)  

### 2.2 수식 개요  

```math
\mathbf{f}_{test}^\ell = g_\ell(I_{test}),\quad
\mathbf{f}_{ref}^\ell = g_\ell(I_{ref})
```

```math
\tilde{\mathbf{f}}^\ell = \mathrm{Upsample}(\mathbf{f}_{test}^\ell)\oplus\mathrm{Upsample}(\mathbf{f}_{ref}^\ell)
\quad(\oplus:\text{concat+1×1 conv})
```

$$
\mathbf{F} = \sum_{\ell=1}^3 \tilde{\mathbf{f}}^\ell,\quad
\hat{Y} = \mathrm{softmax}(\mathbf{F})
$$

## 3. 성능 향상 및 한계  
- **정확도**: VL-CMU-CD 테스트 구간에서  
  - 픽셀 정확도 98.3%, 클래스별 IoU 77.35%, AUC 88.9%  
- **타 기법 대비 우위**:  
  - Supre-pixel(23% f1)·CDnet(55% f1)을 크게 앞서는 79% f1 달성  
- **한계**:  
  - 작은 객체(ex. 교통콘, 간판) 검출·분류 정확도 저하  
  - Occlusion 및 배경 변화가 심한 GSV 데이터셋에서는 f-score 성능 저하  

## 4. 모델 일반화 성능 향상 가능성  
- **멀티-도메인 사전학습**: 드론·위성·도로뷰 등 다양한 촬영 환경 데이터로 ResNet 백본 사전 학습  
- **약한 지도 학습(Weakly Supervised Learning)**: 전역 시멘틱 레이블만 있는 대용량 레이블 데이터 활용해 작은 객체 변화도 포착  
- **어텐션 메커니즘**: 공간·채널 어텐션을 도입해 변화에 민감한 특징 강조  
- **도메인 적응(Domain Adaptation)**: 계절·조명 도메인 간 성능 갭 줄이는 기법 적용  

## 5. 향후 연구 영향 및 고려 사항  
- **영향**:  
  - **시맨틱 변화를 동시에 감지**하는 단일 네트워크 아키텍처를 제시함으로써, 스마트 시티·재난 모니터링·시설 유지보수 분야의 자동화 시스템 발전 촉진  
  - 다중 스케일·다중 레이어 특징 융합의 효용을 입증  
- **고려 점**:  
  - **데이터 다양성** 확보: 객체 크기·형태가 다양한 훈련 샘플 보강  
  - **실시간 처리**: 모델 경량화 및 하드웨어 최적화  
  - **연속적 변화 추적**: 단일 프레임 대비 시계열 데이터 융합으로 변화 정황 해석  

***

**주요 키워드**: 시암 네트워크, 다중 스케일 특징 결합, 시맨틱 변화 검출, ResNet-50 전이 학습, ChangeNet.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b5258291-7947-48b6-b35b-393ddfd2dadb/Varghese_ChangeNet_A_Deep_Learning_Architecture_for_Visual_Change_Detection_ECCVW_2018_paper.pdf
