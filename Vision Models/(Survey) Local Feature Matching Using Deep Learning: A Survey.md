# Local Feature Matching Using Deep Learning: A Survey

## 1. 핵심 주장과 주요 기여

이 논문은 딥러닝 기반 지역 특징 매칭(Local Feature Matching)에 대한 포괄적인 서베이 연구로, 2018년 이후 제안된 최신 방법들을 체계적으로 분류하고 분석합니다.

**핵심 주장:**
- 딥러닝의 도입으로 지역 특징 매칭 분야가 크게 발전했으나, 여전히 시점 변화, 조명 변화 등으로 인한 정확도와 견고성 향상에 과제가 남아있음
- 탐지기 유무에 따른 체계적 분류를 통해 각 방법론의 장단점을 명확히 구분할 필요가 있음

**주요 기여사항:**
1. **체계적 분류 체계 제시**: Detector-based와 Detector-free 방법론으로 구분하여 각각을 세부 카테고리로 분류
2. **실제 응용 분야 분석**: SfM, 원격 , 의료 영상 등록 등 다양한 실제 응용 사례 검토
3. **데이터셋 및 평가 지표 표준화**: 주요 데이터셋 분류 및 성능 평가 메트릭 정리
4. **정량적 성능 비교**: 핵심 방법들의 벤치마크 성능 비교 분석
5. **미래 연구 방향 제시**: 8가지 주요 도전과제와 연구 기회 제시

## 2. 해결하고자 하는 문제와 제안 방법

### 해결하려는 문제
- **시점 변화와 조명 변화**에 따른 매칭 정확도 저하
- **텍스처가 없는 영역**과 **반복 패턴**에서의 매칭 실패
- **계산 복잡도**와 **실시간 처리** 요구사항 간의 균형
- **도메인 간 일반화** 성능의 한계

### 제안하는 분류 체계

**1. Detector-based Models**
- **Detect-then-Describe**: 키포인트 탐지 후 특징 기술
  - Repeatability 지표: $$\text{Repeatability} = \frac{M}{\min(F_1, F_2)} \times 100$$
- **Joint Detection and Description**: 탐지와 기술의 동시 학습
- **Describe-then-Detect**: 특징 기술 후 키포인트 선택
- **Graph Based**: 그래프 신경망 기반 매칭

**2. Detector-free Models**
- **CNN Based**: 4D 대응 볼륨 활용
- **Transformer Based**: Self-attention과 Cross-attention 메커니즘
- **Patch Based**: 패치 단위 매칭 후 정제

### 주요 모델 구조

**SuperGlue (Graph-based)**:
- 키포인트를 노드로 하는 그래프 구조
- Self-attention과 Cross-attention을 교대로 적용
- Sinkhorn 알고리즘으로 최적 할당 결정

**LoFTR (Detector-free)**:
- Transformer 기반 coarse-to-fine 매칭
- 저해상도에서 고해상도로 점진적 정제
- 4D 대응 볼륨 대신 attention 메커니즘 활용

## 3. 성능 향상 및 한계

### 성능 향상
**HPatches 데이터셋 결과 (AUC@10px)**:
- RoMa: 89.1% (최고 성능)
- DKM: 88.5%
- PMatch: 88.5%
- SuperPoint+SuperGlue: 81.7%

**주요 성능 향상 요인**:
1. **Attention 메커니즘**: 전역 정보 활용으로 매칭 정확도 향상
2. **Coarse-to-fine 전략**: 다중 스케일 매칭으로 정밀도 개선
3. **기하학적 제약 활용**: 에피폴라 제약 등으로 견고성 향상

### 한계점
1. **계산 복잡도**: Transformer 기반 방법들의 $$O(n^2)$$ 복잡도
2. **극한 환경 조건**: 악천후, 극심한 조명 변화에서 성능 저하
3. **데이터셋 편향**: 특정 도메인에 편향된 훈련 데이터
4. **실시간 처리**: 고해상도 이미지에서의 처리 속도 한계

## 4. 일반화 성능 향상 관련 내용

### Foundation Models 활용
**SAM, DINO, DINOv2** 등 대규모 기반 모델의 zero-shot 일반화 능력 활용:
- 카테고리에 구애받지 않는 의미론적 지식 제공
- 다양한 장면과 객체에 대한 광범위한 일반화 능력
- 전통적 분할 모델 대비 우수한 적응성

### Weakly Supervised Learning
정밀한 주석 의존성 감소를 통한 일반화 향상:
- **카메라 포즈 정보** 활용한 에피폴라 기하 제약
- **자기 지도 학습**을 통한 더 큰 데이터셋 활용
- **도메인 적응** 기법으로 cross-domain 성능 향상

### Adaptation Strategy
**동적 네트워크 조정**:
- 시각적 겹침과 외관 변화에 따른 네트워크 깊이/너비 조절
- 매칭 난이도 기반 적응형 attention span 조정
- 스케일 변화에 따른 적응형 패치 세분화

### Geometric Information 활용
**기하학적 prior 통합**:
- 상대적 깊이와 카메라 포즈 예측
- 에피폴라 제약을 통한 관련 없는 영역 필터링
- 3D 기하학적 일관성 강화

## 5. 미래 연구에 미치는 영향과 고려사항

### 미래 연구에 미치는 영향

**1. 연구 방향 제시**
- **Efficient Attention**: 계산 복잡도 감소를 위한 새로운 attention 메커니즘
- **Foundation Model 통합**: 대규모 모델과의 효과적 결합 방법론
- **Multi-modal Fusion**: 다양한 센서 데이터의 통합 활용

**2. 평가 기준 표준화**
- 도메인별 특화된 평가 프로토콜 개발
- Cross-domain 일반화 성능 평가 지표 정립
- 실제 응용에서의 성능 검증 방법론 구축

**3. 응용 분야 확장**
- 문화유산 보존, 증강현실 등 새로운 응용 영역
- 의료, 원격감지 등 전문 도메인에서의 특화 기법
- 실시간 시스템에서의 최적화 방법론

### 향후 연구 시 고려사항

**1. 기술적 측면**
- **효율성-성능 트레이드오프**: 실용적 응용을 위한 균형점 탐색
- **견고성 확보**: 다양한 환경 조건에서의 안정적 성능
- **확장성**: 대규모 이미지 컬렉션에서의 처리 능력

**2. 데이터 및 평가**
- **포괄적 데이터셋**: 극한 조건을 포함한 다양한 환경 데이터
- **표준화된 벤치마크**: 공정한 방법론 간 비교를 위한 통일된 평가 기준
- **실제 성능 검증**: 실험실 환경과 실제 환경 간의 격차 해소

**3. 융합 접근법**
- **전통-딥러닝 결합**: 각각의 장점을 살린 하이브리드 방법론
- **다중 모달리티**: RGB, 깊이, 열화상 등 다양한 센서 정보 통합
- **도메인 적응**: 한 도메인에서 학습한 모델의 다른 도메인 적용

이 서베이는 지역 특징 매칭 분야의 현재 상태를 종합적으로 정리하고, 향후 연구의 나침반 역할을 할 것으로 기대됩니다. 특히 Foundation Model과의 통합, 효율적인 Attention 메커니즘 개발, 그리고 실제 응용에서의 견고성 확보가 중요한 연구 방향으로 제시되었습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/adcfd766-d33d-47cc-9a7a-f8925e48d173/2401.17592v2.pdf

# Detector-based Models 

Local Feature Matching에서 **Detector-based Models**는 keypoint 탐지기를 사용하여 이미지에서 중요한 특징점들을 먼저 찾은 후, 이러한 특징점들을 기술하고 매칭하는 방법론입니다[1][2]. 전통적인 컴퓨터 비전 파이프라인을 따르며, 크게 4가지 하위 카테고리로 분류됩니다.

## 1. 기본 개념과 전체 구조

Detector-based 방법론은 **sparse-to-sparse 특징 매칭**을 사용하며, 다음과 같은 핵심 특징을 가집니다[1]:

- **탐지기 의존성**: keypoint detector의 성능에 크게 의존
- **효율성**: nearest neighbor search가 효율적이고 메모리 사용량이 적음
- **한계**: day-night와 같은 극한 조건에서 성능 저하 가능

## 2. 4가지 주요 카테고리

### 2.1 Detect-then-Describe (탐지 후 기술)

**가장 전통적인 접근 방식**으로, 먼저 keypoint를 탐지한 후 descriptor를 추출합니다[1][3].

#### 2.1.1 Fully-Supervised Methods
- **L2Net**: 점진적 샘플링 전략과 중간 특징 맵에 대한 추가 감독 사용
- **H중하여 보조 손실 항 제거
- **SOSNet**: HardNet을 확장하여 2차 유사성 정규화 도입
- **ALIKE**: 미분 가능한 keypoint 탐지(DKD) 모듈 제안

#### 2.1.2 Weakly Supervised Methods
- **AffNet**: affine shape 학습에 집중
- **CAPS**: 카메라 포즈 감독을 활용한 약한 감독 학습 프레임워크
- **DISK**: 강화 학습을 활용한 end-to-end 파이프라인

### 2.2 Joint Detection and Description (공동 탐지 및 기술)

**탐지와 기술을 동시에 학습**하는 통합된 접근 방식입니다[1][4].

#### 주요 특징:
- 극한 변화(day-night, 계절 변화, 약한 텍스처)에서 우수한 성능
- 정보 공유를 통한 일관성 향상
- 고수준 정보 활용 가능

#### 대표적인 방법들:
- **SuperPoint[5]**: 자기 감독 방식으로 keypoint 위치와 descriptor를 동시에 결정
- **D2-Net[6]**: 공유 파라미터를 사용한 공동 탐지-기술 접근법
- **R2D2[7]**: 그리드 피크 탐지와 descriptor 신뢰성 예측 결합
- **ASLFeat**: multi-level 특징 맵의 채널 및 공간 피크 사용

### 2.3 Describe-then-Detect (기술 후 탐지)

**먼저 특징을 기술한 후 keypoint를 선택**하는 역전된 접근 방식입니다[1][8].

#### 핵심 아이디어:
- Descriptor 공간의 정보를 활용하여 keypoint 위치 제안
- 높은 정보 내용을 가진 salient 위치를 keypoint로 선택

#### 주요 방법들:
- **D2D**: descriptor의 상대적 및 절대적 현저성 측정을 통한 keypoint 정의
- **PoSFeat**: 약한 감독 하에서 분리된 훈련 접근법
- **SCFeat**: 공유 결합 다리 전략 사용

### 2.4 Graph Based (그래프 기반)

**Graph Neural Network(GNN)을 사용**하여 keypoint 간의 관계를 모델링합니다[1][9].

#### 작동 원리:
1. Keypoint를 노드로 하는 그래프 구조 생성
2. Self-attention과 Cross-attention 레이어 교대 적용
3. Sinkhorn 알고리즘으로 최적 할당 결정

#### 주요 방법들:
- **SuperGlue[10]**: attention graph neural network와 optimal transport 방법 채택
- **SGMNet[11]**: 시딩 모듈을 통한 부분 매칭점 처리
- **ClusterGNN[12]**: 그래프 노드 클러스터링 알고리즘 활용
- **LightGlue**: 매칭 난이도에 따른 적응적 네트워크 깊이/너비 조정

## 3. 성능 및 특징 비교

### 장점:
- **효율성**: 탐지된 sparse keypoint만 처리하므로 계산 효율적
- **메모리 효율성**: dense 방법 대비 메모리 사용량 적음
- **성숙한 기술**: 오랜 연구를 통해 안정적인 방법론 확립

### 한계점:
- **탐지기 의존성**: keypoint detector 성능에 매칭 결과가 크게 좌우
- **극한 조건 취약**: 심한 조명 변화, 텍스처 부족 상황에서 성능 저하
- **반복 패턴 문제**: 유사한 패턴이 반복되는 영역에서 매칭 실패 가능

## 4. 최신 동향과 발전 방향

### 성능 향상 기법:
- **Foundation Model 활용**: SAM, DINO 등 대규모 모델의 zero-shot 일반화 능력 활용[1]
- **Efficient Attention**: 계산 복잡도 감소를 위한 새로운 attention 메커니즘 개발
- **Weakly Supervised Learning**: 정밀한 주석 의존성 감소를 통한 일반화 향상

### 적응형 전략:
- **동적 네트워크 조정**: 매칭 난이도에 따른 네트워크 깊이/너비 조절
- **기하학적 정보 활용**: 에피폴라 제약 등을 통한 견고성 향상

Detector-based Models는 여전히 많은 실용적 응용에서 핵심적인 역할을 하고 있으며, 딥러닝 기술과의 결합을 통해 지속적으로 발전하고 있습니다. 특히 Graph-based 방법들은 전역적 정보 교환을 통해 기존 방법들의 한계를 극복하며 새로운 가능성을 보여주고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/adcfd766-d33d-47cc-9a7a-f8925e48d173/2401.17592v2.pdf
[2] https://arxiv.org/html/2401.17592v2
[3] https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Chang_Structured_Epipolar_Matcher_for_Local_Feature_Matching_CVPRW_2023_paper.pdf
[4] https://ieeexplore.ieee.org/document/10204558/
[5] http://arxiv.org/pdf/2410.22710.pdf
[6] https://www.sciencedirect.com/science/article/pii/S0303243422001210
[7] https://www.mdpi.com/2076-3417/14/23/11098
[8] https://openaccess.thecvf.com/content/ACCV2020/papers/Tian_D2D_Keypoint_Extraction_with_Describe_to_Detect_Approach_ACCV_2020_paper.pdf
[9] https://psarlin.com/superglue/
[10] https://arxiv.org/html/2503.05122v1
[11] https://arxiv.org/html/2304.14845
[12] http://journals.sagepub.com/doi/10.1177/17298806231158298
[13] https://www.tandfonline.com/doi/full/10.1080/01431161.2018.1528402
[14] https://arxiv.org/abs/2301.02993
[15] https://ieeexplore.ieee.org/document/10655054/
[16] https://ieeexplore.ieee.org/document/10203946/
[17] https://ieeexplore.ieee.org/document/10550550/
[18] https://ieeexplore.ieee.org/document/9578008/
[19] https://ieeexplore.ieee.org/document/10377620/
[20] https://ieeexplore.ieee.org/document/10205230/
[21] https://www.nature.com/articles/s41598-025-90955-8
[22] https://pmc.ncbi.nlm.nih.gov/articles/PMC10283127/
[23] https://zju3dv.github.io/loftr/
[24] https://blog.roboflow.com/what-is-feature-matching/
[25] http://nnw.cz/doi/2022/NNW.2022.32.017.pdf
[26] https://arxiv.org/abs/2005.13605
[27] https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_RetinaTrack_Online_Single_Stage_Joint_Detection_and_Tracking_CVPR_2020_paper.pdf
[28] https://arxiv.org/abs/2401.17592
[29] https://blog.lomin.ai/d2net-a-trainable-cnn-for-joint-description-and-detection-of-local-features-33632
[30] https://www.sciencedirect.com/science/article/abs/pii/S0262885620300640
[31] https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Guiding_Local_Feature_Matching_with_Surface_Curvature_ICCV_2023_paper.pdf
[32] https://arxiv.org/abs/1905.03561
[33] https://github.com/vignywang/Awesome-Local-Feature-Matching
[34] https://pubs.acs.org/doi/10.1021/acs.analchem.0c03693
[35] https://ieeexplore.ieee.org/document/10662506/
[36] https://ieeexplore.ieee.org/document/10229788/
[37] https://ieeexplore.ieee.org/document/8935498/
[38] https://www.mdpi.com/2072-4292/14/18/4595
[39] https://ieeexplore.ieee.org/document/8586755/
[40] http://scholarpublishing.org/index.php/AIVP/article/view/5619
[41] https://milvus.io/ai-quick-reference/what-is-feature-matching-in-image-search
[42] https://www.themoonlight.io/review/yolopoint-joint-keypoint-and-object-detection
[43] https://scispace.com/papers/superglue-learning-feature-matching-with-graph-neural-3t2e18buwg
[44] https://github.com/verlab/DALF_CVPR_2023
[45] https://arxiv.org/abs/1911.11763
[46] https://pmc.ncbi.nlm.nih.gov/articles/PMC10347320/
[47] https://arxiv.org/html/2308.08479v3
[48] https://blog.lomin.ai/superglue-learning-feature-matching-with-graph-neural-networks-33640
[49] https://europe.naverlabs.com/wp-content/uploads/2019/09/R2D2-Repeatable-and-Reliable-Detector-and-Descriptor-2.pdf
[50] https://learnopencv.com/feature-matching/
[51] https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
[52] https://www.sciencedirect.com/science/article/pii/S0303243421002415
[53] https://www.cs.cornell.edu/courses/cs4670/2018sp/pa3/sp18_index.html
[54] https://github.com/magicleap/SuperGluePretrainedNetwork
[55] https://jseobyun.tistory.com/350
[56] https://linkinghub.elsevier.com/retrieve/pii/S0031320323007914
[57] https://linkinghub.elsevier.com/retrieve/pii/S0950705125003971
[58] https://arxiv.org/pdf/2308.08479.pdf
[59] https://arxiv.org/pdf/2302.05846.pdf
[60] http://arxiv.org/pdf/2103.08573.pdf
[61] https://downloads.hindawi.com/journals/wcmc/2021/8927822.pdf
[62] http://arxiv.org/pdf/1905.03561.pdf
[63] https://arxiv.org/html/2504.04497v1
[64] http://arxiv.org/pdf/2203.09645v2.pdf
[65] https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/25/2020/isprs-annals-V-2-2020-25-2020.pdf
[66] https://arxiv.org/pdf/2401.17592.pdf
[67] https://www.sciencedirect.com/science/article/abs/pii/S0950705125003971
[68] https://www.semanticscholar.org/paper/af061c42083cf9c8f9294b68e2cc547229210e78
[69] https://www.tandfonline.com/doi/pdf/10.1080/10095020.2020.1843376?needAccess=true
[70] https://arxiv.org/pdf/2104.00680.pdf
[71] https://arxiv.org/abs/1912.00623
[72] http://arxiv.org/pdf/2405.18872.pdf

# Detector-free Models 

Detector-free Models는 **키포인트 탐지 과정을 생략**하고, 원본 이미지에서 **조밀(dense)한 특징(descriptor)** 직접 추출하여 대응점을 찾는 방법입니다. 이로 인해 시점 변화나 텍스처가 적은 영역에서도 보다 안정적인 매칭이 가능합니다. 주요 세 가지 접근 방식이 있습니다.

## 1. CNN 기반 방법 (CNN Based)

- **4D 상관 맵(correlation volume)**을 생성  
  두 이미지에서 추출한 특징 맵 $$f_A, f_B$$의 각 위치를 대응시켜 4차원(tensor) 상관 맵을 만듭니다.  
- **동작 원리**  
  1. 이미지 A와 B를 CNN으로 피처 맵 $$f_A$$, $$f_B$$로 변환  
  2. 모든 위치 쌍 $$(i,j)$$ vs. $$(k,l)$$에 대해 유사도 점수를 계산해 4D tensor $$c_{ijkl}$$ 생성  
  3. 이 tensor를 CNN/GNN으로 가공해 유효 매칭만 골라냄  
- **대표 모델**  
  - NCNet: 4D correlation을 활용한 최초 모델  
  - Sparse-NCNet: 4D tensor를 희소 연산으로 처리해 속도·메모리 절약  
  - PDC-Net(+): 확률적 학습으로 대응점 신뢰도 예측  

이 방식은 **큰 시야각 변화**에서도 견고하지만, 4D tensor 생성과 연산 비용이 크다는 단점이 있습니다.

## 2. Transformer 기반 방법 (Transformer Based)

- **Self-/Cross-Attention**  
  - Self-Attention: 같은 이미지 내 픽셀 간 전역 문맥 교환  
  - Cross-Attention: 두 이미지 간 정보 교환  
- **Coarse-to-Fine 전략**  
  낮은 해상도에서 전체적인 대응을 찾은 뒤, 점진적으로 해상도를 높여 정밀 매칭  
- **대표 모델**  
  - LoFTR: detector 없이 coarse-to-fine attention으로 픽셀 대응  
  - Aspanformer: 매칭 난이도 따라 attention 범위(adaptive span) 조절  
  - SE2-LoFTR: steerable CNN으로 회전 불변성 추가  

Transformer는 **전역 문맥**을 활용해, 저텍스처 영역에서도 매칭 성능이 뛰어납니다. 다만 attention 연산의 $$O(n^2)$$ 비용을 완화하는 연구가 활발합니다.

## 3. 패치 기반 방법 (Patch Based)

- **이미지를 격자(patch) 단위로 분할**하여 각 패치별 descriptor를 매칭  
- **Match Refinement** 단계로 포인트 단위 정밀 매칭  
- **대표 모델**  
  - Patch2Pix: 패치 단위 매칭 후 기하 변환 기반 정제  
  - AdaMatcher, PATS: 패치 스케일·할당을 학습해 극한 변형 대응  

패치 기반은 **큰 이동**에도 강인하며, 좁은 지역 내 밀집 매칭이 용이하지만, 연산량·메모리 요구가 높아 최적화가 과제입니다.

### 장점과 한계

| 장점                                         | 한계                                               |
|--------------------------------------------|---------------------------------------------------|
| -  탐지기 의존성 제거 → 시점·조명 변화에 견고       | -  4D tensor나 attention 연산의 높은 계산·메모리 비용      |
| -  저텍스처·반복 패턴에서도 대응점 찾기 가능       | -  실시간 응용에는 추가적인 최적화 필요                  |
| -  전역 문맥 활용으로 매칭 정확도 향상             | -  모델 구조가 복잡해 학습·추론 속도 저하 우려             |

Detector-free Models는 **전역 정보를 활용해 조밀한 대응을 생성**함으로써, 기존 detector-based 한계를 극복하고 있습니다. 다만 **연산 효율화** 연구가 중요하며, **Coarse-to-Fine**, **Sparse 구조**, **Adaptive Attention** 등 다양한 최적화 기법이 핵심입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/adcfd766-d33d-47cc-9a7a-f8925e48d173/2401.17592v2.pdf
