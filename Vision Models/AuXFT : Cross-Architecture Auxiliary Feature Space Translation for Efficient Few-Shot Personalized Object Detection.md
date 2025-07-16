# Cross-Architecture Auxiliary Feature Space Translation for Efficient Few-Shot Personalized Object Detection | Object detection

## 핵심 주장과 주요 기여

이 논문은 **Instance-level Personalized Object Detection (IPOD)**라는 새로운 과제를 제시하며, 이를 해결하기 위한 **AuXFT (Auxiliary Feature Space Translation)** 프레임워크를 제안합니다[1]. 핵심 주장은 기존 객체 탐지 시스템이 **Neural Collapse 현상**으로 인해 일반적인 클래스(예: 개)와 사용자 특정 인스턴스(예: 사용자의 개) 구별에 실패한다는 것입니다[1].

**주요 기여:**

1. **Auxiliary Feature Space 생성**: 탐지 성능에 영향을 주지 않으면서 개인화를 위한 보조 특성 공간을 생성하는 파이프라인 설계[1]

2. **Cross-Architecture Knowledge Distillation**: CNN 기반 탐지기(YOLOv8)와 Vision Transformer(DINOv2) 간의 지식 전이를 가능하게 하는 Translator Block 도입[1]

3. **Conditional Coarse-to-Fine FSL**: 계산 효율성을 위해 조건부 분류를 수행하는 Few-Shot Learning 모듈 설계[1]

4. **실용적 성과**: 상한선의 80% 성능을 추론 시간 32%, VRAM 13%, 모델 크기 19%로 달성[1]

## 해결하고자 하는 문제

### 1. Neural Collapse 현상

**Neural Collapse**는 Cross-Entropy 손실로 사전 훈련된 모델의 특성이 클래스 중심 주변으로 붕괴되어 클래스 내 분산을 잃는 현상입니다[1][2]. 이는 개인화에 필요한 인스턴스 간 구별 능력을 크게 저하시킵니다[3][4].

### 2. 프라이버시 및 계산 비용 문제

- **프라이버시 문제**: 개인화 작업은 중앙 서버에서 많은 샘플과 모델 튜닝이 필요하여 개인정보 보호 우려를 야기합니다[1]
- **계산 비용**: Foundation Models 기반 접근법은 높은 계산 비용으로 인해 온디바이스 응용에 적합하지 않습니다[1]

### 3. 아키텍처 간 특성 불일치

CNN(YOLOv8)과 Vision Transformer(DINOv2) 간의 특성 맵 수, 채널 깊이, 해상도 차이로 인한 지식 전이 어려움[1]

## 제안하는 방법

### 1. Translator Block 구조

**Channel Differential (DC) 모듈**:
- 탐지기 특성 공간을 오라클 특성 공간의 채널 깊이로 매핑: $$D_C : \mathbb{R}^{H' \times W' \times l_E} \rightarrow \mathbb{R}^{H' \times W' \times l_O} $$  [1]
- 3×3 컨볼루션을 사용하여 성능과 계산 비용의 균형을 맞춤[1]

**Spatial Differential (DS) 모듈**:
- 리샘플링 비율 ρ에 따른 적응적 보간 전략:
  - ρ < 1-δ: area 보간 (다운샘플링)
  - 1-δ ≤ ρ < 1+δ: bilinear 보간 
  - ρ ≥ 1+δ: bicubic 보간 (업샘플링)[1]

### 2. 잔차 연쇄 구조

보조 특성 생성을 위한 잔차 연쇄:

$$R_i = \sum_{j=i}^{n} D_S(D_C(F_j)) \quad \forall i = 1, 2, \ldots, n $$  [1]

이는 고해상도 특성으로 저해상도 특성의 고주파 성분을 복원합니다[1].

### 3. 지식 증류 손실

l₁과 l₂ 규범의 합을 사용한 재구성 손실:

$$L_R = \sum_{i=1}^{n} \frac{\sum_{p \in H'' \times W''} \|R_i[p] - O[p]\|_1 + \|R_i[p] - O[p]\|_2}{H'' \times W''} $$   [1]

### 4. Detection-Driven Feature Pooling (DDFP)

예측된 바운딩 박스를 사용한 공간 인식 평균 풀링:

$$v_k = \frac{1}{x_1 - x_0} \frac{1}{y_1 - y_0} \sum_{x=x_0}^{x_1-1} \sum_{y=y_0}^{y_1-1} R_1[x, y] $$   [1]

### 5. 조건부 Few-Shot Learning

계산 효율성을 위해 조건부 분류 수행:

$$d = \frac{1}{|D|} \sum_{d \in D} \text{softmax}\left(\frac{1}{d(q, p_i)}; \forall i = 1, \ldots, |P[c]|\right) $$   [1]

$$c' = \arg\max_{i=1,\ldots,|P[c]|} d[i] $$   [1]

여기서 q는 쿼리 벡터, c는 거친 클래스 예측, P[c]는 프로토타입 집합입니다[1].

## 모델 구조

### 전체 아키텍처

1. **인코더-디코더 탐지기**: YOLOv8n을 기본 백본으로 사용[1]
2. **Translator Block**: 두 개의 차별화 모듈(DC, DS)로 구성[1]
3. **오라클 네트워크**: DINOv2 같은 SSL 사전 훈련 모델[1]
4. **FSL 모듈**: 프로토타입 기반 분류기[1]

### 훈련 단계

1. **탐지기 훈련**: 일반적인 객체 탐지 손실로 훈련[1]
2. **지식 증류**: 오라클 특성을 보조 공간으로 증류[1]
3. **FSL 훈련**: 사용자 제공 샘플로 프로토타입 학습[1]

## 성능 향상 및 실험 결과

### 성능 개선

**정량적 성과**:
- PerSeg 데이터셋에서 기준 대비 10.5 mAP 향상[1]
- POD 데이터셋 1-shot에서 9.7 mAP, 5-shot에서 9.4 mAP 향상[1]
- 오라클 상한선 대비 일관된 80% 성능 달성[1]

**효율성 개선**:
- 추론 시간: 32% (221.4ms → 288.7ms)[1]
- VRAM 사용량: 13% (12.2MB → 18.4MB)[1]
- 모델 크기: 19%[1]

### 데이터셋 검증

**PerSeg**: 212개 이미지, 39개 인스턴스 레벨 클래스[1]
**POD**: 자체 제작 벤치마크, 225개 훈련 이미지, 150개 검증 이미지[1]
**iCubWorld**: 로봇 시점 데이터, 검색 정확도로 평가[1]
**CORe50**: 다양한 배경의 객체 인스턴스[1]

## 모델의 일반화 성능 향상 가능성

### 1. 아키텍처 독립성

AuXFT는 다양한 탐지기와 SSL 모델 조합에 적용 가능한 **모델 비의존적** 프레임워크입니다[1]. 이는 다음과 같은 일반화 가능성을 제공합니다:

- **탐지기 다양성**: YOLOv8 외에 다른 CNN 기반 탐지기 적용 가능
- **오라클 모델 확장**: DINOv2 외에 다른 SSL 모델 활용 가능
- **크로스 도메인 적용**: 다양한 응용 도메인에 적용 가능

### 2. 전이학습 능력

**자기지도 학습의 장점**[5][6]:
- 레이블 없는 대규모 데이터로 학습된 SSL 모델의 일반화 능력 활용
- 도메인 간 전이 성능 향상 가능성

**증류된 특성의 견고성**[7]:
- 오라클 모델 없이도 추론 시 일반화 성능 유지
- 다양한 시각적 작업에서 강인한 특성 제공

### 3. 스케일러빌리티

**데이터 효율성**:
- Few-shot 설정에서 효과적인 개인화 가능
- 새로운 인스턴스 클래스 추가 시 최소한의 데이터 필요

**계산 효율성**:
- 온디바이스 배포에 적합한 경량화 설계
- 실시간 처리 가능한 추론 속도

## 한계

### 1. 기술적 한계

**아키텍처 제약**:
- CNN과 Transformer 간의 특성 정렬에 한계
- 복잡한 번역 과정으로 인한 정보 손실 가능성

**프로토타입 기반 한계**:
- 클래스 내 변동성이 높은 경우 단일 프로토타입의 한계[8]
- 프로토타입 품질에 따른 성능 의존성

### 2. 데이터 및 도메인 한계

**데이터셋 크기**:
- 상대적으로 작은 규모의 데이터셋에서 검증
- 대규모 실제 환경에서의 성능 검증 필요

**도메인 특수성**:
- 개인용 기기 환경에 최적화된 설계
- 산업용 또는 전문 분야 적용 시 성능 검증 필요

### 3. 실용적 한계

**메모리 효율성**:
- 프로토타입 저장 및 검색 과정의 메모리 오버헤드
- 클래스 수 증가에 따른 확장성 문제

**업데이트 메커니즘**:
- 새로운 인스턴스 추가 시 재훈련 필요
- 연속 학습 능력의 한계

## 향후 연구에 미치는 영향

### 1. 개인화 객체 탐지 분야 확산

**새로운 연구 방향**:
- **Instance-level Personalized Object Detection** 분야의 체계적 정의[1]
- 개인화 시스템의 평가 기준 및 벤치마크 제시

**응용 분야 확장**:
- 스마트 홈 시스템에서의 개인 물품 인식[1]
- 로봇 어시스턴트의 개인화 서비스
- 증강현실/가상현실 응용

### 2. Cross-Architecture 지식 전이 연구

**아키텍처 간 호환성**:
- CNN-Transformer 간 지식 전이 방법론 발전[9]
- 다양한 모델 아키텍처 조합 연구 촉진

**효율적 증류 기법**:
- 계산 효율성을 고려한 지식 증류 방법 개발
- 모바일 및 엣지 디바이스 배포를 위한 경량화 연구

### 3. Few-Shot Learning 발전

**조건부 학습**:
- 계산 효율성을 위한 조건부 FSL 접근법 확산
- 다단계 분류 체계의 활용 증가

**실용적 응용**:
- 실제 환경에서의 FSL 적용 사례 증가
- 산업 표준 벤치마크 개발

## 앞으로 연구 시 고려할 점

### 1. 기술적 개선 방향

**아키텍처 최적화**:
- 더 효율적인 특성 정렬 방법 개발
- 정보 손실 최소화를 위한 번역 기법 연구

**Neural Collapse 해결**:
- 근본적인 NC 현상 해결 방안 연구[10][11]
- 사전 훈련 단계에서의 NC 방지 기법 개발

### 2. 확장성 고려사항

**대규모 시스템**:
- 수천 개 인스턴스 클래스를 다루는 시스템 설계
- 분산 학습 및 추론 방법론 개발

**연속 학습**:
- 새로운 인스턴스 지속적 추가 능력[12]
- 망각 없는 점진적 학습 방법론

### 3. 실용적 배포 고려사항

**프라이버시 보호**:
- 연합 학습 기반 개인화 시스템 구축
- 차분 프라이버시 기법 적용

**품질 보증**:
- 실제 환경에서의 강건성 검증
- 안전성 및 신뢰성 평가 기준 확립

### 4. 학제간 연구 촉진

**인간-컴퓨터 상호작용**:
- 사용자 경험 관점에서의 개인화 시스템 설계
- 직관적인 인터페이스 개발

**윤리적 고려사항**:
- 개인화 기술의 편향성 및 공정성 연구
- 프라이버시 보호와 성능 간의 균형점 모색

이 논문은 개인화 객체 탐지라는 새로운 패러다임을 제시하며, 실용적인 해결책을 통해 향후 연구 방향에 중요한 영향을 미칠 것으로 예상됩니다. 특히 온디바이스 AI 시스템의 개인화 요구가 증가하는 현재 상황에서 매우 시의적절한 연구로 평가됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/1d6532fa-4e0c-4124-81ff-0ab2fcf7625f/2407.01193v1.pdf
[2] https://arxiv.org/abs/2506.08562
[3] https://academic.oup.com/jcde/article/12/1/300/7935522
[4] https://arxiv.org/abs/2310.06823
[5] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/3045886/CXR-DINO--paving-the-way-for-a-medical-vision/10.1117/12.3045886.full
[6] https://ieeexplore.ieee.org/document/9709990/
[7] https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/
[8] https://milvus.io/ai-quick-reference/what-is-a-prototype-network-in-fewshot-learning
[9] https://arxiv.org/html/2502.10691v1
[10] https://www.semanticscholar.org/paper/5d6d0bb2df73f3c6da10884481303f165ae8d7c2
[11] https://arxiv.org/abs/2209.08378
[12] https://www.mdpi.com/1424-8220/24/11/3456
[13] https://www.sec.gov/Archives/edgar/data/1760903/000164117225001259/form10-k.htm
[14] https://www.sec.gov/Archives/edgar/data/1760903/000164117225013251/forms-1a.htm
[15] https://www.sec.gov/Archives/edgar/data/1760903/000164117225002670/forms-1a.htm
[16] https://www.sec.gov/Archives/edgar/data/1760903/000164117225003661/forms-1a.htm
[17] https://www.sec.gov/Archives/edgar/data/1760903/000164117225011535/forms-1.htm
[18] https://www.sec.gov/Archives/edgar/data/1760903/000149315225004865/forms-1.htm
[19] https://www.sec.gov/Archives/edgar/data/1760903/000164117225016825/form8-k.htm
[20] https://arxiv.org/abs/2210.11173
[21] https://arxiv.org/abs/2402.04672
[22] https://ieeexplore.ieee.org/document/10807360/
[23] https://arxiv.org/html/2404.01397v1
[24] https://openreview.net/forum?id=5opsMLx5LC
[25] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003179956
[26] https://www.ai4europe.eu/research/ai-catalog/few-shot-object-detection-fsdet-training-tools-custom-data
[27] https://openaccess.thecvf.com/content/ACCV2022/papers/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.pdf
[28] https://arxiv.org/html/2506.08562v1
[29] https://blog.si-analytics.ai/7
[30] https://github.com/sebastian-hofstaetter/neural-ranking-kd
[31] https://research.samsung.com/blog/Object-conditioned-Bag-Of-Instances-For-Few-shot-Personalized-Instance-Recognition
[32] https://arxiv.org/abs/2207.05273
[33] https://www.sciencedirect.com/science/article/abs/pii/S1568494625000663
[34] https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/CUSTOM.md
[35] https://arxiv.org/abs/2404.16386
[36] https://openreview.net/pdf?id=SwIp410B6aQ
[37] https://www.sciencedirect.com/science/article/pii/S0262885624004840
[38] https://www.sec.gov/Archives/edgar/data/1760903/000164117225015276/form8-k.htm
[39] https://www.sec.gov/Archives/edgar/data/1760903/000164117225010835/form10-q.htm
[40] https://scik.org/index.php/cmbn/article/view/9048
[41] https://sol.sbc.org.br/index.php/sbcas/article/view/35530
[42] https://ieeexplore.ieee.org/document/10794473/
[43] https://arxiv.org/abs/2404.16818
[44] https://link.springer.com/10.1007/978-3-031-80871-5_6
[45] https://ieeexplore.ieee.org/document/10793755/
[46] https://learnopencv.com/dinov2-self-supervised-vision-transformer/
[47] https://dl.acm.org/doi/10.5555/3294996.3295163
[48] https://en.wikipedia.org/wiki/You_Only_Look_Once
[49] https://www.cs.toronto.edu/~zemel/documents/prototypical_networks_nips_2017.pdf
[50] https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf
[51] https://www.labellerr.com/blog/unlocking-the-power-of-self-supervised-learning-in-computer-vision-with-dino/
[52] https://arxiv.org/abs/1703.05175
[53] https://docs.ultralytics.com/yolov5/tutorials/architecture_description/
[54] https://osintteam.blog/dino-vs-dinov2-a-comprehensive-comparison-of-meta-ais-self-supervised-learning-models-fa162ca94e5a
[55] https://www.v7labs.com/blog/yolo-object-detection
[56] https://thecho7.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-DINOv2-Learning-Robust-Visual-Features-without-Supervision-%EC%84%A4%EB%AA%85
[57] https://rhcsky.tistory.com/9
[58] https://www.datacamp.com/blog/yolo-object-detection-explained
[59] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dinov2/
[60] https://www.sec.gov/Archives/edgar/data/1774170/000162828025033105/aiot-20250331.htm
[61] https://www.sec.gov/Archives/edgar/data/1443669/000162828025007052/prlb-20241231.htm
[62] https://www.sec.gov/Archives/edgar/data/1467623/000146762325000011/dbx-20241231.htm
[63] https://www.sec.gov/Archives/edgar/data/42682/000143774924005519/grc20231231_10k.htm
[64] https://www.sec.gov/Archives/edgar/data/763532/000143774923025488/lyts20230630_10k.htm
[65] https://www.sec.gov/Archives/edgar/data/1387467/000138746723000049/aosl-20230630.htm
[66] https://www.sec.gov/Archives/edgar/data/1590976/000159097623000069/mbuu-20230630.htm
[67] https://ieeexplore.ieee.org/document/10802460/
[68] https://ieeexplore.ieee.org/document/10516600/
[69] https://ieeexplore.ieee.org/document/10659003/
[70] https://www.mdpi.com/2072-4292/16/7/1203
[71] https://arxiv.org/abs/2412.19165
[72] https://ieeexplore.ieee.org/document/10614889/
[73] https://linkinghub.elsevier.com/retrieve/pii/S0262885624002609
[74] https://arxiv.org/abs/2309.01086
[75] https://eccv.ecva.net/virtual/2024/poster/1155
[76] https://www.isca-archive.org/interspeech_2019/liu19d_interspeech.pdf
[77] https://openreview.net/forum?id=caE5faFVT1
[78] https://arxiv.org/abs/2404.04799
[79] https://aclanthology.org/2025.loresmt-1.15.pdf
[80] https://www.themoonlight.io/ko/review/object-conditioned-bag-of-instances-for-few-shot-personalized-instance-recognition
[81] https://arxiv.org/html/2504.10685v1
[82] https://arxiv.org/html/2202.13393v3
[83] https://arxiv.org/abs/2407.01193
[84] https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Generalized_Few-Shot_Object_Detection_Without_Forgetting_CVPR_2021_paper.pdf
[85] https://www.nature.com/articles/s41598-025-91152-3
[86] https://seungwooham.tistory.com/entry/Sequence-Level-Knowledge-Distillation-%EC%9A%94%EC%95%BD-%EB%B0%8F-%EC%84%A4%EB%AA%85
[87] https://towardsdatascience.com/instance-level-recognition-6afa229e2151/
[88] https://www.sciencedirect.com/science/article/pii/S156625352400085X
[89] https://www.sciencedirect.com/science/article/abs/pii/S088523082300102X
[90] https://arxiv.org/abs/2307.01951
[91] https://arxiv.org/html/2310.06823v3
[92] https://arxiv.org/html/2311.01479v3
[93] https://downloads.hindawi.com/journals/cin/2020/8825197.pdf
[94] https://arxiv.org/pdf/1706.06969.pdf
[95] https://pmc.ncbi.nlm.nih.gov/articles/PMC3387543/
[96] https://arxiv.org/pdf/2209.09211.pdf
[97] https://pmc.ncbi.nlm.nih.gov/articles/PMC9247008/
[98] https://elifesciences.org/articles/84797
[99] https://pmc.ncbi.nlm.nih.gov/articles/PMC3174808/
[100] https://www.mdpi.com/2076-3425/13/10/1387/pdf?version=1695973053
[101] https://goombalab.github.io/blog/2024/distillation-part1-mohawk/
[102] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5222345
[103] https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_How_Far_Pre-trained_Models_Are_from_Neural_Collapse_on_the_ICCV_2023_paper.pdf
[104] https://www.sciencedirect.com/science/article/pii/S2214914722001660
[105] https://ieeexplore.ieee.org/document/10314722/
[106] https://ieeexplore.ieee.org/document/10587077/
[107] https://arxiv.org/pdf/2403.03273.pdf
[108] https://arxiv.org/html/2411.19331v1
[109] https://arxiv.org/html/2407.06298
[110] https://arxiv.org/html/2412.16334v1
[111] https://arxiv.org/html/2410.19836v1
[112] http://arxiv.org/pdf/2304.07193.pdf
[113] https://arxiv.org/pdf/2209.07819.pdf
[114] https://arxiv.org/html/2502.08769v2
[115] http://arxiv.org/pdf/2312.02366.pdf
[116] https://arxiv.org/abs/2503.21187
[117] https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
[118] https://encord.com/blog/yolo-object-detection-guide/
[119] https://github.com/facebookresearch/dinov2
[120] https://kalelpark.tistory.com/60
[121] https://www.mdpi.com/2227-7390/11/20/4277
[122] https://ieeexplore.ieee.org/document/10144007/
[123] http://arxiv.org/pdf/2404.01397.pdf
[124] http://arxiv.org/pdf/2112.03641.pdf
[125] https://arxiv.org/html/2405.17859v3
[126] http://arxiv.org/pdf/2407.01193.pdf
[127] https://arxiv.org/pdf/2309.01086.pdf
[128] http://arxiv.org/pdf/1807.00119.pdf
[129] http://arxiv.org/pdf/2409.16073.pdf
[130] https://arxiv.org/pdf/2303.06674.pdf
[131] https://arxiv.org/pdf/2107.05005.pdf
[132] https://arxiv.org/html/2211.11612v2
[133] https://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf
[134] https://eval.ai/web/challenges/challenge-page/2270/overview
