# RetinaNet : Focal Loss for Dense Object Detection | Object detection

## 1. 핵심 주장과 주요 기여

**핵심 주장**: 1-stage detector가 2-stage detector에 비해 정확도가 낮은 근본적인 원인은 학습 중 발생하는 극심한 클래스 불균형(class imbalance) 문제이며, 이를 해결하기 위해 제안된 Focal Loss가 효과적인 해결책이다[1].

**주요 기여**:
- **새로운 손실 함수 도입**: Cross Entropy Loss를 개선한 Focal Loss로 easy negative sample의 가중치를 줄이고 hard negative sample에 집중[1]
- **RetinaNet 설계**: Focal Loss를 활용한 1-stage detector로 속도와 정확도를 모두 달성[1]
- **COCO 데이터셋에서 SOTA 달성**: 39.1 AP를 기록하며 기존 1-stage, 2-stage detector 모두를 능가[1]

## 2. 해결하고자 하는 문제

### 클래스 불균형 문제
1-stage detector는 이미지 전체를 조밀하게 샘플링하여 약 100k개의 anchor를 생성하지만, 실제 객체가 포함된 positive sample은 극소수에 불과하다[2][3]. 대부분이 쉽게 분류되는 배경(easy negative)이므로:

- **비효율적 학습**: 대부분의 샘플이 유용한 학습 신호를 제공하지 못함[3]
- **모델 성능 저하**: Easy negative의 압도적인 수가 학습에 악영향을 미쳐 gradient 계산을 왜곡[2]

### 기존 해결 방법의 한계
- **2-stage detector**: Region proposal과 sampling heuristic으로 해결하지만 연산 속도가 느림[1]
- **Hard example mining**: 일부 개선은 있으나 근본적 해결책이 되지 못함[1]

## 3. 제안하는 방법: Focal Loss

### 수식 정의
기본 Cross Entropy Loss:

$$ CE(p_t) = -\log(p_t) $$

**Focal Loss**:

$$ FL(p_t) = -(1-p_t)^{\gamma}\log(p_t) $$

**α-balanced Focal Loss**:

$$ FL(p_t) = -\alpha_t(1-p_t)^{\gamma}\log(p_t) $$

여기서:
- $$p_t$$: 정답 클래스에 대한 예측 확률
- $$\gamma$$: focusing parameter (논문에서는 γ=2 사용)
- $$\alpha_t$$: 클래스 가중치 (논문에서는 α=0.25 사용)[1][2]

### 작동 원리
- **Easy example (p_t > 0.5)**: (1-p_t)^γ 항이 작아져 손실 기여도가 감소
- **Hard example (p_t < 0.5)**: (1-p_t)^γ 항이 커져 손실에 더 큰 영향[4][5]

논문의 실험 결과에 따르면, γ=2일 때 p_t=0.9인 easy example은 일반 CE 대비 100배 낮은 손실을, p_t≈0.968일 때는 1000배 낮은 손실을 갖는다[1].

## 4. 모델 구조: RetinaNet

### 백본 네트워크
- **Feature Pyramid Network (FPN)**: ResNet 위에 구축된 다중 스케일 특징 피라미드[1]
- **P3-P7 레벨**: 다양한 크기의 객체를 효과적으로 탐지[1]

### 서브네트워크
- **Classification Subnet**: 객체 존재 확률 예측 (4개의 3×3 conv + ReLU + 최종 conv + sigmoid)[1]
- **Box Regression Subnet**: Bounding box 회귀 (동일한 구조, 4개 출력)[1]

### Anchor 설계
- **9개 anchor per location**: 3개 스케일 × 3개 aspect ratio {1:2, 1:1, 2:1}[1]
- **32-813 픽셀 범위**: 다양한 크기의 객체 커버[1]

## 5. 성능 향상 및 실험 결과

### COCO 데이터셋 성능
| 모델 | AP | AP50 | AP75 | 추론시간 |
|------|----|----- |------|---------|
| RetinaNet-101-800 | 39.1 | 59.1 | 42.3 | 198ms |
| Faster R-CNN+FPN | 36.2 | 59.1 | 39.0 | 172ms |
| SSD513 | 31.2 | 50.4 | 33.3 | 125ms |

### Focal Loss 효과 분석
- **α-balanced CE vs Focal Loss**: γ=2일 때 34.0 AP vs 31.1 AP로 2.9 AP 향상[1]
- **OHEM 대비**: 최고 성능 OHEM(32.8 AP) 대비 3.2 AP 향상[1]

### 속도-정확도 트레이드오프
RetinaNet은 COCO test-dev에서 기존 모든 1-stage, 2-stage detector의 상위 envelope를 형성하며, 특히 AP<25 저정확도 구간을 제외하고는 최고 성능을 달성했다[1].

## 6. 일반화 성능 향상 가능성

### 도메인 적응성
Focal Loss는 클래스 불균형이 존재하는 다양한 도메인에서 활용 가능하다:
- **의료 영상**: 정상 vs 비정상 영역의 극심한 불균형 해결[6][7]
- **농업**: 다양한 성장 단계의 잡초 탐지에서 검증된 효과[8]
- **보안**: 총기 탐지 등 희귀 객체 탐지에서 성능 향상[9]

### 모델 아키텍처 독립성
Focal Loss는 특정 네트워크 구조에 의존하지 않는 범용적인 손실 함수로, 다양한 backbone과 결합 가능하다[10]:
- ResNet, ResNeXt, EfficientNet 등 다양한 백본에서 검증
- 다른 객체 탐지 프레임워크(YOLO, SSD 등)와 통합 가능

### 전이 학습 효과
사전 훈련된 모델에서 Focal Loss를 적용하면 다음과 같은 이점이 있다:
- **도메인 적응 가속화**: 새로운 도메인의 클래스 불균형에 빠른 적응[11]
- **Few-shot 학습 향상**: 제한된 데이터에서도 안정적인 학습[12]

## 7. 한계점

### 하이퍼파라미터 민감성
- **γ와 α의 상호작용**: 두 파라미터가 함께 조정되어야 하며, 최적값 탐색이 복잡[1][4]
- **데이터셋 의존성**: 클래스 불균형 정도에 따라 최적 파라미터가 달라짐[2]

### 계산 복잡도
- **추가 연산**: Modulating factor 계산으로 인한 추가 오버헤드[13]
- **메모리 사용량**: 모든 anchor에 대한 손실 계산으로 메모리 증가[14]

### 특정 시나리오 한계
- **균형 잡힌 데이터**: 클래스가 균등한 경우 오히려 성능 저하 가능성[5]
- **극도로 작은 객체**: 여전히 탐지 어려움 존재[15]

## 8. 미래 연구에 미치는 영향

### 손실 함수 연구의 새로운 패러다임
Focal Loss는 단순한 re-weighting을 넘어 **동적 가중치 조정**의 개념을 도입하여 후속 연구들에 영감을 제공했다[16]:
- **Gradient-based 손실 함수**: 학습 진행에 따른 동적 조정
- **Multi-scale 손실 함수**: 다양한 크기 객체에 대한 적응적 가중치

### 1-stage Detector의 부활
논문 이후 1-stage detector 연구가 활발해지며 다음과 같은 발전을 이루었다[17][16]:
- **YOLOv4-v8**: Focal Loss 개념을 활용한 성능 향상
- **EfficientDet**: Compound scaling과 결합하여 효율성 극대화[13]

### 클래스 불균형 해결의 일반화
Focal Loss의 성공으로 다른 컴퓨터 비전 태스크에서도 클래스 불균형 해결에 대한 관심이 증가했다[18]:
- **Semantic Segmentation**: 픽셀 단위 불균형 해결
- **Instance Segmentation**: 객체 크기별 불균형 해결

## 9. 앞으로 연구 시 고려사항

### 기술적 측면
1. **하이퍼파라미터 자동 조정**: AutoML을 활용한 γ, α 최적화 연구 필요[19][20]
2. **적응적 Focal Loss**: 학습 진행에 따른 동적 파라미터 조정[21][22]
3. **다중 작업 학습**: Detection과 Segmentation 동시 수행 시 손실 함수 밸런싱[23]

### 실용적 측면
1. **엣지 디바이스 최적화**: 모바일/IoT 환경에서의 경량화 연구[24][25]
2. **실시간 처리**: 비디오 스트림에서의 효율적인 추론 방법[26][27]
3. **도메인 적응**: 의료, 자율주행 등 특수 도메인에서의 최적화[12][11]

### 미래 연구 방향
1. **Transformer 기반 Detection**: Vision Transformer와 Focal Loss 결합 연구[16][28]
2. **3D Object Detection**: 3차원 데이터에서의 클래스 불균형 해결[25][27]
3. **Self-supervised Learning**: 라벨 없는 데이터에서의 Focal Loss 활용[29][23]

RetinaNet과 Focal Loss는 객체 탐지 분야에서 **클래스 불균형 문제에 대한 근본적 해결책**을 제시하며, 이후 연구의 중요한 기준점이 되었다. 특히 1-stage detector의 실용성을 크게 향상시켜 산업계에서의 객체 탐지 기술 도입을 가속화했으며, 현재도 다양한 컴퓨터 비전 응용에서 핵심 기술로 활용되고 있다[18].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/98fb8c49-8649-4fab-bc76-a751f8a7c3c0/1708.02002v2.pdf
[2] https://sonstory.tistory.com/126
[3] https://mj-thump-thump-story.tistory.com/entry/Model-RetinaNet
[4] https://velog.io/@jeongjae96/Focal-Loss
[5] https://node-softwaredeveloper.tistory.com/52
[6] https://ieeexplore.ieee.org/document/10457004/
[7] https://ieeexplore.ieee.org/document/10138413/
[8] https://ieeexplore.ieee.org/document/10903355/
[9] https://ieeexplore.ieee.org/document/10532867/
[10] https://www.mdpi.com/2674-1024/3/3/31
[11] https://www.sec.gov/Archives/edgar/data/1981462/000121390024112300/ea0225264-20f_leddar.htm
[12] https://www.sec.gov/Archives/edgar/data/1835654/000117891325000817/zk2532805.htm
[13] https://mvje.tistory.com/250
[14] https://ieeexplore.ieee.org/document/10512864/
[15] https://sofee.tistory.com/38
[16] https://deepdaiv.stibee.com/p/60
[17] https://velog.io/@chang0517/Object-Detection-%EA%B8%B0%EC%88%A0-%EB%B0%9C%EC%A0%84-%EB%8F%99%ED%96%A5
[18] https://blog-ko.superb-ai.com/top-5-vision-ai-trends-2025/
[19] https://shashacode.tistory.com/23
[20] https://datadive1004.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EC%84%B1%EB%8A%A5-%ED%96%A5%EC%83%81%EC%9D%84-%EC%9C%84%ED%95%9C-%EA%B3%A0%EA%B8%89-%EA%B8%B0%EB%B2%95
[21] https://blog-ko.superb-ai.com/how-to-optimize-deep-learning-models/
[22] https://tristanchoi.tistory.com/662
[23] https://seo.goover.ai/report/202503/go-public-report-ko-3dbccc4a-10f5-4b14-84e5-5d2af11448ac-0-0.html
[24] https://mvje.tistory.com/151
[25] https://kmoeum.com/entry/%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B4%EB%A9%B0-%EC%9E%A5%EC%A0%90%EA%B3%BC-%EB%8B%A8%EC%A0%90-%ED%8A%B8%EB%A0%8C%EB%93%9C%EC%99%80-%EC%A0%84%EB%A7%9D
[26] http://e-jamet.org/_common/do.php?a=full&b=52&bidx=615&aidx=7898
[27] https://www.ultralytics.com/ko/blog/everything-you-need-to-know-about-computer-vision-in-2025
[28] https://www.toolify.ai/ko/ai-news-kr/2025-3463081
[29] https://seo.goover.ai/report/202504/go-public-report-ko-c09d49c5-5a48-4631-b155-3e789a76838d-0-0.html
[30] https://www.sec.gov/Archives/edgar/data/928876/000117891325001537/zk2533083.htm
[31] https://www.sec.gov/Archives/edgar/data/1691221/000164117225000289/form20f.htm
[32] https://www.sec.gov/Archives/edgar/data/1758009/000121390025061351/ea0248083-s1_quantum.htm
[33] https://www.sec.gov/Archives/edgar/data/1969302/000141057825000895/pony-20241231x20f.htm
[34] https://www.sec.gov/Archives/edgar/data/1280263/000095017025046499/amba-20250131.htm
[35] https://www.sec.gov/Archives/edgar/data/1844505/000184450525000038/qti-20241231.htm
[36] https://www.sec.gov/Archives/edgar/data/1815776/000181577625000019/lenz-20241231.htm
[37] http://ieeexplore.ieee.org/document/8237586/
[38] https://ieeexplore.ieee.org/document/8417976/
[39] https://www.semanticscholar.org/paper/da60e046aac895b5775ed34bde45beb86aad0fe8
[40] https://ieeexplore.ieee.org/document/9792391/
[41] https://ieeexplore.ieee.org/document/9577494/
[42] https://ieeexplore.ieee.org/document/9879409/
[43] https://linkinghub.elsevier.com/retrieve/pii/S0925231221019615
[44] https://arxiv.org/abs/2212.11542
[45] https://small0753.tistory.com/30
[46] https://check-this.tistory.com/139
[47] https://eehoeskrap.tistory.com/460
[48] https://velog.io/@pabiya/Focal-Loss-for-Dense-Object-Detection
[49] https://mynameisoh.tistory.com/51
[50] https://mokssi.tistory.com/52
[51] https://woochan-autobiography.tistory.com/929
[52] https://velog.io/@puwoo0314/RetinaNet%EA%B3%BC-EfficientDet
[53] https://healthy-ai.tistory.com/64
[54] https://gaussian37.github.io/dl-concept-focal_loss/
[55] https://yscho.tistory.com/120
[56] https://www.kim2kie.com/res/html/0_formula/00%20AI/Loss.html
[57] https://recordai.tistory.com/53
[58] https://hyeon827.tistory.com/79
[59] https://ffighting.net/deep-learning-paper-review/object-detection/focal-loss/
[60] https://herbwood.tistory.com/19
[61] https://www.sec.gov/Archives/edgar/data/1543623/000121390025057524/ea0239793-10k_usnuclear.htm
[62] https://www.sec.gov/Archives/edgar/data/355811/000035581125000009/gntx-20241231.htm
[63] https://www.sec.gov/Archives/edgar/data/1009922/000165495424005090/nxt_20f.htm
[64] https://www.sec.gov/Archives/edgar/data/828146/000141057825000511/link-20241231x10k.htm
[65] https://www.sec.gov/Archives/edgar/data/1941029/000157587225000138/vc062_s1a.htm
[66] https://www.sec.gov/Archives/edgar/data/1941029/000157587225000091/vc052_s1a.htm
[67] https://www.sec.gov/Archives/edgar/data/1941029/000157587225000219/advb037_424b4.htm
[68] https://arxiv.org/html/2403.07113
[69] https://www.mdpi.com/2079-9292/11/8/1183/pdf
[70] https://arxiv.org/html/2306.16539
[71] https://arxiv.org/pdf/2006.01413.pdf
[72] https://pmc.ncbi.nlm.nih.gov/articles/PMC11300732/
[73] https://arxiv.org/pdf/1909.00169.pdf
[74] https://arxiv.org/pdf/2403.15127.pdf
[75] https://www.mdpi.com/2076-3417/11/14/6310/pdf
[76] https://lamttic.github.io/2024/03/02/01.html
[77] https://atonlee.tistory.com/98
[78] https://study4silver.tistory.com/535
[79] https://www.themoonlight.io/ko/review/class-imbalance-in-object-detection-an-experimental-diagnosis-and-study-of-mitigation-strategies
[80] https://velog.io/@hsbc/one-stage-detector-VS-two-stage-detector
[81] https://bommbom.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EC%9D%BC%EB%B0%98%ED%99%94regularization-%EC%B4%9D%EC%A0%95%EB%A6%AC
[82] https://eehoeskrap.tistory.com/398
[83] https://stackoverflow.com/questions/65942471/one-stage-vs-two-stage-object-detection
[84] https://wikidocs.net/237358
[85] https://gils-lab.tistory.com/2
[86] https://velog.io/@qtly_u/Object-Detection-Architecture-1-or-2-stage-detector-%EC%B0%A8%EC%9D%B4
[87] https://yeonco.tistory.com/57
[88] https://keyog.tistory.com/40
[89] https://wikidocs.net/215129
[90] https://glanceyes.com/entry/Deep-Learning-%EC%B5%9C%EC%A0%81%ED%99%94Optimization
[91] https://velog.io/@glad415/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%EB%AC%B8%EC%A0%9C-8takrl2s
[92] https://www.sec.gov/Archives/edgar/data/1789029/000095017024032299/aeva-20231231.htm
[93] https://www.sec.gov/Archives/edgar/data/1789029/000095017023009764/aeva-20221231.htm
[94] https://www.sec.gov/Archives/edgar/data/1789029/000095017022002379/aeva-20211231.htm
[95] https://www.sec.gov/Archives/edgar/data/1789029/000095017025042849/aeva-20241231.htm
[96] https://www.sec.gov/Archives/edgar/data/1794621/000119312522143748/d337860ds1a.htm
[97] https://www.sec.gov/Archives/edgar/data/1794621/000119312522143739/d297031ds1a.htm
[98] https://www.sec.gov/Archives/edgar/data/1758057/000119312521022835/d118634ds1a.htm
[99] https://www.sec.gov/Archives/edgar/data/1872964/000121390022022287/f20f2021_maristechltd.htm
[100] https://link.springer.com/10.1007/s10661-023-11612-z
[101] https://linkinghub.elsevier.com/retrieve/pii/S2352938524001617
[102] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003186767
[103] https://memesoo99.tistory.com/76
[104] https://talktato.tistory.com/13
[105] http://journal.dcs.or.kr/xml/40977/40977.pdf
[106] https://mint-lab.github.io/mint-lab/papers/Seo24_icros_mot.pdf
[107] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201864236535536
[108] https://small0753.tistory.com/10
[109] https://www.sec.gov/Archives/edgar/data/1583708/000158370825000051/s-20250131.htm
[110] https://www.sec.gov/Archives/edgar/data/1824920/000095017025065699/ionq-20250331.htm
[111] https://www.sec.gov/Archives/edgar/data/2030781/000119312525023930/d906470ds1a.htm
[112] https://www.sec.gov/Archives/edgar/data/2030781/000119312525019604/d885522ds1a.htm
[113] https://www.sec.gov/Archives/edgar/data/1824920/000095017025027722/ionq-20241231.htm
[114] https://www.semanticscholar.org/paper/a36fa1e3d8d0aecebc896c187bb6cc14aa1581ef
[115] https://www.semanticscholar.org/paper/5acd3ab13799579ada6205c28d0087daf723f8af
[116] https://www.semanticscholar.org/paper/67c702911667b740f3140572a1660f714ce0a891
[117] https://www.semanticscholar.org/paper/955e3d25e9b1efeb9652696069dc239a56792eb9
[118] https://www.semanticscholar.org/paper/0a02907b8f32b063cc189b1b2727d8375e073140
[119] https://www.semanticscholar.org/paper/c423856442ff7c62b93f11c725724d1ba3a76e32
[120] http://koreascience.or.kr/journal/view.jsp?kj=BSGHC3&py=2014&vnc=v19n3&sp=329
[121] https://velog.io/@rnrnfjwl11/A-comprehensive-review-of-object-detection-with-deep-learning-%EC%A0%95%EB%A6%AC-%EC%9E%90%EB%A3%8C
[122] https://inhopp.github.io/paper/Paper2/
[123] https://velog.io/@woojinn8/Object-Detection-0.-Object-Detection-%EC%86%8C%EA%B0%9C
[124] https://velog.io/@letsdoit/%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%AA%A8%EB%8D%B8%EC%84%B1%EB%8A%A5%EA%B0%9C%EC%84%A0
[125] https://www.nature.com/articles/s41598-023-42896-3
[126] https://iopscience.iop.org/article/10.1088/1742-6596/2425/1/012019
[127] https://arxiv.org/pdf/1708.02002.pdf
[128] https://arxiv.org/pdf/2006.04388.pdf
[129] https://arxiv.org/html/2212.11542v3
[130] https://arxiv.org/pdf/1904.09048.pdf
[131] http://arxiv.org/pdf/1803.01114.pdf
[132] https://arxiv.org/pdf/1809.06065.pdf
[133] https://modulabs.co.kr/blog/machine_learning_loss_function
[134] https://rahites.tistory.com/167
[135] https://johyeongseob.tistory.com/34
[136] https://velog.io/@skhim520/RetinaNet-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
[137] https://pmc.ncbi.nlm.nih.gov/articles/PMC7841761/
[138] https://arxiv.org/pdf/1305.1707.pdf
[139] https://journal.universitasbumigora.ac.id/index.php/matrik/article/download/2515/1264
[140] https://arxiv.org/pdf/2204.03719.pdf
[141] https://pmc.ncbi.nlm.nih.gov/articles/PMC9571891/
[142] http://arxiv.org/pdf/2406.06099.pdf
[143] http://arxiv.org/pdf/2308.14181.pdf
[144] https://arxiv.org/pdf/1710.05381.pdf
[145] https://pmc.ncbi.nlm.nih.gov/articles/PMC10557500/
[146] http://arxiv.org/pdf/2502.08149.pdf
[147] https://ganghee-lee.tistory.com/34
[148] https://jbluke.tistory.com/554
[149] https://goodgodgd.github.io/ian-flow/archivers/hyu-detection-lecture3
[150] https://ojs.unud.ac.id/index.php/lontar/article/view/109624
[151] https://ieeexplore.ieee.org/document/10603235/
[152] https://facerain.github.io/improve-dl-performance/
[153] https://brunch.co.kr/@donghyungshin/161
[154] https://da-journal.tistory.com/entry/Paper-Object-Detection-in-20-Years-A-Survey
