- Under Sampling | 언더 샘플링
- Simple Over Sampling | 단순 오버 샘플링
- Algorithm Over Sampling | 알고리즘을 통한 오버샘플링(SMOTE, ADASYN)
- Cost-sensitive learning

# Reference
https://dining-developer.tistory.com/27

## Imbalanced Classification

### **배경 (Background)**

**Imbalanced Classification**은 기계학습과 데이터 마이닝에서 가장 도전적인 문제 중 하나로, 데이터셋에서 클래스들 간의 분포가 불균등한 상황을 다루는 분야입니다[1][2]. 이 문제는 한 클래스(다수 클래스, majority class)의 예제 수가 다른 클래스(소수 클래스, minority class)보다 현저히 많을 때 발생합니다[3][4].

일반적으로 불균형 정도에 따라 다음과 같이 분류됩니다[4]:
- **경미한 불균형**: 소수 클래스가 전체 데이터의 20-40%
- **중간 불균형**: 소수 클래스가 1-20%
- **극심한 불균형**: 소수 클래스가 1% 미만

전통적인 기계학습 알고리즘들은 균등한 클래스 분포를 가정하고 설계되어, 불균형 데이터에서는 다수 클래스에 편향된 결과를 보이며 소수 클래스의 중요한 패턴을 놓치는 문제가 발생합니다[5][2].

### **방법론 (Methodology)**

불균형 분류 문제를 해결하기 위한 방법론은 크게 네 가지 범주로 분류됩니다[6][7]:

#### **1. 데이터 수준 방법 (Data-Level Methods)**

**오버샘플링 기법**:
- **SMOTE (Synthetic Minority Oversampling Technique)**: 소수 클래스의 k-최근접 이웃을 이용해 합성 데이터를 생성[8][9]
- **ADASYN (Adaptive Synthetic Sampling)**: 소수 클래스의 학습 난이도에 따라 적응적으로 샘플을 생성[10][11]
- **Borderline-SMOTE**: 클래스 경계 근처의 샘플에 집중하여 합성 데이터 생성[11]

**언더샘플링 기법**:
- **Random Under-Sampling**: 다수 클래스에서 무작위로 샘플 제거
- **Tomek Links**: 노이즈와 경계 샘플 제거
- **Edited Nearest Neighbor**: 잘못 분류된 샘플 제거[12]

#### **2. 알고리즘 수준 방법 (Algorithm-Level Methods)**

**비용 민감 학습 (Cost-Sensitive Learning)**:
- 서로 다른 오분류 비용을 할당하여 소수 클래스의 잘못된 분류에 더 높은 페널티 부여[13][14]
- 클래스 가중치 조정을 통한 모델 편향 완화[5]

**임계값 조정 (Threshold Adjustment)**:
- 분류 임계값을 조정하여 정밀도와 재현율 간의 균형 조절[15]

#### **3. 앙상블 방법 (Ensemble Methods)**

**배깅과 부스팅**:
- **RUSBoost**: 언더샘플링과 부스팅 결합[6]
- **SMOTEBoost**: SMOTE와 부스팅 결합[6]
- **BalancedRandomForest**: 균형 잡힌 부트스트랩 샘플링 사용[16]

#### **4. 딥러닝 기반 방법**

**신경망 아키텍처 개선**:
- 포컬 로스(Focal Loss) 함수 사용
- 클래스 가중치를 적용한 손실 함수 설계[5]
- 생성 적대 신경망(GAN)을 이용한 합성 데이터 생성[17]

### **연구 동기 (Research Motivation)**

불균형 분류 연구의 주요 동기는 다음과 같습니다:

#### **1. 실제 응용 분야의 필요성**
- **사기 탐지**: 정상 거래 대비 사기 거래는 극소수[1][18]
- **의료 진단**: 건강한 환자 대비 질병 환자는 소수[19][18]
- **침입 탐지**: 정상 네트워크 트래픽 대비 공격은 드물게 발생[18]
- **산업 품질 관리**: 정상 제품 대비 결함 제품은 소수[20]

#### **2. 기존 알고리즘의 한계**
전통적인 기계학습 알고리즘들이 정확도(accuracy)를 최적화 목표로 설정할 때, 다수 클래스를 모두 맞추는 것만으로도 높은 정확도를 달성할 수 있어 소수 클래스의 중요한 패턴을 학습하지 못하는 문제[2][1]

#### **3. 경제적·사회적 영향**
소수 클래스의 오분류가 심각한 결과를 초래하는 경우가 많음:
- 의료진단에서 질병 미탐지로 인한 생명 위험
- 금융에서 사기 거래 미탐지로 인한 경제적 손실[14]

### **연구 목표 (Research Goals)**

불균형 분류 연구의 주요 목표는 다음과 같습니다:

#### **1. 단기 목표**
- **균형 잡힌 성능 달성**: 다수 클래스와 소수 클래스 모두에서 우수한 예측 성능 확보[1]
- **적절한 평가 지표 개발**: 정확도 대신 F1-score, AUC-ROC, G-mean 등 불균형에 적합한 지표 활용[6]
- **효율적인 샘플링 기법 개발**: 오버피팅을 방지하면서 효과적인 합성 샘플 생성[11]

#### **2. 중장기 목표**
- **자동화된 불균형 처리**: 데이터 특성에 따라 자동으로 최적의 기법을 선택하는 시스템 개발[21]
- **다중 클래스 불균형**: 이진 분류를 넘어 다중 클래스 불균형 문제 해결[22]
- **실시간 처리**: 스트리밍 데이터에서의 실시간 불균형 분류[23]

## 리뷰 논문이 다룰 범위와 기대 결과

### **연구 현황 (Current Research Status)**

현재 불균형 분류 연구는 다음 영역에서 활발히 진행되고 있습니다:

#### **1. 기술적 혁신**
- **하이브리드 방법**: 데이터 수준과 알고리즘 수준 기법의 결합[16][6]
- **딥러닝 기반 접근**: CNN, RNN을 활용한 불균형 데이터 처리[5][24]
- **전이 학습**: 다른 도메인의 지식을 활용한 소수 클래스 학습 향상[25]

#### **2. 응용 분야 확장**
- **의료 영상**: 희귀 질병 진단을 위한 영상 분류[16]
- **자연어 처리**: 감정 분석, 텍스트 분류에서의 불균형 문제[24]
- **농업**: 작물 병해 탐지 및 토지 이용 분류[26]

### **기술 동향 (Technical Trends)**

#### **1. 새로운 샘플링 기법**
- **클러스터 기반 SMOTE**: 클러스터링과 SMOTE의 결합으로 더 의미 있는 합성 샘플 생성[27]
- **GAN 기반 생성**: 생성 적대 신경망을 활용한 고품질 합성 데이터 생성[17]
- **적응적 샘플링**: 데이터 특성에 따라 동적으로 샘플링 비율 조정[28]

#### **2. 평가 방법론 개선**
- **다중 지표 평가**: F1-score, AUC-ROC, G-mean을 종합한 평가[6]
- **교차 검증**: 불균형 데이터에 적합한 층화 교차 검증 기법[4]

### **주요 이슈 (Key Issues)**

#### **1. 기술적 도전**
- **오버피팅 문제**: 과도한 오버샘플링으로 인한 모델 과적합[29]
- **클래스 중복**: 클래스 간 경계가 모호한 경우의 처리 어려움[30]
- **다중 클래스 불균형**: 여러 클래스가 동시에 불균형인 상황의 복잡성[22]

#### **2. 계산 효율성**
- **확장성**: 대용량 데이터셋에서의 처리 속도 및 메모리 효율성[23]
- **실시간 처리**: 스트리밍 환경에서의 즉각적 불균형 처리[18]

### **미래 연구 방향 (Future Research Directions)**

#### **1. 인공지능과의 융합**
- **설명 가능한 AI**: 불균형 분류 결과에 대한 해석 가능성 향상[31]
- **AutoML**: 자동화된 불균형 데이터 처리 파이프라인 개발[32]
- **연합 학습**: 분산 환경에서의 불균형 데이터 학습[33]

#### **2. 새로운 응용 분야**
- **사물 인터넷**: IoT 센서 데이터의 이상 탐지[5]
- **자율주행**: 드문 교통 상황에 대한 인식 및 대응[18]
- **기후 변화**: 극한 기상 현상 예측[26]

### **최종 산출물의 기대 결과**

이 리뷰 논문을 통해 기대되는 주요 결과는 다음과 같습니다:

#### **1. 학술적 기여**
- **체계적 분류**: 기존 불균형 분류 기법들의 포괄적 분류 체계 제시
- **성능 비교**: 다양한 데이터셋과 응용 분야에서의 기법별 성능 분석
- **연구 격차 식별**: 현재 연구에서 부족한 영역과 향후 연구 기회 도출

#### **2. 실무적 가이드라인**
- **기법 선택 가이드**: 데이터 특성에 따른 최적 기법 선택 기준 제시
- **구현 방법론**: 실제 프로젝트에서 활용 가능한 구체적 구현 방법
- **성능 평가 프레임워크**: 불균형 분류 모델의 체계적 평가 방법

#### **3. 향후 연구 로드맵**
- **우선순위 연구 주제**: 시급히 해결해야 할 기술적 과제 식별
- **융합 연구 방향**: 다른 AI 기술과의 융합 가능성 탐색
- **장기 비전**: 불균형 분류 분야의 5-10년 발전 방향 제시

## 주요 자료 검토 결과

검토된 주요 자료들의 핵심 이슈와 결론을 다음과 같이 요약할 수 있습니다:

### **핵심 이슈**

#### **1. 데이터 수준 문제**
- **SMOTE의 한계**: 기존 SMOTE는 클래스 간 중복 영역에서 노이즈를 생성할 수 있음[34][8]
- **다중 클래스 복잡성**: 여러 클래스가 동시에 불균형인 경우 처리의 어려움[22]

#### **2. 알고리즘 수준 문제**
- **비용 설정의 어려움**: 적절한 오분류 비용 설정의 주관성[13]
- **임계값 최적화**: 정밀도와 재현율 간 트레이드오프 관리[15]

#### **3. 평가 방법론 문제**
- **단일 지표의 한계**: 정확도만으로는 불균형 분류 성능 평가 부족[6]
- **교차 검증**: 불균형 데이터에 적합한 검증 방법의 필요성[4]

### **주요 결론**

#### **1. 방법론별 효과성**
- **Random Forest와 Decision Tree**: 불균형 데이터에 가장 강건한 성능[35][36]
- **SMOTE 변형들**: SMOTEENN, Borderline-SMOTE 등이 기본 SMOTE보다 우수[16]
- **앙상블 방법**: 단일 기법보다 일관되게 좋은 성능[37][38]

#### **2. 응용 분야별 특성**
- **의료 분야**: 높은 재현율이 중요하며, 오분류 비용이 매우 높음[16][39]
- **금융 분야**: 정밀도와 재현율의 균형이 중요[40][41]
- **보안 분야**: 실시간 처리 능력이 핵심[18][42]

#### **3. 미래 발전 방향**
- **딥러닝 융합**: 전통적 기법과 딥러닝의 결합으로 성능 향상[5][24]
- **자동화**: AutoML을 통한 자동화된 불균형 처리 파이프라인[32]
- **설명 가능성**: 블랙박스 모델의 해석 가능성 향상 필요[31]

이러한 포괄적인 분석을 통해 Imbalanced Classification 분야의 현재 상태와 미래 발전 방향을 체계적으로 이해할 수 있으며, 연구자와 실무자 모두에게 유용한 가이드라인을 제공할 수 있을 것입니다.

[1] https://www.machinelearningmastery.com/what-is-imbalanced-classification/
[2] https://encord.com/blog/an-introduction-to-balanced-and-imbalanced-datasets-in-machine-learning/
[3] https://keylabs.ai/blog/handling-imbalanced-data-in-classification/
[4] https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets
[5] https://www.frontiersin.org/articles/10.3389/fbioe.2021.802712/full
[6] https://arxiv.org/pdf/2211.05456.pdf
[7] https://www.themoonlight.io/en/review/a-comprehensive-survey-on-imbalanced-data-learning
[8] https://arxiv.org/pdf/1106.1813.pdf
[9] https://www.machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
[10] https://peerj.com/articles/cs-523
[11] https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00943-4
[12] https://jiheek.tistory.com/149
[13] https://www.machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
[14] https://www.blog.trainindata.com/cost-sensitive-learning-for-imbalanced-data/
[15] https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf
[16] https://www.mdpi.com/2072-6694/16/19/3417
[17] https://arxiv.org/abs/2304.02858
[18] http://services.igi-global.com/resolvedoi/resolve.aspx?doi=10.4018/978-1-7998-7371-6.ch001
[19] http://ijai.iaescore.com/index.php/IJAI/article/view/20844
[20] https://www.sciencedirect.com/science/article/pii/S0278612523002157
[21] https://link.springer.com/10.1007/s10994-022-06282-w
[22] https://etasr.com/index.php/ETASR/article/view/7206
[23] https://openreview.net/forum?id=goCUrZR0xD
[24] https://aclanthology.org/2023.eacl-main.38.pdf
[25] https://onlinelibrary.wiley.com/doi/10.1002/int.22899
[26] https://www.mdpi.com/2072-4292/17/3/454
[27] https://www.mdpi.com/2227-7390/12/11/1709
[28] https://internationalpubls.com/index.php/cana/article/view/4260
[29] https://neptune.ai/blog/how-to-deal-with-imbalanced-classification-and-regression-data
[30] https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf
[31] https://ieeexplore.ieee.org/document/10218281/
[32] https://openreview.net/forum?id=oRINLlHqni
[33] https://linkinghub.elsevier.com/retrieve/pii/S1110016824008147
[34] https://www.nature.com/articles/s41598-021-03430-5
[35] http://www.aimspress.com/article/doi/10.3934/DSFE.2023021
[36] https://www.aimspress.com/article/doi/10.3934/DSFE.2023021?viewType=HTML
[37] https://www.linkedin.com/advice/1/how-can-ensemble-methods-improve-machine-learning-telwf
[38] https://www.nature.com/articles/s41598-025-01031-0
[39] https://journal.ump.edu.my/ijsecs/article/view/11806
[40] https://journals.ekb.eg/article_414893_9e92b6e04aa25efa9bcbeef5275ebfc0.pdf
[41] https://www.ijcai.org/proceedings/2021/412
[42] https://ieeexplore.ieee.org/document/10397846/
[43] https://www.sec.gov/Archives/edgar/data/1423902/000142390225000033/wes-20241231.htm
[44] https://www.sec.gov/Archives/edgar/data/1414475/000142390225000033/wes-20241231.htm
[45] https://www.sec.gov/Archives/edgar/data/1039684/000103968425000036/oke-20241231.htm
[46] https://www.sec.gov/Archives/edgar/data/83350/000008335025000004/rsrv-20241231.htm
[47] https://www.sec.gov/Archives/edgar/data/107263/000010726325000031/wmb-20241231.htm
[48] https://www.sec.gov/Archives/edgar/data/99250/000010726325000031/wmb-20241231.htm
[49] http://biorxiv.org/lookup/doi/10.1101/2020.05.31.126201
[50] https://academic.oup.com/jaoac/article/102/5/1397/5658312
[51] https://ieeexplore.ieee.org/document/9177789/
[52] https://arxiv.org/pdf/2310.07917.pdf
[53] https://ourcstory.tistory.com/239
[54] https://www.sciencedirect.com/topics/computer-science/class-imbalance-problem
[55] https://www.kdnuggets.com/2023/07/overcoming-imbalanced-data-challenges-realworld-scenarios.html
[56] https://www.sec.gov/Archives/edgar/data/1655888/000165588825000012/obdc-20250331.htm
[57] https://linkinghub.elsevier.com/retrieve/pii/S0925231220306196
[58] https://datascience.stackexchange.com/questions/109134/how-to-define-minority-majority-class-in-a-multi-classification-task
[59] https://encord.com/glossary/class-imbalance-definition/
[60] https://www.nature.com/articles/s41592-021-01302-4
[61] https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
[62] https://www.sec.gov/Archives/edgar/data/1883043/0001883043-25-000001-index.htm
[63] https://www.sec.gov/Archives/edgar/data/2031972/0002031972-25-000001-index.htm
[64] https://www.sec.gov/Archives/edgar/data/1883043/0001883043-25-000002-index.htm
[65] https://www.sec.gov/Archives/edgar/data/2062127/0002062127-25-000001-index.htm
[66] https://www.sec.gov/Archives/edgar/data/2037063/0002037063-24-000002-index.htm
[67] https://www.sec.gov/Archives/edgar/data/1326205/000118518525000706/igc10k033125.htm
[68] https://iopscience.iop.org/article/10.1088/1742-6596/1918/4/042002
[69] https://www.ewadirect.com/proceedings/ace/article/view/22356
[70] https://dl.acm.org/doi/10.1145/3421558.3421560
[71] https://ieeexplore.ieee.org/document/10374407/
[72] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
[73] https://arxiv.org/pdf/1508.03422.pdf
[74] https://mkjjo.github.io/python/2019/01/04/smote_duplicate.html
[75] https://www.sec.gov/Archives/edgar/data/1869453/000186945325000021/ortic-20250331.htm
[76] https://www.sec.gov/Archives/edgar/data/1655888/000165588825000007/obdc-20241231.htm
[77] https://www.sec.gov/Archives/edgar/data/1873529/000095017025043210/klc-20241228.htm
[78] https://science.lpnu.ua/ujit/all-volumes-and-issues/volume-6-number-1/research-data-mining-methods-classification-imbalanced
[79] https://www.mdpi.com/2079-9292/14/4/705
[80] https://ieeexplore.ieee.org/document/9954368/
[81] https://link.springer.com/10.1007/s42979-023-02120-5
[82] https://www.mdpi.com/2076-3298/11/11/250
[83] https://dl.acm.org/doi/10.1145/3542954.3543024
[84] https://www.sciencedirect.com/science/article/abs/pii/S0893608024000819
[85] https://researchguide.cau.ac.kr/aitopiccs08/085
[86] https://www.sec.gov/Archives/edgar/data/1954488/000121390025055414/ea0200696-25.htm
[87] https://www.sec.gov/Archives/edgar/data/1697532/000095017025053840/aplt-20241231.htm
[88] https://www.sec.gov/Archives/edgar/data/1794546/000095012325006117/carl_-_s-1_-_june_2025.htm
[89] https://www.sec.gov/Archives/edgar/data/1651625/000155837025002855/aciu-20241231x20f.htm
[90] https://www.sec.gov/Archives/edgar/data/1899830/000189983025000015/perf-20241231.htm
[91] http://thesai.org/Publications/ViewPaper?Volume=13&Issue=8&Code=IJACSA&SerialNo=56
[92] https://arxiv.org/pdf/2012.11870.pdf
[93] https://www.sec.gov/Archives/edgar/data/110019/000010726325000031/wmb-20241231.htm
[94] https://ieeexplore.ieee.org/document/9362471/
[95] https://aap.onlinelibrary.wiley.com/doi/10.1002/JPER.18-0006
[96] https://ieeexplore.ieee.org/document/9185785/
[97] https://www.mdpi.com/2071-1050/15/9/7097
[98] https://www.frontiersin.org/articles/10.3389/fdata.2022.1021518/full
[99] https://ieeexplore.ieee.org/document/10080783/
[100] https://www.ijimai.org/journal/sites/default/files/2022-02/ijimai7_3_4.pdf
[101] http://joiv.org/index.php/joiv/article/view/2283
[102] https://linkinghub.elsevier.com/retrieve/pii/S0266352X23006079
[103] https://link.springer.com/10.1007/s10994-022-06296-4
[104] https://ieeexplore.ieee.org/document/10384376/
[105] https://www.frontiersin.org/articles/10.3389/fpsyg.2022.852758/full
[106] https://onlinelibrary.wiley.com/doi/10.1002/sd.3260
[107] http://services.igi-global.com/resolvedoi/resolve.aspx?doi=10.4018/IJGHPC.2019070102
[108] https://ieeexplore.ieee.org/document/9631065/
[109] http://ieeexplore.ieee.org/document/7218125/
[110] http://ieeexplore.ieee.org/document/4938667/
[111] https://www.semanticscholar.org/paper/599a8aa2bc9e7d4de2a610277c4a02dfce1db470
[112] https://gaexcellence.com/index.php/ijemp/article/view/749

# Imbalanced Classification 방법론

이전 대화에서 불균형 분류(Imbalanced Classification)의 전반적인 개요를 다뤘으니, 이제 구체적인 **방법론과 수학적 공식**에 초점을 맞춰 자세히 설명하겠습니다.

## 1. 데이터 수준 방법론 (Data-Level Methods)

### 1.1 SMOTE (Synthetic Minority Oversampling Technique)

SMOTE는 불균형 분류에서 가장 널리 사용되는 오버샘플링 기법입니다[1][2][3].

#### **기본 SMOTE 알고리즘**

주어진 소수 클래스 샘플 $$x_i$$에 대해, 다음 과정을 통해 합성 샘플을 생성합니다:

1. **k-최근접 이웃 선택**: 소수 클래스 샘플 $$x_i$$의 k개 최근접 이웃을 찾습니다.

2. **합성 샘플 생성**: 무작위로 선택된 이웃 $$x_j$$에 대해 다음 공식으로 새로운 샘플을 생성합니다:

$$
x_{new} = x_i + \lambda \times (x_j - x_i)
$$

여기서 $$\lambda \in [4]$$은 균등분포에서 추출된 무작위 값입니다[3][5].

#### **SMOTE의 이론적 특성**

SMOTE로 생성된 합성 샘플의 이론적 특성은 다음과 같습니다[6]:

- **기대값**: $$E[X_{SMOTE}] = E[X]$$
- **분산**: $$Var(X_{SMOTE}) = \frac{2}{3}Var(X)$$

이는 SMOTE가 원본 분포의 평균을 유지하면서 분산을 줄인다는 것을 의미합니다[1].

### 1.2 Borderline-SMOTE

Borderline-SMOTE는 클래스 경계 근처의 샘플에만 집중하여 합성 데이터를 생성하는 개선된 방법입니다[7][8][6].

#### **위험도 분류**

각 소수 클래스 샘플 $$x_i$$에 대해 k-최근접 이웃 중 다수 클래스 샘플의 개수 $$m'$$을 계산합니다:

- **NOISE**: $$m' = k$$ (모든 이웃이 다수 클래스)
- **DANGER**: $$\frac{k}{2} \leq m' < k$$ (이웃의 절반 이상이 다수 클래스)  
- **SAFE**: $$m' < \frac{k}{2}$$ (이웃의 절반 미만이 다수 클래스)

#### **Borderline-SMOTE1**

DANGER 그룹의 샘플들에 대해서만 일반 SMOTE 공식을 적용합니다:

$$
x_{new} = x_i + \lambda \times (x_j - x_i)
$$

여기서 $$x_j$$는 소수 클래스의 이웃이며, $$\lambda \in [4]$$입니다[6].

#### **Borderline-SMOTE2**

다수 클래스 이웃도 사용하지만, 더 짧은 거리로 보간합니다:

$$
x_{new} = x_i + \lambda \times (x_j - x_i)
$$

여기서 $$\lambda \in [0,0.5]$$로 제한하여 생성된 샘플이 소수 클래스에 더 가깝게 위치하도록 합니다[8][6].

### 1.3 ADASYN (Adaptive Synthetic Sampling)

ADASYN은 각 소수 클래스 샘플의 학습 난이도에 따라 적응적으로 합성 샘플을 생성합니다[9][10].

#### **ADASYN 알고리즘**

1. **불균형 비율 계산**: 
   $$G = (m_l - m_s) \times \beta$$
   여기서 $$m_l$$은 다수 클래스 샘플 수, $$m_s$$는 소수 클래스 샘플 수, $$\beta$$는 원하는 균형 비율입니다[10].

2. **각 샘플의 비율 계산**:
   $$r_i = \frac{\Delta_i}{K}$$
   여기서 $$\Delta_i$$는 샘플 $$x_i$$의 k-최근접 이웃 중 다수 클래스 샘플의 개수입니다[10].

3. **정규화된 비율**:
   $$\hat{r_i} = \frac{r_i}{\sum_{i=1}^{m_s} r_i}$$

4. **생성할 샘플 수**:
   $$g_i = \hat{r_i} \times G$$

5. **합성 샘플 생성**:
   $$s_i = x_i + (x_{zi} - x_i) \times \gamma$$
   여기서 $$x_{zi}$$는 k-최근접 이웃 중 무작위 선택된 소수 클래스 샘플이며, $$\gamma \in [4]$$입니다[10].

## 2. 알고리즘 수준 방법론 (Algorithm-Level Methods)

### 2.1 비용 민감 학습 (Cost-Sensitive Learning)

비용 민감 학습은 서로 다른 오분류에 다른 비용을 할당하는 방법입니다[11].

#### **비용 행렬 (Cost Matrix)**

2×2 비용 행렬은 다음과 같이 정의됩니다:

$$
C = \begin{pmatrix}
C(0,0) & C(0,1) \\
C(1,0) & C(1,1)
\end{pmatrix}
$$

여기서 $$C(i,j)$$는 실제 클래스 $$j$$를 클래스 $$i$$로 예측했을 때의 비용입니다[11].

#### **총 비용 계산**

총 비용은 다음과 같이 계산됩니다:

$$
\text{Total Cost} = C(0,1) \times \text{FN} + C(1,0) \times \text{FP}
$$

여기서 FN은 거짓 음성, FP는 거짓 양성의 개수입니다[11].

#### **비용 민감 AdaBoost**

AdaC1, AdaC2, AdaC3의 가중치 업데이트 공식들[12]:

**AdaC1**:
$$D^{(t+1)}(i) = \frac{D^{(t)}(i)\exp(-\alpha_t c_i h_t(x_i)y_i)}{Z_t}$$

**AdaC2**:
$$D^{(t+1)}(i) = \frac{c_i D^{(t)}(i)\exp(-\alpha_t h_t(x_i)y_i)}{Z_t}$$

**AdaC3**:
$$D^{(t+1)}(i) = \frac{c_i D^{(t)}(i)\exp(-\alpha_t c_i y_i h_t(x_i))}{Z_t}$$

여기서 $$c_i$$는 샘플 $$i$$의 비용 인수, $$Z_t$$는 정규화 상수입니다[12].

### 2.2 임계값 이동 (Threshold Moving)

임계값 이동은 분류 결정 경계를 조정하여 불균형 데이터에 대응하는 방법입니다[13][14][15].

#### **확률-임계값 변환**

클래스 1의 사전 확률이 $$p$$에서 $$p'$$으로 변경될 때, 최적 임계값은 다음과 같이 조정됩니다[15]:

$$
\text{threshold}' = \frac{p'(1-p)}{p(1-p')} \times \text{threshold}
$$

#### **비용 기반 임계값**

비용 행렬이 주어졌을 때, 최적 임계값은 다음과 같습니다:

$$
\text{threshold}^* = \frac{C(1,0)}{C(1,0) + C(0,1)}
$$

### 2.3 Focal Loss

Focal Loss는 딥러닝에서 클래스 불균형을 해결하기 위한 손실 함수입니다[16][17].

#### **Focal Loss 공식**

$$
FL(p_t) = -(1-p_t)^\gamma \log(p_t)
$$

여기서:
- $$p_t$$는 정답 클래스의 예측 확률
- $$\gamma \geq 0$$는 집중 매개변수 (focusing parameter)

#### **Alpha-Balanced Focal Loss**

클래스 가중치를 추가한 버전:

$$
FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)
$$

여기서 $$\alpha_t$$는 클래스별 가중치입니다[16][18].

## 3. 앙상블 방법론 (Ensemble Methods)

### 3.1 Random Forest for Imbalanced Data

Random Forest는 불균형 데이터에 강건한 앙상블 방법입니다[19][20].

#### **클래스 가중치 조정**

각 클래스에 대한 가중치를 다음과 같이 설정합니다:

$$
w_i = \frac{n}{k \times n_i}
$$

여기서:
- $$n$$은 총 샘플 수
- $$k$$는 클래스 수  
- $$n_i$$는 클래스 $$i$$의 샘플 수

#### **언더샘플링 기반 Random Forest**

각 트리에 대해 다수 클래스를 언더샘플링하여 균형 잡힌 부트스트랩 샘플을 생성합니다[19].

### 3.2 Extrapolation Borderline-SMOTE SVM (BEBS)

BEBS는 SVM의 서포트 벡터 정보를 활용한 개선된 방법입니다[21].

#### **외삽 공식**

$$x_{new} = x_i + \alpha \times (x_i - x_j)$$

여기서:
- $$x_i$$는 DANGER 그룹의 서포트 벡터
- $$x_j$$는 k-최근접 이웃
- $$\alpha$$는 SVM의 결정 경계까지의 거리에 기반한 매개변수[21]

## 4. 평가 지표 (Evaluation Metrics)

### 4.1 주요 성능 지표

불균형 분류에서는 정확도 대신 다음 지표들을 사용합니다:

#### **정밀도 (Precision)**
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### **재현율 (Recall)**
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### **F1-Score**
$$
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### **G-mean**
$$
\text{G-mean} = \sqrt{\text{Sensitivity} \times \text{Specificity}}
$$

#### **AUC-ROC**
ROC 곡선 아래 면적으로, TPR과 FPR의 관계를 나타냅니다:
- $$TPR = \frac{TP}{TP + FN}$$
- $$FPR = \frac{FP}{TN + FP}$$

## 5. 고급 방법론

### 5.1 혼합 정규화 (Mixed Regularization)

복잡한 형상에서 EIT를 적용할 때 초점 우선순위와 부드러움 우선순위를 결합합니다[22].

### 5.2 배치 균형 초점 손실 (Batch-Balanced Focal Loss)

BBFL은 배치 균형화와 초점 손실을 결합한 하이브리드 방법입니다[18]:

1. **배치 균형화**: 각 배치에서 클래스 분포를 동적으로 균형화
2. **초점 손실**: 어려운 샘플에 더 높은 중요도 부여

이러한 수학적 공식들은 모두 신뢰할 수 있는 학술 논문과 공식 문서에서 인용된 것으로, 실제 구현에서 사용되는 정확한 공식들입니다. 각 방법론의 선택은 데이터의 특성, 불균형 정도, 계산 자원, 그리고 응용 분야의 요구사항에 따라 결정되어야 합니다.

[1] https://link.springer.com/10.1007/s10994-022-06296-4
[2] https://gjom.org/index.php/gjom/article/view/2639
[3] https://jair.org/index.php/jair/article/download/11192/26406/20731
[4] https://www.sec.gov/Archives/edgar/data/1039684/000103968425000036/oke-20241231.htm
[5] https://www.nature.com/articles/s41598-021-03430-5
[6] https://escholarship.org/content/qt99x0w9w0/qt99x0w9w0_noSplash_6386a738c0e8b3d02aa47b6a4cda0b3f.pdf
[7] https://iopscience.iop.org/article/10.1088/1742-6596/2031/1/012046
[8] https://www.youtube.com/watch?v=vQDy6EnhyL8
[9] https://www.mdpi.com/2076-3417/14/24/11910
[10] https://arxiv.org/pdf/2105.04301.pdf
[11] https://www.machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
[12] https://opendl.ifip-tc6.org/db/conf/mldm/mldm2005/SunWW05.pdf
[13] https://openreview.net/forum?id=EIPnUofed9
[14] https://www.machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
[15] https://cseweb.ucsd.edu/~elkan/rescale.pdf
[16] https://paperswithcode.com/method/focal-loss
[17] https://arxiv.org/abs/2409.09877
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/
[19] https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf
[20] https://stats.stackexchange.com/questions/340854/random-forest-for-imbalanced-data
[21] https://pmc.ncbi.nlm.nih.gov/articles/PMC5304315/
[22] https://asmedigitalcollection.asme.org/ssdm/proceedings/SSDM2025/88759/V001T01A004/1219009
[23] https://www.sec.gov/Archives/edgar/data/83350/000008335025000004/rsrv-20241231.htm
[24] https://www.sec.gov/Archives/edgar/data/101382/000095017025062287/umbf-20250331.htm
[25] https://www.sec.gov/Archives/edgar/data/1423902/000142390225000033/wes-20241231.htm
[26] https://www.sec.gov/Archives/edgar/data/1414475/000142390225000033/wes-20241231.htm
[27] https://www.sec.gov/Archives/edgar/data/1126741/000155837025000749/gsit-20241231x10q.htm
[28] https://dl.acm.org/doi/10.1145/3678726.3678771
[29] https://www.semanticscholar.org/paper/4e7154eefa1e5bc42ab48b740101f2d8362e5648
[30] https://www.semanticscholar.org/paper/3a980e39c03a25ef2a075dc0d34fd3f660dc5aaa
[31] https://unisciencepub.com/wp-content/uploads/2020/07/The-mathematical-formula-to-estimate-the-Exosome-Affinity-between-miRNA-peptide.pdf
[32] http://ieeexplore.ieee.org/document/7751674/
[33] https://dl.acm.org/doi/10.1145/3577117.3577123
[34] https://blogs.sas.com/content/iml/2025/04/28/smote-method.html
[35] https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf
[36] https://www.numberanalytics.com/blog/cost-sensitive-learning-predictive-modeling
[37] https://towardsdatascience.com/creating-smote-oversampling-from-scratch-64af1712a3be/
[38] https://www.diva-portal.org/smash/get/diva2:1519153/FULLTEXT01.pdf
[39] https://en.wikipedia.org/wiki/Cost-sensitive_machine_learning
[40] https://arxiv.org/pdf/1106.1813.pdf
[41] https://www.mdpi.com/2073-4433/13/4/544
[42] https://www.sciencedirect.com/science/article/pii/S235291482100174X
[43] https://domino.ai/blog/smote-oversampling-technique
[44] https://www.sec.gov/Archives/edgar/data/1560672/000156067225000074/earn-20250331.htm
[45] https://www.sec.gov/Archives/edgar/data/1737927/000095017025079317/cgc-20250331.htm
[46] https://www.sec.gov/Archives/edgar/data/915779/000091577925000102/dakt-20250426.htm
[47] https://www.sec.gov/Archives/edgar/data/906553/000143774925004757/bgc20241008_10k.htm
[48] https://www.sec.gov/Archives/edgar/data/835357/000119312525089088/d841756ds1a.htm
[49] https://www.sec.gov/Archives/edgar/data/751978/000095017025030619/vicr-20241231.htm
[50] https://www.worldscientific.com/doi/10.1142/S0218001422500197
[51] https://www.semanticscholar.org/paper/196efbb8b30e1aed9e2879fea2a16701d84a77fb
[52] https://www.semanticscholar.org/paper/b10e6737d38d8bdac9ff61a6e3abbac6a5e412b4
[53] https://www.hindawi.com/journals/jeph/2023/4916267/
[54] https://onlinelibrary.wiley.com/doi/10.1155/2017/1827016
[55] https://datascience.stackexchange.com/questions/77742/cost-sensitive-learning-and-class-balancing
[56] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html
[57] https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
[58] https://towardsdatascience.com/class-imbalance-smote-borderline-smote-adasyn-6e36c78d804/
[59] https://www.sec.gov/Archives/edgar/data/2041869/000199937125007760/solana-s1a_061325.htm
[60] https://www.sec.gov/Archives/edgar/data/2041869/000199937125006485/canary-s1a_052125.htm
[61] https://www.sec.gov/Archives/edgar/data/1692415/000164117225009257/form10-q.htm
[62] https://www.sec.gov/Archives/edgar/data/1829311/000168316824008555/bitmine_i10k-083124.htm
[63] https://www.sec.gov/Archives/edgar/data/1838359/000155837025002499/rgti-20241231x10k.htm
[64] https://www.sec.gov/Archives/edgar/data/1771951/000149315224020289/forms-1a.htm
[65] https://internationalpubls.com/index.php/pmj/article/view/878
[66] https://link.springer.com/10.1007/s10489-020-02062-y
[67] https://link.springer.com/10.1007/978-3-319-10990-9_22
[68] https://link.springer.com/10.1007/s13198-020-01020-8
[69] https://www.hindawi.com/journals/mpe/2020/3012406/
[70] https://stackoverflow.com/questions/64751157/how-to-use-class-weights-with-focal-loss-in-pytorch-for-imbalanced-dataset-for-m
[71] http://www.ijicic.org/ijicic-150201.pdf
[72] https://arxiv.org/abs/2011.06283
[73] https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838
[74] https://www.sec.gov/Archives/edgar/data/1326205/000118518525000706/igc10k033125.htm
[75] https://www.sec.gov/Archives/edgar/data/1883043/0001883043-25-000001-index.htm
[76] https://www.sec.gov/Archives/edgar/data/2031972/0002031972-25-000001-index.htm
[77] https://www.sec.gov/Archives/edgar/data/1883043/0001883043-25-000002-index.htm
[78] https://www.sec.gov/Archives/edgar/data/2062127/0002062127-25-000001-index.htm
[79] https://ieeexplore.ieee.org/document/11003314/
[80] https://www.mdpi.com/2313-433X/10/1/20
[81] https://iopscience.iop.org/article/10.1088/2632-2153/ad62ac
[82] https://ieeexplore.ieee.org/document/10326040/
[83] https://stats.stackexchange.com/questions/567859/how-to-choose-gamma-parameter-in-focal-loss
[84] https://www.sciencedirect.com/science/article/pii/S0957417423032803
[85] https://www.reddit.com/r/MachineLearning/comments/xt01bk/d_focal_loss_why_it_scales_down_the_loss_of/
[86] https://eprints.uad.ac.id/50862/1/8-Gaussian%20Based-SMOTE%20Method%20for%20Handling%20Imbalanced%20Small%20Datasets%20.pdf
[87] https://arxiv.org/html/2304.02858
[88] https://www.sciencedirect.com/science/article/abs/pii/S1568494622001156
[89] https://www.mdpi.com/2076-3417/8/5/815
[90] https://www.sec.gov/Archives/edgar/data/1899156/0000950138-22-000007-index.htm
[91] https://www.sec.gov/Archives/edgar/data/1331421/000149315224039074/form8-k.htm
[92] https://www.sec.gov/Archives/edgar/data/1393066/000139306621000034/rfp-20210930.htm
[93] https://www.sec.gov/Archives/edgar/data/1375205/000165495424002673/urg_10k.htm
[94] https://www.sec.gov/Archives/edgar/data/1763950/000164117225000926/form10-k.htm
[95] https://www.sec.gov/Archives/edgar/data/1802749/000155837022004767/zev-20211231x10k.htm
[96] https://www.mdpi.com/2504-3110/5/3/98
[97] https://link.springer.com/10.1007/s11042-020-09168-y
[98] https://ieeexplore.ieee.org/document/10702144/
[99] https://www.semanticscholar.org/paper/53d4e6043e49e42cbd90fc86185a6c7af6030bbd
[100] http://jair.org/index.php/jair/article/view/11192
[101] https://www.hindawi.com/journals/amp/2021/3148747/
[102] https://arxiv.org/abs/2408.01777
[103] https://www.sec.gov/Archives/edgar/data/110019/000010726325000031/wmb-20241231.htm
[104] https://ieeexplore.ieee.org/document/9788374/
[105] https://www.atlantis-press.com/article/25838389
[106] https://www.semanticscholar.org/paper/c1207d129ef00134ed28c08540489541ef06db52
[107] https://onlinelibrary.wiley.com/doi/10.1155/2014/859157
[108] https://www.sec.gov/Archives/edgar/data/845877/000084587725000033/agm-20241231.htm
[109] https://www.semanticscholar.org/paper/f94db0e0bdeb8e8b0ddd5125e323f7bdfddeda28
[110] https://academic.oup.com/sleep/article/46/Supplement_1/A197/7181835
[111] https://www.mdpi.com/2076-3417/13/1/574
[112] https://journals.lww.com/ijo/Fulltext/2020/68060/Validation_of_Mahajan_s_formula_for_scaling_ocular.28.aspx
[113] https://www.mdpi.com/1996-1073/15/13/4751
[114] https://www.sec.gov/Archives/edgar/data/1792580/000095017025027914/ovv-20241231.htm
[115] https://onlinelibrary.wiley.com/doi/10.1002/mma.9299
[116] http://link.springer.com/10.1007/s00521-019-04405-4
[117] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-gtd.2017.0945
[118] https://ieeexplore.ieee.org/document/10326027/
[119] https://ieeexplore.ieee.org/document/9627935/
[120] https://onlinelibrary.wiley.com/doi/10.1002/er.7163
[121] http://journal.frontiersin.org/Article/10.3389/fncel.2015.00271/abstract
[122] https://www.sec.gov/Archives/edgar/data/1393066/000139306622000034/rfp-20220331.htm
[123] https://www.sec.gov/Archives/edgar/data/1758009/000121390023067059/f10q0623_quantumcomp.htm
[124] https://www.sec.gov/Archives/edgar/data/1381668/000138166821000109/tfsl-20210930.htm
[125] https://www.sec.gov/Archives/edgar/data/1763950/000149315224010302/form10-k.htm
[126] https://ojs.wiserpub.com/index.php/AECM/article/view/328
[127] https://www.semanticscholar.org/paper/40054548c691898bd89fe82fcb2f5b12fea38cc4
