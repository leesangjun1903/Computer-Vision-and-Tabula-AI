# Random Forest

## Random Forest의 핵심 개념과 주요 기여

Random Forest는 Leo Breiman이 2001년 발표한 앙상블 학습 알고리즘으로, **배깅(Bootstrap Aggregating)과 무작위 특성 선택을 결합**하여 개별 의사결정나무의 한계를 극복한 혁신적인 기법입니다[1][2][3].

### 핵심 주장
Random Forest의 가장 중요한 이론적 기여는 **강한 수의 법칙(Strong Law of Large Numbers)에 기반한 수렴성 보장**입니다. Breiman은 트리의 개수가 증가할수록 일반화 오류가 수렴하며, 과적합이 발생하지 않음을 수학적으로 증명했습니다[3][4].

## 해결하고자 하는 문제와 제안 방법

### 해결 대상 문제
1. **개별 의사결정나무의 높은 분산**: 단일 트리는 훈련 데이터의 작은 변화에도 크게 다른 결과를 생성[5][6]
2. **과적합 문제**: 깊은 트리는 훈련 데이터에 과도하게 특화되어 일반화 성능이 저하[7][8]
3. **불안정한 예측**: 개별 트리의 예측 결과가 일관성 없음[9][10]

### 제안 방법

Random Forest는 두 가지 핵심 무작위화 기법을 결합합니다:

#### 1. 배깅(Bootstrap Aggregating)
원본 데이터에서 복원추출로 여러 부트스트랩 샘플을 생성하여 각각 다른 트리를 학습시킵니다[5][11]:

$$ \text{부트스트랩 샘플} = \{(x_i, y_i)\}_{i=1}^n \text{ (복원추출)} $$

#### 2. 무작위 특성 선택
각 노드 분할 시 전체 특성 중 일부만 무작위로 선택합니다[11][12]:

분류: $$m = \sqrt{p} $$ 개 특성 선택  
회귀: $$m = p/3 $$ 개 특성 선택 (여기서 p는 전체 특성 수)

### 수학적 정의

Random Forest는 다음과 같이 정의됩니다[3]:

$$ h(x, \Theta_k) = k\text{번째 트리의 예측} $$

여기서 $$\{\Theta_k\}$$는 독립동일분포를 따르는 무작위 벡터입니다.

**마진 함수(Margin Function)**:

$$ mg(X,Y) = P_\Theta(h(X,\Theta) = Y) - \max_{j \neq Y} P_\Theta(h(X,\Theta) = j) $$

## 모델 구조

### 앙상블 구조
Random Forest는 **다수의 독립적인 의사결정나무**로 구성됩니다[5][13]:

1. **트리 생성**: 각 트리는 부트스트랩 샘플과 무작위 특성 선택으로 학습
2. **예측 집계**: 
   - 분류: 다수결 투표
   - 회귀: 평균값 계산

### Out-of-Bag (OOB) 추정
각 부트스트랩 샘플에서 제외된 약 36.8%의 데이터를 이용하여 **별도의 검증 세트 없이도 모델 성능을 추정**할 수 있습니다[14][15][16].

## 성능 향상 메커니즘

### 수학적 근거: Strength와 Correlation

Breiman은 Random Forest의 일반화 오류에 대한 상한을 다음과 같이 제시했습니다[3][17]:

$$ PE^* \leq \frac{\bar{\rho}(1-s^2)}{s^2} $$

여기서:
- $$s$$: 개별 분류기의 강도(strength)
- $$\bar{\rho}$$: 개별 트리 간 상관계수
- $$PE^*$$: 일반화 오류

### 분산 감소 효과
배깅을 통해 **편향은 유지하면서 분산을 크게 감소**시킵니다[18][19]:

$$ \text{Var}[\text{평균}] = \frac{1}{B}\text{Var}[\text{개별 예측}] $$

여기서 B는 트리의 개수입니다.

## 일반화 성능 향상 가능성

### 수렴성 보장
**강한 수의 법칙**에 의해 트리 개수가 증가할수록 일반화 오류가 수렴하며, 과적합이 발생하지 않습니다[3][4]:

$$ \lim_{B \to \infty} PE^* = P_{X,Y}(P_\Theta(h(X,\Theta) = Y) - \max_{j \neq Y} P_\Theta(h(X,\Theta) = j) < 0) $$

### 편향-분산 트레이드오프 개선
Random Forest는 **편향 증가 없이 분산을 감소**시켜 전체적인 예측 오류를 줄입니다[20][21]:

- **분산 감소**: 여러 트리의 평균화 효과
- **편향 유지**: 개별 트리의 예측 능력 보존
- **노이즈 강건성**: 무작위화를 통한 노이즈 저항성 증대

### 변수 중요도 측정
Random Forest는 **순열 중요도(Permutation Importance)**를 통해 각 특성의 기여도를 측정할 수 있습니다[22][23][24]:

$$ \text{중요도}_j = E[\text{error}(\text{permuted}_j)] - E[\text{error}(\text{original})] $$

## 모델의 한계

1. **해석성 저하**: 개별 의사결정나무 대비 모델 해석이 어려움[6][8]
2. **메모리 사용량**: 다수의 트리 저장으로 인한 높은 메모리 요구량[25][8]
3. **연관된 특성에 대한 편향**: 상관관계가 높은 특성들 간 중요도 분산[26][27]
4. **노이즈가 많은 데이터**: 매우 노이즈가 많은 환경에서는 성능 저하 가능[25][7]

## 향후 연구에 미치는 영향

### 이론적 기여
Random Forest는 **앙상블 학습의 이론적 기반**을 마련했으며, 이후 Gradient Boosting, XGBoost 등 다양한 앙상블 기법 발전의 토대가 되었습니다[28][29].

### 실용적 응용
다양한 분야에서 강력한 성능을 보여주며 **머신러닝의 범용 도구**로 자리잡았습니다:
- 의료 진단[30][7]
- 환경 예측[31][28]
- 자연어 처리[32][33]
- 재료 공학[34][35]

### 방법론적 발전
Random Forest의 핵심 아이디어는 다음과 같은 발전을 이끌었습니다:
- **극한 랜덤 트리(Extremely Randomized Trees)**
- **Isolation Forest** (이상치 탐지)[36]
- **혼합효과 랜덤 포레스트** (다층 데이터 구조 처리)[37][38]

## 앞으로 연구 시 고려사항

1. **하이퍼파라미터 최적화**: 트리 개수, 최대 깊이, 특성 선택 수 등의 체계적 튜닝[35][29]

2. **특성 엔지니어링**: 도메인 지식을 활용한 효과적인 특성 설계[13][39]

3. **불균형 데이터 처리**: SMOTE 등 샘플링 기법과의 결합[30][40]

4. **해석가능성 향상**: SHAP, LIME 등 설명 가능한 AI 기법과의 통합[23][26]

5. **대용량 데이터 처리**: 분산 컴퓨팅 환경에서의 효율적 구현[41][42]

Random Forest는 **이론적 엄밀성과 실용적 효용성을 모두 갖춘** 알고리즘으로, 현대 머신러닝에서 여전히 중요한 위치를 차지하고 있으며, 향후 연구에서도 기준점 역할을 지속할 것으로 예상됩니다.

[1] https://meetingorganizer.copernicus.org/EGU21/EGU21-2105.html
[2] https://www.scribd.com/document/844748011/Breiman-Random-Forests-MachineLearning-bibtex
[3] https://mfm.uchicago.edu/wp-content/uploads/2020/06/Breiman-Random-Forests.pdf
[4] https://www.cmi.ac.in/~madhavan/courses/dmml2025/literature/Breiman_RandomForests_ML_2001.pdf
[5] https://www.mdpi.com/1996-1073/17/8/1926
[6] https://winterflake.tistory.com/17
[7] https://diagnprognres.biomedcentral.com/articles/10.1186/s41512-024-00177-1
[8] https://zephyrus1111.tistory.com/249
[9] https://blog.eunsukim.me/posts/understanding-random-forest
[10] https://velog.io/@imfromk/ML-RandomForest
[11] https://heave.tistory.com/7
[12] https://westlife0615.tistory.com/1112
[13] https://www.nature.com/articles/s41598-024-60066-x
[14] https://en.wikipedia.org/wiki/Out-of-bag_error
[15] https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
[16] https://www.baeldung.com/cs/random-forests-out-of-bag-error
[17] https://kaw-db-saving.tistory.com/6
[18] https://courses.grainger.illinois.edu/ece543/sp2017/projects/Vaishnavi%20Subramanian.pdf
[19] https://www.dailydoseofds.com/why-bagging-is-so-ridiculously-effective-at-variance-reduction/
[20] https://arxiv.org/html/2402.12668v1
[21] https://www.ibm.com/think/topics/bias-variance-tradeoff
[22] https://www.randomforestsrc.org/articles/vimp.html
[23] https://christophm.github.io/interpretable-ml-book/feature-importance.html
[24] https://scikit-learn.org/stable/modules/permutation_importance.html
[25] https://intothedata.com/docs/02.scholars/data_mining/classification/random_forest/
[26] https://hongl.tistory.com/114
[27] https://songseungwon.tistory.com/98
[28] https://link.springer.com/10.1007/s11269-025-04093-x
[29] https://www.mdpi.com/1996-1944/15/12/4193
[30] https://www.scimedjournal.org/index.php/SMJ/article/view/489
[31] https://icce-ojs-tamu.tdl.org/icce/article/view/14746
[32] http://www.journal-labphon.org/articles/10.5334/labphon.216/
[33] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11968991
[34] https://iopscience.iop.org/article/10.1088/1755-1315/1110/1/012072
[35] https://www.mdpi.com/2071-1050/15/12/9170
[36] https://onlinelibrary.wiley.com/doi/10.1002/mma.8570
[37] https://s-space.snu.ac.kr/handle/10371/178434?mode=full
[38] https://www.ejce.org/archive/view_article?doi=10.29221%2Fjce.2022.25.1.223
[39] https://hongl.tistory.com/129
[40] https://ieeexplore.ieee.org/document/9734140/
[41] https://www.mdpi.com/1996-1073/16/2/867
[42] http://ijain.org/index.php/IJAIN/article/view/471
[43] https://www.sec.gov/Archives/edgar/data/1331421/000149315224039074/form8-k.htm
[44] https://www.sec.gov/Archives/edgar/data/1331421/000164117225000860/form10-k.htm
[45] https://www.sec.gov/Archives/edgar/data/1763950/000164117225000926/form10-k.htm
[46] https://www.sec.gov/Archives/edgar/data/1717115/000095017025025603/tem-20241231.htm
[47] https://www.sec.gov/Archives/edgar/data/1717115/000119312524142956/d221145ds1.htm
[48] https://www.sec.gov/Archives/edgar/data/1763950/000149315224010302/form10-k.htm
[49] https://www.sec.gov/Archives/edgar/data/1331421/000149315224046186/form10-q.htm
[50] https://link.springer.com/10.1007/s11801-022-1115-9
[51] https://linkinghub.elsevier.com/retrieve/pii/S0920410518310635
[52] http://dergipark.org.tr/en/doi/10.31127/tuje.669566
[53] https://ieeexplore.ieee.org/document/10544632/
[54] https://mesopotamian.press/journals/index.php/BJML/article/view/417
[55] https://ieeexplore.ieee.org/document/10522226/
[56] https://onlinelibrary.wiley.com/doi/10.1002/spe.2921
[57] https://www.ijitee.org/portfolio-item/L36091081219/
[58] https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/
[59] https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
[60] https://quantdev.ssri.psu.edu/sites/qdev/files/09_EnsembleMethods_2017_1127.html
[61] https://builtin.com/data-science/random-forest-algorithm
[62] https://dl.acm.org/doi/10.1023/A:1010933404324
[63] https://tyami.github.io/machine%20learning/ensemble-2-bagging-random-forest/
[64] https://www.simplilearn.com/tutorials/machine-learning-tutorial/random-forest-algorithm
[65] https://www.scirp.org/reference/referencespapers
[66] https://www.nature.com/articles/nmeth.4438
[67] https://www.ibm.com/think/topics/random-forest
[68] https://arxiv.org/abs/2001.04295
[69] https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/
[70] https://towardsdatascience.com/random-forest-explained-a-visual-guide-with-code-examples-9f736a6e1b3c/
[71] https://www.bibsonomy.org/bibtex/2b8187107bf870043f2f93669958858f1/kdepublication
[72] https://scikit-learn.org/stable/modules/ensemble.html
[73] https://en.wikipedia.org/wiki/Random_forest
[74] https://www.sec.gov/Archives/edgar/data/1840161/000121390023090628/ea189149-8k_forest2.htm
[75] https://www.sec.gov/Archives/edgar/data/1840161/000121390023086933/f10q0923_forestroadacq2.htm
[76] https://www.sec.gov/Archives/edgar/data/1840161/000121390023066242/f10q0623_forestroadacq2.htm
[77] https://www.sec.gov/Archives/edgar/data/1840161/000121390023024098/f10k2022_forestroad2.htm
[78] https://www.sec.gov/Archives/edgar/data/1899156/0000950138-22-000007-index.htm
[79] https://www.sec.gov/Archives/edgar/data/1840161/000121390023039995/f10q0323_forestroadacq2.htm
[80] http://link.springer.com/10.1007/3-540-48219-9_18
[81] https://onlinelibrary.wiley.com/doi/10.1002/1096-9837(200103)26:33.0.CO;2-1
[82] https://referenceworks.brill.com/doi/10.1163/2468-1733_shafr_SIM100120022
[83] https://onlinelibrary.wiley.com/doi/10.1002/mcda.284
[84] https://onlinelibrary.wiley.com/doi/10.1002/sim.852
[85] https://onlinelibrary.wiley.com/doi/10.1002/hyp.171
[86] http://www.cifor.org/library/1058/the-effects-of-indonesias-decentralisation-on-forests-and-estate-crops-case-study-of-riau-province-the-original-districts-of-kampar-and-indragiri-hulu/
[87] https://www.semanticscholar.org/paper/fad4d6c2785baaf11539d1a2b0df8f9d3713f781
[88] https://process-mining.tistory.com/102
[89] https://arxiv.org/pdf/1511.05741.pdf
[90] https://velog.io/@ddangchani/Random-Forest
[91] https://www.stat.berkeley.edu/~breiman/random-forests.pdf
[92] https://mozenworld.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8-%EC%86%8C%EA%B0%9C-4-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-Random-Forest
[93] https://kaggler-tv.github.io/dku-kaggle-class/lectures/07-rf-lgb.html
[94] https://wikidocs.net/236296
[95] https://keylabs.ai/blog/random-forest-ensemble-learning-technique/
[96] https://oceanonx.tistory.com/30
[97] https://www.sciencedirect.com/topics/engineering/random-forest
[98] https://blog-ko.superb-ai.com/3-minute-algorithm-random-forest/
[99] https://www.sec.gov/Archives/edgar/data/1838359/000155837025002499/rgti-20241231x10k.htm
[100] https://www.sec.gov/Archives/edgar/data/1636639/000163663925000022/fihl-20241231.htm
[101] https://www.sec.gov/Archives/edgar/data/1675149/000095017025024242/aa-20241231.htm
[102] http://www.liverpooluniversitypress.co.uk/doi/10.3167/082279401782310808
[103] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13223/3035330/Random-forest-based-vacancy-filling-of-MCD43A3V061-Antarctic-albedo-product/10.1117/12.3035330.full
[104] http://aircconline.com/csit/papers/vol9/csit91808.pdf
[105] https://www.semanticscholar.org/paper/d0d13a5dc54f253b2ee7c2da6e5e62798d1cd876
[106] https://www.naun.org/main/NAUN/ijmmas/2020/a302001-afs.pdf
[107] https://www.semanticscholar.org/paper/82ac827885f0941723878aff5df27a3207748983
[108] https://www.semanticscholar.org/paper/74dd9c269bafe8e930674ca7f65114f29880c9e3
[109] https://dl.acm.org/doi/10.1145/3409334.3452073
[110] https://www.cohorte.co/blog/how-can-ensemble-methods-prevent-model-overfitting
[111] https://arxiv.org/pdf/1610.01271.pdf
[112] https://www.geeksforgeeks.org/machine-learning/how-ensemble-modeling-helps-to-avoid-overfitting/
[113] https://www.geeksforgeeks.org/random-forest-regression-in-python/
[114] https://arxiv.org/abs/0811.3619
[115] https://stats.stackexchange.com/questions/111968/random-forest-how-to-handle-overfitting
[116] https://journal.kci.go.kr/jksci/archive/articleView?artiId=ART002109885
[117] https://www.baeldung.com/cs/random-forest-overfitting-fix
[118] https://www.sciencedirect.com/science/article/pii/S0167865522001416
[119] https://arxiv.org/abs/2401.04425
[120] https://www.lyzr.ai/glossaries/random-forest/
[121] https://www.e-sciencecentral.org/articles/metaView/SCe0000482424/en
[122] https://www.scribd.com/document/399572885/Leo-Breiman-2001-Random-Forest-Algorithm-Weka-Google-Scholar
[123] https://woolulu.tistory.com/28
[124] https://www.sec.gov/Archives/edgar/data/1566657/0001743115-25-000004-index.htm
[125] https://www.sec.gov/Archives/edgar/data/1171890/0001315863-25-000297-index.htm
[126] https://www.sec.gov/Archives/edgar/data/2048862/0002048862-24-000001-index.htm
[127] https://www.sec.gov/Archives/edgar/data/2021111/0002021111-24-000001-index.htm
[128] https://www.sec.gov/Archives/edgar/data/2022916/0002022916-24-000006-index.htm
[129] https://www.sec.gov/Archives/edgar/data/1940456/0000902664-24-005366-index.htm
[130] https://arxiv.org/pdf/1310.1415.pdf
[131] https://arxiv.org/pdf/2412.13020.pdf
[132] https://downloads.hindawi.com/journals/cin/2021/5572781.pdf
[133] https://peerj.com/articles/cs-1775
[134] http://arxiv.org/pdf/2407.04042.pdf
[135] http://arxiv.org/pdf/0811.3619.pdf
[136] https://peerj.com/articles/cs-1041
[137] https://pmc.ncbi.nlm.nih.gov/articles/PMC4710485/
[138] https://minye-lee19.gitbook.io/sw-engineer/business-analytics/class/4-2-bagging-random-forest
[139] https://daehyun-bigbread.tistory.com/173
[140] https://push-and-sleep.tistory.com/80
[141] https://wikidocs.net/266275
[142] https://velog.io/@73syjs/Ensemble-model-Random-Forests
[143] https://sonstory.tistory.com/79
[144] https://gsbang.tistory.com/entry/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5-%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8Random-Forest
[145] https://fish-tank.tistory.com/79
[146] https://comemann.tistory.com/66
[147] https://bluenoa.tistory.com/54
[148] https://topo314.tistory.com/81
[149] https://swalloow.github.io/decision-randomforest/
[150] https://mac-user-guide.tistory.com/241
[151] https://velog.io/@gangjoo/ML-%EB%B6%84%EB%A5%98-%EB%B0%B0%EA%B9%85-Bagging%EA%B3%BC-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-Random-Forest
[152] https://swalloow.tistory.com/92
[153] https://www.minitab.com/ko-kr/solutions/analytics/statistical-analysis-predictive-analytics/random-forests/
[154] https://www.sec.gov/Archives/edgar/data/1440972/000095017025046966/lar-20241231.htm
[155] https://www.sec.gov/Archives/edgar/data/845698/000155837025004125/tmb-20241231x10k.htm
[156] https://www.sec.gov/Archives/edgar/data/1479247/000141057825001151/usci-20250331x10q.htm
[157] https://www.sec.gov/Archives/edgar/data/1001290/000114036125015768/ef20038949_20f.htm
[158] https://www.sec.gov/Archives/edgar/data/1826889/000095017025072601/body-20250331.htm
[159] https://www.sec.gov/Archives/edgar/data/1347426/000119312525087225/d865064d20f.htm
[160] https://arxiv.org/html/2310.12428v3
[161] http://arxiv.org/pdf/2310.09702.pdf
[162] https://www.esaim-ps.org/articles/ps/pdf/2020/01/ps180111.pdf
[163] https://projecteuclid.org/journals/statistics-surveys/volume-3/issue-none/Navigating-Random-Forests-and-related-advances-in-algorithmic-modeling/10.1214/07-SS033.pdf
[164] https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-14/issue-2/Modeling-of-time-series-using-random-forests-Theoretical-developments/10.1214/20-EJS1758.pdf
[165] https://arxiv.org/abs/1407.3939
[166] https://arxiv.org/pdf/1904.07830.pdf
[167] https://arxiv.org/html/2408.05537v1
[168] https://untitledtblog.tistory.com/143
[169] https://bommbom.tistory.com/entry/%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8Random-Forest-%EB%8F%99%EC%9E%91-%EC%9B%90%EB%A6%AC-%EB%B0%8F-OOB
[170] https://happysemicon.tistory.com/20
[171] http://blog.heartcount.io/random-forest-ver-10
[172] https://datasciencebeehive.tistory.com/90
[173] https://whitesoil.tistory.com/96
[174] https://ebbnflow.tistory.com/134
[175] https://self-objectification.tistory.com/28
[176] https://velog.io/@jochedda/%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-Random-Forest
[177] https://hayden-archive.tistory.com/302
[178] https://labex.io/ko/tutorials/ml-random-forest-oob-error-estimation-49119
[179] https://velog.io/@dlskawns/Machine-Learning-Random-Forest-%EC%A0%95%EB%A6%AC-%EA%B5%AC%EC%84%B1%EC%9B%90%EB%A6%AC-%ED%8C%8C%EC%95%85-%EB%B0%8F-%EB%AA%A8%EB%8D%B8-%EC%9E%91%EC%84%B1
[180] https://soobarkbar.tistory.com/19
[181] https://www.blog.data101.io/149
[182] https://school.cbe.go.kr/_cmm/fileDownload/os-h/M010603/ed4f7415ec93ca66bd3c6b2d0ca03bb9
[183] https://m.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=d143e9002176e65ec85d2949c297615a
[184] https://www.sec.gov/Archives/edgar/data/1850767/000164117225002594/form10-k.htm
[185] https://www.sec.gov/Archives/edgar/data/1822791/000143774925008956/clnn20241231_10k.htm
[186] https://www.sec.gov/Archives/edgar/data/1850767/000164117225011721/form10-q.htm
[187] http://www.aimspress.com/article/doi/10.3934/mine.2024013
[188] https://journals.sagepub.com/doi/10.1177/1536867X20909688
[189] http://link.springer.com/10.1007/s11749-016-0481-7
[190] https://www.semanticscholar.org/paper/06ea09512d547e8fbfe3a39c4e72dce506d54fb6
[191] http://www.airccse.org/journal/ijsc/papers/6115ijsc01.pdf
[192] https://www.semanticscholar.org/paper/3ee4a95dca76c1b0854a3ad29a2c71a46f3af779
[193] http://link.springer.com/10.1007/s11749-016-0482-6
[194] https://www.semanticscholar.org/paper/1bc5ba9301d2b23743b09e2210797bc49e5eacea
[195] https://m.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=aa270885b33164b1e9810257f7042666
[196] https://enjoy0life.tistory.com/23
[197] https://dl.acm.org/doi/10.1023/a:1010933404324
[198] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201919163609605
[199] https://bioinformaticsandme.tistory.com/167
[200] https://ai-inform.tistory.com/entry/%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%EB%9E%80-Random-Forest
[201] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002998632
[202] https://ko.wikipedia.org/wiki/%EB%9E%9C%EB%8D%A4_%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8
[203] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=DIKO0015385139
[204] http://ui.adsabs.harvard.edu/abs/2001MachL..45....5B/abstract
[205] https://m.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=2a29f4733797ed48b7998d826d417196
[206] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11580203
[207] https://velog.io/@euisuk-chung/%ED%8A%B8%EB%A6%AC-%ED%8A%B8%EB%A6%AC-%EA%B8%B0%EB%B0%98-ML-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
[208] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10445294
[209] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002990870
[210] https://www.dbpia.co.kr/Journal/ArticleDetail/NODE10672199
[211] https://www.sec.gov/Archives/edgar/data/1120914/000155837025005827/pdfs-20250617xdef14a.htm
[212] https://www.sec.gov/Archives/edgar/data/1844505/000184450525000038/qti-20241231.htm
[213] https://www.sec.gov/Archives/edgar/data/1326205/000118518525000706/igc10k033125.htm
[214] https://www.sec.gov/Archives/edgar/data/1611115/000119312525129904/d785770ds1a.htm
[215] https://www.sec.gov/Archives/edgar/data/2019410/000110465925059635/tm2415719-23_s1a.htm
[216] https://www.sec.gov/Archives/edgar/data/1120914/000155837025001813/pdfs-20241231x10k.htm
[217] https://academic.oup.com/biometrics/article/77/1/23-27/7445044
[218] https://photonics.pl/PLP/index.php/letters/article/view/17-7
[219] https://photonics.pl/PLP/index.php/letters/article/view/17-6
[220] https://zephyrus1111.tistory.com/222
[221] https://stackoverflow.com/questions/64771024/how-to-prevent-overfitting-in-random-forest
[222] https://cran.r-project.org/web/packages/randomForest/randomForest.pdf
[223] https://m.riss.kr/search/detail/DetailView.do?p_mat_type=1a0202e37d52c72d&control_no=3e1fbde8eded3286d18150b21a227875
[224] https://datascience.stackexchange.com/questions/1028/do-random-forest-overfit
[225] http://www.kibme.org/resources/journal/20200414103930886.pdf
[226] https://datajobstest.com/data-science-repo/Random-Forest-%5BFrederick-Livingston%5D.pdf
[227] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART96080212&dbt=NART
[228] https://www.scribd.com/document/852153132/Random-Forests-Machine-Learning
[229] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10820371
[230] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART128781237
[231] https://www.sciencedirect.com/science/article/abs/pii/S0957417423011545
[232] https://iieta.org/journals/ijsse/paper/10.18280/ijsse.140613
[233] https://ieeexplore.ieee.org/document/10141319/
[234] https://jurnal.iaii.or.id/index.php/RESTI/article/view/5795
[235] https://www.linkedin.com/pulse/mastering-model-accuracy-navigating-bias-variance-dilek-celik-phd-689ce
[236] https://www.kaggle.com/code/swahajraza/bias-variance-trade-off-random-forest
[237] https://www.sciencedirect.com/science/article/am/pii/S2352012422009018
[238] https://blackas119.tistory.com/74
[239] https://arxiv.org/html/2402.01502v1
[240] https://towardsdatascience.com/the-bias-variance-tradeoff-for-modeling-5988db08ef91/
[241] https://heung-bae-lee.github.io/2020/05/02/machine_learning_15/
[242] https://www.sec.gov/Archives/edgar/data/1978313/0001012975-25-000140-index.htm
[243] https://www.sec.gov/Archives/edgar/data/2050808/0002050808-25-000001-index.htm
[244] https://www.sec.gov/Archives/edgar/data/1843714/000095017024102903/ck0001843714-20240903.htm
[245] https://www.sec.gov/Archives/edgar/data/1843714/000119312524094887/d795295ds1.htm
[246] https://www.semanticscholar.org/paper/d464e8bfee01386cd22c7a5278bc5379a2f2d782
[247] https://www.semanticscholar.org/paper/43121692ba481ed63a5dcc14b5e7e4fc0cf738fd
[248] https://www.tandfonline.com/doi/full/10.2989/16073606.2024.2403745
[249] https://linkinghub.elsevier.com/retrieve/pii/S0022247X25006328
[250] https://www.semanticscholar.org/paper/5b60187d90a487c28aec26cf602041ddf16731a8
[251] https://www.semanticscholar.org/paper/0b7454c376f15067e039afe3755d488833ba581c
[252] https://arxiv.org/abs/2212.13562
[253] https://proceedings.neurips.cc/paper/2020/file/6925f2a16026e36e4fc112f82dd79406-Paper.pdf
[254] https://www.skillcamper.com/blog/random-forest-why-ensemble-learning-outperforms-individual-models
[255] https://stackoverflow.com/questions/18541923/what-is-out-of-bag-error-in-random-forests
[256] https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-16/issue-1/Rates-of-convergence-for-random-forests-via-generalized-U-statistics/10.1214/21-EJS1958.pdf
[257] https://dea.lib.unideb.hu/bitstreams/b5dcb2d0-e97f-4c8a-a72c-7f07aedb0380/download
[258] https://www.sciencedirect.com/science/article/pii/S2666827021000475
[259] https://stats.stackexchange.com/questions/207815/out-of-bag-error-makes-cv-unnecessary-in-random-forests
[260] https://www.tse-fr.eu/sites/default/files/medias/stories/SEMIN_10_11/STATISTIQUE/biau.pdf
[261] https://mlu-explain.github.io/random-forest/
[262] https://arxiv.org/abs/2505.07958
[263] https://aggregata.de/random-forests/
[264] https://www.sec.gov/Archives/edgar/data/1588671/0001588671-25-000002-index.htm
[265] https://www.sec.gov/Archives/edgar/data/1962461/0001962461-25-000003-index.htm
[266] https://www.sec.gov/Archives/edgar/data/1972756/0001972756-25-000003-index.htm
[267] https://www.sec.gov/Archives/edgar/data/2020913/0002020913-25-000003-index.htm
[268] https://www.sec.gov/Archives/edgar/data/1703766/0001703766-25-000002-index.htm
[269] https://www.sec.gov/Archives/edgar/data/1709467/0001709467-25-000002-index.htm
[270] https://www.sec.gov/Archives/edgar/data/1688311/0001688311-25-000002-index.htm
[271] http://jmes.humg.edu.vn/en/archives?article=1577
[272] https://link.springer.com/10.1007/s41939-023-00314-1
[273] https://iopscience.iop.org/article/10.1088/1755-1315/1110/1/012085
[274] https://link.springer.com/10.1007/s41939-023-00248-8
[275] https://arxiv.org/abs/1810.09746
[276] https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/permutation-variable-importance.html
[277] https://arxiv.org/pdf/1405.2881.pdf
[278] https://yngie-c.github.io/machine%20learning/2021/03/19/random_forest/
[279] https://rpubs.com/stoney/579763
[280] https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
[281] https://sjpyo.tistory.com/41
[282] https://www.sec.gov/Archives/edgar/data/1375205/000165495424002673/urg_10k.htm
[283] https://www.mdpi.com/2072-4292/13/20/4033/pdf?version=1633774904
[284] https://www.mdpi.com/2076-3417/10/22/8237/pdf
[285] https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO202417776451101
[286] https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11741108
[287] https://blog.hyeongeun.com/21
[288] https://codingalzi.github.io/handson-ml3/ensemble_learning_random_forests.html
[289] https://www.koreascience.kr/article/JAKO202417776451101.page
[290] https://bigdaheta.tistory.com/32
[291] https://horizon.kias.re.kr/15780/
[292] https://siroro.tistory.com/101
[293] https://www.nature.com/articles/s41598-024-72051-5
[294] https://linkinghub.elsevier.com/retrieve/pii/S0924271618302090
[295] https://pmc.ncbi.nlm.nih.gov/articles/PMC4957112/
[296] https://www.shs-conferences.org/articles/shsconf/pdf/2021/27/shsconf_icsr2021_00078.pdf
[297] http://thesai.org/Downloads/Volume14No10/Paper_54-Research_on_the_Application_of_Random.pdf
[298] https://pmc.ncbi.nlm.nih.gov/articles/PMC8019375/
[299] https://hul980.tistory.com/15
[300] https://inspirehep.net/literature/1484913
[301] https://www.semanticscholar.org/paper/e54152403da98a0403afef8477d42383d606e1f9
[302] https://arxiv.org/abs/1405.2881
[303] https://arxiv.org/pdf/1805.02587.pdf
[304] https://arxiv.org/pdf/2402.12668.pdf
[305] https://pmc.ncbi.nlm.nih.gov/articles/PMC9605715/
[306] https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab074/38694194/btab074.pdf
[307] https://www.semanticscholar.org/paper/68a4cbe5f45b1bc5f84d897f57d7596c90d593b2
[308] https://www.semanticscholar.org/paper/f031df138fdea7a8fcb5d6d599fa8294269a953c
[309] http://arxiv.org/pdf/2001.04295.pdf
[310] https://www.mdpi.com/1424-8220/24/16/5223
[311] https://www.mdpi.com/1996-1073/15/20/7547/pdf?version=1666670971
[312] http://arxiv.org/pdf/2306.11908.pdf
[313] https://arxiv.org/pdf/2201.06821.pdf
[314] https://arxiv.org/html/2402.14131v1
[315] https://arxiv.org/pdf/2502.10185.pdf
[316] https://arxiv.org/pdf/2405.09832.pdf
[317] https://pubs.rsc.org/en/content/articlepdf/2021/an/d0an02155e
[318] https://arxiv.org/html/2402.19232v1
[319] https://www.mdpi.com/1660-4601/19/10/6111/pdf?version=1652793051
[320] https://itcount.tistory.com/2
[321] https://analytics-yulvely.tistory.com/entry/R-%EC%95%99%EC%83%81%EB%B8%94-1-%EB%B0%B0%EA%B9%85-%EB%B6%80%EC%8A%A4%ED%8C%85-%EB%9E%9C%EB%8D%A4%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8
[322] https://velog.io/@zlddp723/Ensemble%EC%95%99%EC%83%81%EB%B8%94-Random-Forest%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8
[323] https://think-something.tistory.com/32
[324] https://arxiv.org/pdf/2008.07063.pdf
[325] https://arxiv.org/pdf/2310.18814.pdf
[326] https://arxiv.org/pdf/1106.5112.pdf
[327] https://velog.io/@soohee2001/%ED%8C%A8%ED%84%B4%EC%9D%B8%EC%8B%9D-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5Ensemble-Method
[328] https://rpago.tistory.com/56
[329] https://arxiv.org/html/2408.01777v1
[330] https://arxiv.org/pdf/1501.07196.pdf
[331] https://www.jksee.or.kr/m/journal/view.php?doi=10.4491%2FKSEE.2021.43.3.206
[332] https://www.semanticscholar.org/paper/1662455ea7cd144db1ddb3f04359830d8349a7d0
[333] https://arxiv.org/pdf/1604.07143.pdf
[334] https://link.springer.com/10.1007/s42835-023-01680-z
[335] https://dx.plos.org/10.1371/journal.pone.0321263
[336] https://onlinelibrary.wiley.com/doi/10.1155/2015/471371
[337] http://arxiv.org/pdf/1907.08742.pdf
[338] https://pmc.ncbi.nlm.nih.gov/articles/PMC8975250/
[339] http://arxiv.org/pdf/2410.00942.pdf
[340] https://pmc.ncbi.nlm.nih.gov/articles/PMC4387916/
[341] http://arxiv.org/pdf/2410.04297.pdf
[342] https://www.mdpi.com/1099-4300/19/10/520/pdf?version=1506602662
[343] https://hrcak.srce.hr/file/426085
[344] https://datasciencediary.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-Ensemble-method-Bagging-%EB%B0%B0%EA%B9%85-%EC%84%B1%EB%8A%A5-%ED%96%A5%EC%83%81-%EB%B0%A9%EB%B2%95
[345] https://www.semanticscholar.org/paper/fc186fa3abe09508967f692cc0c3f241ba7c410b
[346] https://link.springer.com/10.1007/s10474-023-01350-6
[347] https://arxiv.org/html/2202.04912v5
[348] https://arxiv.org/pdf/1903.05806.pdf
[349] https://arxiv.org/pdf/2212.13562.pdf
[350] https://arxiv.org/html/2501.16589v1
[351] https://arxiv.org/pdf/1503.06388.pdf
[352] http://rspa.royalsocietypublishing.org/content/464/2100/3175.full.pdf
[353] https://arxiv.org/abs/2310.06760
[354] http://www.scirp.org/journal/PaperDownload.aspx?paperID=25544
[355] https://www.mdpi.com/2073-4360/14/11/2270
[356] https://xlink.rsc.org/?DOI=D4NA00405A
[357] http://arxiv.org/pdf/2402.04550.pdf
[358] https://pmc.ncbi.nlm.nih.gov/articles/PMC2889677/
[359] https://arxiv.org/pdf/2006.04709.pdf
[360] https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-16/issue-2/Random-forest-estimation-of-conditional-distribution-functions-and-conditional-quantiles/10.1214/22-EJS2094.pdf
[361] https://www.matec-conferences.org/articles/matecconf/pdf/2018/87/matecconf_cas2018_01020.pdf

# Car Prediction with Randomforest

# Reference
https://www.kaggle.com/code/kagglecombasmasalem/95-accuracy-using-randomforest
