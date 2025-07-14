# WDSR : Wide Activation for Efficient and Accurate Image Super-Resolution | Super resolution

## 1. 핵심 주장과 주요 기여

### 핵심 주장
이 논문의 가장 중요한 발견은 **동일한 파라미터와 계산 복잡도에서 ReLU 활성화 함수 이전의 더 넓은 특성(wide activation)을 가진 모델이 단일 이미지 초해상도(SISR)에서 현저히 더 나은 성능을 보인다**는 것입니다[1]. 이는 기존의 복잡한 skip connection이나 concatenation 대신, 단순히 활성화 이전의 채널 수를 늘리는 것만으로도 상당한 성능 향상을 얻을 수 있음을 보여줍니다.

### 주요 기여
1. **Wide Activation의 효과 입증**: 2× ~ 4× 확장에서 최적 성능을 보이며, linear low-rank convolution을 사용하여 6×~9× 확장까지 가능함을 보였습니다[1].

2. **Weight Normalization 도입**: Batch normalization 대신 weight normalization을 사용할 때 더 나은 성능을 보임을 증명했습니다[1].

3. **경쟁 우승**: DIV2K 벤치마크에서 우수한 성능을 보였으며, NTIRE 2018 Challenge의 세 가지 realistic track에서 모두 1등을 차지했습니다[1].

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의
기존 깊은 SR 네트워크의 핵심 문제는 **얕은 층의 특성 정보를 충분히 활용하지 못한다**는 것입니다. 저자들은 이 원인을 **비선형 ReLU 활성화 함수가 얕은 층에서 깊은 층으로의 정보 흐름을 방해**한다고 분석했습니다[1].

### 제안 방법

#### WDSR-A (Wide Deep Super-Resolution A)
- **슬림한 identity mapping pathway**와 **2×~4× 넓은 활성화 채널**을 특징으로 합니다
- 수학적 관계: $$w_2 = r × w_1$$, 여기서 $$r$$은 확장 비율입니다
- 파라미터 보존을 위해: $$w_1^2 = r × \hat{w}_1^2$$

#### WDSR-B (Wide Deep Super-Resolution B)
- **Linear low-rank convolution**을 도입하여 6×~9× 확장을 계산 오버헤드 없이 달성합니다
- 1×1 convolution으로 채널 수를 줄인 후 3×3 convolution으로 공간적 특성을 추출합니다[1]

#### 수학적 모델링
확장 비율 $$r$$에 대해:
- Identity pathway 너비: $$\hat{w}_1 = w_1/\sqrt{r}$$
- 활성화 확장: $$\hat{w}_2 = \sqrt{r} × w_1$$

이는 동일한 파라미터 복잡도를 유지하면서 활성화 이전의 특성을 넓힐 수 있게 합니다.

### Weight Normalization vs Batch Normalization
논문에서는 SR 작업에서 batch normalization이 부적절한 이유를 세 가지로 제시합니다[1]:
1. **Mini-batch 의존성**: 작은 이미지 패치와 작은 mini-batch 크기로 인한 통계 불안정성
2. **훈련/추론 차이**: 훈련과 추론 시 다른 공식 사용
3. **강한 정규화 효과**: SR 네트워크는 일반적으로 overfitting이 문제가 되지 않음

## 3. 모델 구조 및 성능 향상

### 네트워크 구조 개선
- **전역 residual pathway 단순화**: 여러 convolution 층을 하나의 5×5 convolution으로 교체
- **업샘플링 층 최적화**: 모든 특성 추출을 저해상도 단계에서 수행하여 속도 향상[1]

### 성능 결과
| 모델 | 파라미터 | DIV2K PSNR (×2) |
|------|----------|-----------------|
| EDSR | 0.78M | 34.457 dB |
| WDSR-A | 0.60M | 34.541 dB |
| WDSR-B | 0.60M | 34.536 dB |

논문의 실험 결과에 따르면, WDSR은 더 적은 파라미터로도 기존 EDSR보다 우수한 성능을 보였습니다[1].

## 4. 일반화 성능 향상 가능성

### 이론적 근거
Wide activation의 일반화 성능은 **Neural Tangent Kernel (NTK) 이론**과 연관됩니다[2][3]. 무한폭 네트워크에서 NTK는 다음과 같이 정의됩니다:

$$\Theta^{(L)}\_\infty(x, x') = \Theta^{(L-1)}_\infty(x, x') \dot{\Sigma}^{(L)}(x, x') + \Sigma^{(L)}(x, x')$$

여기서 $$\dot{\Sigma}^{(L)}$$는 활성화 함수의 도함수와 관련된 항입니다.

### 일반화 능력 향상 메커니즘
1. **정보 보존**: 넓은 활성화는 더 많은 정보가 ReLU를 통과하도록 하여 **feature propagation**을 개선합니다[1]
2. **표현력 증가**: 넓은 특성 공간에서 더 풍부한 표현 학습이 가능합니다
3. **안정적 훈련**: Weight normalization을 통해 10배 높은 학습률로 안정적 훈련이 가능합니다[1]

### 실험적 증거
다양한 파라미터 예산에서 일관된 성능 향상을 보였으며, 이는 **일반화 능력의 본질적 향상**을 시사합니다[1].

## 5. 한계점

### 기술적 한계
1. **확장 비율 제한**: $$r > 4$$일 때 성능이 급격히 저하됩니다 (identity pathway가 너무 슬림해짐)[1]
2. **메모리 사용량**: 넓은 활성화로 인한 메모리 사용량 증가
3. **특정 작업 최적화**: 초해상도 외 다른 작업에 대한 일반화 검증 필요

### 이론적 한계
현재의 이론적 분석은 주로 경험적 관찰에 기반하며, wide activation의 수학적 근거에 대한 더 깊은 이해가 필요합니다.

## 6. 향후 연구에 미치는 영향과 고려사항

### 연구 영향
1. **패러다임 전환**: 복잡한 구조 대신 단순한 width expansion의 중요성 부각
2. **효율성 연구**: 계산 효율성과 성능 간의 새로운 균형점 제시
3. **이론적 발전**: Wide networks의 특성 학습 능력 이해에 기여[4][5]

### 향후 연구 방향
1. **다른 도메인 적용**: Denoising, dehazing 등 다른 저수준 이미지 복원 작업 적용[1]
2. **이론적 분석 심화**: Wide activation의 이론적 근거 더 깊이 탐구
3. **효율성 개선**: 계산 효율성과 성능 간의 최적 균형점 찾기
4. **일반화 연구**: 다양한 도메인에서의 wide activation 효과 검증

### 실용적 고려사항
1. **하드웨어 요구사항**: 넓은 활성화로 인한 메모리 요구사항 증가 고려
2. **실시간 응용**: 실시간 초해상도 시스템에서의 적용 가능성 평가
3. **다양한 데이터셋**: 다양한 이미지 도메인에서의 robustness 검증

이 논문은 딥러닝 기반 초해상도 연구에서 **단순함의 힘**을 보여주는 중요한 연구로, 향후 효율적인 네트워크 설계에 대한 새로운 관점을 제시했습니다. 특히 wide activation과 weight normalization의 조합은 다른 컴퓨터 비전 작업에도 광범위하게 적용될 수 있는 잠재력을 가지고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/18a53986-9351-4012-9dd8-bb35129d58c5/1808.08718v2.pdf
[2] https://www.semanticscholar.org/paper/7a84a692327534fd227fa1e07fcb3816b633c591
[3] https://arxiv.org/abs/2302.05933
[4] https://escholarship.org/uc/item/0fp2p8tx
[5] https://arxiv.org/abs/2305.18506
[6] https://www.sec.gov/Archives/edgar/data/1621672/000143774925010233/slgg20241231_10k.htm
[7] https://www.sec.gov/Archives/edgar/data/1878057/000162828025016393/sghc-20241231.htm
[8] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000006/smci-20241231.htm
[9] https://www.sec.gov/Archives/edgar/data/1621672/000143774925017257/slgg20250331_10q.htm
[10] https://www.sec.gov/Archives/edgar/data/1621672/000143774925019131/slgg20250601_8k.htm
[11] https://www.sec.gov/Archives/edgar/data/766792/000149315225010917/form10-k.htm
[12] https://www.sec.gov/Archives/edgar/data/1375365/000137536525000004/smci-20240630.htm
[13] https://www.semanticscholar.org/paper/fb915031b76086cec98fef42a9a1e95226bd05ff
[14] https://ieeexplore.ieee.org/document/10368051/
[15] https://ieeexplore.ieee.org/document/10634806/
[16] https://ieeexplore.ieee.org/document/10473143/
[17] https://dl.acm.org/doi/10.1145/3636534.3690698
[18] https://link.springer.com/10.1007/s10462-023-10648-4
[19] https://arxiv.org/abs/2407.02670
[20] https://www.ewadirect.com/proceedings/ace/article/view/10845
[21] https://www.theobjects.com/dragonfly/dfhelp/2022-1/Content/Resources/PDFs/wdsr.pdf
[22] http://krasserm.github.io/2019/09/04/super-resolution/
[23] https://arxiv.org/abs/1808.08718
[24] https://arxiv.org/abs/2111.13905
[25] https://arxiv.org/pdf/1808.08718.pdf
[26] https://cdn.aaai.org/ojs/16263/16263-13-19757-1-2-20210518.pdf
[27] https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Towards_High-Quality_and_Efficient_Video_Super-Resolution_via_Spatial-Temporal_Data_Overfitting_CVPR_2023_paper.pdf
[28] https://openreview.net/forum?id=G6-oxjbc_mK
[29] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.12734
[30] https://github.com/JiahuiYu/wdsr_ntire2018
[31] https://par.nsf.gov/servlets/purl/10179392
[32] https://psh7286.tistory.com/entry/Batch-Normalization
[33] https://paperswithcode.com/paper/wide-activation-for-efficient-and-accurate
[34] https://www.sec.gov/Archives/edgar/data/1832950/000149315224029170/form8-k.htm
[35] https://www.sec.gov/Archives/edgar/data/1832950/000149315224029922/form8-k.htm
[36] https://www.sec.gov/Archives/edgar/data/1832950/000149315224009610/form10-k.htm
[37] https://www.sec.gov/Archives/edgar/data/803578/000143774925002056/wavd20250122_s1a.htm
[38] https://www.sec.gov/Archives/edgar/data/1832950/000149315224019246/form10-q.htm
[39] https://www.sec.gov/Archives/edgar/data/803578/000143774925000977/wavd20241216_s1a.htm
[40] https://www.sec.gov/Archives/edgar/data/803578/000143774924036621/wavd20241123c_s1a.htm
[41] https://www.semanticscholar.org/paper/dc418c0b5ac24a67fef336323ef0417600ba3718
[42] https://arxiv.org/abs/2407.17120
[43] https://ieeexplore.ieee.org/document/10591437/
[44] https://arxiv.org/abs/2410.05626
[45] https://arxiv.org/abs/2405.15539
[46] https://arxiv.org/abs/2412.18756
[47] https://arxiv.org/abs/2409.05349
[48] http://proceedings.mlr.press/v119/adlam20a/adlam20a.pdf
[49] https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2025.1572428/full
[50] https://openreview.net/forum?id=lycl1GD7fVP
[51] https://www.pnas.org/doi/10.1073/pnas.2208779120
[52] https://koreascience.kr/article/CFKO201920461757445.pdf
[53] https://arxiv.org/abs/1806.07572
[54] https://arxiv.org/abs/2204.14126
[55] https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf
[56] https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf
[57] https://proceedings.neurips.cc/paper/8076-neural-tangen-kernel-convergence-and-generalization-in-neural-networks.pdf
[58] https://www.jmlr.org/papers/volume25/23-0831/23-0831.pdf
[59] https://arxiv.org/abs/1707.02921
[60] https://glanceyes.com/entry/Neural-Tangent-Kernel%EA%B3%BC-Fourier-Features%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-Positional-Encoding-2-Neural-Tangent-Kernel
[61] http://proceedings.mlr.press/v70/nguyen17a/nguyen17a.pdf
[62] https://www.sciencedirect.com/science/article/pii/S2666827021000815
[63] https://arxiv.org/abs/2206.10012
[64] https://www.ewadirect.com/proceedings/ace/article/view/16827
[65] https://iopscience.iop.org/article/10.1088/1742-6596/2891/14/142003
[66] http://arxiv.org/pdf/2208.12052.pdf
[67] http://arxiv.org/pdf/1706.09077.pdf
[68] http://arxiv.org/pdf/2404.03209.pdf
[69] https://arxiv.org/pdf/1203.5871.pdf
[70] https://www.mdpi.com/2072-4292/13/12/2382/pdf
[71] http://arxiv.org/pdf/2101.08987v3.pdf
[72] https://www.epj-conferences.org/articles/epjconf/pdf/2020/14/epjconf_eosam2020_06014.pdf
[73] http://arxiv.org/pdf/2103.14373.pdf
[74] https://www.sciencedirect.com/science/article/pii/S2352938524001976
[75] https://proceedings.neurips.cc/paper_files/paper/2023/file/ac24656b0b5f543b202f748d62041637-Paper-Conference.pdf
[76] https://www.sciencedirect.com/science/article/abs/pii/S0893608020302033
[77] https://arxiv.org/abs/2411.18806
[78] https://arxiv.org/abs/2412.05545
[79] http://arxiv.org/pdf/1901.01608.pdf
[80] http://arxiv.org/pdf/2009.09829.pdf
[81] http://arxiv.org/pdf/1810.05369.pdf
[82] https://arxiv.org/pdf/2103.11558.pdf
[83] https://arxiv.org/pdf/2002.04026.pdf
[84] http://arxiv.org/pdf/2012.04859.pdf
[85] https://arxiv.org/pdf/2412.17518.pdf
[86] https://arxiv.org/pdf/2206.08720.pdf
[87] https://arxiv.org/pdf/2411.02904.pdf
[88] https://arxiv.org/pdf/1902.04760.pdf
[89] https://arxiv.org/abs/2506.22429
[90] https://www.youtube.com/watch?v=vDQWwOqQ7mo
[91] https://www.v7labs.com/blog/neural-networks-activation-functions
[92] https://www.lunit.io/company/blog/review-enhanced-deep-residual-networks-for-single-image-super-resolution-winner-of-ntire-2017-sisr-challenge


# Reference
- https://arxiv.org/abs/1808.08718
