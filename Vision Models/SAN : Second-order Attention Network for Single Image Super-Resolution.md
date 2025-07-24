# SAN : Second-order Attention Network for Single Image Super-Resolution | Super resolution

## 1. 간결 요약  
Second-Order Attention Network (SAN)은 단일 영상 초해상화(SISR) 모델이 **중간 계층(feature) 간 상관 관계**를 충분히 활용하지 못한다는 한계를 지적하고,  
1) **Second-Order Channel Attention (SOCA)** 모듈로 채널별 상관성을 2차 통계량(공분산) 기반으로 학습하며,  
2) **Non-Locally Enhanced Residual Group (NLRG)** 구조로 장거리 공간 정보를 통합한다.  
이를 통해 여러 공개 데이터셋에서 EDSR·RCAN·RDN 등 당시 SOTA 모델보다 더 높은 PSNR/SSIM을 달성하였다[1][2].

## 2. 문제 정의와 수식으로 본 제안 방법  

### 2.1 해결하고자 한 문제  
1. **정보 손실**: 깊은 네트워크가 LR 영상의 저주파 정보를 충분히 전달하지 못함.  
2. **채널 간 상관 무시**: 기존 SE/SENet 기반 채널 어텐션은 평균값(1차 통계)에만 의존.  
3. **한정된 수용 영역**: 공간적 비국소(non-local) 문맥을 충분히 포착하지 못함[1].

### 2.2 SOCA 모듈  
1. 입력 특성 맵 $$F \in \mathbb{R}^{H \times W \times C}$$ → $$X \in \mathbb{R}^{C \times s}$$ ($$s=HW$$) 로 재구성.  
2. **표본 공분산**  

$$
\Sigma = X\left(I - \tfrac{1}{s}\mathbf{11}^{\!\top}\right)X^{\top}
$$

3. **행렬 제곱근 정규화**  

$$
\hat{Y}= \Sigma^{\frac12}=U\Lambda^{\frac12}U^{\top}
$$

실제 학습에서는 SVD 대신 **Newton-Schulz iteration**으로 근사해 GPU 효율을 확보[1].  
4. **Global Covariance Pooling(GCP)**로 채널 서술자 $$z$$ 계산  

$$
z_c=\frac{1}{C}\sum_{i=1}^{C}\hat{Y}_{c,i}
$$

5. 두 개의 $$1\times1$$ Conv로 게이트 $$w= \sigma\!\big(W_U\mathrm{ReLU}(W_D z)\big)$$ 생성 후 채널 재조정:  

$$\hat{f}_c = w_c \cdot f_c$$ [1].

### 2.3 NLRG 구조  
- **Region-Level Non-Local (RL-NL)** 모듈을 그룹 앞-뒤에 배치해 전역 컨텍스트 확보.  
- **Local-Source Residual Attention Group (LSRAG)** × $$G=20$$: 내부에 residual blocks 10개와 SOCA 1개.  
- **Share-Source Skip Connection**으로 입력 저주파를 직접 전달해 기울기 소실·과적합 완화[1].

## 3. 전체 네트워크 파이프라인  

| 단계 | 구성 | 목적 |
|------|------|------|
| Shallow Feature | 1×Conv | 초기 저주파 추출 |
| **NLRG** (20 LSRAG + RL-NL 앞·뒤) | 깊은 표현 + 전역 문맥 |
| Upscale | ESPCNN(Pixel-Shuffle) | 저해상 → 고해상 |
| Reconstruction | 1×Conv | HR RGB 복원 |

해당 설계로 **400 Conv 층 이상**의 깊이를 가지면서도 학습이 안정적으로 수렴한다[1].

## 4. 성능 향상·한계 분석  

### 4.1 정량 성능  
4× 확대, BI degradation 기준 (crop Y채널):

| 모델 | Set5 | Urban100 |
|------|------|-----------|
| EDSR | 32.46 / 0.8968 | 26.64 / 0.8033 |
| RCAN | 32.62 / 0.9001 | 26.82 / 0.8087 |
| **SAN** | **32.64 / 0.9003** | **26.79 / 0.8068** |
| **SAN+** (self-ensemble) | 32.70 / 0.9013 | 27.23 / 0.8169 |  
PSNR / SSIM, 단위 dB[1].

- **SOCA 기여**: FOCA 대비 +0.04 dB 상승[1].  
- **NLRG 기여**: RL-NL + SSC 조합으로 Base 대비 +0.08 dB[1].

### 4.2 파라미터 효율  
RDN 22.3 M vs. SAN 15.7 M으로 PSNR 우세 혹은 동급[1].

### 4.3 한계  
1. **연산량 증가**: Covariance 계산 및 Newton-Schulz 반복으로 기본 SE 모듈보다 FLOPs 높음.  
2. **2차 통계 현실성**: 고해상 복원에 유리하지만, 강한 압축·노이즈가 있는 도메인에서는 분산 추정이 불안정할 수 있음.  
3. **Still CNN-centric**: Transformer 기반 글로벌 의존성 학습이 대두되면서 receptive field 한계가 재부각[3].

## 5. 일반화 성능 관련 고찰  

| 기법 | 일반화 관점 이점 | 잠재적 리스크 |
|------|-----------------|---------------|
| **SOCA** | 2차 통계는 스타일·조명 변화에 덜 민감 → **도메인 간 전이**에 견고[4][5] | 공분산 추정 편향 시 과적합 가능 |
| **Share-Source Skip** | LR 입력 정보 우회 전달 → **다중 열화 모델(BI·BD)** 상황에서도 안정적[1] | 너무 강한 단순 정보 우회 시 세밀한 패턴 학습 저해 |
| **RL-NL** | 지역 단위 유사 패치 매칭 → **텍스처 다양성** 증가 | 해상도↑ 시 연산량 증가 |

결국 SAN 구조는 통계 기반 정규화와 전역-지역 컨텍스트 병행으로 **도메인 이동**에 비교적 강하나,  
공분산 추정 정확도에 크게 의존하므로 저품질·저휘도 데이터셋에서는 후보 정규화 기법(예: Padé 근사[5])의 도입이 필요하다.

## 6. 향후 연구 영향 및 고려 사항  

1. **고차 통계 Attention의 부상**  
   – SAN은 2차 통계를 CNN-attention에 최초로 결합, 이후 **Mixed Second-Order Attention**[6]·**Holistic Attention**[7] 같은 변형 연구로 확장.  
2. **효율적 행렬 함수 계산**  
   – Newton-Schulz 가속[8] → 고차 근사·Durand iteration[9] 등으로 이어져, **저연산/저메모리** 초해상화의 연구 축을 형성.  
3. **Transformer SR과의 융합**  
   – 글로벌 self-attention과 SOCA를 결합하는 시도가 등장(TaylorIR[10]). 미래에는 **pixel-level transformer + second-order pooling** 하이브리드가 유망.  
4. **평가 지표 재고**  
   – 고차 통계 기반 모델은 PSNR 대비 **주관적 품질**이 더욱 향상되는 경향이 있어, RQI 등 새로운 지표 연구가 병행되어야 함[11].  
5. **실제 열화 모델(blind SR) 대응**  
   – 공분산 정규화가 열화 파라미터의 불확실성에 민감하므로, **데이터 불확실성 추정 + SOCA** 결합이 필요.

## 7. 결론  
SAN은 **2차 통계 Attention**과 **비국소 잔차 구조**를 통합해 SISR 성능을 끌어올린 선구적 연구이다.  
비록 연산량·공분산 추정 신뢰성이라는 과제가 남아 있지만, **채널-공간-통계** 통합 설계라는 발상을 제시해 후속 모델과  
범용 영상 복원·표현 학습 분야 전반에 실질적 영감을 제공하였다.

[1] https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf
[2] https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf
[3] https://arxiv.org/abs/2501.07855
[4] https://serp.ai/posts/global-second-order-pooling-convolutional-networks/
[5] https://openaccess.thecvf.com/content/ICCV2021/papers/Song_Why_Approximate_Matrix_Square_Root_Outperforms_Accurate_SVD_in_Global_ICCV_2021_paper.pdf
[6] https://ieeexplore.ieee.org/document/9598826/
[7] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570188.pdf
[8] https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0475.pdf
[9] https://arxiv.org/pdf/2208.04068.pdf
[10] https://arxiv.org/abs/2411.10231
[11] https://arxiv.org/html/2503.13074v2
[12] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5844b09f-8e3c-466a-bf69-d5dcbec9e999/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf
[13] https://www.atlantis-press.com/article/125919477
[14] https://ieeexplore.ieee.org/document/11018875/
[15] https://www.semanticscholar.org/paper/34734140256017cab69065aedbe982dd867d01db
[16] https://www.mdpi.com/1099-4300/23/11/1398/pdf
[17] https://arxiv.org/pdf/2110.14638.pdf
[18] https://arxiv.org/pdf/2210.05960.pdf
[19] https://arxiv.org/html/2411.17513v1
[20] https://kozistr.tech/2020-03-14-SAN/
[21] https://github.com/daitao/SAN
[22] https://www.kibme.org/resources/journal/20201216163518051.pdf
[23] https://pmc.ncbi.nlm.nih.gov/articles/PMC9003536/
[24] https://peerj.com/articles/cs-1196/
[25] https://arxiv.org/html/2308.12880v2
[26] https://www.sciencedirect.com/science/article/pii/S1569843222000280
[27] https://www.mdpi.com/2079-9292/11/24/4159
[28] https://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html
[29] https://www.mdpi.com/2079-9292/10/10/1187
[30] https://par.nsf.gov/servlets/purl/10134410
[31] https://bellzero.tistory.com/6
[32] https://ieeexplore.ieee.org/document/10712887/
[33] https://ijaems.com/detail/a-comprehensive-review-and-comparison-of-image-super-resolution-techniques/
[34] https://pubs.acs.org/doi/10.1021/cbmi.4c00019
[35] https://arxiv.org/abs/2502.00404
[36] https://arxiv.org/abs/2505.23248
[37] https://www.ijraset.com/best-journal/gan-based-super-resolution-algorithm-for-high-quality-image-enhancement
[38] http://mclab.unipv.it/Master/Image%20Super-Resolution%20Hystorical%20Overview%20and%20Future%20Challenges.pdf
[39] https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf
[40] https://displaydaily.com/wp-content/uploads/2020/07/Image-super-resolution-The-techniques-applications-and-future.pdf
[41] https://www.mdpi.com/2076-3417/13/12/7160
[42] https://pmc.ncbi.nlm.nih.gov/articles/PMC11134610/
[43] https://openaccess.thecvf.com/content/CVPR2021W/UG2/papers/Han_Two-Stage_Network_for_Single_Image_Super-Resolution_CVPRW_2021_paper.pdf
[44] https://www.mdpi.com/2072-4292/14/21/5423
[45] https://arxiv.org/pdf/1904.06836.pdf
[46] https://www.sciencedirect.com/science/article/abs/pii/S0925231220307748
[47] https://jkms.kms.or.kr/journal/view.html?volume=55&number=6&spage=1529
[48] https://www.bohrium.com/paper-details/second-order-attention-network-for-single-image-super-resolution/812690184888909824-100927
[49] https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Wang_NTIRE_2024_Challenge_on_Light_Field_Image_Super-Resolution_Methods_and_CVPRW_2024_paper.pdf
[50] https://www.tandfonline.com/doi/full/10.1080/09540091.2023.2182487
[51] https://arxiv.org/pdf/1810.11579.pdf
[52] https://arxiv.org/pdf/2501.15774.pdf
[53] https://arxiv.org/pdf/2308.08142.pdf
[54] http://arxiv.org/pdf/2205.04437.pdf
[55] https://www.mdpi.com/1424-8220/23/20/8574/pdf?version=1697697631
[56] https://arxiv.org/pdf/2208.11247.pdf
[57] https://arxiv.org/html/2412.02234v1
[58] https://arxiv.org/pdf/1904.07523.pdf
[59] https://www.mdpi.com/2079-9292/13/1/194/pdf?version=1704174929
[60] https://arxiv.org/pdf/2401.05633.pdf
[61] https://arxiv.org/html/2311.16512v4
[62] https://www.sciencedirect.com/science/article/abs/pii/S0020025518308703
[63] https://dl.acm.org/doi/10.1007/978-3-030-57321-8_15
[64] https://www.sciencedirect.com/science/article/pii/S1569843224001481
[65] https://arxiv.org/pdf/2303.06373.pdf
[66] https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2313102
[67] https://ieeexplore.ieee.org/document/10714352/
[68] https://arxiv.org/pdf/2501.07855.pdf
[69] http://arxiv.org/pdf/2102.09351.pdf
[70] https://arxiv.org/html/2410.22830
[71] http://arxiv.org/pdf/2502.09654.pdf
[72] https://pmc.ncbi.nlm.nih.gov/articles/PMC11043356/
[73] https://downloads.hindawi.com/journals/cin/2022/8628402.pdf
[74] http://arxiv.org/pdf/2212.14181.pdf
[75] https://www.mdpi.com/2072-4292/13/20/4180/pdf
[76] https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2022/591/2022/isprs-annals-V-3-2022-591-2022.pdf
[77] https://docs.modula.systems/algorithms/newton-schulz/
[78] https://www.sciencedirect.com/science/article/abs/pii/S0925231222004076
[79] https://www.sciencedirect.com/science/article/abs/pii/S1566253521001792
[80] https://www.mdpi.com/2227-7390/11/14/3161
