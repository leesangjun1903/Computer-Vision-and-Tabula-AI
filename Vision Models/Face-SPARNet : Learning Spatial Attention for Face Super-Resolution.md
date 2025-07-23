# Face-SPARNet : Learning Spatial Attention for Face Super-Resolution | Super resolution

# 핵심 요약

**Learning Spatial Attention for Face Super-Resolution**는  
-  얼굴 초해상도(Face SR)가 놓치기 쉬운 **눈·코·입 등의 ‘희소한’ 핵심 구조**를 자동으로 강조하는 **SPatial Attention Residual Network(SPARNet)**를 제안한다.  
-  기존 방법과 달리 **추가 라벨(랜드마크·파싱 맵)** 없이, **Residual Block 내부에 ‘Face Attention Unit(FAU)’**를 넣어 **공간적 주의(attention)** 를 학습한다[1].  
-  16 × 16 입력을 128 × 128로 복원할 때 PSNR·SSIM·정체성 유사도·랜드마크 검출 정확도 모두에서 최고 성능을 달성한다[1].  
-  SPARNet을 **다중 스케일 판별기**가 있는 **SPARNetHD**로 확장해 **512 × 512 고해상도**까지 생성하고, 합성 데이터로 학습해도 **실제 저화질 얼굴**에 강한 **일반화**를 보인다[1].

## 1. 문제가 되는 지점

| 기존 접근 | 한계 |
|-----------|------|
| (a) 일반 SR 네트워크 | 얼굴 세부 구조 복원 불충분·과도한 블러 |
| (b) 다중 과제 학습(파싱·랜드마크)[1] | 수동 라벨 필요, LR에서 정확한 파싱·랜드마크 예측 자체가 난제 |
| (c) 채널 어텐션·패치 어텐션[2] | 공간적 위치 정보를 세밀하게 반영하지 못함 |

**과제:** 추가 감독 없이, 매우 저해상도 얼굴에서 **국소적·공간적** 얼굴 구조를 선택적으로 증폭해 고해상도로 복원.

## 2. 제안 방법

### 2.1 Face Attention Unit(FAU)

Residual Block에 **Dual Branch** 삽입:

```math
\begin{aligned}
&f_j = F_{\text{feat}}(x_{j-1}) \\
&\alpha_j = \sigma \bigl(F_{\text{att}}(f_j)\bigr) \\
&x_j = x_{j-1} + \alpha_j \otimes f_j  
\end{aligned}
```

* **Feature branch** $$F_{\text{feat}}$$: Pre-activation Residual Unit(PReLU + BN).  
* **Attention branch** $$F_{\text{att}}$$: Hourglass block → Conv → Sigmoid → **spatial map** $$\alpha_j$$ (0–1).  
* 다중 해상도 특징을 이용해 **눈·입·윤곽** 등 영역별로 다른 주의 지도를 학습.

### 2.2 SPARNet 구조

1. **Down-scale module**: 2× stride Conv + FAU 반복(10개).  
2. **Feature extraction**: FAU 스택(10개)으로 고차 특징 추출.  
3. **Up-scale module**: Nearest Upsample + Conv + FAU 반복(10개).  
4. **Loss**: 단순 L2 (픽셀)  

$$
\mathcal{L}\_{\text{pix}}=\frac1N\sum_i\|F_{\text{SPAR}}(I^{\text{LR}\!\uparrow}_i)-I^{\text{HR}}_i\|_2^2.
$$

### 2.3 SPARNetHD 확장

| 구성 요소 | 역할 |
|-----------|------|
| **Generator** | SPARNet 채널 수 증가, 출력 512² |
| **3-scale Discriminators** $$D_1,D_2,D_3$$ | 512²·256²·128² 해상도별 사실감 학습 |
| **Total loss** |  $$\lambda_\text{pix} \mathcal{L}\_{\text{pix}} +\lambda_\text{adv} \mathcal{L}\_{\text{GAN}} +\lambda_\text{fm} \mathcal{L}\_{\text{FM}} +\lambda_\text{pcp} \mathcal{L}\_{\text{per}}$$  |

* **Feature-matching**와 **VGG perceptual loss**로 구조·텍스처 균형 유지[1].  

## 3. 성능 및 일반화

| 데이터셋 · 입력| 지표 | Bicubic | RCAN | Wavelet-SRNet | **SPARNet** |
|---------------|------|---------|------|--------------|-------------|
| Helen 16→128 | PSNR(dB) | 23.52 | 26.40 | 26.42 | **26.59**[1] |
|               | SSIM    | .641 | .765 | .771 | **.772**[1] |
|               | 랜드마크 AUC↑ | 4.4% | 56.5% | 58.4% | **58.5%**[1] |
| UMD-Face ID   | Cos sim↑ | .185 | .537 | .515 | **.555**[1] |

* **어텐션 유닛 개수 증가** → PSNR·SSIM 상승(26.30→26.60 dB)[1].  
* **다중 스케일 어텐션**(bottleneck 4×4) > 단일 스케일[1].  
* **SPARNetHD**: FFHQ로 합성 학습 후 **실제 저화질**(CelebA-TestN)에서 FID 27.2, SFTGAN보다 10 ↓[1].

### 일반화 비결

1. **라벨 없는 공간 어텐션** → 데이터셋 간 구조 편향 학습, 라벨 노이즈 영향 ↓.  
2. **Hourglass 기반 멀티스케일** → 시야 범위 적응, 미소·주름 등 다양한 해상도에 대응.  
3. **GAN + Perceptual 혼합 손실** → 합성·실제 도메인 간 텍스처 분포 간극 완화.  
4. **Synthetic degradation 다양화**(Gaussian/Median/Motion blur, JPEG, 노이즈)로 **도메인 랜덤화** 학습[1].

## 4. 한계

* **16× 업스케일** 등 극단적 배수에서 채널/계산량 급증.  
* 주의 지도는 **채널 간 상호작용**을 무시(전역 채널 어텐션 부재).  
* GAN 기반 SPARNetHD는 **안정적 학습 세팅**(λ 조정)에 민감.  
* 얼굴 외 객체·프로필 초과 기울기에서는 구성 요소 혼란 가능.

## 5. 향후 연구 영향 및 고려 사항

* **Label-free attention**이 보여준 효과는 이후 **Transformer·Mix-attention FSR** 연구의 기반이 되었다[3][4].  
* **멀티-스케일 판별기** + 공간 어텐션 조합은 **고해상도 합성**(StyleGAN 계열)에도 적용 가능.  
* 후속 연구에서는  
  1. **채널·주파수 어텐션**을 결합해 구조-텍스처 동시 강화,  
  2. **Implicit NeRF·INR**와의 융합으로 임의 배율·시점 대응,  
  3. **Privacy·Bias** 측면: 정체성 유지가 향상됨에 따른 **악용 방지** 가이드라인 마련,  
  4. 의료·안전 영상 등 **도메인 특화 약식 라벨**과의 결합 등이 중요 과제가 될 것이다.

**참고 출처**: 논문 원문 PDF[1] 및 IEEE DL 메타데이터[5][6].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5aa6ae9b-b17b-4528-b454-0039fe7450d7/2012.01211v2-1.pdf
[2] https://arxiv.org/abs/2408.05205
[3] https://ieeexplore.ieee.org/document/10205328/
[4] https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Spatial-Frequency_Mutual_Learning_for_Face_Super-Resolution_CVPR_2023_paper.pdf
[5] https://ieeexplore.ieee.org/document/9293182/
[6] https://dl.acm.org/doi/abs/10.1109/TIP.2020.3043093
[7] https://ieeexplore.ieee.org/document/10207832/
[8] https://ieeexplore.ieee.org/document/10145603/
[9] https://dl.acm.org/doi/10.1145/3664647.3681088
[10] https://ieeexplore.ieee.org/document/10485196/
[11] https://ojs.aaai.org/index.php/AAAI/article/view/28339
[12] https://ieeexplore.ieee.org/document/10409565/
[13] https://arxiv.org/abs/2012.01211
[14] https://hub.hku.hk/bitstream/10722/301191/1/Content.pdf
[15] https://www.nature.com/articles/s41598-025-97451-z
[16] https://github.com/chaofengc/Face-SPARNet
[17] https://www.emergentmind.com/articles/2012.01211
[18] https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Residual_Attention_Network_CVPR_2017_paper.pdf
[19] https://www.mdpi.com/2079-9292/14/10/2070
[20] https://dlaiml.tistory.com/entry/Residual-Attention-Network-for-Image-Classification
[21] https://openaccess.thecvf.com/content/WACV2024/papers/Tsai_Arbitrary-Resolution_and_Arbitrary-Scale_Face_Super-Resolution_With_Implicit_Representation_Networks_WACV_2024_paper.pdf
[22] https://shamra-academia.com/show/5d398ada519b3
[23] https://arxiv.org/html/2409.00591v1
[24] https://www.mdpi.com/2076-3417/14/16/7154
[25] https://dlgari33.tistory.com/24
[26] https://dl.acm.org/doi/10.1109/TIP.2020.3043093
[27] https://ieeexplore.ieee.org/document/10483821/
[28] https://arxiv.org/pdf/2012.01211.pdf
[29] https://arxiv.org/abs/2304.02923
[30] https://arxiv.org/html/2407.19768v2
[31] https://arxiv.org/abs/2306.02277
[32] https://arxiv.org/pdf/2109.13626.pdf
[33] https://onlinelibrary.wiley.com/doi/pdfdirect/10.1049/ipr2.12863
[34] https://downloads.hindawi.com/journals/mpe/2021/6648983.pdf
[35] http://arxiv.org/pdf/2210.06002.pdf
[36] https://arxiv.org/abs/1908.08239
[37] https://arxiv.org/pdf/2101.03749.pdf
[38] https://www.sciencedirect.com/science/article/abs/pii/S0893608023000060
