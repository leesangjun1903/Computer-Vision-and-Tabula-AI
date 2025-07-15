# DeepSIM : Image Shape Manipulation from a Single Augmented Training Sample | Image generation, Image manipulation

## 1. 핵심 주장 및 주요 기여  
**DeepSIM**은 단 하나의 이미지-프리미티브 쌍만으로도 조건부 생성 모델(cGAN)을 학습시켜 복잡한 이미지 편집을 수행할 수 있음을 보인다[1].  
- **싱글 샷 학습**: 대규모 데이터셋 없이 단일 이미지 쌍만으로도 고해상도 조작을 학습.  
- **TPS 기반 증강**: Thin-Plate-Spline(TPS) 변환을 활용한 비선형 증강으로 단일 샘플의 다양성을 확보.  
- **범용성**: 에지(edge), 세그멘테이션, 조합된 프리미티브 모두에 적용 가능하며, 소·대형 물체의 형태 변형, 추가·제거, 애니메이션 생성에 활용.  

## 2. 문제 정의, 방법론, 모델 구조, 성능, 한계  

### 문제 정의  
- 유니크한 단일 이미지를 원하는 대로 조작할 때, 데이터 부족으로 일반적인 cGAN이 과적합되어 새로운 프리미티브 변형에 대응 불가.  

### 제안 방법  
1. 입력: 이미지 프리미티브 x (에지·세그멘테이션·조합)와 타깃 이미지 y  
2. **손실 함수**  
   - 재구성(VGG 퍼셉추얼) 손실:  

$$ \ell_{\text{rec}} = \ell_{\text{perc}}(G(x), y) $$  
   - 조건부 GAN 손실:  

$$ \ell_{\text{adv}} = \log D(x,y) + \log\bigl(1 - D(x, G(x))\bigr) $$  
   - 최종:  

$$ \ell_{\text{total}} = \ell_{\text{rec}} + \alpha\,\ell_{\text{adv}} $$  

3. **TPS 증강**  
   - 3×3 격자 제어점에 무작위 이동 적용, 전체 이미지와 프리미티브에 일관되게 변환  
   - 이동 범위는 이미지 최소 축의 10% 이내로 제한[1].

# Thin-Plate Spline (TPS) 변환 

**핵심 요약**  
Thin-Plate Spline(TPS) 변환은 2차원 공간에서 비선형 워핑을 수행하는 대표적인 스플라인 기반 방법으로, ‘얇은 철판’이 구속된 제어점(knots) 사이를 최소한의 굽힘 에너지(bending energy)로 부드럽게 이어지도록 하는 변환입니다.  

## 1. 물리적 유추와 목적  
TPS는 얇은 금속판이 고정된 지점들 사이를 부드럽게 휘어질 때 발생하는 물리적 현상을 모사합니다.  
- **제어점**: 2차원 평면상의 $$K$$개의 대응점 $$\{(x_i,y_i)\}_{i=1}^K$$.  
- **변환 목표**: 각 제어점이 대응점 $$\{(x'_i,y'_i)\}$$로 이동하도록 전체 평면을 워핑.  
- **에너지 최소화**: 금속판의 굽힘 에너지를 최소화하면서 대응점을 정확히 잇는 매핑 함수를 찾음[1].

## 2. 수학적 공식  

### 2.1. 에너지 함수  
TPS는 두 가지 항의 합을 최소화하는 함수 $$f\colon\mathbb R^2\to\mathbb R^2$$를 구합니다.  
1. **재구성 오차**

$$
     E_{\mathrm{data}} = \sum_{i=1}^K \bigl\|y_i - f(x_i)\bigr\|^2,
   $$
   
   여기서 $$x_i=(x_i,y_i)$$, $$y_i=(x'_i,y'_i)$$.  
2. **굽힘 에너지**  

$$
     E_{\mathrm{bend}} = \iint_{\mathbb R^2} \Bigl( f_{xx}^2 + 2f_{xy}^2 + f_{yy}^2 \Bigr)\,dx\,dy.
   $$ 
   
3. **총 에너지**  
   
$$
     E_{\mathrm{total}}(f) = E_{\mathrm{data}} + \lambda\,E_{\mathrm{bend}},
   $$ 
   
   $$\lambda\ge0$$는 스무딩 강도를 조절하는 하이퍼파라미터입니다[1].

### 2.2. 해의 형태 및 라디얼 베이시스 표현  
변환 $$f(x)$$는 전역 선형부(affine)와 국소 비선형부 합으로 표현됩니다:  

$$
  f(x,y) = 
  \underbrace{
    \begin{bmatrix} a_0 + a_1 x + a_2 y \\ b_0 + b_1 x + b_2 y \end{bmatrix}
  }\_{\text{affine}}
  + 
  \sum_{i=1}^K 
  w_i \,\varphi\bigl(\| (x,y)-(x_i,y_i)\|\bigr),
$$

여기서 $$\varphi(r)=r^2\log r$$는 TPS 특유의 라디얼 베이시스 함수입니다[1][2].  
- $$\{a_j,b_j\}$$: 전역 아핀 파라미터  
- $$\{w_i\}$$: 국소 변형 가중치  

해를 구하려면, 다음 선형 시스템을 풀면 됩니다:  

```math
\begin{pmatrix}
K + \lambda I & P \\
P^\mathsf{T} & 0
\end{pmatrix}
\begin{pmatrix} w \\ c \end{pmatrix}
=
\begin{pmatrix} y \\ 0 \end{pmatrix},
```

여기서  
- $$K_{ij} = \varphi\bigl(\|x_i-x_j\|\bigr)$$  
- $$P_{i} = [1,\,x_i,\,y_i]$$  
- $$\lambda I$$는 스무딩 항(정규화 추가)  
- $$c$$는 아핀 파라미터 벡터  

## 3. 구현 절차  

1. **제어점 설정**  
   - 사용자 혹은 자동화 방식으로 $$\{x_i\}$$와 대응 $$\{y_i\}$$ 결정.  
2. **커널·행렬 구성**  
   - $$K$$, $$P$$ 행렬 계산.  
3. **선형 시스템 풀이**  
   - 위 블록 행렬을 구성하여 $$\{w,c\}$$를 해 구하기.  
4. **워핑 적용**  
   - 임의 $$x$$좌표에 대해 $$f(x)$$를 계산하여 변환된 위치 산출.  

대규모 제어점에서는 시스템 크기가 $$O(K)\times O(K)$$이므로 계산 비용이 커질 수 있습니다. 이때는 서브샘플링, 근사법(예: 분할정복, 랜덤 피처) 등을 활용합니다.

## 4. 응용 예시  
- **이미지 워핑**: 얼굴 랜드마크 매칭, 왜곡 보정  
- **의료 영상 정합**: 해부학적 구조 정렬  
- **그래픽스**: 3D 메시 디포메이션  
- **단일 샷 이미지 증강**: Thin-Plate-Spline 증강을 통한 단일 이미지 GAN 훈련[3].

**참고문헌**  
[1] “Thin plate spline – Wikipedia,” https://en.wikipedia.org/wiki/Thin_plate_spline  
[2] A. Djellouli, “Thin Plate Spline Interpolation,” May 24, 2025.

[1] https://en.wikipedia.org/wiki/Thin_plate_spline
[2] https://adamdjellouli.com/articles/numerical_methods/6_regression/thin_plate_spline_interpolation
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b46f4d29-1f3c-41b0-a3f2-380201225856/2007.01289v2.pdf
[4] https://www.sec.gov/Archives/edgar/data/1385849/000138584925000004/efr-20241231.htm
[5] https://www.sec.gov/Archives/edgar/data/1049659/000129281425001805/sidform20f_2024.htm
[6] https://www.sec.gov/Archives/edgar/data/1018281/000143774925019879/itkg20240630c_10k.htm
[7] https://www.sec.gov/Archives/edgar/data/1131903/000109690625001012/fccn-20241231.htm
[8] https://www.sec.gov/Archives/edgar/data/2029967/0002029967-24-000001-index.htm
[9] https://www.sec.gov/Archives/edgar/data/1531177/000095017025035500/sght-20241231.htm
[10] https://www.sec.gov/Archives/edgar/data/1342874/000134287425000024/tx-20241231.htm
[11] https://ieeexplore.ieee.org/document/9956462/
[12] https://ieeexplore.ieee.org/document/10566012/
[13] https://arxiv.org/abs/2408.09131
[14] https://arxiv.org/abs/2305.05322
[15] https://ieeexplore.ieee.org/document/10377387/
[16] https://ojs.aaai.org/index.php/AAAI/article/view/33195
[17] https://dl.acm.org/doi/10.1145/3503161.3551597
[18] https://www.semanticscholar.org/paper/1495a985b1736317fb1d358b9dc2164e4bf0df27
[19] https://www.palass.org/sites/default/files/media/palaeomath_101/article_20/article_20.pdf
[20] https://flimagingmanual.fourthlogic.co.kr/01_ImageProcessing/Warping/ThinPlateSplineWarping/ImageProcessing_Warping_ThinPlateSplineWarping.html
[21] https://www.publichealth.columbia.edu/research/population-health-methods/thin-plate-spline-regression
[22] https://scikit-image.org/docs/0.24.x/auto_examples/transform/plot_tps_deformation.html
[23] https://cseweb.ucsd.edu/~sjb/eccv_tps.pdf
[24] https://ui.adsabs.harvard.edu/abs/2019JGeod..93.1251K/abstract
[25] https://mathematica.stackexchange.com/questions/213532/how-to-interpolate-xi-yi-xn-yn-with-minimum-bending-energy
[26] https://kr.mathworks.com/matlabcentral/fileexchange/37576-3d-thin-plate-spline-warping-function
[27] https://flimagingmanual.fourthlogic.co.kr/06_Foundation/Mapping/ThinPlateSplineMapping/Foundation_Mapping_ThinPlateSplineMapping.html
[28] https://github.com/djeada/Numerical-Methods/blob/master/notes/6_regression/thin_plate_spline_interpolation.md
[29] https://arxiv.org/abs/2401.13432
[30] https://scikit-image.org/docs/0.25.x/auto_examples/transform/plot_tps_deformation.html
[31] https://www.mdpi.com/2227-7390/10/9/1562
[32] https://ostin.tistory.com/65
[33] https://www.semanticscholar.org/paper/5eb1a79f8a559ca37e8766c64f974e49c2314ca0
[34] https://www.semanticscholar.org/paper/7ee4758112c504e901fdc8c14a4f15c419d0c2b1
[35] https://arxiv.org/pdf/2401.13432.pdf
[36] https://arxiv.org/pdf/1705.05178.pdf
[37] https://arxiv.org/html/2302.10442v3
[38] https://library.oapen.org/bitstream/20.500.12657/41274/1/2020_Book_SpectralAndHighOrderMethodsFor.pdf
[39] https://arxiv.org/pdf/2207.13931.pdf
[40] https://arxiv.org/abs/1907.10978
[41] https://arxiv.org/pdf/2404.01902.pdf
[42] https://www.emis.de/journals/SIGMA/2018/083/sigma18-083.pdf
[43] https://arxiv.org/pdf/2302.12974.pdf
[44] http://arxiv.org/abs/2110.12826
[45] https://www.css.cornell.edu/faculty/dgr2/_static/files/R_PDF/exTPS.pdf
[46] https://khanhha.github.io/posts/Thin-Plate-Splines-Warping/
[47] https://user.engineering.uiowa.edu/~aip/papers/bookstein-89.pdf

4. **학습**  
   - Pix2PixHD 아키텍처 기반, TPS 증강 하에 16,000 iter 수행  
   - 한 번 학습된 G는 다양한 프리미티브 조작에 재사용 가능  

### 모델 구조  
- Generator: Pix2PixHD 스타일의 U-Net 계열, VGG 기반 퍼셉추얼 손실과 PatchGAN 판별기 사용  
- Discriminator: 프리미티브/이미지 쌍을 입력받아 진위 판별  

### 성능 향상  
- **LPIPS, SIFID** 영상 품질 지표에서 crop-flip 대비 TPS 증강이 대폭 개선  
- Cityscapes: LPIPS 0.342→0.104, SIFID 0.292→0.104  
- LRS2 비디오 프레임: 평균 LPIPS 0.46→0.14, SIFID 0.44→0.06  
- 사용자 연구에서 생성물의 진위 혼동률 42.6%[1].  

### 한계  
- **훈련 속도**: 단일 이미지당 49분 필요, 매 이미지별 재학습 필수  
- **미지 객체 생성 실패**: 학습에 없는 물체 생성 시 왜곡  
- **배경 복제 오류**: 제거·이동 시 주변 패턴 중복  
- **공백 보간 오류**: 가이드 없는 영역에서 부정확한 세부 묘사[1].  

## 3. 모델의 일반화 성능 향상 가능성  
- **증강 융합**: TPS 외에도 색상, 노이즈, 기하학적 변형 조합 시 일반화 확대  
- **메타-러닝**: 다중 이미지 단일샷 학습을 통한 빠른 적응  
- **프리트레이닝**: 대규모 도메인 일반화된 백본 활용 후 단일 샷 파인튜닝  
- **제한적 재사용**: 유사 이미지들 간 전이 학습으로 초기 가중치 공유  

## 4. 연구의 향후 영향 및 고려 사항  
- **개인화된 편집 도구**: 사용자별 맞춤 이미지 편집기 개발  
- **초저샷 비디오 합성**: 단일 프레임 기반 애니메이션 제작 파이프라인  
- **고속 학습 알고리즘**: 단일샷 GAN 학습 시간 단축 연구  
- **신뢰성·안정성**: 공백 영역 보간과 미지 객체 생성을 위한 추가 제약 도입  
- **윤리적 고려**: 단일 이미지 편집의 오남용 방지 및 워터마킹  

---
[1] Yael Vinker et al., “Image Shape Manipulation from a Single Augmented Training Sample,” arXiv:2007.01289v2, 2020.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b46f4d29-1f3c-41b0-a3f2-380201225856/2007.01289v2.pdf
