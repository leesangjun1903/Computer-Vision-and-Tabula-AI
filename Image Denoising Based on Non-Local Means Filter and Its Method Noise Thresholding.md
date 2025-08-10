# Image Denoising Based on Non-Local Means Filter and Its Method Noise Thresholding  
**B. K. Shreyamsha Kumar (2012)**  

## 1. 핵심 주장 및 주요 기여  
이 논문은 **비국소 평균화 필터(Non-Local Means, NL-Means)의 성능 저하(특히 고잡음 환경에서의 블러링 및 세부 정보 손실)를 보완**하기 위해, NL-Means가 제거한 “방법 노이즈(method noise)”를 **웨이브릿 기반 방법 노이즈(thresholded method noise)** 로 정제하여 다시 이미지에 더해 주는 하이브리드 기법(NLFMT)을 제안한다.  
- NL-Means가 놓친 **미세 구조·디테일**를 웨이브릿 영역에서 분리된 **세부 성분(detail coefficients)** 으로 복원  
- **BayesShrink** soft-thresholding을 이용해 방법 노이즈의 웨이브릿 계수를 적응적으로 임계 처리  
- 기존 기법(웨이브릿 단독, 양쪽 필터, 다중 해상도 BF, NL-Means, BM3D) 대비 **PSNR, IQI, 시각 품질, 방법 노이즈 저감** 모두에서 우수 또는 동등 성능 입증  

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- NL-Means는 **자기유사성(self-similarity)** 을 기반으로 노이즈를 제거하나, 필터링 과정에서 **유사 패치들도 함께 노이즈화**되어 고잡음(저 SNR) 환경에서 결과가 블러링되고 디테일이 손실됨.  

### 2.2 제안 알고리즘 개요  
1. **입력 노이즈 영상** $$I = A + Z$$ 에 NL-Means 필터 적용 → 중간 결과 $$I_F$$  
2. NL-Means가 제거한 “방법 노이즈” $$\;MN = I - I_F$$ 계산(식 8)  
3. 방법 노이즈 $$MN$$ 를 **웨이브릿 변환**하여 계수 $$Y$$ 획득  
4. 각 서브밴드에 대해 **BayesShrink** soft-threshold $$T = \sigma_n^2/\sigma_w$$ 적용(식 11–13)  
5. 역웨이브릿 재구성으로 **추정된 디테일 $$\hat D$$** 획득  
6. 최종 복원 영상 $$B = I_F + \hat D$$  

### 2.3 핵심 수식  
- NL-Means 복원:

$$
I_F(i)=\sum_{j\in S_i} w(i,j)\,I(j),
\quad
w(i,j)\propto \exp\Bigl(-\|I(N_i)-I(N_j)\|^2/(h^2)\Bigr)
$$

- 방법 노이즈:  

$$
MN = I - I_F
$$

- BayesShrink 임계값:  

$$
T = \frac{\sigma_n^2}{\sigma_w},\quad
\hat\sigma_n = \frac{\mathrm{Median}(|Y_{HH1}|)}{0.6745},\quad
\hat\sigma_w^2=\max(\hat\sigma_y^2-\hat\sigma_n^2,0)
$$

## 3. 모델 구조 및 성능 향상  
- **단계적 구조**: NL-Means → 방법 노이즈 추출 → 웨이브릿 임계 처리 → 디테일 재합성  
- **성능 향상**  
  - **PSNR/IQI**: 다수의 실험에서 WT, BF, MRBF, NL-Means 보다 평균 1–2 dB, IQI 0.01–0.03 상승  
  - **시각 품질**: 디테일 보존 및 블러링 최소화  
  - **방법 노이즈**: 순수 랜덤 노이즈 형태로 정제, 디테일 성분은 복원 영상으로 이전  

### 3.1 한계  
- NL-Means 및 BayesShrink 파라미터(패치 크기, 검색 창, h, 웨이브릿 종류)에 민감  
- 고잡음(σ ≥ 50) 시 BM3D 수준에는 미치지 못함  
- 계산 복잡도: 이중 필터링 및 변환 단계로 실행 시간 증가  

## 4. 일반화 성능 및 확장성  
- **다양한 웨이브릿** 실험(sym8, db16, coif5, bior6.8, DCHWT)에서 DCHWT 사용 시 PSNR 최대, coif5/IQI 최대 확인  
- **다른 노이즈 모델**(비가우시안, 색상 영상)에도 적용 가능성 있으며,  
- **BM3D** 의 협업 필터링, **adaptive NL-Means** 커널, **shift-invariant wavelet** 결합을 통해 일반화 성능 및 계산 효율 향상 전망  

## 5. 향후 영향 및 연구 고려사항  
- **영향**: NL-Means 기반 계층적 하이브리드 필터링 전략을 제시, **방법 노이즈** 개념 확장  
- **고려점**:  
  - **파라미터 자동 최적화**(SURE 기반 자동 h, threshold)  
  - **실시간 적용**을 위한 가속화(GPU, 근사 알고리즘)  
  - **다중 채널·다중 센서** 영상, 동영상 시퀀스로 확장하여 **시간적 자기유사성** 활용 연구  
  - **비가우시안/비독립 노이즈** 모델 대응 위한 통계 기반 threshold 개선  

이 하이브리드 프레임워크는 NL-Means의 강점인 자기유사성 활용과 웨이브릿의 멀티해상도 디테일 복원이 결합된 방식으로, 추후 **더 정교한 thresholding** 및 **협업 필터링 통합** 연구 방향을 열어 주었다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/658fffd5-7919-4bb9-bd52-425801260b83/s11760-012-0389-y.pdf
