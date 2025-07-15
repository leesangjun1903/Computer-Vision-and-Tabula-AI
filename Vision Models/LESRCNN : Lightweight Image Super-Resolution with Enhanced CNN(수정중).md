# LESRCNN : Lightweight Image Super-Resolution with Enhanced CNN | Super resolution

## 1. 핵심 주장 및 주요 기여 요약  
**핵심 주장:**  
LESRCNN(Lightweight Enhanced Super-Resolution CNN)은 정보 추출·강화 블록(IEEB), 재구성 블록(RB), 정보 정제 블록(IRB)을 연속적으로 결합해 단일 모델로 다양한 배율(×2, ×3, ×4)에서 고품질 SR을 실현하면서도 파라미터 수와 연산량을 크게 줄인다.  

**주요 기여:**  
- IEEB: 3×3/1×1 Conv+ReLU의 17개 계층으로 LR 특징을 계층적으로 추출·집계해 얕은 층 기억 능력을 강화하고, 이질적(concatenated heterogeneous) 구조로 압축 효율을 높임.  
- RB: sub-pixel convolution을 중간에 배치해 LR→HR 특징 변환 시 글로벌(1층)·로컬(17층) 특징을 잔차합(fRB = ReLU(S(O₁)+S(O₁₇)))으로 융합, 장기 의존 문제 완화.  
- IRB: 4×(3×3 Conv+ReLU)+1×(3×3 Conv)로 거친 HR 특징을 정제해 최종 SR 이미지 생성.  
- 모델 하나로 멀티 스케일 지원(LESRCNN-S) 및 단일 스케일 최적화(LESRCNN), GPU 한 쌍(1080Ti×2)에서 ▲PSNR+0.22 dB, ▲SSIM+0.0013 향상, 파라미터·FLOPs 대폭 저감[1].

## 2. 문제 정의, 제안 방법, 구조, 성능, 한계  
### 2.1 해결 과제  
- 깊은 SR 네트워크는 성능↑에 따른 파라미터·메모리·연산 급증  
- 모바일·임베디드 환경에 부적합  
- LR 입력을 미리 업샘플링 시 복잡도 추가  

### 2.2 제안 방법  
1. **Loss(평균 제곱 오차):**  

$$
     \ell(p) = \frac{1}{2T}\sum_{i=1}^T \|f_{\mathrm{LESRCNN}}(I_i^{LR}) - I_i^{HR}\|^2
$$  

2. **IEEB (17 layers):**  
   - 홀수층: 3×3 Conv+ReLU, 짝수층: 1×1 Conv+ReLU  
   - 잔차 연결로 모든 짝수 이전 홀수층 출력을 합산:

$$
O_j =
\begin{cases}
\text{ReLU}\left(O_j^c + \sum_{i < j, i \text{ odd}} O_i^c\right) & \text{if } j \text{ is odd}, \
\text{ReLU}(O_j^c) & \text{if } j \text{ is even}
\end{cases}
$$

3. **RB:** sub-pixel convolution $$S(\cdot)$$ 삽입,  

$$
O_{RB} = \mathrm{ReLU}\bigl(S(O_1)+S(O_{17})\bigr)
$$  

4. **IRB (5 layers):**  

$$
     I^{SR} = C_{3\times3}\Bigl(\mathrm{ReLU}\bigl(\cdots \mathrm{ReLU}(C_{3\times3}(O_{RB}))\bigr)\Bigr)
$$  

### 2.3 모델 구조  
- 총 23층: IEEB(17층) → RB(1층 sub-pixel) → IRB(5층)  
- 단일 스케일/멀티 스케일 학습 분리  

### 2.4 성능 향상  
- Set5–×4에서 PSNR 31.88 dB(기존 대비＋0.14 dB), SSIM 0.8903[1]  
- U100–×2에서 PSNR 31.45 dB(＋0.22 dB), SSIM 0.9206(＋0.0013)[1]  
- 파라미터 516K, FLOPs 3.08G로 VDSR(665K/10.9G) 대비 경량[1]  

### 2.5 한계  
- IEEB의 잔차 합산 방식이 계층 간 “특징 왜곡” 가능성  
- 단일 모델로 극고배율(×8 이상) 확장성 미검증  
- 실세계 흐릿함(blur)·노이즈 조건에 대한 강인성 검증 부족  

## 3. 일반화 성능 향상 가능성  
- **계층적 특징 집계**: 얕은층 정보가 깊은층으로 전달되며 overfitting 감소  
- **이질적 구조**: 1×1 Conv로 채널 간 다양성 확보 → 새로운 도메인 적응성 향상  
- **잔차 융합**: 글로벌·로컬 특징 결합으로 다양한 해상도 배율에 유연 대응  
- 추후 도메인 간 전이학습, 저작업(memory) 환경에서 추가 평가 시 일반화 한계 완화 기대  

## 4. 향후 연구 영향 및 고려 사항  
**영향:**  
- 경량 SR 모델 설계 패러다임 제시: 계층적 잔차·sub-pixel 중간 배치  
- 멀티 스케일 SR 통합 학습 가능성 확장  

**고려 사항:**  
- 극단적 블러·노이즈·비등방성 다운샘플링에 대한 견고성 평가  
- Transformers·채널 주의(attention) 기법 결합으로 일반화·시각 품질 향상  
- 연산 예산이 극히 제한된 엣지 단말 최적화(양자화·프루닝) 연구  

――  
[1] C. Tian et al., “Lightweight Image Super-Resolution with Enhanced CNN,” *arXiv* (2020).

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d098f69d-4702-4eab-9ae4-d63bae5db602/2007.04344v3.pdf
