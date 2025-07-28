# The Nonsubsampled Contourlet Transform: Theory, Design, and Applications

## 1. 핵심 주장 및 주요 기여  
이 논문[1]은 **이 컨투어렛 변환(contourlet transform)**의 비한차 샘플링(non-subsampled) 버전인 **NSCT (Nonsubsampled Contourlet Transform)** 를 제안한다.  
- 완전한 **이동 불변성(shift-invariance)**, **다중 해상도(multiscale)** 및 **다방향성(multidirectionality)** 을 동시에 만족시키는 과잉급수(redundant) 변환을 설계  
- 기존 컨투어렛 변환의 다운/업샘플러 제거를 통해 **아 트루(à trous) 알고리즘**기반의 효율적 필터 구현 제시  
- 2채널 비한차 샘플링 필터 뱅크(NSFB) 설계 문제를 1차원 필터 매핑(mapping)을 통해 단순화  
- **완전 재구성(perfect reconstruction)**, **좋은 주파수 선택성**, **프레임(frame) 근접 타이트성**, **선형상 위상(linear phase)**, **정규성(regularity)** 을 모두 만족하는 필터 디자인  
- 도입된 NSCT가 향상된 **이미지 디노이징(denoising)** 및 **영상 개선(enhancement)** 성능을 입증  

## 2. 문제 정의  
- **기존 파형(wavelet)** 계열 변환은 에지·곡선 구조를 효과적으로 포착하지 못하고 이동에 민감(shift-variant)  
- **컨투어렛 변환**은 다방향·다중 해상도 특성을 가짐에도 다운샘플링으로 인해 이동 불변성이 결여되고, 2-D 필터 설계가 복잡  

## 3. 제안 방법  
1) **비한차 샘플링 피라미드(NSP)**  
   - Laplacian 피라미드의 다운/업샘플러 제거 → 아 트루 필터링으로 multiscale 분해  
   - 각 스케일마다 하나의 밴드패스 생성 → redundancy = *(L+1)* (L은 분해 수준)  

2) **비한차 방향 필터 뱅크(NSDFB)**  
   - DFB(tree-structured directional filter bank)의 다운/업샘플러 제거 및 필터 업샘플링  
   - 스케일별 방향 해상도 조절(고차 스케일에 ↑방향 수)  

3) **2채널 비한차 샘플링 필터 뱅크(NSFB)**  
   - 분석/합성 필터 {H₀,H₁;G₀,G₁}가 Bézout 항등식  
     $$H₀(z)G₀(z)+H₁(z)G₁(z)=1$$[1]  
   - 1차원 필터를 매핑( McClellan mapping )하여 2-D 필터 생성  
   - Bézout 조건 자동 충족, 선형 위상 및 원하는 주파수 응답 제어  

4) **필터 설계 예시**  
   - 1차원 프로토타입 H₀(x),H₁(x) 설계 → ladder/리프팅 구조로 분해(식(6)) → 2-D 매핑  
   - maximally-flat 매핑 다항식(식(11), Table II) 적용 → line zeros 및 flat response 보장  
   - 설계된 필터는 거의 tight frame(예: frame bounds ≈1.03), regular scaling function 확보[Table I]  

## 4. 성능 향상  
- **디노이징 (AWGN 제거)**  
  - NSWT, Curvelet과 비교 시 하드 쓰레숄딩에서 PSNR 0.5–2 dB 향상[Fig. 10, Table III]  
  - 간단한 Local Adaptive Shrinkage 기법 적용 시 BLS-GSM(최첨단)와 유사 성능 달성  

- **영상 개선 (Enhancement)**  
  - NSWT 대비 textured 영역 약한 에지 강조 및 잡음 억제 개선  
  - DV(Detail Variance) 증가, BV(Background Variance) 과도 증가 억제[Table V, Fig. 12]  

## 5. 한계 및 일반화 성능  
- **중복도(redundancy)** 가 높아 실시간 응용엔 계산·메모리 부담  
  - 대안으로 critically-sampled DFB 연동 시 중복도 ↓ (0.5–0.8 dB 성능 저하)[Table IV]  
- **필터 매핑** 기반 설계가 대체로 매크로 주파수 대역엔 유리, 극단적 비등방 패턴엔 성능 제한 가능  
- **프레임 tightness** 완전 달성 불가 → IIR 합성필터 필요, FIR 근사 시 설계 복잡도 상승  
- **일반화(generalization)**  
  - 자연 이미지 외 의료·위성 영상 등 다양한 데이터에 대한 튜닝 필요  
  - 노이즈 모델(비가우시안)·손상 유형(압축, 블러) 대비 일반화 성능 검증 필요  

## 6. 향후 연구 영향 및 고려사항  
- **경량화/실시간 구현**: 필터 근사·리프팅 최적화, GPU/FPGA 가속  
- **비가우시안·적응 노이즈** 대응: 통계 모델링 결합 및 비선형 shrinkage 확장  
- **딥러닝 결합**: NSCT 기반 특징 맵을 신경망 입력으로 활용하여 일반화 성능 향상  
- **다중 모달 영상처리**: 의료·원격탐사 등 다양한 주파수 대역 구조에 대한 검증  
- **프레임 이론 심화**: tight frame 근사 오차 최소화 및 IIR→FIR 합성 개선 연구  

[1] A. L. da Cunha, J. Zhou, and M. N. Do, “The Nonsubsampled Contourlet Transform: Theory, Design, and Applications,” IEEE Trans. Image Process., vol. 15, no. 10, pp. 3089–3101, Oct. 2006.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/41e5cc13-e201-47d5-b693-4a3b168f0139/The-Nonsubsampled-Contourlet-Transform-Theory-Design-and-Applications.pdf

# II. 비한차 샘플링 컨투어렛 변환(NSCT) 및 필터 뱅크 이해하기  

NSCT는 “이동 불변성(shift-invariance”, “다중 해상도(multiscale)”, “다방향성(multidirectionality)”을 모두 만족하며 완전 재구성 가능한 과잉급수(oversampled) 변환입니다. 크게 두 블록으로 나뉩니다.  

1. **비한차 샘플링 피라미드(NSP)**  
   - Laplacian 피라미드에서 사용되는 다운/업샘플러를 제거하고, à trous 알고리즘(“구멍 뚫린 필터링”)에 따라 필터 계수를 희소하게(“구멍”을 뚫듯) 적용합니다.  
   - 각 스케일마다 하나의 저역통과(LL)와 하나의 고역통과(HP) 밴드를 생성하여, $$L$$단계 분해 시 중복도(redundancy) = $$L + 1$$을 가집니다.  
   - 1단계 필터를 설계한 뒤, 상위 스케일에서는 필터 계수를 업샘플링(삽입)함으로써 추가 설계 없이 다중 해상도 분해를 얻습니다.  

2. **비한차 샘플링 방향 필터 뱅크(NSDFB)**  
   - Bamberger–Smith의 방향 필터 뱅크(critically-sampled DFB)에서 역시 다운/업샘플러를 제거하고 각 필터를 업샘플링하여 트리 구조를 유지합니다.  
   - 2채널 NSFB(Non-Subsampled Filter Bank) 여러 개가 트리 형태로 연결되어, 원하는 만큼의 방향(예: 8방향, 16방향)을 스케일별로 지원합니다.  
   - 모든 채널에서 필터 설계 및 구현 복잡도는 동일하며, perfect reconstruction을 만족합니다.  

3. **NSCT 결합 구조**  
   - 먼저 NSP로 다중 해상도 분해를 수행한 뒤, 각 스케일의 고역 밴드만을 NSDFB로 추가 분해하여 방향 정보를 얻습니다.  
   - 조합 시, 고차 스케일(저주파역)에서는 NSDFB 필터의 “나쁜” 주파수 응답 영역과 겹칠 수 있어 aliasing이 발생하므로, 해당 스케일 필터만 추가로 업샘플링하여 aliasing을 방지합니다.  
   - 전체 NSCT의 중복도는  

$$
       \text{Redundancy} = \sum_{i=1}^{L} l_i 
     $$  

여기서 $$L$$은 피라미드 레벨 수, $$l_i$$는 각 스케일에서 방향 필터 뱅크의 채널 수입니다.  

4. **프레임 해석**  
   - NSCT는 에너지 보존과 역변환 안정성을 위해 프레임(frame) 확장으로 볼 수 있습니다.  
   - 각 NSFB(피라미드·팬 필터 뱅크)가 프레임 양 바운드를 $$[A,B]$$로 갖는다면, 전체 NSCT의 프레임 바운드는 $$A^2$$에서 $$B^2$$ 사이로 보장됩니다.  
   - Tight frame(바운드가 같음)에 가까울수록, 분석·합성 필터의 FIR 근사가 용이해집니다.  

# III. 필터 설계 및 구현  

NSCT의 성능과 효율성은 NSP와 NSDFB에서 쓰이는 2-채널 NSFB 설계에 달려 있습니다. 설계 목표는  
-  완전 재구성 (Bezout 항등식 만족)  
-  선형 상 위상(linear phase)  
-  좋은 주파수 선택성(sharp frequency selectivity)  
-  프레임 바운드가 tight에 근접  
-  구현 효율성 (lifting/ladder 구조, 1-D 필터링만으로 구현 가능)  

1. **매핑(mapping) 설계 기법**  
   - 2-D 필터 설계를 1-D 프로토타입 필터와 2-D 매핑 함수 $$f(z_1,z_2)$$로 분리.  
   - 1-D 프로토타입 $$\{h_0(x),\,h_1(x)\}$$이 Bezout 조건

$$
       h_0(x)\,g_0(x) + h_1(x)\,g_1(x)=1
     $$  

을 만족하면, 임의의 2-D 다항식 $$f$$에 대해  

$$
       H_i(z_1,z_2)=h_i\bigl(f(z_1,z_2)\bigr),
       \quad G_i(z_1,z_2)=g_i\bigl(f(z_1,z_2)\bigr)
     $$  

가 동일한 완전 재구성을 보장합니다.  

2. **Maximally-Flat 매핑 함수**  
   - 피라미드용 매핑: $$(1+w)^\alpha (1-w)^\beta $$ 형태로, $$w=f(z)$$ 에 대해 $$w=\tfrac12(z+z^{-1})$$와 같은 line-zero(선형 영점) 삽입 가능.  
   - 팬 필터용 매핑: 피라미드 매핑에 주파수 변수 이동(예: $$z_2\to -z_2$$)을 추가하여 다이아몬드→팬 형태 주파수 응답 확보.  
   - 표(Table II)에 주요 매핑 함수 계수 예시 제시.  

3. **Lifting/Ladder 구조 구현**  
   - 1-D 프로토타입 필터를 Euclidean 알고리즘으로 계단식(“ladder”)으로 분해:  

$$
       \begin{bmatrix} H_0 \\ H_1 \end{bmatrix}
       = 
       \underbrace{\begin{bmatrix} 1 & S_1 \\ 0 & 1 \end{bmatrix}
                     \begin{bmatrix} 1 & 0 \\ S_0 & 1 \end{bmatrix}}_{\text{ladder stages}}
       \begin{bmatrix} 1 & 0 \\ 0 & x^{-n} \end{bmatrix}
       \begin{bmatrix} G_0 \\ G_1 \end{bmatrix}
     $$  

- 매핑 함수가 모노미얼 형태(지수 단위 계수)일 경우, 2-D 업샘플링 후에도 1-D 필터링만으로 구현 가능.  

4. **설계 예시**  
   -  **Example 1** (피라미드 필터):  
     – 1-D 프로토타입 $$h_0,\,h_1$$을 거의 동일하게 설계해 tight frame에 근접  
     – maximally-flat 매핑으로 line-zero 4차 삽입  
     – 최종 2-D 필터 지원 크기: $$13\times13$$, $$19\times19$$  
     – 프레임 바운드 ≈1.03 → tight frame 근접  
   -  **Example 2** (팬 필터):  
     – Example 1 프로토타입과 Table II 매핑 함수를 이용, diamond→fan 변환  
     – 지원 크기 $$21\times21$$, $$31\times31$$  

5. **기능적 고려사항**  
   - **정규성(regularity)**: scaling 함수 연속성 보장 위해 매핑 함수의 영점 차수와 프로토타입 차수를 조절  
   - **계산 복잡도**: lifting 구조 채택 시 direct form 대비 곱셈 횟수 절반 이하, monomial 매핑 시 전 단계 1-D 연산만으로 가능  
   - **중복도 vs. 성능**: 중복도를 낮추려면 NSP에만 NSDFB 대신 critically-sampled DFB 사용 가능하나 PSNR 0.5–0.8 dB 손실 발생  

이처럼 NSCT의 II장·III장은 **비한차 샘플링 구조**를 이용해 “완전 이동 불변”이면서 “다중 해상도·다방향” 특성을 갖춘 컨투어렛 변환을 제시하고, **매핑+lifting** 기법으로 2-D 필터를 효율적으로 설계·구현하는 과정을 구체적으로 다루고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/41e5cc13-e201-47d5-b693-4a3b168f0139/The-Nonsubsampled-Contourlet-Transform-Theory-Design-and-Applications.pdf
