# Instant-NGP : Instant Neural Graphics Primitives with a Multiresolution Hash Encoding | 3D reconstruction

## 핵심 주장과 주요 기여

**Instant Neural Graphics Primitives (NGP)**는 2022년 SIGGRAPH에서 발표된 혁신적인 논문으로, 기존 신경망 그래픽 프리미티브의 훈련 및 평가 비용을 획기적으로 줄이는 **multiresolution hash encoding** 기법을 제안했습니다[1]. 

### 주요 기여점

**1. 혁신적인 입력 인코딩**: 품질 손실 없이 더 작은 네트워크 사용을 가능하게 하는 다목적 입력 인코딩 방법론을 개발했습니다[1].

**2. 병렬화 친화적 구조**: 현대 GPU에서 병렬화하기 쉬운 단순한 아키텍처를 통해 해시 충돌을 신경망이 스스로 해결하도록 설계했습니다[1].

**3. 대폭적인 성능 향상**: 수십 배의 속도 향상을 달성하여 고품질 신경 그래픽 프리미티브를 몇 초 만에 훈련하고, 1920×1080 해상도에서 수십 밀리초 내 렌더링을 가능하게 했습니다[1].

## 해결하고자 하는 문제

### 기존 방법론의 한계
기존의 완전 연결 신경망으로 매개변수화된 신경 그래픽 프리미티브는 다음과 같은 문제점을 가지고 있었습니다:

- **높은 계산 비용**: 훈련과 평가에 막대한 비용 소모[1]
- **복잡한 데이터 구조**: 휴리스틱과 구조적 수정(pruning, splitting, merging)에 의존[1]
- **작업별 특화**: 특정 작업에만 제한되거나 GPU 성능을 제한하는 문제[1]

## 제안하는 방법론: Multiresolution Hash Encoding

### 핵심 수식과 원리

**1. 레벨별 해상도 결정**[2]:

$$ N_l := \lfloor N_{min} \cdot b^l \rfloor $$

여기서 성장 인자 $$b$$는:

$$ b := \exp\left(\frac{\ln N_{max} - \ln N_{min}}{L-1}\right) $$

**2. 공간 해시 함수**[2]:

$$ h(x) = \left(\bigoplus_{i=1}^{d} x_i \pi_i\right) \bmod T $$

여기서 ⊕는 비트 단위 XOR 연산이며, $$\pi_i$$는 고유한 큰 소수들입니다.

### 알고리즘 구조

**1단계**: 입력 좌표 x를 L개의 해상도 레벨에서 처리[2]
- 거친 해상도: 1:1 매핑 (충돌 없음)
- 세밀한 해상도: 해시 테이블 사용 (의도적 충돌 허용)

**2단계**: 각 레벨에서 특징 벡터를 d-선형 보간[2]

**3단계**: 모든 레벨의 결과를 연결하여 신경망에 입력[2]

## 모델 구조

### 하이퍼파라미터 설정[2]
| 파라미터 | 기호 | 값 |
|---------|------|-----|
| 레벨 수 | L | 16 |
| 레벨당 최대 엔트리 | T | 2¹⁴ ~ 2²⁴ |
| 특징 차원 수 | F | 2 |
| 최소 해상도 | N_min | 16 |
| 최대 해상도 | N_max | 512 ~ 524,288 |

### 네트워크 아키텍처
- **일반 작업**: 2개의 은닉층, 각각 64개 뉴런[2]
- **NeRF**: 밀도 MLP(1개 은닉층) + 색상 MLP(2개 은닉층)[2]
- **활성화 함수**: ReLU 사용[2]

## 성능 향상

### 정량적 성과

**1. NeRF 비교**[2]:
- 훈련 시간: 기존 수 시간 → **5초~5분**
- PSNR: 기존 NeRF와 경쟁적 성능 유지
- 특히 15초 만에 mip-NeRF와 유사한 품질 달성

**2. 다양한 작업에서의 성능**[2]:
- **Gigapixel 이미지**: ACORN 대비 1000배 이상 빠른 훈련
- **SDF**: NGLOD와 유사한 품질, 더 빠른 속도
- **Neural Radiance Caching**: 147 FPS → 133 FPS로 소폭 감소하지만 품질 대폭 향상

### 메모리 효율성
- 기존 밀집 그리드 대비 **20배 적은 파라미터**로 유사한 품질 달성[1]
- 해시 테이블 크기 T를 통한 성능-메모리 트레이드오프 조절 가능[2]

## 한계점

### 기술적 한계

**1. 해시 충돌로 인한 아티팩트**[2]:
- 미세한 "grainy" 마이크로구조 발생
- 특히 SDF에서 표면 거칠음으로 나타남
- 무작위로 분산된 해시 충돌이 완전히 해결되지 않음

**2. 하드웨어 의존성**[2]:
- RTX 3090 GPU의 6MB L2 캐시 한계에서 성능 절벽 발생
- T > 2¹⁹일 때 성능 급격히 저하

**3. 작업별 한계**:
- 복잡한 시점 의존적 반사가 있는 장면에서 mip-NeRF보다 성능 저하[2]
- 생성형 설정에서의 적용 어려움 (특징들이 규칙적 패턴으로 배치되지 않음)[2]

## 일반화 성능 향상 가능성

### 온라인 적응성
multiresolution hash encoding은 **자동 적응 기능**을 제공합니다[2]:
- 훈련 중 입력 분포가 변경되면 더 세밀한 그리드 레벨에서 충돌이 감소
- 구조적 데이터 구조 유지보수 없이도 트리 기반 인코딩의 장점 상속
- Neural Radiance Caching에서 애니메이션 시점과 3D 콘텐츠에 지속적으로 적응

### 작업 독립적 특성
**단일 구현으로 다양한 작업 지원**[1]:
- Gigapixel 이미지, SDF, NRC, NeRF 등에 동일한 하이퍼파라미터 사용
- 오직 해시 테이블 크기만 조정하여 품질-성능 트레이드오프 제어

### 확장 가능성 연구 사례
최근 연구들이 Instant-NGP의 일반화 성능을 다양한 분야로 확장하고 있습니다:

- **의료 영역**: 방광내시경 3D 렌더링에 적용[3]
- **로보틱스**: 초음파/적외선 센서와 결합한 VIRUS-NeRF[4]
- **위성 영상**: SAT-NGP로 다중 날짜 위성 이미지 처리[5]
- **연속 학습**: C-NGP로 여러 씬을 하나의 모델에 점진적 인코딩[6]

## 향후 연구에 미치는 영향과 고려사항

### 긍정적 영향

**1. 연구 패러다임 변화**:
- **실시간 신경 렌더링** 시대 개막
- 수 시간의 반복 시간을 수 초로 단축하여 연구 효율성 대폭 향상[1]

**2. 새로운 응용 분야 개척**:
- VR/AR에서의 실시간 NeRF 렌더링[7]
- 실시간 경로 추적과 결합한 신경 방사 캐싱
- 모바일 및 엣지 디바이스에서의 신경 그래픽 응용

### 향후 연구 고려사항

**1. 해시 함수 최적화**[2]:
- 연속적 인덱싱 공식화를 통한 미분 가능한 해시 함수 개발
- 진화적 최적화 알고리즘을 활용한 이산 함수 공간 탐색
- 사전 훈련된 해시 함수의 전이 학습 가능성

**2. 아티팩트 해결 방안**[2]:
- 해시 테이블 룩업 필터링 기법 개발
- 손실 함수에 추가적인 평활성 사전 조건 부과
- 충돌 해결을 위한 명시적 방법론 연구

**3. 메모리 압축 기술**:
- Context-aware NeRF 압축 (CNC) 프레임워크[8]
- 해시 충돌과 점유 그리드를 활용한 컨텍스트 모델 정확도 향상

**4. 물리 정보 통합**:
- Physics-Informed Neural Networks (PINNs)와의 결합[9]
- 유한 차분법을 통한 불연속적 도함수 처리

### 일반화 성능 관련 주요 연구 방향

**1. 멀티 씬 학습**: 단일 모델로 여러 장면을 학습하는 연속 학습 프레임워크 개발[6]

**2. 도메인 적응**: 의료 영상, 위성 이미지, 로보틱스 등 특화 도메인으로의 확장[5][4][3]

**3. 하이브리드 접근**: 전통적 컴퓨터 비전 기법과의 결합을 통한 강건성 향상[10]

Instant-NGP는 신경 그래픽 분야에 패러다임 변화를 가져왔으며, 실시간 고품질 렌더링의 새로운 기준을 제시했습니다. 해시 충돌과 같은 기술적 한계가 있지만, 속도와 품질의 혁신적 균형을 통해 다양한 분야로의 확장 가능성을 보여주었습니다. 향후 연구는 이러한 한계를 극복하면서도 일반화 성능을 더욱 향상시키는 방향으로 진행될 것으로 예상됩니다.

[1] https://dl.acm.org/doi/10.1145/3528223.3530127
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f45bed94-2ee0-4824-8333-33b98af1e836/2201.05989v2.pdf
[3] http://www.auajournals.org/doi/10.1097/01.JU.0001008580.58088.27.06
[4] https://ieeexplore.ieee.org/document/10802852/
[5] https://ieeexplore.ieee.org/document/10641775/
[6] https://www.semanticscholar.org/paper/3862e2808bc57225937e17bba2a5611e9aca5682
[7] https://ieeexplore.ieee.org/document/10108686/
[8] https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf
[9] https://arxiv.org/abs/2302.13397
[10] https://ieeexplore.ieee.org/document/10161117/
[11] https://ieeexplore.ieee.org/document/10675956/
[12] https://www.mdpi.com/1424-8220/23/14/6245
[13] http://www.auajournals.org/doi/10.1097/01.JU.0001008580.58088.27.07
[14] http://www.auajournals.org/doi/10.1097/01.JU.0001008580.58088.27.08
[15] http://papers.cumincad.org/cgi-bin/works/paper/caadria2024_248
[16] https://arxiv.org/abs/2201.05989
[17] https://arxiv.org/pdf/2306.04166.pdf
[18] http://arxiv.org/pdf/2309.12642.pdf
[19] https://arxiv.org/abs/2302.14683v2
[20] https://arxiv.org/html/2407.10482v1
[21] https://arxiv.org/html/2312.17241v1
[22] http://arxiv.org/pdf/2303.16884.pdf
[23] https://journals.iucr.org/paper?S2059798325002025
[24] https://arxiv.org/pdf/2212.05231.pdf
[25] http://arxiv.org/pdf/2207.11620.pdf
[26] https://nvlabs.github.io/instant-ngp/
[27] https://www.themoonlight.io/en/review/a-new-perspective-to-understanding-multi-resolution-hash-encoding-for-neural-fields
[28] https://velog.io/@rlaalsthf02/Instant-Neural-Graphics-Primitives-with-a-Multiresolution-Hash-Encoding
[29] https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding
[30] https://dippingtodeepening.tistory.com/141
[31] https://yai-yonsei.tistory.com/43
[32] https://github.com/NVlabs/instant-ngp
[33] https://velog.io/@damab/%EB%85%BC%EB%AC%B8-Instant-Neural-Graphics-Primitives
[34] https://openaccess.thecvf.com/content/CVPR2024W/VISOD/papers/Liu_BAA-NGP_Bundle-Adjusting_Accelerated_Neural_Graphics_Primitives_CVPRW_2024_paper.pdf
[35] https://kyujinpy.tistory.com/85
[36] https://jungsoo-ai-study.tistory.com/31
[37] https://hsejun07.tistory.com/319
[38] https://dl.acm.org/doi/10.1145/3610548.3618167
[39] https://jaeyeol816.github.io/neural_representation/ingp-about-instant-ngp/
[40] https://blog.outta.ai/258
[41] https://www.youtube.com/watch?v=EYWHjrW-Xoo
[42] https://dl.acm.org/doi/10.1145/3658155
[43] https://arxiv.org/abs/2402.05568
[44] https://ieeexplore.ieee.org/document/10611272/
[45] https://linkinghub.elsevier.com/retrieve/pii/S0143816624001933
[46] http://arxiv.org/pdf/2401.16318.pdf
[47] http://arxiv.org/pdf/2402.05916.pdf
[48] https://arxiv.org/pdf/2403.12143.pdf
[49] https://pmc.ncbi.nlm.nih.gov/articles/PMC7971515/
[50] https://www.frontiersin.org/articles/10.3389/frai.2021.618372/pdf
[51] https://arxiv.org/pdf/1806.07572.pdf
[52] https://arxiv.org/pdf/2108.06530.pdf
[53] https://arxiv.org/pdf/2209.14863.pdf
[54] http://arxiv.org/pdf/2209.01610.pdf
[55] https://openreview.net/pdf?id=8nXkyH2_s6
[56] https://arxiv.org/html/2505.03042v1
[57] https://thenumb.at/Neural-Graphics/
[58] https://arxiv.org/abs/2101.05490
[59] https://journals.iucr.org/paper?sor5003
[60] https://arxiv.org/html/2406.04101v1
[61] https://theaisummer.com/nerf/
[62] https://www.reddit.com/r/MachineLearning/comments/s5grvj/r_instant_neural_graphics_primitives_with_a/
[63] https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
[64] https://openaccess.thecvf.com/content/WACV2025/papers/Walker_Spatially-Adaptive_Hash_Encodings_for_Neural_Surface_Reconstruction_WACV_2025_paper.pdf
