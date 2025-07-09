# RegNet : Designing Network Design Spaces | Image classification

**주요 주장 및 기여**  
Facebook AI Research의 Radosavovic et al.(2020)는 개별 네트워크 구조가 아닌, 네트워크 *디자인 공간*(design space)을 직접 설계함으로써 모델 **모집단**의 일반적 설계 원칙을 도출할 것을 제안한다. 이들은 복잡도와 성능을 동시에 고려해 단순·규칙적 모델군 RegNet을 정립하였으며, 그 핵심은 “좋은 네트워크의 폭(width)과 깊이(depth)는 양자화된 선형 함수로 설명 가능하다”는 **정량적 설계 원칙**이다[1].

## 1. 해결 문제  
기존 수동 디자인과 NAS(neural architecture search)는  
- 수동 디자인: 단일 아키텍처 인스턴스 설계, 일반화 한계  
- NAS: 단일 검색 공간에서 최적 모델 탐색, 설계 원칙 부재  

두 패러다임은 **단일 모델**에 집중하므로, 다양한 연산 예산(regime)에 일관된 설계 원칙 부재 및 해석성 부족 문제를 가짐[1].

## 2. 제안 방법 및 수식

1. **디자인 공간 설계**  
   - 초기 디자인 공간 AnyNet(X): 4개 스테이지(stage), 각 스테이지마다  
     - 블록 수 $$d_i$$, 폭 $$w_i$$, bottleneck 비율 $$b_i$$, 그룹 폭 $$g_i$$  
   - 모델 복잡도(플롭스) 제어를 위해 $$n=500$$ 샘플링→10 에폭 간 훈련→오류 분포(EDF) 분석[1].

2. **연속적 공간 정제**  
   - 공유 bottleneck $$b_i=b$$ → AnyNetXB  
   - 공유 그룹 폭 $$g_i=g$$ → AnyNetXC  
   - 폭·깊이 증가 제약 $$w_{i+1}\ge w_i$$, $$d_{i+1}\ge d_i$$ → AnyNetXD/E  
   - 각 단계에서 EDF(오차 경험적 분포 함수) 개선 확인.

3. **양자화 선형 파라미터화(RegNet)**  
   - 블록 인덱스 $$j$$에 따른 예측 폭
   
  $$
       u_j = w_0 + w_a\cdot j,\quad 0\le j<d
  $$
   
   - 로그 스케일 양자화:
   
   $$
       s_j=\log_{w_m}\tfrac{u_j}{w_0},\quad w_j = w_0\cdot w_m^{\lfloor s_j\rceil}
   $$  
   - 파라미터 $$(d,w_0,w_a,w_m,b,g)$$ 6차원 설계 공간으로 축소, 크기 약 $$3\times10^8$$ → 고성능 모델 집중[1][2].

## 3. 모델 구조  
- **Stem**: $$3\times3$$ 합성곱(채널 32),  
- **Body**: 4단계(stage), 각 단계가 $$d_i$$개의 동일 블록(Residual bottleneck+그룹 합성곱)  
- **Head**: 평균 풀링+FC 출력  
- RegNet에서는 블록 종류 고정, 파라미터로 폭·깊이·양자화(step) 등만 변경.

## 4. 성능 향상 및 일반화  
- **검색 효율성**: 30개 랜덤 샘플만으로도 우수 모델 탐색 가능.  
- **일관된 성능**: 플롭스(200MF–32GF), 에폭(10→50), 스테이지(4→5), 블록 종류 변경에도 순위 유지[1][3].  
- **비교 우위**:  
  - Mobile(∼600MF)에서 RegNetY(24.5%↓)가 NAS·Man-design보다 경쟁적[Table 2].  
  - ResNe(X)t(3.2GF) 대비 RegNetX-3.2GF가 23.2%→21.7%[3.2GF][ResNet-50선택] 개선[Table 3a].  
  - EfficientNet 대비 동등 조건에서 5× 빠른 추론 속도, 0.2–12.8GF 전 구간에서 비슷하거나 낮은 오차[Figure 18, Table 4].

## 5. 한계 및 고려 사항  
- **구조만 최적화**: 연산자, 활성화 함수(Swish vs. ReLU), 해상도 스케일링 등은 고정. 일부 설정(해상도 224 고정)이 최적이며, 블록별 다른 연산자 적용은 추가 연구 필요[Figure 14].  
- **양자화·Pruning 미포함**: 모델 경량화 추가 방법(양자화·지식 증류)과의 결합 효과 미검증.  
- **하드웨어 의존성**: GPU 추론 속도 측정에 초점, 임베디드·모바일 하드웨어 일반화 추가 분석 필요.  

## 6. 향후 연구 방향 및 영향  
- **디자인 공간 확장**: 연산자(operator), 활성화 함수, 합성곱 형태 등 *더 풍부한* 디자인 공간 설계[결론 제언].  
- **하드웨어 특화 NAS**: RegNet 설계원칙을 기반으로 실제 임베디드 NPU·FPGA에 최적화된 검색 공간 정의.  
- **범용 설계 원칙**: 다양한 비전 태스크(분할·검출) 및 시계열 데이터에도 RegNet 원칙 적용 가능성 연구.  
- **자동화 고도화**: 자동 디자인 과정에 해석 가능한 통계 기법(EDF 등)을 결합, *인간-머신 협업* 설계 도구 개발.  

RegNet은 “네트워크 설계 공간을 직접 설계”하여 간단한 선형 규칙으로 모델 구조를 통일하고, 해석 가능한 범용 설계 원칙을 제시한 획기적 작업이다. 향후 NAS·수동 설계를 아우르는 **하이브리드 디자인 패러다임** 발전에 핵심적 영향이 기대된다.

[1] https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf
[2] https://huggingface.co/docs/transformers/en/model_doc/regnet
[3] https://ai.meta.com/research/publications/designing-network-design-spaces/
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0244df10-1e70-4eee-9ce3-ab6abb196c58/2003.13678v1.pdf
[5] https://www.sec.gov/Archives/edgar/data/1094324/000157587225000398/sify-20250331.htm
[6] https://www.sec.gov/Archives/edgar/data/1158324/000141057825000226/ccoi-20241231x10k.htm
[7] https://www.sec.gov/Archives/edgar/data/1821424/000121390025036276/ea0238772-20f_ucommune.htm
[8] https://www.sec.gov/Archives/edgar/data/1158324/000141057825001105/ccoi-20250331x10q.htm
[9] https://www.sec.gov/Archives/edgar/data/1066119/000162828025008767/viv-20241231.htm
[10] https://www.sec.gov/Archives/edgar/data/1528849/000155837025004329/rh-20250201x10k.htm
[11] https://www.sec.gov/Archives/edgar/data/1876183/000155837025003172/tmb-20241231x20f.htm
[12] https://www.semanticscholar.org/paper/8a975596be1a9c0b4a494749c06ed2da0872e4cc
[13] https://ieeexplore.ieee.org/document/9156494/
[14] https://ieeexplore.ieee.org/document/9010853/
[15] https://ieeexplore.ieee.org/document/9906585/
[16] https://link.springer.com/10.1007/978-3-030-88520-5_9
[17] https://dl.acm.org/doi/10.1145/3459637.3481944
[18] https://arxiv.org/abs/2401.14652
[19] https://onlinelibrary.wiley.com/doi/10.1002/mmce.23462
[20] https://arxiv.org/abs/2003.13678
[21] https://cocopambag.tistory.com/47
[22] https://arxiv.org/abs/2101.00590
[23] https://www.slideshare.net/slideshow/designing-network-design-spaces/239104133
[24] https://cdn.who.int/media/docs/default-source/wash-documents/regnet/regnet-flyer_en.pdf?sfvrsn=b47087b2_4&download=true
[25] https://yunmorning.tistory.com/36
[26] https://github.com/iVishalr/RegNetX
[27] https://cdn.who.int/media/docs/default-source/wash-documents/regnet/regnet_brochure__en.pdf?sfvrsn=ebfd4192_7
[28] https://freddiekim.tistory.com/28
[29] https://jackyoon5737.tistory.com/245
[30] https://regnet.anu.edu.au/about-us-1
[31] https://gaussian37.github.io/dl-concept-regnet/
[32] https://regnet.anu.edu.au
[33] https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings/IDETC-CIE2024/88377/V03BT03A009/1208870
[34] https://www.mdpi.com/2071-1050/14/11/6877
[35] https://arxiv.org/pdf/2003.13678.pdf
[36] https://arxiv.org/html/2503.21297
[37] https://arxiv.org/pdf/1512.03770v2.pdf
[38] http://arxiv.org/pdf/2407.18502.pdf
[39] https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2014.0881
[40] http://arxiv.org/pdf/2501.09954.pdf
[41] https://journals.sagepub.com/doi/pdf/10.1177/21582440221091248
[42] http://arxiv.org/pdf/2110.03760.pdf
[43] https://arxiv.org/pdf/2107.01101.pdf
[44] https://arxiv.org/html/2502.16228v1
[45] https://github.com/mberkay0/cnn-registration-with-regnet
[46] https://academic.oup.com/bioinformatics/article/34/2/308/4100161
