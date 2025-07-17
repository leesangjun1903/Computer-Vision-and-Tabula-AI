# Deep Learning on Image Denoising: An Overview

**주요 주장 및 기여**  
Deep Learning on Image Denoising: An Overview(2020)는 이미지 잡음 제거 분야에서 딥러닝 기법을 종합·분류·비교 분석한 최초의 체계적 리뷰이다.  
1. **딥러닝 프레임워크 정리**: CNN, GAN, 딥 신경망, 최적화 모델 등 주요 구조와 발전사 정리.  
2. **잡음 유형별 분류**:  
   - Additive White Noisy Images(가우시안·포아송 등)  
   - Real Noisy Images(실제 카메라 잡음)  
   - Blind Denoising(잡음 분포 미지)  
   - Hybrid Noisy Images(잡음+블러+저해상도)  
3. **성능 비교**: 공용 데이터셋(SET12, BSD68, CBSD68 등)에서 200편 이상의 방법을 질적·양적으로 비교.  
4. **미해결 과제 제시**: 일반화, 비지도·준지도 학습, 효율화, 실제 잡음 모델링 등 향후 연구 방향 제안.

## 1. 해결하고자 하는 문제  
- **다양한 잡음 유형**: 기존 최적화·통계 기반 기법은 단일 잡음(가우시안 등)에 최적화되어 실제 촬영 환경의 복합 잡음에 취약.  
- **수작업 매개변수 설정**: 전통 기법은 테스트 단계마다 최적의 파라미터를 수동 설정해야 함.  
- **대용량 데이터 필요**: 딥러닝 모델 학습·일반화를 위해 실제 잡음 데이터를 충분히 확보하기 어려움.

## 2. 제안하는 방법  
본 논문은 개별 모델을 제안하는 대신, 기존 문헌을 **분류·비교**하고 다음 구조별 특징을 분석한다.  

### 2.1 핵심 모델 구조  
1) **Convolutional Neural Networks (CNN)**  
   – 입력 $$y=x+\mu$$를 받아, 잔차 학습(residual learning)과 Batch Normalization 적용  
   – 기본 수식:  

$$
     \hat{x} = y - f_\theta(y)
$$  

2) **GAN 기반**  
   – Generator: denoising 네트워크  
   – Discriminator: 실제 vs 복원 결과 구분  
   – Loss:  

$$
     \min_G \max_D \,\mathbb{E}\_{x,y}[\log D(x)] + \mathbb{E}_y[\log(1-D(G(y)))]
$$  

4) **Optimization + CNN**  
   – Half-Quadratic Splitting(HQS), MAP 추정과 CNN 결합  
   – 예: HQS 내부에서 CNN으로 denoiser prior 학습  
5) **Multi-scale·Non-local·Attention 기법**  
   – dilated conv, non-local self-similarity, 채널·공간 주의(attention) 통합  

# Deep Learning Techniques in Image Denoising 

## 개요

Deep learning techniques in image denoising은 전통적인 이미지 잡음 제거 방법의 한계를 극복하기 위해 발전된 딥러닝 기반 접근법입니다[1][2][3]. 이 분야는 다양한 유형의 잡음에 대응하는 딥러닝 기법들을 체계적으로 분류하고 분석합니다.

## 주요 분류 체계

### 1. Additive White Noisy Images (AWNI) 처리 기법

#### 1.1 CNN/NN 기반 AWNI 잡음 제거
- **DnCNN (Denoising CNN)**: 잔차 학습(residual learning)과 배치 정규화(batch normalization)를 활용한 대표적인 모델[1]
- **딜레이티드 컨볼루션(Dilated Convolution)**: 수용 영역(receptive field)을 확장하여 더 많은 컨텍스트 정보를 포착[1]
- **멀티스케일 기법**: 서로 다른 해상도에서 특징을 추출하여 성능 향상[1]

#### 1.2 CNN/NN과 특징 추출 방법의 결합
- **웨이블릿 변환과 CNN**: 주파수 도메인에서의 잡음 제거 향상[1]
- **비국소 자기유사성(Non-local Self-similarity)**: 이미지 내 유사한 패치들을 활용한 잡음 제거[1]
- **주성분 분석(PCA)과 CNN**: 차원 축소를 통한 효율적인 잡음 제거[1]

#### 1.3 최적화 방법과 CNN/NN의 결합
- **MAP (Maximum A Posteriori) 추정**: 베이지안 접근법을 통한 잡음 추정[1]
- **전변분 정규화(Total Variation Regularization)**: 에지 정보 보존을 위한 정규화 기법[1]
- **반복 축소 임계값(Iterative Shrinkage-Thresholding)**: 희소 표현 기반 잡음 제거[1]

### 2. Real Noisy Images 처리 기법

#### 2.1 단일 End-to-End CNN 방법
- **네트워크 아키텍처 변경**: 실제 잡음 특성에 맞는 구조 설계[1]
- **멀티스케일 지식 활용**: 다양한 스케일에서의 특징 추출[1]
- **주의 메커니즘(Attention Mechanism)**: 중요한 특징에 집중하는 메커니즘[1]

#### 2.2 사전 지식과 CNN의 결합
- **카메라 파이프라인 시뮬레이션**: 실제 카메라 노이즈 특성 모델링[1]
- **반교사 학습(Semi-supervised Learning)**: 제한된 라벨 데이터로 학습[1]
- **GAN 기반 노이즈 모델링**: 생성적 적대 신경망을 통한 현실적인 노이즈 생성[4][5]

### 3. Blind Denoising 기법

#### 3.1 잡음 레벨 추정
- **FFDNet**: 잡음 레벨을 입력으로 받아 적응적 잡음 제거[1]
- **소프트 축소(Soft Shrinkage)**: 잡음 레벨에 따른 적응적 임계값 설정[1]
- **잔차 학습**: 복잡한 잡음에 대한 강건한 대응[1]

#### 3.2 무감독 학습 방법
- **Noise2Noise**: 깨끗한 이미지 없이 잡음 제거 학습[1]
- **Noise2Void**: 단일 잡음 이미지만으로 학습 가능[6]
- **Self-supervised Learning**: 자기 감독 학습을 통한 잡음 제거[1]

### 4. Hybrid Noisy Images 처리 기법

#### 4.1 다중 열화 처리
- **잡음 + 블러 + 저해상도**: 복합적인 이미지 열화 동시 처리[1]
- **캐스케이드 방법**: 단계적 처리를 통한 성능 향상[1]
- **플러그 앤 플레이 방법**: 모듈식 접근법으로 다양한 열화 처리[1]

#### 4.2 버스트 이미지 처리
- **다중 프레임 활용**: 시간적 정보를 활용한 잡음 제거[1]
- **커널 예측 네트워크**: 적응적 필터링을 통한 잡음 제거[1]
- **순환 신경망(RNN)**: 시퀀스 데이터의 시간적 의존성 활용[1]

## 핵심 기술 요소

### 1. 네트워크 구조 설계
- **잔차 연결(Residual Connection)**: 기울기 소실 문제 해결[1]
- **배치 정규화**: 학습 안정성 향상[1]
- **주의 메커니즘**: 중요한 특징에 집중[1]

### 2. 손실 함수 설계
- **지각 손실(Perceptual Loss)**: 인간의 시각 특성 반영[1]
- **적대적 손실(Adversarial Loss)**: 더 현실적인 결과 생성[1]
- **구조적 유사성(SSIM)**: 구조적 정보 보존[1]

### 3. 데이터 증강 기법
- **합성 데이터 생성**: 다양한 잡음 패턴 생성[1][4]
- **도메인 적응**: 합성 데이터를 실제 데이터에 적응[1]
- **메타 학습**: 새로운 잡음 유형에 빠른 적응[1]

## 성능 평가 및 벤치마크

### 1. 정량적 평가 지표
- **PSNR (Peak Signal-to-Noise Ratio)**: 신호 대 잡음 비율[1]
- **SSIM (Structural Similarity Index)**: 구조적 유사성[1]
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 학습된 지각 유사성[1]

### 2. 주요 데이터셋
- **합성 데이터**: BSD68, Set12, CBSD68[1]
- **실제 데이터**: SIDD, DND, CC[1]
- **의료 영상**: 저선량 CT, MRI[7][8]

## 최신 연구 동향

### 1. 자기 감독 학습 발전
- **Noise2Noise**: 깨끗한 이미지 없이 학습[1]
- **Noise2Void**: 단일 잡음 이미지 학습[6]
- **Neighbor2Neighbor**: 인접 픽셀을 활용한 학습[9]

### 2. 트랜스포머 기반 방법
- **주의 메커니즘**: 전역적 의존성 모델링[8]
- **멀티헤드 어텐션**: 다양한 특징 추출[8]
- **계층적 구조**: 효율적인 연산[8]

### 3. 확산 모델 적용
- **조건부 확산 모델**: 잡음 제거를 위한 확산 과정[10]
- **노이즈 스케줄링**: 단계별 잡음 제거[10]
- **샘플링 기법**: 효율적인 추론 과정[10]

## 실용적 응용 분야

### 1. 의료 영상
- **저선량 CT**: 방사선 노출 최소화[7]
- **MRI**: 스캔 시간 단축[7]
- **현미경 영상**: 실시간 관찰[11][12]

### 2. 산업 응용
- **품질 검사**: 제조업에서의 결함 탐지[13]
- **원격 감지**: 위성 영상 처리[1]
- **보안 시스템**: 감시 카메라 영상 개선[1]

### 3. 일상 응용
- **스마트폰 카메라**: 실시간 잡음 제거[1]
- **소셜 미디어**: 사진 품질 향상[1]
- **디지털 카메라**: 저조도 성능 개선[1]

## 향후 연구 방향

### 1. 효율성 개선
- **경량화 모델**: 모바일 환경에서의 실시간 처리[1]
- **하드웨어 최적화**: 특수 목적 하드웨어 활용[1]
- **지식 증류**: 큰 모델의 성능을 작은 모델로 전달[1]

### 2. 일반화 성능 향상
- **도메인 적응**: 다양한 환경에서의 강건성[1]
- **메타 학습**: 새로운 잡음 유형에 빠른 적응[1]
- **연속 학습**: 기존 지식 보존하며 새로운 학습[1]

### 3. 해석 가능성 향상
- **주의 시각화**: 모델의 결정 과정 이해[1]
- **특징 분석**: 학습된 특징의 의미 파악[1]
- **오류 분석**: 실패 사례 분석 및 개선[1]

딥러닝 기반 이미지 잡음 제거는 전통적인 방법의 한계를 극복하며 다양한 응용 분야에서 혁신적인 성과를 보여주고 있습니다. 특히 자기 감독 학습, 트랜스포머, 확산 모델 등의 최신 기법들이 통합되면서 더욱 강력하고 실용적인 솔루션들이 개발되고 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/48cdd7df-565b-4ddd-9a35-8c22141924f6/1912.13171v4.pdf
[2] https://arxiv.org/abs/1912.13171
[3] http://lab.semi.ac.cn/ailab/upload/files/2020/8/11115111487.pdf
[4] https://openaccess.thecvf.com/content/ACCV2020/papers/Tran_GAN-based_Noise_Model_for_Denoising_Real_Images_ACCV_2020_paper.pdf
[5] https://arxiv.org/abs/2204.02844
[6] http://dmqa.korea.ac.kr/activity/seminar/450
[7] https://academic.oup.com/bjr/article/97/1156/812/7609040
[8] https://www.degruyterbrill.com/document/doi/10.1515/mim-2024-0033/html
[9] https://paperswithcode.com/task/image-denoising
[10] https://www.ndt.net/search/docs.php3?id=31384
[11] https://academic.oup.com/mam/article/27/6/1431/6887994
[12] https://rodin.uca.es/bitstream/handle/10498/34709/Denoising.pdf?sequence=1&isAllowed=y
[13] https://iopscience.iop.org/article/10.1088/1361-6501/ac85d2
[14] https://link.springer.com/10.1007/s00500-022-07166-w
[15] https://link.springer.com/10.1007/978-3-030-31332-6
[16] https://www.semanticscholar.org/paper/b91c0756e13422d17e4c09e6e7f09ec1e4cd8aa5
[17] https://www.semanticscholar.org/paper/a0bb19709340bcce2b0d48bbdc07d8b67512c40e
[18] https://www.studypool.com/documents/46742369/image-denoising-paper
[19] https://pubmed.ncbi.nlm.nih.gov/32829002/
[20] https://fugumt.com/fugumt/paper_check/1912.13171v4
[21] https://www.sciencedirect.com/science/article/abs/pii/S0893608020302665
[22] https://search.proquest.com/openview/d4df47205b2d1ae56a00f82de958c9b1/1?pq-origsite=gscholar&cbl=33692
[23] https://velog.io/@jihyeheo/inmuck9denoising
[24] https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16833
[25] https://arxiv.org/abs/2301.03362
[26] https://arxiv.org/pdf/1912.13171.pdf
[27] http://arxiv.org/pdf/2106.09311.pdf
[28] https://arxiv.org/pdf/1810.05052.pdf
[29] https://arxiv.org/pdf/2209.09214.pdf
[30] http://arxiv.org/pdf/2103.09962.pdf
[31] https://arxiv.org/pdf/1801.06756.pdf
[32] https://arxiv.org/pdf/2304.01627.pdf
[33] http://arxiv.org/pdf/2409.05118.pdf
[34] https://ojs.istp-press.com/jait/article/download/101/107
[35] http://arxiv.org/pdf/2403.12382.pdf
[36] https://arxiv.org/pdf/1811.10980.pdf
[37] https://arxiv.org/pdf/2401.02831.pdf
[38] https://www.cs.toronto.edu/~lindell/teaching/2529/past_projects/2022/report/jasper_zhang-zhiling_zou.pdf
[39] https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.pdf
[40] https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf
[41] http://dmqa.korea.ac.kr/uploads/seminar/%5B240510%5DDMQA_OpenSeminar_Image_Denoising.pdf

### 2.2 모델 일반화 성능 향상 기법  
- **Residual / Recursive 구조**: 깊은 네트워크에서 기울기 소실 완화 및 로컬·글로벌 정보 결합  
- **Self-supervised 학습**: Noise2Noise, Noise2Self, Blind-spot Network로 실제 잡음 레이블 불필요  
- **데이터 합성 + Camera Pipeline 모방**: 실제 ISO 잡음 특성 재현한 합성 데이터로 훈련  
- **Domain Adaptation**: unpaired real-noisy→clean 매핑을 GAN으로 학습  

## 3. 성능 향상 및 한계  
| 잡음 유형       | 대표 모델               | PSNR 향상                                    | 한계                                    |
|----------------|--------------------------|-----------------------------------------------|----------------------------------------|
| AWGN           | DnCNN, FFDNet, MemNet    | +0.5–1 dB over BM3D                           | 고정 시그마 가정, 실제 잡음 불반영       |
| Real Noisy     | CBDNet, VDN, DRDN        | SIDD: 39.6 dB (DRDN)                          | 데이터셋 편중, 보기 힘든 잡음 분포       |
| Blind Denoising| FFDNet, ADNet            | BSD68: 29.2 dB @σ=25                          | 정확한 잡음 추정 없이 불안정한 결과      |
| Hybrid Noise   | WarpNet, DPSR            | VggFace2: +2 dB over DnCNN                     | 복합 모델 구조 과대/배포 어려움         |

**한계**  
- 과도한 네트워크 깊이·파라미터로 메모리·실행 속도 부담  
- 비지도·준지도 학습 안정화 미완  
- PSNR·SSIM 한계: 인간 지각 품질 반영 부족  

## 4. 향후 연구 영향 및 고려 사항  
- **일반화 강화**: Self-supervised·Meta-learning 기법으로 unseen 잡음 적응력 향상  
- **경량화 모델**: Neural Architecture Search, 지식 증류(Knowledge Distillation)로 실시간 적용  
- **실제 환경 데이터**: 다양한 기기·조명·ISO 조합의 대규모 캡처 데이터 필요  
- **지각 품질 평가**: Learned perceptual metrics(예: LPIPS) 기반 손실 함수 도입  
- **보안·사생활**: 영상 복원 과정에서 개인정보 노출 이슈 검토  

이 리뷰는 딥러닝 기반 이미지 복원 연구의 전반적 지형도를 제시하며, 새 모델 개발 시 “일반화·효율·실제성”을 핵심 목표로 삼도록 이끈다는 점에서 큰 영향을 미칠 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/48cdd7df-565b-4ddd-9a35-8c22141924f6/1912.13171v4.pdf
