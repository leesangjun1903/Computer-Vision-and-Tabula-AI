# 3DiM : Novel View Synthesis with Diffusion Models | Novel View Synthesis, 3D generation

## 핵심 주장과 주요 기여

**3DiM (3D Diffusion Models)**[1]은 단일 입력 이미지로부터 3D 일관성을 가진 새로운 시점의 이미지들을 생성하는 geometry-free 확산 모델을 제안합니다[2]. 이 연구의 **핵심 주장**은 기존의 명시적인 3D 기하학적 표현 없이도 확산 모델을 통해 고품질의 새로운 시점 합성이 가능하다는 것입니다.

**주요 기여**는 다음과 같습니다:

1. **Stochastic Conditioning 샘플링 알고리즘**: 3D 일관성을 향상시키는 새로운 조건부 샘플링 기법[2][3]
2. **X-UNet 아키텍처**: 새로운 시점 합성에 특화된 개선된 UNet 구조[4][5]
3. **3D Consistency Scoring**: 기하학 정보 없이 성을 정량적으로 평가하는 새로운 방법론[6]
4. **Geometry-free 접근법**: 복잡한 3D 표현이나 테스트 시간 최적화 없이 단일 모델로 다양한 장면에 확장 가능[7]

## 해결하고자 하는 문제와 제안하는 방법

### 문제 정의

기존 Novel View Synthesis 방법들의 한계:
- **NeRF 기반 방법들**: 장면별 개별 훈련이 필요하고, 적은 수의 입력 이미지로는 품질이 떨어짐[1]
- **회귀 기반 모델들**: 다양한 가능성을 모델링하지 못하고 흐릿한 결과 생성[1]
- **기존 확산 모델들**: 3D 일관성 부족으로 서로 다른 시점에서 일관되지 않은 결과 생성[1]

### 제안하는 방법

#### 1. Pose-conditional Image-to-Image 확산 모델

기본 목적 함수는 다음과 같습니다:

$$ L = E_{q(x_1,x_2)} E_{\lambda,\epsilon} \|\epsilon_\theta(z_2^{(\lambda)}, x_1, \lambda, p_1, p_2) - \epsilon\|_2^2 $$

여기서:
- $$z_2^{(\lambda)} = \sigma(\lambda)^{1/2} x_2 + \sigma(-\lambda)^{1/2} \epsilon$$ (잡음이 추가된 이미지)
- $$x_1$$은 조건부 뷰, $$x_2$$는 타겟 뷰
- $$p_1, p_2$$는 각각의 카메라 포즈
- $$\epsilon_\theta$$는 잡음 예측 네트워크[1]

#### 2. Stochastic Conditioning 샘플링

전통적인 자기회귀 생성의 근사치로, 각 디노이징 단계에서 조건부 뷰를 무작위로 선택:

$$ \hat{x}\_{k+1} = \frac{1}{\sigma(\lambda_t)^{1/2}} \left( z_{k+1}^{(\lambda_t)} - \sigma(-\lambda_t)^{1/2} \epsilon_\theta(z_{k+1}^{(\lambda_t)}, x_i) \right) $$

여기서 $$i \sim \text{Uniform}(\{1, ..., k\})$$는 각 단계에서 재샘플링됩니다[1][3].

## 모델 구조: X-UNet

### 주요 특징

1. **Weight Sharing**: 두 입력 프레임 간 가중치 공유로 대칭성 활용[1][4]
2. **Cross-Attention**: 조건부 뷰와 타겟 뷰 간의 정보 교환을 위한 교차 주의 메커니즘[5]
3. **Pose Encoding**: 카메라 레이를 통한 포즈 정보 인코딩[1]

### 아키텍처 개선점

기존 Concat-UNet 대비:
- **매개변수**: ~471M (X-UNet) vs ~421M (Concat-UNet)
- **성능**: PSNR 21.01 vs 17.21 (cars), FID 8.99 vs 21.54 (cars)[1]

## 성능 향상 및 실험 결과

### 정량적 성과

SRN ShapeNet 데이터셋에서의 결과:
- **FID 점수**: 8.99 (cars), 6.57 (chairs) - 기존 최고 성능 대비 현저한 개선[1]
- **Sharp 샘플 생성**: 확산 모델의 특성상 회귀 모델보다 선명한 결과[1]

### Ablation Study 결과

- **Stochastic Conditioning 제거 시**: 3D 일관성 현저히 저하[1]
- **Regression 모델**: PSNR/SSIM은 높지만 시각적 품질(흐림) 저하[1]
- **X-UNet vs Concat-UNet**: 모든 메트릭에서 X-UNet 우수[1]

## 일반화 성능 향상 가능성

### 확장성 장점

1. **Single Model Scalability**: 하나의 모델로 전체 ShapeNet 데이터셋 학습 가능[2][7]
2. **Geometry-free**: 명시적 3D 정보 없이도 작동하여 다양한 도메인에 적용 가능[8]
3. **Feed-forward Inference**: 테스트 시간 최적화 불필요[9]

### 한계점

1. **Out-of-distribution Poses**: 훈련 데이터와 다른 스케일의 포즈에서 성능 저하[1]
2. **실제 데이터 적용**: 합성 데이터로 훈련되어 실제 이미지에서는 성능 제한[10]
3. **카메라 매개변수 의존성**: 정확한 포즈 정보 필요[1]

## 3D Consistency Scoring 평가 방법론

기하학 정보 없는 모델 평가를 위한 새로운 메트릭:
1. 생성된 뷰들로 Neural Field 훈련
2. Hold-out 뷰에서 렌더링 품질 측정
3. 3D 일관성이 높을수록 Neural Field 성능 향상[1][6]

## 향후 연구에 미치는 영향

### 긍정적 영향

1. **Diffusion 기반 3D 생성의 새로운 패러다임**: 이후 SV3D[11], ViewCrafter[12] 등에 영감 제공
2. **Geometry-free 접근법 확산**: Scene Representation Transformer[9] 등과 함께 새로운 연구 방향 제시
3. **평가 방법론 개선**: 3D Consistency Scoring이 후속 연구의 표준 평가 방법으로 활용[13]

### 기술적 발전

- **Stochastic Conditioning**: 다양한 확산 기반 3D 생성 모델에서 채택[14][15]
- **Cross-attention Architecture**: 멀티뷰 생성 모델의 표준 구조로 발전[16]

## 향후 연구 시 고려사항

### 개선 필요 영역

1. **실제 데이터 적응**: 합성-실제 도메인 갭 해결[10][17]
2. **포즈 추정 통합**: 정확한 카메라 포즈 없이도 작동하는 시스템[18][19]
3. **효율성 개선**: 훈련 시간 단축 (Efficient-3DiM[20]에서 10일 → 1일로 단축)
4. **동적 장면 확장**: 정적 객체에서 동적 장면으로 확장[21][22]

### 응용 분야 확장

1. **VR/AR 콘텐츠 생성**: 몰입형 경험을 위한 실시간 렌더링[12]
2. **로보틱스**: 인식 및 상호작용을 위한 3D 이해[22]
3. **자율주행**: 차량 에셋 생성 및 시뮬레이션[10]

3DiM은 확산 모델을 3D 비전에 성공적으로 도입한 선구적 연구로, geometry-free 접근법의 가능성을 입증하고 후속 연구의 토대를 마련했습니다. 특히 stochastic conditioning과 3D consistency scoring은 현재까지도 관련 연구에서 널리 활용되고 있어, 이 분야의 중요한 이정표로 평가됩니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/257e0cf5-2093-44f7-97b3-e5194911fe54/2210.04628v1.pdf
[2] https://arxiv.org/abs/2210.04628
[3] https://3d-diffusion.github.io
[4] https://arxiv.org/pdf/2210.04628.pdf
[5] https://www.emergentmind.com/articles/2210.04628
[6] https://openreview.net/pdf?id=HtoA0oT30jC
[7] https://openreview.net/forum?id=HtoA0oT30jC
[8] https://compvis.github.io/geometry-free-view-synthesis/
[9] https://ieeexplore.ieee.org/document/9878781/
[10] https://arxiv.org/abs/2412.14494
[11] https://arxiv.org/abs/2403.12008
[12] https://arxiv.org/abs/2409.02048
[13] https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_ConsistDreamer_3D-Consistent_2D_Diffusion_for_High-Fidelity_Scene_Editing_CVPR_2024_paper.pdf
[14] https://ieeexplore.ieee.org/document/10378576/
[15] https://huggingface.co/papers/2304.02602
[16] https://arxiv.org/abs/2408.14211
[17] https://pure.kaist.ac.kr/en/publications/let-2d-diffusion-model-know-3d-consistency-for-robust-text-to-3d-
[18] https://ieeexplore.ieee.org/document/10654914/
[19] https://arxiv.org/abs/2312.07246
[20] https://arxiv.org/abs/2310.03015
[21] https://dl.acm.org/doi/10.1145/3680528.3687681
[22] https://arxiv.org/abs/2405.14868
[23] https://ieeexplore.ieee.org/document/10376843/
[24] https://openreview.net/forum?id=gVbPYihQag
[25] https://openaccess.thecvf.com/content/CVPR2024W/CV4MR/papers/Kohler_fMPI_Fast_Novel_View_Synthesis_in_the_Wild_with_Layered_CVPRW_2024_paper.pdf
[26] https://dataphoenix.info/novel-view-synthesis-with-diffusion-models/
[27] https://arxiv.org/html/2402.16506v2
[28] https://huggingface.co/learn/computer-vision-course/unit8/3d-vision/nvs
[29] https://www.columbia.edu/~wt2319/CDG.pdf
[30] https://pure.korea.ac.kr/en/publications/stochastic-conditional-diffusion-models-for-robust-semantic-image
[31] https://www.alphaxiv.org/overview/2210.04628v1
[32] https://arxiv.org/abs/2402.16506
[33] https://arxiv.org/html/2309.11525v3
[34] https://www.themoonlight.io/ko/review/stochastic-conditional-diffusion-models-for-robust-semantic-image-synthesis
[35] https://arxiv.org/abs/2410.22817
[36] https://ieeexplore.ieee.org/document/10657381/
[37] https://arxiv.org/abs/2301.04650
[38] https://openreview.net/forum?id=iO6tcLJEwA
[39] https://openaccess.thecvf.com/content/WACV2024/papers/Lee_PoseDiff_Pose-Conditioned_Multimodal_Diffusion_Model_for_Unbounded_Scene_Synthesis_From_WACV_2024_paper.pdf
[40] https://openaccess.thecvf.com/content/CVPR2022/papers/Sajjadi_Scene_Representation_Transformer_Geometry-Free_Novel_View_Synthesis_Through_Set-Latent_Scene_CVPR_2022_paper.pdf
[41] https://openreview.net/forum?id=Wf0OI5bpGu
[42] https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Consistent3D_Towards_Consistent_High-Fidelity_Text-to-3D_Generation_with_Deterministic_Sampling_Prior_CVPR_2024_paper.pdf
[43] https://ivpg.github.io/humanLDM/
[44] https://ku-cvlab.github.io/3DFuse/
[45] https://arxiv.org/abs/2411.12872
[46] https://openaccess.thecvf.com/content/ICCV2021/papers/Rombach_Geometry-Free_View_Synthesis_Transformers_and_No_3D_Priors_ICCV_2021_paper.pdf
[47] https://arxiv.org/abs/2412.14531
[48] https://arxiv.org/abs/2303.07937
[49] https://grail.cs.washington.edu/projects/dreampose/
[50] https://arxiv.org/abs/2411.07765
[51] https://arxiv.org/abs/2402.02906
[52] https://arxiv.org/abs/2304.02602
[53] https://arxiv.org/html/2502.12752v1
[54] https://arxiv.org/pdf/2310.03015.pdf
[55] http://arxiv.org/pdf/2409.02048.pdf
[56] https://arxiv.org/pdf/2404.03652.pdf
[57] https://arxiv.org/pdf/2302.10109.pdf
[58] https://arxiv.org/html/2411.14384
[59] http://arxiv.org/pdf/2310.17994v2.pdf
[60] http://arxiv.org/pdf/2303.11328.pdf
[61] https://www.alphaxiv.org/ko/overview/2210.04628v1
[62] https://openaccess.thecvf.com/content/ICCV2023/papers/Chan_Generative_Novel_View_Synthesis_with_3D-Aware_Diffusion_Models_ICCV_2023_paper.pdf
[63] https://github.com/mlvlab/SCDM
[64] https://arxiv.org/abs/2407.08280
[65] https://arxiv.org/abs/2406.08920
[66] https://arxiv.org/abs/2210.01602
[67] https://arxiv.org/html/2411.16680
[68] http://arxiv.org/pdf/2209.14819.pdf
[69] http://arxiv.org/pdf/2406.09801.pdf
[70] https://arxiv.org/pdf/2103.15407.pdf
[71] https://arxiv.org/html/2312.07246v2
[72] https://arxiv.org/html/2410.04402v1
[73] https://arxiv.org/pdf/2312.04551.pdf
[74] https://arxiv.org/html/2312.02255
[75] https://papers.neurips.cc/paper_files/paper/2023/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf
