# HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models | Super resolution

## 핵심 주장과 주요 기여

HiDiffusion은 사전 훈련된 확산 모델을 **추가 훈련 없이** 고해상도 이미지 생성에 적용할 수 있는 혁신적인 프레임워크입니다[1]. 이 연구의 핵심 기여는 다음과 같습니다:

**주요 발견**: 고해상도 이미지를 직접 생성할 때 발생하는 **객체 중복 문제**가 U-Net 깊은 블록에서의 **특징 중복**에서 비롯된다는 것을 발견했습니다[1][2]. 또한 생성 시간 증가는 U-Net 상위 블록의 **자기 주의 메커니즘 중복성**에서 기인한다는 것을 확인했습니다[2].

**기술적 혁신**: 
- **Resolution-Aware U-Net (RAU-Net)**: 특징 맵 크기를 동적으로 조정하여 객체 중복을 해결
- **Modified Shifted Window Multi-head Self-Attention (MSW-MSA)**: 최적화된 윈도우 어텐션으로 계산량 감소[2][3]

**성능 개선**: 기존 방법 대비 **1.5-6배 빠른 추론 속도**를 달성하며, 최대 **4096×4096 해상도**까지 확장 가능합니다[1][2].## 해결하고자 하는 문제와 제안 방법### 문제 정의기존 확산 모델의 두 가지 주요 한계를 해결합니다:

1. **실행 가능성 문제**: 훈련 해상도보다 높은 해상도로 직접 생성할 때 **비합리적인 객체 중복**과 **설명할 수 없는 객체 겹침** 발생[1]
2. **효율성 문제**: 해상도 증가에 따른 **기하급수적인 생성 시간 증가**[1]

### 제안 방법
#### 1. Resolution-Aware U-Net (RAU-Net)

**핵심 아이디어**: 생성된 이미지의 구조가 U-Net 깊은 블록의 특징 맵과 높은 상관관계를 가지며, 특징 중복이 객체 중복을 야기한다는 관찰에 기반합니다[1].

**수학적 정의**:
- **Resolution-Aware Downsampler (RAD)**: 

  $$ \mathcal{RAD}(x, \alpha) = \mathcal{R}(\mathcal{C}_{3,1,2,1}(x), \alpha) $$

여기서 $$\mathcal{R}(\mathcal{C}\_{3,1,2,1}(x), \alpha) = \mathcal{C}_{3,p,\alpha,d}(x) $$

- **Resolution-Aware Upsampler (RAU)**: 

  $$\mathcal{RAU}(x, \beta) = \mathcal{C}_{3,1,1,1}(\text{interp}(x,\beta)) $$

여기서 α는 다운샘플링 팩터, β는 업샘플링 팩터입니다[1].

- **Switching Threshold**: 구조 생성(초기 단계)에서는 RAU-Net을 사용하고, 세부 사항 생성(후기 단계)에서는 바닐라 U-Net을 사용하는 임계값 T₁을 도입했습니다[1].

#### 2. Modified Shifted Window Multi-head Self-Attention (MSW-MSA)

**관찰**: 상위 블록의 전역 자기 주의 메커니즘이 **놀라운 지역성**을 보인다는 발견[1].

**수학적 정의**:

$$ y = \text{MSW-MSA}(x, w, s(t)) + x $$

여기서:
- w: 윈도우 크기 (H/2, W/2)
- s(t): 타임스텝에 따른 이동 보폭 함수[1]

**계산 복잡도 개선**:
- **기존 MSA**: $$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C $$
- **제안 W-MSA**: $$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC $$

여기서 M이 고정되어 있어 선형 복잡도를 달성합니다[1].

### 모델 구조
**통합 프레임워크**: HiDiffusion은 RAU-Net과 MSW-MSA를 결합하여 **tuning-free** 고해상도 이미지 생성을 위한 통합 프레임워크를 구성합니다[1].

## 성능 향상 및 실험 결과### 정량적 성능 개선**속도 향상**:
- SD 1.5: 1024×1024에서 **1.96배**, 2048×2048에서 **2.83배** 빠름
- SDXL: 2048×2048에서 **1.58배**, 4096×4096에서 **2.68배** 빠름[1]

**품질 지표**:
- **FID 점수**: 기존 방법 대비 현저한 개선 (예: SD 1.5 COCO에서 78.53 → 28.93)
- **CLIP Score**: 텍스트-이미지 정렬 향상[1]

### 정성적 개선

**객체 중복 해결**: 직접 추론에서 발생하는 객체 중복 문제를 효과적으로 해결하며, 더 풍부한 세부 사항을 가진 이미지 생성[1].

**다양한 종횡비 지원**: 고정된 정사각형 이미지뿐만 아니라 다양한 종횡비(예: 512×2048, 2048×4096)에서도 안정적인 성능 발휘[1].

### 비교 분석**기존 방법 대비 우위**:
- **ScaleCrafter**: 계산량 감소와 품질 향상 동시 달성
- **DemoFusion**: 4-6배 빠른 속도로 비슷하거나 더 나은 품질
- **초해상도 모델**: 단일 단계 생성으로 더 풍부한 세부 사항 제공[1]

## 한계 및 제약사항
### 기술적 한계

**모델 의존성**: Stable Diffusion의 내재적 한계(예: 프롬프트 엔지니어링 필요)가 여전히 존재합니다[1].

**정보 손실**: RAU-Net의 다운샘플링/업샘플링 과정에서 일정한 정보 손실이 발생하며, 이는 Switching Threshold로 부분적으로 완화됩니다[1].

### 적용 범위 제한

**특정 해상도 범위**: 논문에서 검증된 해상도는 주로 1024×1024부터 4096×4096까지이며, 더 극한의 해상도에서의 성능은 명확하지 않습니다[1].

## 일반화 성능 향상 가능성
### 모델 확장성

**범용적 적용**: HiDiffusion은 다양한 사전 훈련된 확산 모델(SD 1.5, SD 2.1, SDXL, SDXL Turbo)에 **추가 훈련 없이** 적용 가능합니다[1].

**아키텍처 독립성**: U-Net 기반 확산 모델이라면 모델 아키텍처에 관계없이 통합할 수 있는 범용성을 보입니다[1].

### 이론적 통찰

**특징 중복 해결**: 객체 중복의 근본 원인을 특징 중복으로 규명하고 이를 해결함으로써, 다른 생성 모델의 해상도 확장 문제에도 적용 가능한 일반적 원리를 제시합니다[1].

**지역성 활용**: 전역 어텐션의 지역성을 활용한 효율성 개선 방법은 다른 트랜스포머 기반 모델에도 적용 가능한 일반적 접근법입니다[4][5].

## 앞으로의 연구에 미치는 영향
### 이론적 기여

**확산 모델 확장성**: 사전 훈련된 저해상도 확산 모델이 **추가 훈련 없이** 고해상도 생성에 사용될 수 있다는 것을 입증하여, 확산 모델의 확장성 연구에 새로운 방향을 제시합니다[1].

**계산 효율성**: 어텐션 메커니즘의 지역성을 활용한 효율성 개선 방법은 다른 비전 트랜스포머 모델의 최적화 연구에 영향을 미칠 것입니다[4][6].

### 실용적 함의

**산업 응용**: 추가 훈련 비용 없이 기존 모델을 고해상도로 확장할 수 있어, 실제 산업 응용에서의 도입 장벽을 크게 낮췄습니다[1].

**리소스 효율성**: 계산 자원 절약을 통해 더 많은 연구자와 개발자가 고해상도 이미지 생성 연구에 참여할 수 있는 기회를 제공합니다[1].

## 앞으로 연구 시 고려할 점

### 기술적 발전 방향

**초해상도 모델과의 통합**: 논문에서 제안한 바와 같이 초해상도 모델과의 더 나은 통합 방법을 탐구하여 더 높은 해상도와 놀라운 이미지 생성 결과를 달성할 수 있습니다[1].

**다양한 생성 작업 확장**: 이미지 생성을 넘어 비디오 생성, 3D 생성 등 다른 생성 작업에의 적용 가능성을 탐구할 필요가 있습니다[7].

### 이론적 심화 연구

**일반화 이론**: 확산 모델의 일반화 성능에 대한 이론적 이해를 깊화하여, 언제 그리고 왜 모델이 훈련 해상도를 넘어 확장될 수 있는지에 대한 더 명확한 이론적 기반을 마련해야 합니다[8][9].

**최적화 전략**: 다양한 해상도와 종횡비에서 최적의 RAD/RAU 파라미터와 Switching Threshold를 자동으로 결정하는 적응적 방법론 개발이 필요합니다[1].

### 실용적 개선 방향

**메모리 효율성**: 더 높은 해상도에서의 메모리 사용량 최적화를 통해 제한된 하드웨어 자원에서도 활용 가능한 방법론 개발이 중요합니다[10][11].

**품질 보장**: 다양한 도메인과 스타일에서의 생성 품질 일관성을 보장하기 위한 강건성 연구가 필요합니다[12][13].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fa196fa1-3719-4c0f-914c-81d94549ab2f/2311.17528v2.pdf
[2] https://link.springer.com/10.1007/978-3-031-72983-6_9
[3] https://hidiffusion.github.io
[4] https://ieeexplore.ieee.org/document/10658428/
[5] https://arxiv.org/abs/2405.18428
[6] https://arxiv.org/abs/2406.08552
[7] https://arxiv.org/abs/2312.06662
[8] https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/
[9] https://openreview.net/forum?id=ANvmVS2Yr0
[10] https://openaccess.thecvf.com/content/ACCV2024/papers/Yin_UNet--_Memory-Efficient_and_Feature-Enhanced_Network_Architecture_based_on_U-Net_with_ACCV_2024_paper.pdf
[11] https://arxiv.org/html/2412.18276v1
[12] https://arxiv.org/abs/2409.02919
[13] https://arxiv.org/html/2409.02919
[14] https://arxiv.org/abs/2310.07702
[15] https://ieeexplore.ieee.org/document/10943773/
[16] https://arxiv.org/abs/2412.02099
[17] https://arxiv.org/abs/2503.18446
[18] https://arxiv.org/abs/2311.17528
[19] https://arxiv.org/abs/2402.10491
[20] https://arxiv.org/abs/2412.09626
[21] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06764.pdf
[22] https://pmc.ncbi.nlm.nih.gov/articles/PMC7785874/
[23] https://arxiv.org/pdf/2103.14030.pdf
[24] https://cheatsheet.md/stable-diffusion/hiDiffusion
[25] https://www.kibme.org/resources/journal/20241002113154238.pdf
[26] https://rahites.tistory.com/180
[27] https://pmc.ncbi.nlm.nih.gov/articles/PMC11076390/
[28] https://velog.io/@jiwon-km/CV-Study-WEEK04-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
[29] https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2020.605132/full
[30] https://stevenkim1217.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EC%A0%95%EB%A6%AC-Swin-Transformer-Hierarchical-Vision-Transformer-using-Shifted-Windows
[31] https://arxiv.org/html/2311.17528v2
[32] https://www.tandfonline.com/doi/abs/10.1080/01431161.2021.1986239
[33] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/swin-transformer/
[34] https://www.reddit.com/r/StableDiffusion/comments/1cbaxsu/introducing_hidiffusion_increase_the_resolution/
[35] https://ki-it.com/xml/37687/37687.pdf
[36] https://arxiv.org/abs/2404.01709
[37] https://arxiv.org/abs/2403.12470
[38] https://arxiv.org/abs/2404.04544
[39] http://arxiv.org/pdf/2408.11001.pdf
[40] http://www.arxiv.org/pdf/2402.10491.pdf
[41] https://arxiv.org/html/2412.02099v1
[42] https://huggingface.co/papers/2412.09626
[43] https://milvus.io/ai-quick-reference/how-do-diffusion-models-perform-on-highresolution-image-generation-tasks
[44] https://with-ahn-ssu.tistory.com/37
[45] https://eccv.ecva.net/virtual/2024/poster/1431
[46] https://arxiv.org/abs/2408.11001
[47] https://github.com/christianversloot/machine-learning-articles/blob/main/u-net-a-step-by-step-introduction.md
[48] https://openaccess.thecvf.com/content/WACV2025/papers/Wu_MegaFusion_Extend_Diffusion_Models_towards_Higher-Resolution_Image_Generation_without_Further_WACV_2025_paper.pdf
[49] https://arxiv.org/pdf/2311.17791.pdf
[50] https://github.com/ali-vilab/FreeScale
[51] https://openreview.net/forum?id=u48tHG5f66
[52] https://towardsdatascience.com/understanding-u-net-61276b10f360/
[53] https://www.themoonlight.io/ko/review/freescale-unleashing-the-resolution-of-diffusion-models-via-tuning-free-scale-fusion
[54] https://github.com/huggingface/diffusers/issues/593
[55] https://www.sciencedirect.com/science/article/abs/pii/S0933365724000423
[56] https://www.semanticscholar.org/paper/0cbc0f29b9b8cd13e23ecb50c2cf88b882a2e6d9
[57] https://arxiv.org/abs/2306.13776
[58] https://arxiv.org/abs/2408.02615
[59] https://arxiv.org/abs/2407.15886
[60] https://proceedings.neurips.cc/paper_files/paper/2024/file/0267925e3c276e79189251585b4100bf-Paper-Conference.pdf
[61] https://proceedings.mlr.press/v201/duman-keles23a/duman-keles23a.pdf
[62] https://arxiv.org/html/2504.21292v1
[63] https://arxiv.org/abs/2209.04881
[64] https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf
[65] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02391.pdf
[66] https://nyuscholars.nyu.edu/en/publications/on-the-computational-complexity-of-self-attention
[67] https://arxiv.org/html/2311.01797
[68] https://arxiv.org/html/2406.08552v1
[69] https://vds.sogang.ac.kr/wp-content/uploads/2023/02/2023-%EB%8F%99%EA%B3%84%EC%84%B8%EB%AF%B8%EB%82%98_%EC%9C%A0%ED%98%84%EC%9A%B0.pdf
[70] https://arxiv.org/html/2505.20123v1
[71] https://ojs.aaai.org/index.php/AAAI/article/view/26502/26274
[72] https://amber-chaeeunk.tistory.com/96
[73] https://milvus.io/ai-quick-reference/what-role-do-attention-mechanisms-play-in-diffusion-models
[74] https://dl.acm.org/doi/10.1145/3703187.3703224
[75] http://arxiv.org/pdf/2406.07792.pdf
[76] http://arxiv.org/pdf/2409.19589.pdf
[77] https://arxiv.org/html/2407.06079v1
[78] https://arxiv.org/html/2411.12072
[79] https://arxiv.org/html/2405.16759
[80] https://arxiv.org/html/2409.16488v1
[81] http://arxiv.org/pdf/2305.15357.pdf
[82] https://arxiv.org/html/2404.01709
[83] https://pajamacoder.tistory.com/18
[84] https://github.com/megvii-research/HiDiffusion
[85] https://www.mdpi.com/2306-5354/12/2/140
[86] https://arxiv.org/html/2503.18446v2
[87] http://arxiv.org/pdf/2410.13807.pdf
[88] http://arxiv.org/pdf/2405.17025.pdf
[89] http://arxiv.org/pdf/2501.01039.pdf
[90] https://arxiv.org/html/2502.06155v2
[91] https://arxiv.org/html/2405.18428v1
[92] https://arxiv.org/html/2406.08552
[93] https://arxiv.org/pdf/2401.05907.pdf
[94] http://arxiv.org/pdf/2409.04005.pdf
[95] http://arxiv.org/pdf/2407.01425.pdf
[96] https://arxiv.org/pdf/2203.15380.pdf
[97] https://stackoverflow.com/questions/65703260/computational-complexity-of-self-attention-in-the-transformer-model
[98] https://neurips.cc/virtual/2023/poster/70803
[99] https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Diffusion_Models_Without_Attention_CVPR_2024_paper.pdf
[100] https://www.reddit.com/r/LanguageTechnology/comments/9gulm9/complexity_of_transformer_attention_network/
