# SUPIR : Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild | Super resolution

## **1. 핵심 주장과 주요 기여**

**SUPIR(Scaling-UP Image Restoration)**는 이미지 복원 분야에서 **모델 스케일링의 혁신적 활용**을 통해 실세계 이미지 복원 성능을 획기적으로 향상시킨 연구입니다[1][2]. 본 논문의 핵심 주장은 다음과 같습니다:

### **주요 기여**
- **최대 규모의 이미지 복원 모델**: 26억 개의 파라미터를 가진 SDXL(Stable Diffusion XL)을 기반으로 한 가장 큰 규모의 이미지 복원 모델 구축[1]
- **텍스트 프롬프트 기반 제어**: 텍스트 설명을 통해 복원 과정을 제어할 수 있는 최초의 대규모 이미지 복원 시스템[2]
- **대규모 학습 데이터**: 2천만 장의 고해상도 이미지와 텍스트 설명 쌍으로 구성된 대규모 데이터셋 구축[1]
- **Negative-Quality 프롬프트**: 품질 향상을 위한 부정적 품질 프롬프트 도입[2]

## **2. 해결하고자 하는 문제**

### **기존 이미지 복원 방법의 한계**
1. **제한된 생성 능력**: 기존 방법들은 특정 degradation에 대해서만 효과적이며, 실세계의 복잡한 degradation을 처리하는 데 한계가 있음[1]
2. **스케일링 문제**: 이미지 복원 분야에서 모델 스케일링이 제대로 탐구되지 않았으며, 대규모 모델의 잠재력이 충분히 활용되지 못함[2]
3. **지능성 부족**: 텍스트 기반 제어나 semantic understanding이 부족하여 사용자의 의도를 반영하기 어려움[1]

## **3. 제안하는 방법론**

### **3.1 모델 스케일링 (Model Scaling Up)**

**Degradation-Robust Encoder**
저품질 이미지를 latent space로 매핑하기 위해 기존 SDXL encoder를 fine-tuning합니다:

$$ L_E = \|D(E_{dr}(x_{LQ})) - D(E_{dr}(x_{GT}))\|_2^2 $$

여기서 $$E_{dr}$$는 degradation-robust encoder, $$D$$는 고정된 decoder, $$x_{GT}$$는 ground truth입니다[1].

**Large-Scale Adaptor Design**
기존 ControlNet의 한계를 극복하기 위해 새로운 adaptor를 설계했습니다:
- **Trimmed ControlNet**: 각 encoder block에서 ViT block의 절반을 제거하여 효율성 증대[2]
- **ZeroSFT Connector**: 픽셀 수준의 정밀한 제어를 위해 spatial feature transfer(SFT) 연산과 group normalization을 포함한 새로운 연결 방식[1]

### **3.2 복원 가이드 샘플링 (Restoration-Guided Sampling)**

생성 모델의 fidelity 문제를 해결하기 위해 EDM 샘플링 방법을 수정했습니다:

$$ z_{t-1} = \hat{z}\_{t-1} + k_t(z_{LQ} - \hat{z}_{t-1}) $$

여기서 $$k_t = (\sigma_t/\sigma_T)^{\tau_r}$$이며, $$\tau_r$$은 복원 가이드 강도를 제어하는 하이퍼파라미터입니다[2].

**Algorithm 1: Restoration-Guided Sampling**

```
Input: H, {σt}T_t=1, zLQ, c
For t ∈ {T, ..., 1}:
    kt ← (σt/σT)^τr
    ẑt-1 ← H(ẑt, zLQ, σ̂t, c)
    dt ← (ẑt - (ẑt-1 + kt(zLQ - ẑt-1)))/σ̂t
    zt-1 ← ẑt + (σt-1 - σ̂t)dt
```

### **3.3 Negative-Quality 프롬프트**

CFG(Classifier-Free Guidance)를 활용하여 품질 향상을 도모합니다:

$$ z_{t-1} = z_{t-1}^{pos} + \lambda_{cfg} \times (z_{t-1}^{pos} - z_{t-1}^{neg}) $$

여기서 positive 프롬프트는 고품질 이미지 특성을, negative 프롬프트는 "oil painting, cartoon, blur, dirty, messy, low quality, deformation"과 같은 저품질 특성을 나타냅니다[2].

## **4. 모델 구조**

SUPIR의 전체 구조는 다음과 같습니다:

1. **Degradation-Robust Encoder**: 저품질 이미지를 latent space로 매핑[1]
2. **Multi-Modal Language Model**: LLaVA-13B를 사용하여 이미지 내용 이해 및 텍스트 프롬프트 생성[2]
3. **Trimmed ControlNet**: 효율적인 제어를 위한 경량화된 ControlNet[1]
4. **ZeroSFT Connector**: 픽셀 수준 제어를 위한 새로운 연결 방식[2]
5. **Pre-trained SDXL**: 26억 개 파라미터의 강력한 생성 모델[1]

## **5. 성능 향상 및 실험 결과**

### **정량적 성능**
- **Non-reference 메트릭**에서 기존 방법들을 크게 상회하는 성능 달성[1]
- **ManIQA**: 0.4738, **ClipIQA**: 0.8049, **MUSIQ**: 73.83으로 최고 성능 기록[2]

### **정성적 평가**
- **사용자 연구**에서 84.17%의 선호도를 얻어 다른 방법들을 압도적으로 능가[1]
- 복잡한 실세계 degradation에서도 높은 품질의 복원 결과 달성[2]

## **6. 일반화 성능 향상**

### **6.1 대규모 데이터의 효과**
- **2천만 장의 고해상도 이미지**로 학습하여 다양한 degradation에 대한 robust한 성능 확보[1]
- **텍스트 어노테이션**을 통해 semantic understanding 능력 향상[2]

### **6.2 텍스트 기반 제어**
- **Multi-modal 접근법**으로 다양한 복원 시나리오에 적응 가능[2]
- **Negative-quality 샘플링**으로 다양한 품질 요구사항에 대응[1]

### **6.3 실세계 적용성**
- **실세계 테스트 데이터**에서 기존 방법들보다 우수한 generalization 성능 입증[3]
- **복잡한 degradation 조합**에서도 안정적인 성능 유지[2]

## **7. 한계**

### **7.1 기술적 한계**
- **Full-reference 메트릭**에서는 기존 방법들보다 낮은 점수를 기록하여 fidelity 측면에서 개선 필요[1]
- **계산 비용**이 높아 실시간 처리에는 제약이 있음[2]
- **Negative 프롬프트의 한계**: 명확한 semantic이 없는 경우 artifacts 발생 가능[1]

### **7.2 데이터 의존성**
- **대규모 데이터셋**에 의존하여 학습 비용이 높음[2]
- **텍스트 어노테이션**의 품질이 성능에 직접적 영향을 미침[1]

## **8. 앞으로의 연구에 미치는 영향**

### **8.1 모델 스케일링 패러다임**
- **이미지 복원 분야에서 모델 스케일링의 효과**를 입증하여 향후 연구 방향을 제시[2]
- **생성 모델과 복원 모델의 융합**에 대한 새로운 접근법 제시[1]

### **8.2 멀티모달 이미지 처리**
- **텍스트 기반 이미지 복원**의 가능성을 보여주어 관련 연구들이 활발히 진행됨[4][5]
- **Instruction-based 이미지 처리**의 새로운 패러다임 제시[6]

### **8.3 실세계 적용성 강화**
- **실세계 degradation 처리**에 대한 벤치마크 제시[3]
- **일반화 성능 향상**을 위한 새로운 방법론 제시[7]

## **9. 향후 연구 고려사항**

### **9.1 효율성 개선**
- **모델 경량화** 및 **추론 속도 향상**을 위한 연구 필요[8]
- **LoRA 기반 fine-tuning** 등 효율적인 adaptation 방법 탐구[9]

### **9.2 품질 및 fidelity 균형**
- **perceptual quality와 fidelity 간의 균형**을 맞추는 새로운 메트릭 개발 필요[1]
- **restoration-guided sampling**의 개선을 통한 fidelity 향상 방안 모색[2]

### **9.3 범용성 확장**
- **다양한 degradation type**에 대한 처리 능력 확장[10]
- **domain adaptation** 기법을 통한 실세계 적용성 향상[11]

**결론적으로, SUPIR은 이미지 복원 분야에서 모델 스케일링의 혁신적 활용을 통해 새로운 패러다임을 제시한 중요한 연구입니다. 특히 텍스트 기반 제어와 대규모 학습을 통한 일반화 성능 향상은 향후 연구 방향에 중요한 영향을 미칠 것으로 예상됩니다.**

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8f51e9d7-306c-41c4-ba97-b0c11d69984c/2401.13627v2.pdf
[2] https://ieeexplore.ieee.org/document/10654855/
[3] https://arxiv.org/abs/2412.00878
[4] https://arxiv.org/abs/2306.13090
[5] https://arxiv.org/abs/2312.06162
[6] https://github.com/mv-lab/InstructIR
[7] https://arxiv.org/abs/2408.15143
[8] https://arxiv.org/abs/2408.17060
[9] https://huggingface.co/papers/2408.17060
[10] https://arxiv.org/abs/2401.03379
[11] https://ieeexplore.ieee.org/document/10223252/
[12] https://arxiv.org/abs/2411.18588
[13] https://arxiv.org/abs/2403.10336
[14] https://arxiv.org/abs/2412.05043
[15] https://arxiv.org/abs/2408.10145
[16] https://ace.ewapublishing.org/article/97492cac39a4406498100d2e9e28720f
[17] https://ieeexplore.ieee.org/document/10167688/
[18] https://arxiv.org/html/2401.13627v1
[19] https://openaccess.thecvf.com/content/ACCV2024/html/Tan_DiffLoss_Unleashing_Diffusion_Model_as_Constraint_for_Training_Image_Restoration_ACCV_2024_paper.html
[20] https://www.themoonlight.io/en/review/efficient-image-restoration-through-low-rank-adaptation-and-stable-diffusion-xl
[21] https://huggingface.co/papers/2401.13627
[22] https://openreview.net/forum?id=6t8SUcA4sI
[23] https://www.runcomfy.com/comfyui-workflows/supir-in-comfyui-realistic-image-video-upscaling
[24] https://arxiv.org/abs/2403.07319
[25] https://arxiv.org/abs/2401.13627
[26] https://openaccess.thecvf.com/content/CVPR2024/html/Ye_Learning_Diffusion_Texture_Priors_for_Image_Restoration_CVPR_2024_paper.html
[27] https://github.com/huggingface/diffusers/issues/5956
[28] https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Scaling_Up_to_Excellence_Practicing_Model_Scaling_for_Photo-Realistic_Image_CVPR_2024_paper.pdf
[29] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05684.pdf
[30] https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Luo_Photo-Realistic_Image_Restoration_in_the_Wild_with_Controlled_Vision-Language_Models_CVPRW_2024_paper.pdf
[31] https://ostin.tistory.com/415
[32] https://github.com/iSEE-Laboratory/DiffUIR
[33] http://biorxiv.org/lookup/doi/10.1101/2024.02.10.579780
[34] https://ieeexplore.ieee.org/document/10415233/
[35] https://arxiv.org/abs/2404.10358
[36] https://paperswithcode.com/paper/hierarchical-information-flow-for-generalized
[37] https://openreview.net/forum?id=KqTzfiNjWU
[38] https://arxiv.org/html/2312.06162v1
[39] https://arxiv.org/abs/2411.15295
[40] https://openreview.net/forum?id=C0Ubo0XBPn
[41] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05700.pdf
[42] https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_Deep_Equilibrium_Diffusion_Restoration_with_Parallel_Sampling_CVPR_2024_paper.pdf
[43] https://arxiv.org/abs/2501.14014
[44] https://arxiv.org/html/2412.00878v1
[45] https://www.mdpi.com/1424-8220/24/12/3917
[46] https://www.themoonlight.io/ko/review/frequency-guided-posterior-sampling-for-diffusion-based-image-restoration
[47] https://www.themoonlight.io/ko/review/scaling-up-to-excellence-practicing-model-scaling-for-photo-realistic-image-restoration-in-the-wild
[48] https://www.sciencedirect.com/science/article/pii/S0952197625009819
[49] https://www.nature.com/articles/s40494-025-01693-z
[50] https://ojs.aaai.org/index.php/AAAI/article/view/27907
[51] https://arxiv.org/abs/2309.06023
[52] https://arxiv.org/html/2401.13627v2
[53] https://arxiv.org/pdf/2401.00523.pdf
[54] https://arxiv.org/pdf/2110.15655.pdf
[55] https://arxiv.org/pdf/2404.02154.pdf
[56] https://arxiv.org/html/2408.17060v1
[57] https://arxiv.org/pdf/2312.11232.pdf
[58] https://peerj.com/articles/cs-2679
[59] https://pmc.ncbi.nlm.nih.gov/articles/PMC11888847/
[60] https://arxiv.org/html/2411.11906v1
[61] http://arxiv.org/pdf/2412.11468.pdf
[62] https://openreview.net/forum?id=di52zR8xgf
[63] https://github.com/Fanghua-Yu/SUPIR
[64] https://arxiv.org/abs/2409.10353
[65] https://arxiv.org/abs/2410.06551
[66] https://arxiv.org/abs/2412.09324
[67] https://arxiv.org/html/2412.21063v1
[68] https://arxiv.org/html/2412.01427
[69] http://arxiv.org/pdf/2312.08881.pdf
[70] https://arxiv.org/pdf/2404.00807.pdf
[71] https://arxiv.org/html/2408.10145v1
[72] https://arxiv.org/pdf/2302.09554.pdf
[73] https://arxiv.org/html/2403.11423
[74] https://www.mdpi.com/2072-4292/15/14/3490/pdf?version=1689068473
[75] https://arxiv.org/html/2407.12273
[76] https://openaccess.thecvf.com/content/CVPR2022/papers/Mou_Deep_Generalized_Unfolding_Networks_for_Image_Restoration_CVPR_2022_paper.pdf
[77] https://www.sciencedirect.com/science/article/abs/pii/S0952197625009819
