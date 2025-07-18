## SAM-DiffSR: Structure-Modulated Diffusion Model for Image Super-Resolution | Super resolution

### **핵심 주장 및 주요 기여**

SAM-DiffSR는 기존 확산 모델의 한계를 극복하는 혁신적인 접근법을 제안합니다[1]. 기존 확산 모델들은 단일 분포에서 노이즈를 샘플링하여 실제 장면과 복잡한 텍스처를 처리하는 능력이 제한적이었습니다. 본 연구의 핵심 기여는 다음과 같습니다:

**1. 구조적 정보 통합**: SAM(Segment Anything Model)의 세밀한 구조 정보를 확산 모델의 노이즈 샘플링 과정에 통합하여 추론 단계에서 추가 계산 비용 없이 이미지 품질을 향상시킴[1]

**2. 효율적인 아키텍처**: 훈련 단계에서만 SAM을 활용하고, 추론 단계에서는 원본 확산 모델만 사용하여 계산 비용을 최소화함[1]

**3. 우수한 성능**: DIV2K 데이터셋에서 기존 확산 기반 방법들보다 최대 0.74dB의 PSNR 향상을 달성하며, 아티팩트 억제에서도 탁월한 성능을 보임[1]

### **해결하고자 하는 문제**

기존 확산 기반 초해상도 모델들의 근본적인 한계를 해결하고자 합니다:

**1. 균일한 노이즈 분포 문제**: 기존 모델들은 이미지의 모든 영역에 동일한 노이즈 분포를 적용하여 서로 다른 의미 영역 간의 정보 혼재를 야기함[1]

**2. 구조적 세부사항 복원 부족**: 복잡한 텍스처와 구조적 영역에서 세부사항 복원이 미흡하여 왜곡된 구조와 아티팩트 발생[1]

**3. 높은 계산 비용**: SAM을 직접 통합할 경우 추론 단계에서 상당한 계산 비용이 발생하는 문제[1]

### **제안하는 방법론 (수식 포함)**

SAM-DiffSR는 구조적 위치 인코딩과 노이즈 모듈레이션을 통한 혁신적인 접근법을 제안합니다:

**1. 구조적 노이즈 모듈레이션**

기존 확산 과정을 수정하여 구조적 정보를 통합합니다:

$$ q(x_t|x_{t-1}, E_{SAM}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}E_{SAM}, \beta_t I) $$

여기서 $$E_{SAM}$$은 SAM에서 생성된 구조적 위치 인코딩 마스크입니다[1].

**2. 조건부 분포 유도**

직접 샘플링을 위한 조건부 분포:

$$ q(x_t|x_0, E_{SAM}) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}\_t}x_0 + \phi_t E_{SAM}, (1-\bar{\alpha}_t)I) $$

여기서 $$\bar{\alpha}\_t = \prod_{i=1}^t \alpha_i$$, $$\phi_t = \sum_{i=1}^t \sqrt{\frac{\bar{\alpha}_t}{\bar{\alpha}_i}}\beta_i$$입니다[1].

**3. 역확산 과정 최적화**

베이즈 정리를 이용한 후방 분포:

$$ \tilde{\mu}\_t(x_t, x_0, E_{SAM}) = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}\_t}}(\frac{\sqrt{1-\bar{\alpha}\_t}}{\sqrt{\beta_t}}E_{SAM} + \epsilon)) $$

**4. 손실 함수**

구조적 노이즈 추정을 위한 손실 함수:

$$ L(\theta) = \mathbb{E}\_{t,x_0,\epsilon}[\|\frac{\sqrt{1-\bar{\alpha}\_t}}{\sqrt{\beta_t}}E_{SAM} + \epsilon - \epsilon_\theta(x_t, x_{LR}, t)\|_2^2] $$

### **모델 구조**

**1. Structural Position Encoding (SPE) 모듈**

- **RoPE 기반 위치 인코딩**: 각 분할 영역에 고유한 위치 정보를 할당
- **구조적 정보 통합**: $$E_{SAM} = \sum_i M_{SAM,i} \cdot \text{mean}(x_{RoPE,i})$$

**2. 구조적 노이즈 모듈레이션**

- **훈련 단계**: SAM 마스크를 사용하여 영역별 노이즈 평균을 독립적으로 조정
- **추론 단계**: 학습된 모델이 구조적 정보를 내재화하여 SAM 없이 동작

### **성능 향상 및 실험 결과**

**1. 정량적 성능**

- **PSNR**: DIV2K에서 29.34dB (기존 SRDiff 28.60dB 대비 0.74dB 향상)[1]
- **FID**: 0.3809 (기존 SRDiff 0.4649 대비 18% 향상)[1]
- **MANIQA**: 0.5959 (지각적 품질 개선)[1]

**2. 아티팩트 억제**

모든 테스트 데이터셋에서 가장 낮은 아티팩트 수준을 달성:
- Set5: 0.1322 (기존 SRDiff 0.1821 대비 27% 감소)[1]
- Urban100: 1.1453 (기존 SRDiff 1.4163 대비 19% 감소)[1]

**3. 계산 효율성**

- **훈련 시간**: 기존 SRDiff와 거의 동일 (10h21min vs 10h16min)[1]
- **추론 시간**: 추가 비용 없음 (37.62s vs 37.64s)[1]
- **매개변수**: 기존 모델과 동일한 12M 매개변수 유지[1]

### **한계점 및 고려사항**

**1. 분할 모델 의존성**

SAM의 분할 품질에 따라 성능이 좌우되며, 저해상도 영역에서는 구조 식별에 한계가 있을 수 있습니다[1].

**2. 메모리 사용량**

훈련 과정에서 모든 이미지에 대해 SAM 마스크를 사전 생성하여 저장해야 하므로 추가 메모리 요구사항이 있습니다[1].

**3. 도메인 특화성**

일반적인 자연 이미지에 최적화되어 있어 의료 영상이나 위성 이미지 등 특수 도메인에서의 일반화 성능에 대한 검증이 필요합니다[1].

### **일반화 성능 향상 가능성**

**1. 구조적 선행 지식 활용**

SAM의 강력한 일반화 능력을 활용하여 다양한 도메인과 객체에 대한 구조적 정보를 효과적으로 캡처할 수 있습니다[1].

**2. 스케일별 적응성**

논문에서 제시된 X2 스케일 실험 결과, 다양한 확대 비율에서도 일관된 성능 향상을 보여줍니다[1].

**3. 마스크 품질 의존성**

고품질 마스크(SAM with HR images)에서 중품질 마스크(MobileSAM with HR images), 저품질 마스크(MobileSAM with LR images) 순으로 성능이 향상되어, 분할 품질이 일반화 성능에 직접적인 영향을 미칩니다[1].

### **향후 연구에 미치는 영향**

**1. 구조적 정보 통합의 새로운 패러다임**

SAM과 확산 모델의 효율적인 통합 방식은 향후 다양한 생성 모델에서 구조적 정보를 활용하는 새로운 표준을 제시할 것으로 예상됩니다[2][3].

**2. 계산 효율성 최적화**

훈련 단계에서만 구조적 정보를 활용하고 추론 단계에서는 추가 비용 없이 성능을 향상시키는 접근법은 실용적인 AI 시스템 개발에 중요한 통찰을 제공합니다[1].

**3. 멀티모달 학습 확장**

구조적 정보와 확산 모델의 통합 방식은 텍스트-이미지 생성, 3D 생성 등 다양한 멀티모달 작업으로 확장될 수 있는 잠재력을 보여줍니다[3].

### **향후 연구 시 고려사항**

**1. 분할 모델 진화 대응**

SAM2와 같은 차세대 분할 모델과의 통합을 통해 더욱 정교한 구조적 정보 활용 방안을 모색해야 합니다[4].

**2. 도메인 특화 최적화**

의료 영상, 위성 이미지, 과학 이미지 등 특수 도메인에서의 적용을 위한 도메인별 최적화 방안을 연구해야 합니다[5].

**3. 실시간 처리 요구사항**

모바일 및 엣지 디바이스에서의 실시간 처리를 위한 경량화 방안과 하드웨어 가속 기법 연구가 필요합니다[6].

**4. 윤리적 고려사항**

고품질 이미지 생성 기술의 오남용(딥페이크 등) 방지를 위한 기술적, 제도적 대응 방안을 함께 고려해야 합니다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/da942c1c-770a-4864-bded-12b78746982f/2402.17133v2.pdf
[2] https://arxiv.org/abs/2402.17133
[3] https://arxiv.org/abs/2505.07071
[4] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/segment-anything-2/
[5] https://ieeexplore.ieee.org/document/10947187/
[6] https://arxiv.org/abs/2406.05723
[7] https://ieeexplore.ieee.org/document/10678132/
[8] https://ieeexplore.ieee.org/document/10678598/
[9] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12930/3008456/AniRes2D--anisotropic-residual-enhanced-diffusion-for-2D-MR-super/10.1117/12.3008456.full
[10] https://www.semanticscholar.org/paper/801779a7964d515a31d430eb3501bf5c7db8c065
[11] https://www.emergentmind.com/papers/2402.17133
[12] https://arxiv.org/html/2402.17133v2
[13] https://www.themoonlight.io/ko/review/semantic-guided-diffusion-model-for-single-step-image-super-resolution
[14] https://arxiv.org/html/2402.17133v1
[15] https://openreview.net/pdf/1a30154772764f5f142c853e35c628ac87103544.pdf
[16] https://github.com/continue-revolution/sd-webui-segment-anything
[17] https://www.themoonlight.io/en/review/sam-diffsr-structure-modulated-diffusion-model-for-image-super-resolution
[18] https://paperreading.club/page?id=211735
[19] https://ostin.tistory.com/199
[20] https://public.thinkonweb.com/sites/2024s/media?key=site%2F2024s%2Fabs%2F0569-SRPKR.pdf
[21] https://github.com/lose4578/SAM-DiffSR
[22] https://huggingface.co/papers?q=Fine-resolution+inference
[23] https://www.sciencedirect.com/science/article/pii/S221486042500154X
[24] https://openreview.net/forum?id=5cYTAcZAgt
[25] https://proceedings.neurips.cc/paper_files/paper/2024/file/3685de48976169ca9fd68cb4c8e48b76-Paper-Conference.pdf
[26] https://arxiv.org/abs/2503.08915
[27] https://arxiv.org/abs/2505.02784
[28] http://arxiv.org/pdf/2409.19589.pdf
[29] https://arxiv.org/pdf/2307.12348.pdf
[30] https://arxiv.org/html/2410.22830
[31] https://arxiv.org/pdf/2308.07977.pdf
[32] https://arxiv.org/pdf/2311.14760.pdf
[33] http://arxiv.org/pdf/2305.15357.pdf
[34] http://arxiv.org/pdf/2404.10688.pdf
[35] http://arxiv.org/pdf/2408.07476.pdf
[36] https://arxiv.org/html/2411.12072
[37] https://paperswithcode.com/task/image-super-resolution/latest?page=17&q=
[38] https://dblp.org/rec/journals/corr/abs-2402-17133
[39] https://dl.acm.org/doi/10.1016/j.neucom.2024.128911
