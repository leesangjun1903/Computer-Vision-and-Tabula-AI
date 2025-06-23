# Your Diffusion Model is Secretly a Zero-Shot Classifier

## 1. 배경 및 동기  
대규모 텍스트-투-이미지(diffusion) 모델은 아름다운 이미지를 생성하지만, 실제로 **분류(classification)**에도 활용할 수 있습니다. 본 논문은 Stable Diffusion과 같은 대규모 확산 모델(diffusion model)의 **조건부 밀도 추정**을 이용해 추가 학습 없이 **제로샷(zero-shot) 분류기**를 구현하는 **Diffusion Classifier**를 제안합니다[1].

---

## 2. 확산 모델(DDPM) 사전 지식  
1) **Forward Process**: 입력 이미지 $$x_0$$에 단계별로 가우시안 노이즈를 더해 $$x_T$$를 생성합니다.  
2) **Reverse Process**: 학습된 네트워크 $$\epsilon_\theta(x_t,c)$$가 노이즈 $$\epsilon$$를 예측하며 $$x_t$$에서 $$x_{t-1}$$를 복원합니다[1].  
3) **ELBO (Evidence Lower Bound)**: 모델은 변분 하한(ELBO)을 최대화하도록 학습되며,
   
$$\text{ELBO} \approx -\mathbb{E}\_{t,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t,c)\|^2\big] + \text{const}$$

로 나타낼 수 있습니다[1].

---

## 3. Diffusion Classifier 수학적 유도  
1) **Bayes 법칙 적용**

$$p(c \mid x) \propto p(c)\,p_\theta(x\mid c).$$  

   균등 사전확률 $$p(c)=\text{const}$$를 가정하면 $$p_\theta(x\mid c)$$만 최대화하면 됩니다[1].  
4) **ELBO로 근사**  
   
$$\log p_\theta(x\mid c)\approx -\mathbb{E}\_{t,\epsilon}\big[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x+\sqrt{1-\bar\alpha_t}\epsilon,\;c)\|^2\big].$$  

5) **Monte Carlo 추정**  
   * 각 클래스 $$c_i$$에 대해 여러 시도(trial)에서 $$\epsilon$$-예측 오차를 계산하고 평균을 구합니다.  
   * 최종 분류는 오류가 가장 낮은 $$c_i$$를 선택합니다[1].

---

## 4. 분산 감소(Variance Reduction) 기법  
- **Paired Difference Test 유사**: 서로 다른 클래스 간 절대 오차 대신 **오차 차이**만 필요하므로, 동일한 $$(t,\epsilon)$$ 샘플을 모든 클래스에 재사용해 통계적 분산을 크게 줄입니다[1].  
- 이 방법으로 매끄러운 클래스 간 비교가 가능해집니다.

---

## 5. 실전 적용 고려사항  
### 5.1. 타임스텝 선택  
- 중간 노이즈 수준($$t\approx500$$)에서 분류 성능이 가장 높으며, 균일 샘플링(uniform sampling)이 최적의 확장성을 보입니다[1].  

### 5.2. 효율적 분류(Adaptive Evaluation)  
- 클래스 수가 많을 때는 **단계별(stage-wise)**로 후보를 점진적 제거합니다.  
  1. 모든 클래스에 적은 시도를 수행, 오차 상위 클래스 제거  
  2. 남은 클래스에 더 많은 시도를 집중  
- 이를 통해 ImageNet(1,000 클래스)도 부분적으로 대응 가능하나 여전히 수백 초가 소요됩니다[1].

---

## 6. 주요 실험 결과  
### 6.1. 제로샷 분류 성능  
| 방법                    | zero-shot | CIFAR-10 | Pets   | Flowers | STL-10 | ImageNet | ObjectNet |
|-------------------------|-----------|----------|--------|---------|--------|----------|-----------|
| Synthetic SD Data       | ✓         | 35.3%    | 31.3%  | 22.1%   | 38.0%  | 18.9%    |  5.2%     |
| SD Features             | ✗         | 84.0%    | 75.9%  | 70.0%   | 87.2%  | 56.6%    | 10.2%     |
| **Diffusion Classifier**| ✓         | **88.5%**|**87.3%**|66.3%    |**95.4%**|61.4%    |**43.4%**  |
| CLIP ResNet-50          | ✓         | 75.6%    | 85.4%  | 65.9%   | 94.3%  | 58.2%    | 40.0%     |
| OpenCLIP ViT-H/14       | ✓         | 97.3%    | 94.6%  | 79.9%   | 98.3%  | 76.8%    | 69.2%     |

- Stable Diffusion 기반 제로샷 분류가 **Synthetic SD Data** 대비 대폭 성능 향상을 보이며, **CLIP ResNet-50**을 넘어서고 **OpenCLIP**과 경쟁할 정도로 발전함을 확인했습니다[1].

### 6.2. 구성(reasoning) 능력 (Winoground)  
| 모델                    | Object | Relation | Both  | 평균   |
|-------------------------|--------|----------|-------|--------|
| Random                  | 25.0%  | 25.0%    | 25.0% | 25.0%  |
| CLIP ViT-L/14           | 27.0%  | 25.8%    | 57.7% | 28.2%  |
| OpenCLIP ViT-H/14       | 39.0%  | 26.6%    | 57.7% | 33.0%  |
| **Diffusion Classifier**|**46.1%**|**29.2%**|**80.8%**|**38.5%**|

- Stable Diffusion의 제너레이티브 특성으로 인해 **관계 중심(Relation)** 구성 과제에서도 타 방법을 상당히 앞서며, 전반적 구성(reasoning) 능력이 검증되었습니다[1].

---

## 7. 결론 및 향후 과제  
- **Diffusion Classifier**는 확산 모델을 제로샷 분류기로 활용하여 **추가 학습 없이** 강력한 성능을 달성합니다.  
- **구성적 추론 능력**과 **분포 이동(robustness)**에서도 뛰어난 결과를 보이며, 생성을 넘어서는 응용성을 제시합니다[1].  
- 향후 **추론 속도** 개선, **프롬프트 튜닝**, 그리고 더 다양한 데이터셋으로 **사전학습 영역 확장**이 중요한 연구 방향이 될 것입니다.

[1] https://ieeexplore.ieee.org/document/10376944/
[2] https://arxiv.org/pdf/2303.16203.pdf
[3] http://arxiv.org/pdf/2406.03736.pdf
[4] https://arxiv.org/html/2412.17219v2
[5] http://arxiv.org/pdf/2402.02316.pdf
[6] https://arxiv.org/pdf/2402.06559.pdf
[7] https://arxiv.org/html/2308.16534
[8] http://arxiv.org/pdf/2308.12469.pdf
[9] https://arxiv.org/abs/2303.16203
[10] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[11] https://openreview.net/pdf/a1ae5b7782b9f642d012ef077e717a1b620f0b9a.pdf
[12] https://ar5iv.labs.arxiv.org/html/2303.16203
[13] https://paperswithcode.com/paper/your-diffusion-model-is-secretly-a-zero-shot
[14] https://diffusion-classifier.github.io
[15] https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[16] https://arxiv.org/html/2412.12594v1
[17] https://arxiv.org/html/2403.13652
[18] https://arxiv.org/pdf/2406.02929.pdf
[19] https://www.youtube.com/watch?v=t5Daou0eT-g
[20] https://openreview.net/forum?id=fxNQJVMwK2

### 핵심 접근법  
기존 텍스트-이미지 확산 모델(예: Stable Diffusion)이 이미지 생성 외에 **조건부 확률 밀도 추정** 을 통해 제로샷 분류가 가능함을 입증했습니다[1][3]. 이 방법은 별도 학습 없이 사전 훈련된 모델의 density estimate를 활용해 클래스 간 상대적 가능도를 비교하며, 이를 **Diffusion Classifier** 로 명명했습니다[4][7].

### 주요 강점  
1. **다중모달 추론 능력**:  
   텍스트와 이미지 간 구성적 관계 이해에서 CLIP 등의 판별적 모델을 능가합니다(예: "빨간색 사과 vs 녹색 사과" 구분)[5][7].  
2. **벤치마크 성능**:  
   CIFAR-10(77.9%), Flowers(86.2%), ImageNet(58.9%) 등에서 기존 확산 모델 기반 분류기 대비 우수한 성적[2][5].  
3. **효율적 강건성**:  
   이미지넷 분류기 추출 시 약한 데이터 증강만으로도 분포 변화에 강인한 특성 보임[3][4].  

### 적용 사례  
- **이미지넷 분류기 변환**: 클래스 조건부 확산 모델을 전통적인 분류기로 변환 가능[4][7]  
- **합성 데이터 활용**: 확산 모델이 생성한 합성 데이터로 분류기 훈련 시 성능 향상[2][6]  

### 한계 및 비교  
- **계산 비용**: 실시간 추론에는 여전히 고비용(이미지당 1-2분 소요)[6]  
- **CLIP 대비 성능 격차**: 일부 벤치마크에서 OpenCLIP ViT-H/14 대비 10-15%p 낮은 정확도[2][5]  

이 연구는 생성 모델이 판별 작업에서도 유용함을 입증하며, 향후 다중모달 AI 시스템 개발에 새로운 방향성을 제시했습니다[1][7]. 특히 데이터 증강 없이도 분포 변화에 강인한 분류 가능성은 실제 응용 분야에서 주목할 만한 결과입니다[3][4].

[1] https://arxiv.org/abs/2303.16203
[2] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[3] https://openreview.net/forum?id=Ck3yXRdQXD
[4] https://github.com/diffusion-classifier/diffusion-classifier
[5] https://paperswithcode.com/paper/your-diffusion-model-is-secretly-a-zero-shot
[6] https://www.jetir.org/papers/JETIR2411561.pdf
[7] https://huggingface.co/papers/2303.16203
[8] https://papers.nips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[9] https://www.computer.org/csdl/proceedings-article/iccv/2023/071800c206/1TJjVcPg24g
[10] https://www.youtube.com/watch?v=t5Daou0eT-g

