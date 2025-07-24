# P2 weighting : Perception Prioritized Training of Diffusion Models | Image generation

## 1. 핵심 주장과 주요 기여

**Perception Prioritized Training of Diffusion Models**는 확산 모델(diffusion models)의 훈련 목적 함수에서 가중치 스킴을 재설계하여 모델 성능을 획기적으로 향상시킨 연구입니다[1][2][3].

### 핵심 주장

논문의 핵심 주장은 확산 모델이 각 노이즈 레벨에서 학습하는 내용을 분석한 결과, **특정 노이즈 레벨에서의 복원 작업이 모델이 풍부한 시각적 개념을 학습하는데 더 적합한 사전 작업(pretext task)을 제공한다**는 것입니다[1][4]. 연구진은 신호 대 잡음비(Signal-to-Noise Ratio, SNR)에 따라 확산 과정을 세 단계로 분류했습니다:

- **거친 특징 단계(Coarse stage)**: SNR 0-10⁻² - 전역적 색상 구조 등 거친 특징 학습[1]
- **내용 단계(Content stage)**: SNR 10⁻²-10⁰한 내용 학습[1]
- **정리 단계(Clean-up stage)**: SNR 10⁰-10⁴ - 지각할 수 없는 세부 사항 제거[1]

### 주요 기여

1. **P2(Perception Prioritized) 가중치 스킴 제안**: 지각적으로 중요한 노이즈 레벨에 더 높은 가중치를 부여하는 새로운 훈련 목적 함수 설계[1][3]

2. **확산 모델의 학습 메커니즘 분석**: 각 노이즈 레벨에서 모델이 학습하는 시각적 개념을 체계적으로 분석[4]

3. **광범위한 성능 향상 입증**: 다양한 데이터셋, 모델 구조, 샘플링 전략에서 일관된 성능 향상 달성[2][3]

## 2. 해결하고자 하는 문제

### 문제 정의

기존 확산 모델은 Ho et al.[5]이 제안한 표준 가중치 스킴을 사용하여 훈련되었지만, **왜 이 방법이 효과적인지, 그리고 샘플 품질 향상을 위한 더 나은 가중치 스킴이 존재하는지는 알려지지 않았습니다**[1]. 연구진은 기존 방법이 여전히 지각할 수 없는 세부 사항 학습에 과도한 초점을 맞추어 지각적으로 풍부한 콘텐츠 학습을 방해한다고 지적했습니다[4].

### 설계 어려움

가중치 스킴 설계의 어려움은 두 가지 요인에서 기인합니다:
1. **수천 개의 노이즈 레벨**: 전수 탐색이 불가능[1]
2. **불분명한 학습 정보**: 각 노이즈 레벨에서 모델이 학습하는 정보가 명확하지 않아 우선순위 결정이 어려움[1]

## 3. 제안하는 방법: P2 가중치 스킴

### P2 가중치 수식

연구진이 제안한 P2 가중치 스킴은 다음과 같습니다:

$$ \lambda'_t = \frac{\lambda_t}{(k + \text{SNR}(t))^{\gamma}} $$

여기서:
- $$\lambda_t$$: 기존 표준 가중치 스킴 $$\lambda_t = -\text{SNR}(t)/\text{SNR}'(t)$$[1]
- $$\gamma$$: 지각할 수 없는 세부 사항 학습에 대한 하향 가중치의 강도를 제어하는 하이퍼파라미터[3]
- $$k$$: 극도로 작은 SNR에 대한 가중치 폭발을 방지하고 가중치 스킴의 선명도를 결정하는 하이퍼파라미터[3]

### 구현 세부사항

- $$k = 1$$로 설정: $$1/(1 + \text{SNR}(t)) = 1 - \alpha_t$$이므로 쉬운 배포를 위함[4]
- $$\gamma = 0.5$$ 또는 $$\gamma = 1$$ 사용[4]
- $$\gamma > 2$$일 때는 정리 단계에 거의 0에 가까운 가중치를 할당하여 노이즈 아티팩트가 발생[4]

### 기존 방법과의 관계

P2 가중치 스킴은 Ho et al.[5]의 널리 사용되는 가중치 스킴의 일반화로, $$\gamma = 0$$일 때 기존 방법과 동일해집니다[4]. 즉, 기존 확산 모델에 $$\sum_t \lambda_t L_t$$를 $$\sum_t \lambda'_t L_t$$로 교체하여 적용할 수 있습니다[3].

## 4. 모델 구조

### 기반 아키텍처

연구진은 **ADM(Ablated Diffusion Model)[6]을 기반**으로 구현했습니다. ADM은 잘 설계된 아키텍처와 효율적인 샘플링을 제공합니다[4]. 

### 모델 구성 요소

- **U-Net 스타일 아키텍처**: 3개 입력 채널, 6개 출력 채널 (노이즈 예측 $$\epsilon$$과 분산 $$\sigma_t$$)[1]
- **BigGAN 잔여 블록**: 큰 채널 차원을 가진 U-Net[1]
- **멀티 해상도 어텐션 및 멀티 헤드 어텐션**: 고정된 헤드당 채널 수[1]
- **적응적 그룹 정규화(AdaGN)**: 시간 단계 $$t$$를 그룹 정규화의 스케일과 편향으로 변환[1]

### 경량화된 구성

효율성을 위해 다음과 같이 경량화했습니다[1]:
- 더 적은 기본 채널 수
- 더 적은 잔여 블록
- 16×16 해상도에서만 셀프 어텐션
- **총 94M 파라미터** (최근 연구들이 500M+ 사용하는 것과 대비)[1]

## 5. 성능 향상 결과

### 정량적 성능 향상

P2 가중치를 사용한 모델은 여러 데이터셋에서 일관되게 기준선을 상회하는 성능을 보였습니다[1]:

**FID-50k 점수 비교**:
- **FFHQ**: 7.86 → 6.92 (1000 스텝), 8.41 → 6.97 (500 스텝)[1]
- **CUB**: 9.60 → 6.95 (1000 스텝), 10.26 → 6.32 (250 스텝)[1]
- **AFHQ-Dogs**: 12.47 → 11.55 (1000 스텝)[1]
- **MetFaces**: 44.34 → 36.80 (250 스텝)[1]

### 최첨단 성능 달성

- **CelebA-HQ**: FID 6.91로 최첨단 성능 달성[1]
- **Oxford Flowers**: FID 17.29로 최첨단 성능 달성[1]
- **FFHQ**: StyleGAN2를 제외한 대부분의 모델보다 우수한 성능[1]

### 정성적 개선

기준선으로 훈련된 모델은 색상 편이(color shift) 아티팩트에 취약한 반면, P2 목적 함수는 전역적이고 전체적인 개념을 학습하도록 모델을 장려합니다[1].

## 6. 일반화 성능 향상

### 모델 구성에 대한 견고성

P2 가중치는 **다양한 모델 구성에서 일관되게 효과적**입니다[1][4]:

- BigGAN 잔여 블록을 Ho et al.[5]의 잔여 블록으로 교체
- 16×16에서 셀프 어텐션 제거  
- 두 개의 BigGAN 잔여 블록 사용
- 다른 학습률(2.5×10⁻⁵) 사용

특히 셀프 어텐션이 제거된 경우에 더욱 효과적이며, 이는 **P2가 전역적 의존성 학습을 장려함**을 시사합니다[4].

### 샘플링 단계에 대한 견고성

**다양한 샘플링 단계에서 일관된 성능 향상**을 보입니다[4]:
- P2로 훈련된 모델은 기준선 대비 절반의 샘플링 단계로도 더 나은 성능 달성
- DDIM 샘플러 사용 시에도 효과적

### 데이터셋 간 일반화

P2 가중치는 **데이터셋, 아키텍처, 샘플링 전략에 관계없이 일관된 개선**을 보여줍니다[2][3]. 특히 제한된 데이터(MetFaces 1k 이미지)에서 더 큰 개선을 보이는데, 이는 **모델 용량을 지각할 수 없는 세부 사항에 낭비하는 것이 제한된 데이터에서 더 해롭다**는 가설을 뒷받침합니다[1].

## 7. 한계점

### 샘플링 속도

확산 모델은 여전히 **다중 샘플링 단계가 필요**합니다[1]. DDIM 샘플러를 사용해도 최소 25번의 순전파가 필요하여 실시간 애플리케이션에는 부적합합니다. 하지만 P2 방법으로 기준선 대비 절반의 단계로도 더 나은 FID를 달성할 수 있습니다[1].

### 하이퍼파라미터 민감성

$$\gamma > 2$$일 때 노이즈 아티팩트가 발생하므로 **하이퍼파라미터 선택에 주의**가 필요합니다[4]. 연구진은 실험적으로 $$\gamma = 0.5$$ 또는 $$\gamma = 1$$을 권장합니다.

### 이론적 최적성 보장 부족

P2는 경험적으로 효과적이지만, **이론적으로 최적임을 보장하지 않습니다**. 더 정교한 가중치 스킴 설계는 향후 연구 과제로 남겨져 있습니다[4].

## 8. 향후 연구에 미치는 영향

### 후속 연구 동향

P2 논문은 확산 모델 훈련 목적 함수 최적화 분야에 새로운 방향을 제시했습니다. 이후 **Min-SNR-γ 가중치 전략**[7][8][9] 등 유사한 접근법들이 제안되어 **3.4배 빠른 수렴 속도**를 달성했습니다.

### 응용 분야 확장

P2 가중치 스킴은 다양한 응용 분야로 확장되고 있습니다:
- **의료 영상**: 교차 모달리티 의료 영상 변환에서 P2W(Perception Prioritized Weight) 적용[10]
- **수중 영상**: 해양 생물 영상의 시각적 인식 향상[11]
- **초분광 영상**: 단일 초분광 영상 초해상도에서 효과적인 스펙트럼 정보 인식[6]

### 이론적 발전

P2 연구는 **확산 모델이 각 노이즈 레벨에서 학습하는 내용에 대한 체계적 분석**을 제공하여, 후속 연구들이 더 정교한 가중치 스킴을 개발하는 기반을 마련했습니다[7][12].

## 9. 앞으로 연구 시 고려사항

### 다중 작업 학습 관점

P2의 성공은 **확산 훈련을 다중 작업 학습 문제로 접근**하는 것의 유효성을 입증했습니다[7]. 향후 연구에서는 각 타임스텝을 개별 작업으로 다루어 작업 간 충돌을 완화하는 방법을 고려해야 합니다.

### 적응형 가중치 전략

**고정된 가중치 대신 훈련 과정에서 적응적으로 조정되는 가중치**를 탐구할 필요가 있습니다[13]. 각 타임스텝에서 경사 업데이트의 영향을 추적하여 목적 함수를 효과적으로 최소화할 수 있는 타임스텝을 적응적으로 선택하는 방법이 유망합니다.

### 계산 효율성과 안정성

**Pareto 최적화 같은 일반적인 다중 작업 학습 방법은 확산 훈련에 부적합**합니다[7]. 수천 개의 작업(타임스텝)이 있는 희소성, 제한된 샘플로 인한 노이즈 경사로 인한 불안정성, 계산 비효율성 때문입니다. 따라서 **고정된 가중치 전략이 더 실용적**입니다.

### 노이즈 스케줄과의 상관관계

**가중치 스킴과 노이즈 스케줄은 상관관계가 있지만 동등하지 않습니다**[4]. 노이즈 스케줄은 가중치와 MSE 항 모두에 영향을 미치므로, 두 요소의 통합적 설계를 고려해야 합니다.

### 실제 배포 고려사항

- **하이퍼파라미터 민감성**: $$\gamma$$ 값 선택이 성능에 미치는 영향을 신중히 고려
- **데이터셋 특성**: 제한된 데이터에서 P2의 효과가 더 크므로 데이터셋 크기에 따른 적응적 적용
- **샘플링 효율성**: 더 적은 샘플링 단계로도 높은 품질을 달성하는 방법 탐구

P2 연구는 단순한 가중치 스킴 재설계만으로도 **확산 모델의 성능을 크게 향상**시킬 수 있음을 보여주었으며, 이는 확산 모델 훈련 최적화 분야의 새로운 연구 기회를 열었습니다[4].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b4cb478b-2f2d-4170-8b43-d9ed444822da/2204.00227v1.pdf
[2] https://ieeexplore.ieee.org/document/9879163/
[3] https://openaccess.thecvf.com/content/CVPR2022/papers/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.pdf
[4] https://ar5iv.labs.arxiv.org/html/2204.00227
[5] https://proceedings.neurips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Supplemental.pdf
[6] https://ieeexplore.ieee.org/document/11036681/
[7] https://arxiv.org/html/2303.09556v3
[8] https://arxiv.org/abs/2303.09556
[9] https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/
[10] https://ieeexplore.ieee.org/document/10508481/
[11] https://link.springer.com/10.1007/s40747-025-01832-w
[12] https://openreview.net/forum?id=ylHLVq0psd
[13] https://arxiv.org/abs/2411.09998
[14] https://ieeexplore.ieee.org/document/10658487/
[15] https://dl.acm.org/doi/10.1145/3636534.3690684
[16] https://dl.acm.org/doi/10.1145/3581783.3611851
[17] https://github.com/jychoi118/P2-weighting
[18] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/min-snr/
[19] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/p2weight/
[20] https://milvus.io/ai-quick-reference/what-loss-functions-are-typically-used-when-training-diffusion-models
[21] https://proceedings.neurips.cc/paper_files/paper/2024/file/38d67d1df644cf2efe9ebd5521741dc5-Paper-Conference.pdf
[22] https://arxiv.org/abs/2204.00227
[23] https://huggingface.co/docs/diffusers/en/tutorials/basic_training
[24] https://neurips.cc/virtual/2023/poster/71843
[25] https://scholar.google.co.kr/citations?view_op=view_citation&user=6qTcgH0AAAAJ&citation_for_view=6qTcgH0AAAAJ%3ALkGwnXOMwfcC
[26] https://arxiv.org/abs/2504.09000
[27] https://ieeexplore.ieee.org/document/10938396/
[28] https://arxiv.org/pdf/2305.10924.pdf
[29] http://arxiv.org/pdf/2310.08442.pdf
[30] http://arxiv.org/pdf/2409.19128.pdf
[31] http://arxiv.org/pdf/2403.13304.pdf
[32] https://arxiv.org/html/2411.08034v3
[33] http://arxiv.org/pdf/2303.09556.pdf
[34] https://arxiv.org/html/2411.05005
[35] http://arxiv.org/pdf/2502.01117.pdf
[36] https://arxiv.org/html/2406.17636v1
[37] https://openreview.net/forum?id=UEP8yRxTfV
[38] https://openreview.net/pdf?id=WihqyNqGFb
[39] https://cris.unibo.it/bitstream/11585/969479/4/1-s2.0-S1077314224001073-mmc1.pdf
[40] https://bmva-archive.org.uk/bmvc/2024/papers/Paper_727/paper.pdf
