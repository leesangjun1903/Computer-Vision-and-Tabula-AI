# Pix2Pix-zero : Zero-shot Image-to-Image Translation | Image generation

# 핵심 요약 및 기여

**Zero-shot Image-to-Image Translation**(pix2pix-zero)은 **사전 학습된 텍스트-투-이미지 확산 모델(Stable Diffusion)을 활용해, **추가 학습 없이** 사용자가 입력한 **도메인 간 변환 지시(예: cat→dog)**만으로도 실시간에 가까운 이미지 편집을 가능케 한다.  
주요 기여는 다음과 같다:  
1. **무프롬프트 편집 방향 탐색**: GPT-3로 생성한 다수의 문장 간 CLIP 임베딩 차이의 평균으로 견고한 편집 방향 $$\Delta c_{\text{edit}}$$을 자동 계산.  
2. **크로스-어텐션 가이던스**: 확산 과정 중 편집 전·후의 크로스-어텐션 맵 $$M_t$$ 간 L₂ 손실  

$$
     L_{\text{xa}} = \|M_t^{\text{edit}} - M_t^{\text{ref}}\|_2^2
   $$  

을 적용해 구조 보존.  

3. **잡음 역전파 정규화**: DDIM 역전파 시 **자가상관 정규화** $$L_{\text{auto}} = L_{\text{pair}} + \lambda L_{\text{KL}}$$로 잡음의 가우시안 속성 유지.  
4. **조건부 GAN 증류**: 15,000쌍의 편집 전·후 이미지를 CoMod-GAN으로 학습해 ∼3,800× 속도 가속.  

# 문제 정의와 제안 기법

## 해결하고자 하는 문제  
- **실제 이미지 편집**: 텍스트-투-이미지 모델은 신규 합성에 강하나, 실제 입력 이미지를 구조 손실 없이 편집하기 어려움.  
- **프롬프트 작성 난이도**: 입력 이미지에 어울리는 완전한 자연어 묘사를 매번 작성하는 부담.  
- **모델별 파인튜닝 비효율**: 다양한 이미지·편집 유형마다 대규모 모델을 재학습할 수 없음.  

## 제안 방법 흐름  
1. **이미지 인버전 (Section 3.1)**  
   - DDIM 역확산으로 입력 잠재 $$x_0$$를 노이즈 맵 $$x_{\text{inv}}$$으로 역전파.  
   - 잡음의 통계(자기상관)를 보존하기 위해

$$
       L_{\text{pair}} = \sum_{p}\frac{1}{S_p^2}\sum_{\delta=1}^{S_p-1}\sum_{x,y,c}\eta^p_{x,y,c}\,\eta^p_{x-\delta,y,c}
       + \eta^p_{x,y,c}\,\eta^p_{x,y-\delta,c},
     $$  

$$
       L_{\text{KL}} = \mathrm{KL}\bigl(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,1)\bigr)
     $$  

를 조합해 $$L_{\text{auto}}=L_{\text{pair}}+\lambda L_{\text{KL}}$$ 로 최적화.  

2. **편집 방향 발견 (Section 3.2)**  
   - 소스 단어 $$s$$, 타겟 단어 $$t$$로 GPT-3 생성 문장군 $$\{S\},\{T\}$$ 구성.  
   - CLIP 텍스트 임베딩 $$\phi(\cdot)$$ 차이의 평균  

$$
       \Delta c_{\text{edit}} = \frac1{|\{T\}|}\sum_{u\in T}\phi(u)
       - \frac1{|\{S\}|}\sum_{u\in S}\phi(u).
     $$  

3. **교차-어텐션 기반 구조 보존 (Section 3.3)**  
   - 편집 전 텍스트 피처 $$c$$로 생성 과정 중 레퍼런스 크로스-어텐션 맵 $$M_t^{\text{ref}}$$ 수집.  
   - 편집된 피처 $$c_{\text{edit}}=c+\Delta c_{\text{edit}}$$로 생성하면서 매 스텝  

$$
       x_t \leftarrow x_t - \lambda_{\text{xa}}\nabla_{x_t}\|M_t^{\text{edit}}-M_t^{\text{ref}}\|_2^2
     $$  

를 적용해 구조 손실 최소화.  

4. **모델 증류 (Section 4.5)**  
   - 15,000쌍의 (원본, 편집) 이미지를 수집해 CoMod-GAN 학습.  
   - 추론 속도를 1.8 ms/이미지로 단축(∼3,800× 가속).  

# 모델 구조 및 수식 정리

- **DDIM 역전파**:

$$
    x_{t+1} = \sqrt{\bar\alpha_{t+1}}\,f_\theta(x_t,t,c)
    + \sqrt{1-\bar\alpha_{t+1}}\,\epsilon_\theta(x_t,t,c),
  $$  

$$
    f_\theta(x_t,t,c) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t,c)}{\sqrt{\bar\alpha_t}}.
  $$

- **크로스-어텐션**:  

$$
    M = \mathrm{Softmax}\bigl(QK^T/\sqrt{d}\bigr),\quad
    Q=W_Q\phi(x_t),\;K=W_Kc,\;V=W_Vc.
  $$

- **크로스-어텐션 가이던스 손실**:  

$$
    L_{\text{xa}} = \sum_t\|M_t^{\text{edit}} - M_t^{\text{ref}}\|_2^2.
  $$

# 성능 및 한계

- **CLIP-Acc**: cat→dog 92.4%, horse→zebra 75.2%.  
- **Structure Dist**: cat→dog 0.044, horse→zebra 0.066.  
- **BG LPIPS**(배경 보존): cat→dog 0.182, horse→zebra 0.194.  
- **속도**: 확산 모델 1.8 s→GAN 1.8 ms.  

**한계**  
- 크로스-어텐션 해상도(64×64) 한계로 미세 구조(꼬리·사지 위치) 제어 어려움.  
- 비정형 자세·복잡 배경에서는 구조 보존 실패 사례 존재.  

# 일반화 및 향후 연구 고려사항

- **다양한 도메인 적용**: 제안된 무프롬프트 편집 방향 탐색은 어떠한 단어 쌍에도 일반화 가능하므로, 풍부한 사전학습 모델에 확대 적용 시 미지 도메인 간 번역 성능 향상 기대.  
- **고해상도 어텐션**: 어텐션 맵 해상도 증대로 더욱 정교한 구조 제어 가능.  
- **멀티-모달 조건부 편집**: 텍스트 외에 마스크·스케치 등 추가 조건 결합으로 세밀한 국부 수정 연구.  
- **실시간 대화형 시스템**: 속도 가속된 증류 모델을 챗봇·인터랙티브 툴에 통합해 실시간 사용자 피드백 기반 편집 연구.  

이 논문은 **사전학습 확산 모델을 훈련·프롬프트 없이 실질적 이미지 편집**에 활용하는 방향을 제시함으로써, 향후 제너레이티브 AI 기반 편집 도구의 **즉시성·유연성·확장성**을 크게 확장할 것으로 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eee9abab-b098-467f-8799-cf980c566eb9/2302.03027v1.pdf
