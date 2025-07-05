# Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention | Image Inpainting

이 보고서에서는 “Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention” 논문의 핵심 아이디어와 방법을 이해하기 쉽게 정리합니다. 본 연구는 개인별 얼굴 정체성을 유지하면서도 언어 제어가 가능한 얼굴 인페인팅 기법을 제안합니다.

---

## 1. 연구 배경  
일반적인 얼굴 인페인팅 기법은 마스크나 손상된 부위를 채우는 데 초점을 맞추지만, 인물의 *고유 얼굴 정체성(identity)*을 보존하는 것은 여전히 어려운 과제입니다[1].  
- **기존 방법**:  
  - *MyStyle*은 StyleGAN을 개인별 Fine-tuning하여 얼굴 정체성을 보존하지만, 40장 이상의 참조 이미지가 필요하며 연산 비용이 크다[1].  
  - *Custom Diffusion* 등 확산 모델 기반 방법은 소수의 이미지로 개인화를 가능하게 하나, 수 시간의 Fine-tuning이 요구된다[1].

---

## 2. 제안 기법: Parallel Visual Attention (PVA)  
PVA는 확산 모델(denoising UNet)에 **병렬 시각 어텐션 모듈**과 **피드포워드 정체성 인코더(identity encoder)**를 추가하여, 기존 네트워크 파라미터는 고정(frozen)한 채 소수의 Fine-tuning만으로 개인화된 인페인팅을 달성합니다[1].

### 2.1. 정체성 인코더  
- **역할**: 참조 이미지 5장을 FaceNet 기반 인식 네트워크로 추출한 특징(feature)을 Transformer 쿼리로 변환.  
- **효과**: Fine-tuning 초기값(initialization)을 학습함으로써, 빠른 개인화 업데이트(40 steps) 가능[1].

### 2.2. 병렬 시각 어텐션 모듈  
- **구성**: 기존 텍스트-조건용 Cross-Attention에 병렬로 {Q′,K′,V′}를 추가.  
- **동작**:  
  1. 텍스트 어텐션 점수  
  2. 시각(정체성) 어텐션 점수  
  3. 두 점수를 Softmax로 합산하여 가중합  
- **장점**: 참조 이미지가 없어도 원본 네트워크 동작과 동일하며, 시각 조건 추가 시 정체성 보존 강화[1].

---

## 3. 데이터셋 및 평가  
### 3.1. CelebAHQ-IDI  
- **구성**: CelebAHQ 고해상도 얼굴 이미지를 대상으로, ID별 5장의 참조 이미지와 다양한 얼굴부위 마스크(전체·하단·눈썹 등)를 생성[1].  
- **통계**: 1,963개 ID × 참조 5장, 테스트 ID는 훈련에 사용되지 않음[1].

### 3.2. 평가 지표  
- **정체성 보존**: CosFace R100 기반 특징 유사도  
- **이미지 품질**: FID(Fréchet Inception Distance), KID(Kernel Inception Distance)  
- **언어 제어**: CLIP 점수(텍스트-이미지 정합도)  

---

## 4. 실험 결과  
| 방법                 | Fine-tune Time | ID 유사도↑ | FID↓   | KID↓     |
|----------------------|----------------|------------|--------|----------|
| Latent Diffusion     | –              | 0.359      | 8.24   | 2.717    |
| Paint by Example     | –              | 0.430      | 11.2   | 6.089    |
| MyStyle              | ~15min         | 0.696      | 27.7   | 5.029    |
| Textual Inversion    | ~6h            | 0.644      | 13.8   | 8.404    |
| Custom Diffusion     | ~3h            | 0.729      | 13.9   | 5.870    |
| **PVA (제안)**       | **~1min**      | **0.741**  | **8.22** | **4.289** |

- PVA는 **최고의 정체성 보존**과 **경쟁력 있는 이미지 품질**을 보였으며, Fine-tuning 시간은 1분 이내로 단축됐다[1].  
- 언어 제어 실험에서도 PVA는 Baseline 대비 정체성 보존이 크게 향상되었고, CLIP 점수는 유사 수준을 유지했다[1].

---

## 5. 결론 및 한계  
PVA는 소수의 Fine-tuning 단계만으로 개인별 얼굴 정체성을 효과적으로 보존하면서도 언어 제어를 지원하는 얼굴 인페인팅을 구현했습니다[1].  
- **강점**:  
  - 빠른 개인화(40 steps, ~1min)  
  - 높은 ID 유사도와 양호한 품질  
- **한계**: 언어 제어 성능은 일부 손해가 있으므로, 향후 정체성 유지와 언어 컨트롤 간 균형 개선이 과제로 남아 있습니다.  

[1] https://ieeexplore.ieee.org/document/10484154/
[2] https://dl.acm.org/doi/10.1145/3532719.3543215
[3] https://dl.acm.org/doi/10.1145/3550454.3555436
[4] https://www.semanticscholar.org/paper/8e5a734ffd7a5ceb249ffcd0b5fc1e7f37f5dbc8
[5] https://ieeexplore.ieee.org/document/10657591/
[6] https://slejournal.springeropen.com/articles/10.1186/s40561-024-00292-y
[7] https://arxiv.org/abs/2412.06753
[8] https://ieeexplore.ieee.org/document/10445705/
[9] https://arxiv.org/abs/2312.03556
[10] https://openaccess.thecvf.com/content/WACV2024/papers/Xu_Personalized_Face_Inpainting_With_Diffusion_Models_by_Parallel_Visual_Attention_WACV_2024_paper.pdf
[11] http://humansensing.cs.cmu.edu/node/571
[12] https://openaccess.thecvf.com/content/WACV2024/supplemental/Xu_Personalized_Face_Inpainting_WACV_2024_supplemental.pdf
[13] https://openreview.net/forum?id=HSS5A2NQux
[14] https://web.eecs.utk.edu/~bmaclenn/Classes/494-594-CCN/presentations/CCN-Computational%20models%20of%20attention.pdf
[15] https://huggingface.co/papers/2312.03556
[16] https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhuang_Parallel_Attention_A_CVPR_2018_paper.pdf
[17] https://ieeexplore.ieee.org/document/10568093/
[18] https://link.springer.com/10.1007/s40747-024-01543-8
[19] https://www.toolify.ai/ai-news/enhance-facial-details-with-inpainting-trick-tutorial-on-inpainting-workflow-966830
[20] https://pmc.ncbi.nlm.nih.gov/articles/PMC7027440/
