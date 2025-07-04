# Class-Attention in Image Transformers | Image classification
"Going deeper with Image Transformers" (CaiT) 논문은 이미지 분류를 위해 더 깊은 트랜스포머 아키텍처를 최적화하는 방법을 제안합니다. 기존 트랜스포머는 이미지 분류에서 CNN의 성능을 뛰어넘었지만, 깊은 네트워크로 확장 시 최적화 문제가 발생했습니다. 이 논문은 **LayerScale**과 **Class-Attention**이라는 두 가지 핵심 기술을 도입해 이러한 한계를 해결했습니다.  

### 🔧 주요 방법론  
1. **LayerScale**  
   - 각 트랜스포머 블록(셀프 어텐션 또는 피드포워드 네트워크)의 출력에 학습 가능한 작은 파라미터(초기값 ≈ 0.1)를 곱합니다[2][3].  
   - 깊은 네트워크에서 그래디언트 소실/폭발 문제를 완화하며, **잔차 연결(residual connection)** 의 안정성을 높입니다[3].  
   - 예시: $$ \text{Output} = \text{Residual} + \lambda \cdot \text{Block}(x) $$, 여기서 $$\lambda$$는 학습된 스칼라 값입니다[2].  

2. **Class-Attention Layers**  
   - 기존 트랜스포머 레이어를 **두 단계로 분리**합니다[3]:  
     - **패치 처리 단계**: 이미지 패치 간 정보 교환에 집중하는 셀프 어텐션 레이어.  
     - **클래스 집계 단계**: 클래스 토큰이 패치 정보를 종합하는 전용 어텐션 레이어.  
   - 클래스 토큰이 패치 간 상호작용에 방해받지 않도록 해 **분류 성능을 최적화**합니다[2][3].  

### 📊 성능 및 결과  
- **ImageNet 1K**에서 **86.5% Top-1 정확도** 달성(외부 데이터 없음)[1][2].  
- 동일 정확도 기준으로 **기존 모델 대비 18% 적은 FLOPs**(329B vs 377B)와 **19% 적은 파라미터**(356M vs 438M) 사용[2].  
- **ImageNet-Real** 및 **ImageNet-V2** 벤치마크에서 SOTA 성능 기록[1][3].  

### 💡 핵심 기여  
- **깊은 트랜스포머 최적화**: LayerScale을 통해 36층 이상의 깊은 네트워크도 안정적으로 학습 가능[2][3].  
- **계층적 정보 처리**: 클래스 토큰과 패치 처리의 역할 분리로 목적별 최적화 가능[3].  
- **효율성**: 복잡도를 줄이면서도 CNN 기반 모델들을 능가하는 정확도 달성[1][2].  

이 연구는 이미지 트랜스포머의 확장성을 입증하며, 이후 비전 트랜스포머 발전의 초석이 되었습니다[1][3].

[1] https://arxiv.org/abs/2103.17239
[2] https://openaccess.thecvf.com/content/ICCV2021/papers/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.pdf
[3] https://storrs.io/cait/
[4] https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452/
[5] https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf
[6] https://velog.io/@letsbe/GOING-Deeper-with-Image-Transformers
[7] https://www.toolify.ai/ai-news/transformers-revolutionize-image-recognition-paper-explained-77860
[8] https://ffighting.net/deep-learning-paper-review/vision-model/vision-transformer/
[9] https://openreview.net/pdf?id=r16Vyf-0-
[10] https://paperswithcode.com/paper/going-deeper-with-image-transformers
