# DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

DreamBooth는 텍스트-이미지 확산 모델을 미세 조정(fine-tuning)하여 특정 주체(subject)를 다양한 상황에서 일관되게 생성하는 혁신적인 방법을 제안합니다[2][3]. 기존 대규모 텍스트-이미지 모델(예: Imagen)은 다양한 이미지 생성을 가능하게 했으나, 참조 이미지 집합의 특정 주체를 정확히 모방하고 새로운 맥락에서 재생성하는 데 한계가 있었습니다[2][3]. 이 논문은 2023년 CVPR에서 Best Student Paper Honorable Mention을 수상했으며(0.25% 수상률)[1], 다음과 같은 핵심 기여를 합니다.

### 🧠 방법론  
1. **미세 조정 프로세스**  
   - 3~5장의 주체 이미지로 사전 학습된 모델을 조정합니다[4].  
   - 고유 식별자(예: `[T]`)와 클래스명(예: "강아지")을 포함한 프롬프트(`"A [T] dog"`)로 저해상도 모델을 학습시킵니다[4].  
   - **Autogenous Prior Preservation Loss**를 적용해 주체의 클래스 특성을 유지하면서도 다양한 변형을 생성합니다[2][3]. 이 손실 함수는 모델의 시맨틱 프라이어(semantic prior)를 활용하여 주체의 핵심 특징을 보존합니다[2].  

2. **고해상도 디테일 보존**  
   - 초해상도(super-resolution) 컴포넌트를 별도로 미세 조정하여 피사체의 세부 사항(예: 질감, 로고)을 고충실도로 재현합니다[4].  

### 🎨 적용 사례  
- **주체 재배치(Recontextualization)**: 참조 이미지에 없는 배경/조명에서 주체 생성[4].  
  ![예시: 가방이 해변에 있는 이미지](https://dreambooth.github.io/assets/반 뷰 합성**: 프롬프트로 주체의 포즈, 각도 변경[2].  
- **예술적 렌더링**: 고흐/워홀 스타일의 강아지 생성[4].  

### ⚙️ 기술적 장점  
- **샘플 효율성**: 3~5장의 이미지만으로도 강력한 개인화 가능[3][4].  
- **일반화**: 학습 데이터에 없는 포즈, 조명, 스타일에서도 주체 특징 유지[2].  
- **확장성**: Imagen 외 다양한 확산 모델에 적용 가능[4].  

### 📊 평가 및 데이터셋  
- **새로운 평가 프로토콜**: 주체 주도 생성 과제를 위한 데이터셋과 정량적 지표 제시[2].  
- **정성적 결과**: 사용자 연구에서 기존 방법 대비 우월한 정확도와 일관성 확인[3].  

이 기술은 E-커머스(제품 가상 배치), 개인화된 아트워크, 가상 스튜디오 등에 적용 가능하며, 생성형 AI의 개인화 영역을 혁신했습니다[2][4]. 프로젝트 페이지와 코드는 [dreambooth.github.io](https://dreambooth.github.io)에서 공개되었습니다[3][4].

[1] https://onlinelibrary.wiley.com/doi/10.1002/9783527809080.cataz06916
[2] https://ieeexplore.ieee.org/document/10204880/
[3] https://arxiv.org/abs/2208.12242
[4] https://dreambooth.github.io
[5] https://github.com/vlgiitr/papers_we_read/blob/master/summaries/DreamBooth.md
[6] https://ieeexplore.ieee.org/document/10392676/
[7] https://ieeexplore.ieee.org/document/10943802/
[8] https://zhangtemplar.github.io/dream-booth/
[9] https://www.youtube.com/watch?v=5Byg3EeOsRc
[10] https://arxiv.org/abs/2504.02612
[11] https://arxiv.org/abs/2411.01179
[12] https://arxiv.org/abs/2305.14720
[13] https://arxiv.org/abs/2306.14153
[14] https://arxiv.org/abs/2312.00079
[15] https://arxiv.org/abs/2304.00186
[16] https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf
[17] https://github.com/google/dreambooth
[18] https://paperswithcode.com/paper/dreambooth-fine-tuning-text-to-image
[19] https://openreview.net/forum?id=bMY2HyXbbQ
[20] https://cvpr.thecvf.com/virtual/2023/poster/23180
