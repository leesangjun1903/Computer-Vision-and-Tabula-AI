# CLIP : Learning Transferable Visual Models From Natural Language Supervision | Image recognization, Image generation
"Learning Transferable Visual Models From Natural Language Supervision" 논문은 이미지와 텍스트 간의 관계를 학습해 범용적인 시각 인식 모델을 구축한 획기적인 연구입니다.

### 📌 핵심 기여  
1. **자연어 감독 학습**  
   - 기존 시각 모델은 고정된 객체 범주에 의존해 추가 라벨링이 필요했음[1][3].  
   - 본 연구는 **400만 개의 인터넷 이미지-텍스트 쌍**을 활용해 자연어 설명만으로 학습함[1][2].  

2. **Contrastive 학습 프레임워크**  
   - 이미지 인코더(ResNet/ViT)와 텍스트 인코더(Transformer)를 병렬로 구성[2][5].  
   - **대조 학습(Contrastive Learning)** 통해 유사한 이미지-텍스트 쌍은 가깝게, 비유사한 쌍은 멀어지도록 임베딩 공간 최적화[2][5].  

3. **제로샷 전이 성능**  
   - 30개 이상의 다양한 태스크(OCR, 동작 인식, 지리 위치 등)에서 평가[1][3].  
   - **별도 미세 조정 없이** ImageNet에서 ResNet-50과 동등한 정확도 달성[1][5].  
   - 예시: "강아지 사진" 텍스트 프롬프트만으로도 개 품종 분류 가능[4][5].  

### ⚙️ 기술적 혁신  
- **효율성**: Bag-of-Words 예측 대비 4배 빠른 학습 속도[2].  
- **확장성**: 텍스트 프롬프트 조정으로 새로운 객체 범주 즉시 인식 가능[4][5].  
- **다중 모달 통합**: 이미지와 텍스트를 동일한 임베딩 공간에 매핑해 시각-언어 상호작용 가능[2][5].  

### 🌐 의의 및 한계  
- **의의**: 라벨 의존성 탈피, 대규모 웹 데이터 활용 가능성 증명[1][3].  
- **한계**:  
  - 텍스트의 모호성(예: "빨간 공"이 축구공/테니스공인지 구분 불확실)[4].  
  - 데이터 내 사회적 편향 재생산 가능성[5].  

이 연구는 **자연어가 시각 인식의 강력한 감독 신호**가 될 수 있음을 입증하며, 이후 CLIP 등 다중 모달 모델 발전의 초석이 되었습니다[1][5].

[1] https://arxiv.org/abs/2103.00020
[2] https://proceedings.mlr.press/v139/radford21a/radford21a.pdf
[3] http://arxiv.org/pdf/2103.00020.pdf
[4] https://molly.polycount.com/library-files/learning-transferable-visual-models-from-natural-language-supervision.pdf
[5] https://github.com/cognitivetech/llm-research-summaries/blob/main/document-processing/Learning-Transferable-Visual-Models-From-Natural-Language-Supervision_2103.00020.md
[6] http://graphics.csie.ncku.edu.tw/2025%20CGAP/Learning-Transferable-Visual-Models-From-Natural-Language-Supervision.pdf
[7] https://strikingloo.github.io/wiki/clip
[8] https://www.scribd.com/document/548666345/Learning-Transferable-Visual-Models-From-Natural-Language-Supervision
[9] https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4
[10] https://paperswithcode.com/paper/learning-transferable-visual-models-from

https://ffighting.net/deep-learning-paper-review/multimodal-model/clip/

https://github.com/openai/CLIP/tree/main

- How is the dataset collected? #23 : https://github.com/openai/CLIP/issues/23
