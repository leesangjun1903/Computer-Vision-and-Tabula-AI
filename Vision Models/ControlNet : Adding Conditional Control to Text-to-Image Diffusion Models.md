# ControlNet : Adding Conditional Control to Text-to-Image Diffusion Models | Image generation

ControlNet은 사전 훈련된 대규모 텍스트-이미지 확산 모델(예: Stable Diffusion)에 **공간적 조건 제어를 추가하는 신경망 아키텍처**입니다. 이 기술은 기존 모델의 강력한 이미지 생성 능력을 유지하면서 사용자가 원하는 구체적인 조건(예: 윤곽선, 깊이, 포즈)을 반영한 이미지를 생성할 수 있게 합니다. 핵심 내용을 다음과 같이 설명합니다:

---

### 1. **핵심 아이디어: 조건부 제어 구조**
- **"잠금-훈련" 이중 구조**:  
  원본 확산 모델의 가중치를 **고정된 복사본(Locked Copy)** 으로 유지하면서, 동시에 **훈련 가능한 복사본(Trainable Copy)** 이 추가 조건(예: 스케치, 깊이 맵)을 학습합니다.  
  → 기존 모델의 안정성을 해치지 않으면서 새로운 제어 기능을 추가합니다[1][2].

- **제로 컨볼루션(Zero Convolution)**:  
  두 구조를 연결하는 1×1 컨볼루션 레이어로, **가중치와 편향을 0으로 초기화**합니다.  
  - 초기에는 출력이 0이므로 원본 모델에 영향 없음.  
  - 훈련 중 점진적으로 학습되어 조건 정보를 주입[1][3][4].

---

### 2. **동작 원리 및 장점**
- **조건 입력 처리**:  
  사용자가 제공한 조건(예: Canny 에지, 인간 포즈)을 인코딩해 확산 모델의 U-Net 블록에 주입합니다.  
  → 텍스트 프롬프트와 결합해 정밀한 이미지 생성 가능[5][6].

- **데이터 효율성**:  
  - 소규모 데이터셋(1M)과 호환되어 산업용 수준의 정확도 달성[7][4].

- **다중 조건 지원**:  
  에지, 깊이, 세그멘테이션 맵 등을 **동시에 적용**해 복합적 제어 가능[1][2].

---

### 3. **실험 결과 및 적용 사례**
- **정량적 평가**:  
  Stable Diffusion 기반 ControlNet이 **깊이→이미지, 포즈→이미지** 변환에서 SOTA 성능 달성[5][6].

- **응용 분야**:  
  - **스케치→사진**: 사용자 드로잉을 사실적 이미지로 변환.  
  - **포즈 제어**: 입력 포즈를 정확히 따르는 인물 생성.  
  - **3D 재구성**: 깊이 맵으로 객체의 3D 구조 제어[1][7].

---

### 4. **기술적 의의**
- **안전한 미세 조정(Fine-Tuning)**:  
  제로 컨볼루션으로 인해 **초기 훈련 시 원본 모델의 출력이 변형되지 않음**.  
  → 대규모 모델을 망가뜨리지 않고 개인 장비에서도 훈련 가능[3][4].

- **확장성**:  
  사전 훈련된 모든 확산 모델(예: SD v1.5, v2)과 호환되어 **플러그인 방식으로 적용** 가능[2].

---

### 결론
ControlNet은 **"조건부 생성"의 정밀도를 혁신**한 기술로, 기존 텍스트-이미지 모델의 한계를 해결합니다. 사용자가 공간적 조건을 구체적으로 지정할 수 있어 디자인, 게임, 가상 현실 등 다양한 분야에서 활용될 수 있습니다. 코드는 [GitHub](https://github.com/lllyasviel/ControlNet)에서 공개되어 있습니다[3][4].

[1] https://arxiv.org/abs/2302.05543
[2] https://paperswithcode.com/paper/adding-conditional-control-to-text-to-image
[3] https://github.com/lllyasviel/ControlNet
[4] https://huggingface.co/papers/2302.05543
[5] https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf
[6] https://www.semanticscholar.org/paper/Adding-Conditional-Control-to-Text-to-Image-Models-Zhang-Rao/efbe97d20c4ffe356e8826c01dc550bacc405add
[7] https://lllyasviel.github.io/misc/202309/cnet_supp.pdf
[8] https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1091_ECCV_2024_paper.php
[9] https://www.studocu.com/en-us/document/cornell-university/advanced-machine-learning/2302-a-rese/64027419
[10] https://arxiv.org/abs/2312.08768


# Reference
https://junia3.github.io/blog/controlnet

https://github.com/lllyasviel/ControlNet?tab=readme-ov-file

https://ffighting.net/deep-learning-paper-review/diffusion-model/controlnet/
