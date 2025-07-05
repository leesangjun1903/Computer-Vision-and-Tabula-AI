# PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance | Image restoration, Face restoration

## 개요  
PGDiff는 사전 학습된 확산 모델(pre-trained diffusion model)의 강력한 생성(prior)을 활용하여, 기존의 복원 과제에서 필요로 하던 명시적 열화 모델(degradation model)을 배제하고도 다양한 얼굴 복원(face restoration) 작업을 수행할 수 있는 **부분 가이드(Partial Guidance)** 방식을 제안한 연구입니다[1].

---

## 1. 배경 및 동기  
전통적으로 이미지 복원(task-specific training) 방식은 특정 열화(degradation) 과정을 알고 있어야만 제대로 작동했으나, 실제 환경에서는 블러, 노이즈, 압축 아티팩트 등이 복합적으로 섞여 있어 정확한 모델링이 어렵습니다[1].  
반면 확산 모델(diffusion model)은 노이즈를 점진적으로 제거하며 고품질 이미지를 생성하는 능력이 뛰어나지만, 기존의 안내(guidance) 기법은 주로 알려진 열화 모델을 활용해 제한된 문제만 해결할 수 있었습니다[1].

---

## 2. 부분 가이드(Partial Guidance) 개념  
1. **열화 프로세스 대신 원하는 속성(Properties) 모델링**  
   - 이미지 구조(structure), 색 통계(color statistics) 등 고품질 이미지에서 쉽게 얻을 수 있는 속성에 주목하여, 역확산(reverse diffusion) 과정 중에 이 속성으로만 안내합니다[1].  
2. **분류기 안내(Classifier Guidance)**  
   - 각 속성에 대응하는 분류기(classifier)를 정의하고, 중간 출력(intermediate output)에 대해 속성 목표값(target)에 대한 손실(loss)을 계산하여 그래디언트를 역전파(back-propagation)합니다[1].  
3. **동적 안내 스킴(Dynamic Guidance Scheme)**  
   - 그래디언트 스케일(gradient scale)을 중간 이미지 변화량에 비례해 동적으로 조정하여 유사도를 높이고,  
   - 각 단계에서 여러 번의 그래디언트 스텝을 수행해 노이즈가 많은 초기 단계에서도 강력한 안내를 보장합니다[1].  

---

## 3. 복원 과제별 가이드 구성  
|작업(task)|모델링 속성(property)|분류기(classifier)|목표(target)|  
|:---:|:---|:---|:---|  
|**맹복원(Blind Restoration)**|부드러운 의미(smooth semantics)|사전 학습된 복원기(restorer)|MSE 예측값|  
|**색채화(Colorization)**|명도(lightness), 색 통계(color stats)|rgb2gray, AdaIN|입력 명도, 평균·분산|  
|**인페인팅(Inpainting)**|마스크된 영역 외부|마스크 곱(mask \*)|입력 이미지|  
|**참조 기반 복원(Ref-based)**|신원(identity)|ArcFace|참조 이미지 특징|  
|**복합 작업(Old photo)**|맹복원 + 색채화 + 인페인팅|–|속성별 가중합|  

각 작업에서는 해당 속성만 안내하고, 디퓨전 모델 자체의 생성 능력으로 나머지 디테일을 복구합니다[1].

---

## 4. 추가 기능 및 확장  
- **복합 작업(Composite Tasks)**: 속성별 손실을 단순 합산하여 복수 과정을 동시에 처리합니다(예: 구사진 복원)[1].  
- **추가 손실 통합**: 지각(perceptual) 손실과 적대적(adversarial) 손실을 분류기 안내에 포함시켜 결과 품질을 더 높일 수 있습니다[1].

---

## 5. 실험 결과  
- **맹복원**: GDP, DDNM 등 기존 확산 기반 기법 대비 실제 열화에서 잔존하는 아티팩트 없이 디테일 복원이 우수함을 확인[1].  
- **색채화**: 명도와 색 통계를 이용한 안내만으로도 풍부한 색감과 다양한 스타일 생성이 가능함[1].  
- **인페인팅**: 마스크 외부만 안내하여 자연스러운 합성이 가능함[1].  
- **구사진 복원**: 복합 안내로 스크래치 제거, 색 복원, 디테일 복원까지 한 번에 처리[1].  
- **참조 기반 복원**: ArcFace 손실 통합으로 인물 고유의 특징(눈동자 색, 주름 등) 복원이 강화됨[1].  

---

## 6. 결론  
PGDiff는 열화 과정을 가정하지 않고도 고품질 이미지 속성을 모델링하는 **부분 가이드** 방식을 통해 확산 모델을 다수의 얼굴 복원 작업에 범용적으로 적용할 수 있음을 보였습니다. 복합 작업에서도 손쉽게 확장 가능하며, 기존 확산 기반·태스크 특화 모델 모두에 비해 경쟁력 있는 성능을 달성합니다[1].

---

## 참고 문헌  
1. Peiqing Yang et al., “PGDiff: Guiding Diffusion Models for Versatile Face Restoration via Partial Guidance,” NeurIPS 2023[1].

[1] https://arxiv.org/abs/2309.10810
[2] https://www.mdpi.com/1424-8220/24/22/7112
[3] https://arxiv.org/abs/2411.17163
[4] https://dl.acm.org/doi/10.1145/3664647.3680853
[5] https://dl.acm.org/doi/10.1145/3647649.3647703
[6] https://arxiv.org/abs/2410.09864
[7] https://ieeexplore.ieee.org/document/10734729/
[8] https://www.semanticscholar.org/paper/2690ee3ac3d586eebb4df346308bc2a5b13a5bb1
[9] https://github.com/pq-yang/PGDiff
[10] https://openreview.net/forum?id=yThjbzhIUP
[11] https://arxiv.org/html/2309.10810
[12] https://chatpaper.com/pt/paper/10878
[13] https://chatpaper.com/zh-CN/chatpaper/paper/10878
[14] https://www.comp.hkbu.edu.hk/wsb2025/slides/Chen_Change_Loy.pdf
[15] https://proceedings.neurips.cc/paper_files/paper/2023/file/661c37f3b098bdee53fd7d9c4ef6964a-Supplemental-Conference.pdf
[16] https://www.semanticscholar.org/paper/27e70a81b9a77902d9625a6b2f525231ecad8d25
[17] https://ieeexplore.ieee.org/document/10822469/
[18] https://dl.acm.org/doi/10.5555/3666122.3667520
