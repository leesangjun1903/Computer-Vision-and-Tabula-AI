# GLAM : All the attention you need: Global-local, spatial-channel attention for image retrieval | Image retrieval

**핵심 주장 및 주요 기여**  
이 논문은 이미지 검색 분야에서 **공간-채널적 주의(attention)** 메커니즘을 **국소(local)와 전역(global)** 두 관점에서 통합하여, 기존 연구가 각각 하나 또는 두 형태만 다룬 데 반해 네 가지 형태의 주의(국소-공간, 국소-채널, 전역-공간, 전역-채널)를 모두 결합한 **Global-Local Attention Module (GLAM)** 을 제안한다. 이를 통해 강력한 전역 디스크립터를 학습하여 표준 벤치마크에서 성능을 크게 향상시킨다.

***

## 1. 해결하고자 하는 문제  
- **인스턴스 수준 이미지 검색**에서:  
  - 전역 디스크립터 기반 방법은 효율적이나, 공간 풀링 및 주의 메커니즘을 단일 형태에만 의존해 디스크립터 표현력에 한계가 존재  
  - 기존 연구들은 국소/전역 주의, 공간/채널 주의 중 일부만을 적용해 상호작용을 종합적으로 이해하지 못함  

***

## 2. 제안하는 방법  
### 2.1 모델 구조  
Backbone(ResNet101) 뒤에 부착되는 GLAM 모듈은 입력 특징 $$F \in \mathbb{R}^{C\times H\times W}$$ 에 대해,  
1. 국소 채널 주의 $$\;A^l_c\in\mathbb{R}^{C\times1\times1}$$  
2. 국소 공간 주의 $$\;A^l_s\in\mathbb{R}^{1\times H\times W}$$  
3. 전역 채널 주의 $$\;A^g_c\in\mathbb{R}^{C\times C}$$  
4. 전역 공간 주의 $$\;A^g_s\in\mathbb{R}^{(HW)\times(HW)}$$  

단계별 특징 계산:  

$$
F^l = (F \odot A^l_c + F) \odot A^l_s + (F \odot A^l_c + F)
$$  

$$
F^g = \bigl(F \odot (V_c A^g_c)\bigr) \odot A^g_s + \bigl(F \odot (V_c A^g_c)\bigr)
$$  

여기서 $$V_c$$는 값(value) 텐서, $$\odot$$는 요소별 곱.  
국소·전역 특징과 원본 $$F$$를 소프트맥스로 가중합해 최종 $$\;F^{gl}\in\mathbb{R}^{C\times H\times W}$$ 을 얻고, GeM 풀링 및 $$\ell_2$$-정규화를 거쳐 512차원 전역 디스크립터를 생성한다.

### 2.2 수식 요약  
- 국소 채널 주의:  
  
$$
    A^l_c = \sigma(\mathrm{Conv1D}(\mathrm{GAP}(F)))
  $$

- 국소 공간 주의:  
  
$$
    A^l_s = \mathrm{Conv1\times1}(\mathrm{Concat}\_{d\in\{1,2,3\}}(\mathrm{DilatedConv}_{3\times3}^{d}(F')))
  $$

- 전역 채널 주의:  
  
$$
    A^g_c = \mathrm{softmax}\bigl(K_c^\top Q_c\bigr),\quad Q_c,K_c=\sigma(\mathrm{Conv1D}(\mathrm{GAP}(F)))
  $$

- 전역 공간 주의:  
  
$$
    A^g_s = \mathrm{softmax}\bigl(K_s^\top Q_s\bigr),\quad Q_s,K_s = \mathrm{reshape}(\mathrm{Conv1\times1}(F))
  $$

- 특징 융합:  
  
$$
    F^{gl} = w_l F^l + w_g F^g + w F,\quad (w_l,w_g,w = \mathrm{softmax}(\alpha_l,\alpha_g,\alpha))
  $$

***

## 3. 성능 향상 및 한계  
- **벤치마크 결과**: Oxford5k, Paris6k, Revisited-Oxford/Paris(1M)에서 기존 SOTA 대비 mAP 최대 +4% 이상 성능 향상  
- **국소 vs 전역 주의**: 전역 주의 단독 적용 시에도 큰 향상(최대 +7.5%), 국소 주의는 전역과 결합 시에만 의미 있는 추가 개선 (+2.8%)  
- **한계**:  
  - 전역 공간 주의 행렬 크기가 $$HW\times HW$$ 로 대규모 계산 부담  
  - 국소 주의의 공간-채널 쌍별 최적화 필요성 검증은 미흡  

***

## 4. 일반화 성능 향상 가능성  
- **다양한 태스크 적용**: GLAM이 생성하는 텐서 표현은 분류·검출·세그멘테이션 등에도 적용 가능  
- **Transformer 및 하이브리드 모델**: Vision Transformer가 전역 공간 주의만 사용하는 반면, 채널 및 국소 주의를 추가해 일반화 저하를 방지하고, 소량 데이터셋에서도 강건한 성능 발휘 기대  
- **경량화 연구**: 전역 공간 주의 압축·근사화 기법을 도입해 모바일·엣지 환경에서도 활용 가능  

***

## 5. 향후 연구 방향 및 고려 사항  
- **효율적 전역 주의 계산**: 쿼리·키 차원 차별화, 샘플링·양자화 기법 연구  
- **다중 모달 확장**: 언어·영상 통합 검색·클러스터링에 채널·공간 주의 융합  
- **적응형 융합 전략**: 데이터 특성에 따라 국소·전역 주의 비중을 동적으로 조절하는 메커니즘 설계  
- **소규모·제로샷 학습**: GLAM을 활용한 프로토타입·프록시 기반 메트릭 학습으로 일반화 성능 평가  

이 논문은 주의 메커니즘의 종합적 통합을 통해 전역 디스크립터 기반 검색의 성능 한계를 뛰어넘었으며, 앞으로 다양한 비전 모델 설계와 효율화 연구에 중요한 토대를 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/52edd610-fd6a-4d91-b17c-23f126c49a48/2107.08000v1.pdf
