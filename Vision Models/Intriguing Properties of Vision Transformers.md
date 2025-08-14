# Intriguing Properties of Vision Transformers

## 1. 논문의 핵심 주장 및 주요 기여
이 논문은 **Vision Transformer(ViT)** 계열 모델이 전통적 CNN 대비 다음과 같은 탁월한 특성을 지닌다는 점을 체계적으로 입증한다.

- **심한 폐색(occlusion)에 강건**: 이미지의 최대 80%를 무작위로 가렸을 때도 Top-1 정확도가 약 60% 수준으로 유지된다.  
- **텍스처 편향(texture bias) 해소**: ViT는 국소 텍스처보다 전체 형상(shape) 정보에 더 민감해, 사람 수준의 형태 인식 능력을 달성한다.  
- **픽셀 수준 지도 없이도 의미론적 분할(segmentation) 가능**: 모양(shape) 정보에 초점을 둔 별도 토큰(shape token) 학습만으로 배경과 전경을 자동 분할한다.  
- **다양한 다운스트림 작업에서 뛰어난 일반화 성능**: 단일 ViT의 여러 블록에서 추출한 토큰을 앙상블해 전통 분류, 소수 샷(few-shot) 학습, 장기 꼬리(long-tail) 분류 등에서 CNN 대비 일관되게 우수한 성능을 보인다.

## 2. 해결 과제 및 제안 기법

### 2.1 해결하고자 하는 문제
- CNN이 분포 변화(distribution shift), 자연·적대적 교란(common corruptions, adversarial perturbations), 공간적 변형(spatial permutations) 등에 취약한 반면, ViT가 이러한 **실세계의 잡음(nuisances)** 에 얼마나 견고한지 체계적으로 분석하고자 함.

### 2.2 제안 방법
1. **PatchDrop**: 이미지 패치를 무작위(Random), 전경(Salient), 배경(Non-salient) 기준으로 일정 비율 M/N만큼 마스킹해( $f(x′) , \text{where}  x′ = PatchDrop(x)$ )  
   – 정보 손실률(IL) = M/N  
   – Top-1 정확도로 견고성 평가  
2. **형상 편향 학습(shape-biased training)**  
   – 스타일라이즈드 ImageNet(SIN)으로 학습:  

$$
       \min_{\theta} \, \mathbb{E}\_{(x,y)\sim \mathrm{SIN}} \mathcal{L}\big(f_\theta(x), y\big)
     $$  
   
   – **Shape Token** 추가: 분류(class) 토큰과 별개로 모양(feature) 전용 토큰을 두어,  

$$
       z_{\mathrm{cls}},\, z_{\mathrm{shape}} = \mathrm{Transformer}(x)
     $$  
     
  두 토큰에 대한 지식 증류(distillation)로 CNN 기반 모양 교사(ResNet50-SIN) 지식을 전수.  

3. **토큰 앙상블(feature ensemble)**  
   – 각 블록의 클래스 토큰 $$z^l_{\mathrm{cls}}$$을 개별 선형 분류기로 학습하거나, 마지막 4개 블록을 연결(concatenate)해 전이 학습에 이용.

### 2.3 모델 구조
- 백본: ViT, DeiT, T2T 등 셀프어텐션 기반 트랜스포머  
- 입력: 14×14 패치(16×16 픽셀) 시퀀스 + 위치 인코딩  
- 추가 구성 요소:  
  1. **분류 토큰(class token)**  
  2. **모양 토큰(shape token)** (shape-distilled 모델에만)  
  3. **어텐션 기반 분할**: 최종 어텐션 맵으로 픽셀 단위 분할 구현  

## 3. 성능 향상 및 한계

### 3.1 성능 향상
- **폐색 강건성**: ResNet50이 50% PatchDrop 시 0.1% 정확도인 데 반해, DeiT-Small은 70% 유지. 90% 마스킹에도 Deit-Base는 37% 정확도 달성.  
- **모양 편향(shape bias)**: SIN 학습 시 ViT는 CNN 대비 모양 정확도 최대 87%[Fig.6]. shape token 추가로 분류 정확도(∼75%)와 모양 편향(∼0.47) 사이 균형 달성.  
- **자동 분할(segmentation)**: 지도 학습 없이 Jaccard 지수 42% 수준(PASCAL VOC12)  
- **전이 학습**  
  – 일반 분류: CUB, Flowers, iNaturalist 등에서 ResNet50 대비 평균 5~10%p 향상  
  – Few-shot: Meta-Dataset 다수 과제에서 ViT 앙상블이 CNN 대비 ≈3%p 우수  

### 3.2 한계 및 트레이드오프
- **모양 편향 vs. 교란 강건성**: SIN 학습된 모델은 자연 교란(common corruptions)과 적대적 공격에 더 취약(Fig.12-13).  
- **대규모 파라미터 의존**: ViT-Large는 수백만 파라미터로 작은 데이터셋만으로는 안정적 학습 어려움. 추가 데이터·증강 필요.  
- **위치 정보 의존도**: 패치 셔플(shuffle) 내성은 높으나, 극단적 변형 시 성능 저하.  

## 4. 일반화 성능 향상 관점
- **동적 수용 영역(dynamic receptive field)**: 어텐션 메커니즘이 남은 정보에 자동 집중, 다양한 잡음에도 안정적 표현 학습.  
- **토큰 앙상블 전략**: 서로 다른 깊이의 특징을 결합해, 세분화된 도메인 적응 능력 및 소수 샷 학습 일반화.  
- **모양·텍스처 이중 학습**: 별도 토큰으로 서로 다른 시각적 단서를 동시에 보존해, 형상 기반 도메인(스케치·페인팅)에서도 일관된 인식 가능.

## 5. 향후 연구 영향 및 고려사항
- **다중 토큰 통합 연구**: shape, texture, attention 토큰 간 상호 보완적 결합 방식 탐색  
- **셀프슈퍼비전 결합**: DINO 등 비지도 기법과 모양 편향 융합으로 분할·분포 적응 강화  
- **윤리·편향 이슈**: 기존 ImageNet 데이터의 인구통계·프라이버시 편향 해소된 데이터셋 활용 필요  
- **경량화 및 효율화**: 작은 데이터셋·리소스 제약 환경에서도 ViT 학습 안정화 위한 증강·정규화 기법 개발  

이 논문은 **트랜스포머 기반 비전 모델의 견고성과 일반화 가능성**을 입증함으로써, 이후 연구에서 **다양한 시각 단서의 분리·통합**, **셀프어텐션의 적절한 활용**, **공정하고 안전한 데이터셋** 구축 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/b748f7ca-9995-4e6d-ac9e-178388e7b0db/2105.10497v3.pdf
