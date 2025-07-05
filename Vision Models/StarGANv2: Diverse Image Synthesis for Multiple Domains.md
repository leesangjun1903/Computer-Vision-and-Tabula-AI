# StarGAN v2: Diverse Image Synthesis for Multiple Domains | Image generation

## 1. 핵심 주장 및 주요 기여  
StarGAN v2는 **하나의 프레임워크**로 여러 도메인 간에 **다양하고 고품질의 이미지 변환**을 동시에 지원하며, 기존 방법 대비 **시각적 품질, 스타일 다양성, 확장성** 면에서 뛰어남[1].  
주요 기여는 다음과 같다[1]:  
- **도메인별 스타일 코드** 도입으로 무작위 잠재벡터 또는 참조 이미지를 통해 다양한 스타일 생성  
- **스타일 매핑 네트워크(mapping network)**와 **스타일 인코더(style encoder)** 설계로 도메인별 스타일 학습  
- CelebA-HQ와 새로 공개된 AFHQ(동물 얼굴) 데이터셋에서 기존 기법 대비 FID 감소 및 LPIPS 증가로 우수성 입증  

## 2. 해결 문제  
StarGAN v2는 다음 두 가지 주요 문제를 해결하고자 함[1]:  
1. **다양성 부족**: 기존 StarGAN은 도메인 레이블만으로 결정적 결과만 생성  
2. **도메인 확장성 문제**: 다수 도메인의 쌍별 모델 학습의 비효율성  

## 3. 제안 방법  
### 3.1 스타일 코드 및 학습 목표  
- **스타일 코드** $$s$$를 매핑 네트워크 $$F$$ 또는 스타일 인코더 $$E$$ 통해 생성  
- **적대 손실(Adversarial Loss)**:

$$
L_{adv} = \mathbb{E}\_{x,y}[\log D_y(x)] + \mathbb{E}\_{x,\tilde y,z}[\log(1 - D_{\tilde y}(G(x,\tilde s)))]
$$  

- **스타일 복원 손실(Style Reconstruction)**:

$$
L_{sty} = \mathbb{E}\_{x,\tilde y,z}\| \tilde s - E_{\tilde y}(G(x,\tilde s))\|_1
$$ 

- **다양성 촉진 손실(Diversity Sensitive Loss)**:

$$
L_{ds} = \mathbb{E}_{x,\tilde y,z_1,z_2}\|G(x,\tilde s_1)-G(x,\tilde s_2)\|_1
$$  

- **사이클 일관성 손실(Cycle Consistency)**:
  
$$
L_{cyc} = \mathbb{E}_{x,y,\tilde y,z}\|x - G(G(x,\tilde s),s)\|_1
$$  

전체 목적함수는  

$$
\min_{G,F,E}\max_D L_{adv} + \lambda_{sty}L_{sty} - \lambda_{ds}L_{ds} + \lambda_{cyc}L_{cyc}
$$  

으로 정의됨[1].  

### 3.2 모델 구조  
- **Generator**: 입력 이미지에 AdaIN을 통해 스타일 주입, 다운샘플·업샘플·잔차 블록 반복[1]  
- **Mapping Network**: 잠재벡터 $$z$$→스타일 코드 $$s$$, 도메인별 출력 브랜치 구조  
- **Style Encoder**: 참조 이미지→도메인별 스타일 코드 추출  
- **Discriminator**: 도메인별 다중 브랜치로 진위 판별  

## 4. 성능 향상 및 한계  
- **시각적 품질**: CelebA-HQ FID 13.7, AFHQ FID 16.2로 기존 대비 2× 이상 개선[1]  
- **다양성**: LPIPS 지표에서 최고 값 달성  
- **한계**: 고해상도(>256×256) 훈련 비용 높음, 극단적 도메인 차이에서는 스타일 추출 실패 가능성 존재[1]  

## 5. 일반화 성능 향상 가능성  
- **공유 파라미터** 학습으로 도메인 불변 특성 강건하게 캡처  
- FFHQ 등 **미지의 데이터셋**에 대한 실험에서 참조 스타일 유연하게 전이 성공  
- 도메인 수 확장 및 소수 샘플 환경에서도 **Few-shot** 방식과 결합 시 일반화 강화 여지 있음[1]  

## 6. 향후 연구 영향 및 고려사항  
- **영향**: 다중 도메인·다양성 동시 달성 모델 설계의 표준 제시  
- **고려점**:  
  - **고해상도 지원**을 위한 경량화 및 분산훈련 필요  
  - **스타일 공간 해석** 강화로 제어 가능성 확대  
  - **소수 샷 및 도메인 편향** 문제 해결을 위한 메타러닝 통합  

---

[1] Y. Choi et al., “StarGAN v2: Diverse Image Synthesis for Multiple Domains,” *arXiv*, 2020.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4156cbc2-59d7-440b-98a4-1e5ec0a4b900/1912.01865v2.pdf
