# FixEfficientNet : Fixing the Train–Test Resolution Discrepancy | Image classification, Data augmentation, preprocessing

## 1. 핵심 주장과 주요 기여  
**핵심 주장**  
훈련 시 데이터 증강(Train-time RandomResizedCrop)과 테스트 시 전처리(CenterCrop) 간에 발생하는 해상도·스케일 불일치가 모델 성능을 저해하므로, 훈련 해상도와 테스트 해상도를 분리하여 최적화해야 한다.  

**주요 기여**  
1. 훈련·테스트 전처리 간 객체 크기(distribution of apparent object size) 불일치를 이론·실험으로 분석  
2. 테스트 해상도를 증가시키고(batch-norm과 classifier 일부만 fine-tuning) 통계적 스케일 편향을 보정하는 간단한 해상도 적응 기법 제안  
3. ResNet-50, PNASNet-5-Large, ResNeXt-101 32×48d 등 다양한 모델에 적용하여 ImageNet 성능 1–2%p 향상 달성  
4. 저해상도 훈련으로 연산량·메모리 절감(최대 3× 속도 향상)과 높은 테스트 해상도 적용의 속도–정확도 트레이드오프 제시  

***

## 2. 문제 정의, 제안 방법, 모델 구성, 성능 향상, 한계

### 2.1 해결하려는 문제  
- 훈련 시 RandomResizedCrop에 따라 객체가 다양한 크기로 확대·축소되나,  
- 테스트 시 CenterCrop으로만 고정된 크기를 사용 → 동일 객체의 CNN 입력 크기가 일치하지 않음  
- 이로 인해 훈련 중 학습된 스케일 불변성 범위와 실제 테스트 분포가 달라져 일반화 성능 저하  

### 2.2 제안 방법  
1) **객체 크기 균형**  
   - 훈련 시 평균 스케일 왜곡 비율 $$E\bigl[\tfrac{r_{\text{test}}}{r_{\text{train}}}\bigr] \approx 0.8$$임을 분석  
   - 테스트 해상도 $$K_{\text{image}}^{\text{test}}, K_{\text{test}}$$를 $$\alpha\approx1.25$$ 배로 늘려 균형 회복  
2) **통계 편향 보정**  
   - 해상도 변화로 인한 global average pooling 후 activation 분포 변화(희소도 증가·분포 확장)를 제어  
   - 배치 정규화와 최종 classifier만 제한적으로 fine-tuning하여 새로운 해상도에 맞춤  

#### 수식 요약  
- 훈련 시 객체 크기:  

$$
r_{\text{train}} = s \cdot r = \frac{k\,K_{\text{train}}}{\sigma}\,r_1
$$

- 테스트 시 객체 크기:  

$$
r_{\text{test}} = k\,K_{\text{image}}^{\text{test}}\,r_1
$$

- 크기 비율:  

$$
\frac{r_{\text{test}}}{r_{\text{train}}}
= \sigma \,\frac{K_{\text{image}}^{\text{test}}}{K_{\text{train}}},\quad 
E\!\bigl[\tfrac{r_{\text{test}}}{r_{\text{train}}}\bigr]\approx0.80
$$

### 2.3 모델 구조  
- 기본 네트워크: ResNet-50, PNASNet-5-Large, ResNeXt-101 32×48d  
- 변경점: **모델 아키텍처는 그대로** 유지하며, 테스트 해상도와 배치 정규화·최종 fully-connected layer만 fine-tuning  

### 2.4 성능 향상  
- ResNet-50 (Train 224→Test 384): Top-1 77.0% → 79.1% (+2.1%p)  
- PNASNet-5-Large (Train 331→Test 480): 82.7% → 83.7% (+1.0%p)  
- ResNeXt-101 32×48d (Train 224→Test 320): 85.4% → 86.4% (+1.0%p)  
- **저해상도 훈련+해상도 적응**: Train 128→Test 224에서도 77.1% 달성, 원래 224 훈련(77.0%)보다 우수  
- 연산 효율: Train 224 해상도 대비 Train 128 + 해상도 적응 방식이 11% 연산 증가로 동일 성능 달성, Train-Test 224→384 fine-tuning은 2.3× 빠름  

### 2.5 한계  
- **fine-tuning 비용**: 별도 fine-tuning 단계가 추가되므로 전체 워크플로우 복잡성 증가  
- **배치 정규화 의존**: 작은 batch size나 BN 비활성화 환경에서는 효과 감소 우려  
- **추가 하이퍼파라미터**: 최적 $$\alpha$$, fine-tuning epoch 수, augmentation 정책 조정 필요  

***

## 3. 일반화 성능 향상 가능성  
- **스케일 일관성 확보**: 훈련·테스트 간 객체 스케일 분포 매칭으로 스케일 민감도 저하  
- **통계 편향 보정**: 배치 정규화 조정만으로도 activation 분포 회복 → overfitting 감소  
- **전이 학습 개선**: iNaturalist·Herbarium fine-grained 분류 대회에서 전이 성능(Top-1) 각각 +1.3~+1.7%p 상승  
- **다양한 모델 적용성**: CNN 계열 전반에 간단히 적용 가능  

***

## 4. 향후 연구에 미치는 영향 및 고려사항  
- **고해상도 테스트 전략**: 기존 224×224 일원화 관행 타파, 해상도 적응이 모델 성능 개선 지름길  
- **경량화 모델에도 적용**: MobileNet, EfficientNet 등 경량 모델에서 훈련 해상도 낮추고 해상도 적응 적용 시 추가 효율 가능  
- **Normalization 기법 연구**: BN 외 LayerNorm·GroupNorm 등 대체 정규화와 해상도 적응 시너지 탐색  
- **End-to-End 해상도 적응**: fine-tuning 없이 훈련 단계부터 멀티해상도 학습 통합 방안 연구  

이 논문은 “훈련과 테스트의 전처리 간 해상도·스케일 간극”을 규명하고, 테스트 해상도 상승과 제한적 fine-tuning만으로 간단하고 효과적으로 이를 보정하는 방법을 제시하여, 향후 스케일 일관성 확보를 위한 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d620e549-45d9-48b8-93d8-fe1191b788a4/1906.06423v4.pdf
