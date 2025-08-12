# SiT: Self-supervised vIsion Transformer

## 1. 핵심 주장 및 주요 기여  
SiT는 **Vision Transformer(ViT)를 위한 최초의 완전 자율(self-supervised) 사전학습 프레임워크**로, 그룹 마스킹된 패치 복원을 통해 비지도(masked image modeling) 방식이 지도(supervised) 사전학습을 능가할 수 있음을 입증했다. 주요 기여는 다음과 같다:  
- Group Masked Model Learning(GMML) 제안: 이미지 패치를 그룹 단위로 마스킹하고 문맥(context)으로부터 복원하도록 학습  
- **경량 디코더** 설계: ViT 인코더 출력에 2-layer MLP만 추가해 오토인코더 구조 완성  
- **멀티태스크 학습**: GMML 기반 재구성 손실 $$L_\text{recons}$$과 대조 학습(contrastive) 손실 $$L_\text{contr}$$을 동시에 최적화  
- 소규모·중규모 데이터셋에서도 SP 대비 최대 +5.4%p 성능 향상, 대규모 데이터셋은 동등 성능 달성  

## 2. 문제 정의 · 제안 기법 · 모델 구조 · 성능 · 한계  

### 2.1 해결하고자 하는 문제  
- ViT는 강력하지만 **대규모 레이블된 데이터와 거대한 모델**이 요구되는 ‘데이터 기아(data-hungry)’ 문제  
- 비지도 사전학습(Self-supervised Pretraining, SSP)이 CNN 분야 대비 성능 격차 존재  

### 2.2 제안 방법  
1. **GMML(그룹 마스킹 재구성)**  
   - 입력 이미지 $$x$$를 패치 $$\{x_i\}$$로 분할 후, 30–70% 패치를 노이즈 또는 다른 이미지 패치로 교체해 $$\bar x$$ 생성  
   - ViT 인코더 $$E$$를 통해 데이터 토큰만 추출, 경량 디코더 $$D$$로 재구성  
   - 재구성 손실:  

$$
       L_\text{recons}(W) = \frac1N \sum_i \|\,x_i - D(E(\bar x_i))\|_1
     $$  

2. **대조 학습**  
   - SimCLR 유사 방식으로, 서로 다른 뷰 $$\tilde x, \bar x$$쌍을 positive pair로, 다른 샘플은 negative로 취급  
   - contrastive head $$\mathrm{Contr}$$ 적용 후 온도 $$\tau$$ softmax 손실:  

$$
       \ell_{i,j} = \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k\neq i} \exp(\mathrm{sim}(z_i,z_k)/\tau)},\quad
       L_\text{contr} = -\frac1N \sum_i \log \ell_{i,\tilde i}
     $$  

3. **종합 손실**  

$$
     L_\text{total} = \alpha\,L_\text{recons} + L_\text{contr},\quad
     \alpha=5\ (\text{소규모}),\ \alpha=1\ (\text{대규모})
   $$  

### 2.3 모델 구조  
- Backbone: ViT-S/16 (224×224→16×16 패치, 12-layer MSA+MLP)  
- Reconstruction head: 2-layer MLP(2048→2048)+트랜스포즈드 컨볼루션  
- Contrastive head: 2-layer MLP(4096→4096→256) + batch-norm + GeLU  
- Momentum encoder: EMA로 업데이트  

### 2.4 성능 향상  
- **소규모 데이터**(Flowers, Pets 등): 지도학습 대비 +20–60%p 대폭 향상  
- **대규모 사전학습 후 전이**(ImageNet-1K→각종 소형 분류): 동급 또는 소폭 개선  
- **다중 레이블 분류**(PASCAL VOC, MS-COCO): mAP +0.5–0.6p 향상  
- **비디오 인스턴스 분할**(DAVIS-2017): J&amp;F 기준 +2–5p 우위  

### 2.5 한계  
- **계산 비용**: ViT 기반 장시간(3000epochs) 학습 필요  
- **디코더 단순화의 제약**: 복잡한 구조적 복원 혹은 고해상도 복원 한계  
- **하이퍼파라미터 민감성**: 마스킹 비율·α 값 조정 필수  

## 3. 일반화 성능 향상 관점  
- GMML은 **로컬 패치 간 통계적 상관관계**를 학습시켜 ViT의 inductive bias 부재 문제 완화  
- 대조 학습과 결합해 **전역 표현도 함께 최적화**, 소량 레이블 데이터로도 높은 전이 성능  
- **t-SNE 시각화**에서 비지도 단계만으로도 뚜렷한 클래스 클러스터 형성, 전이 후 더욱 분리도 향상  
- 다양한 downstream(분류·검출·분할)에서 **동일한 백본**으로 우수한 일반화 확인  

## 4. 향후 연구 영향 및 고려사항  
- **범용 자율 사전학습**: 소·중·대규모 환경에서 ViT 초기화로 광범위 적용 가능  
- **다중 태스크 통합**: reconstruction, contrastive 외에도 **컨텍스트 예측·분리 손실** 등 병합 연구  
- **효율화**: 디코더 경량화, 마스킹 전략 자동화(mask ratio 자동 조정)로 학습 시간·자원 절감  
- **확장성**: 고해상도 영상·3D 포인트 클라우드·멀티모달에 GMML 원리 응용  
- **하이퍼파라미터 일반화**: α·mask 비율 등 데이터 특성 무관한 자동 튜닝 알고리즘 필요

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4e01c20a-1ec2-407f-8ef0-4499ab5f5316/2104.03602v3.pdf
