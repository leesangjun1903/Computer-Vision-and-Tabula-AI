# Transformer in Transformer | Image classification

## 주요 주장 및 기여 요약  
**Transformer in Transformer (TNT)** 논문은 비전 트랜스포머(ViT)의 한계를 극복하기 위해, 이미지 패치를 “시각적 문장(visual sentence)”으로 보고 이를 다시 더 작은 “시각적 단어(visual word)”로 세분화하여 두 단계(Self-attention)로 특징을 추출하는 **내부–외부 이중 트랜스포머 구조**를 제안한다.  
- **핵심 주장**: 이미지의 지역적 세부 정보를 손실 없이 보존하기 위해 패치 내부에도 self-attention을 적용해야 한다.  
- **주요 기여**:  
  1. 패치(16×16)를 다시 4×4 크기 하위 패치(m=16)로 분할하여 “단어” 수준의 세밀한 특징을 추출하는 **내부 트랜스포머** 도입.  
  2. 전체 패치 간 관계를 학습하는 **외부 트랜스포머**와 결합하여 이중 구조로 성능을 향상.  
  3. ImageNet 상 81.5% Top-1 정확도 달성(DeiT-S 대비 +1.7%).  

## 해결하고자 하는 문제  
- ViT 계열 모델은 이미지 패치를 일정 크기로 나누어 global attention만 수행하므로, **패치 내부의 세부 구조 정보**가 충분히 반영되지 않아 작은 객체나 텍스처 등에서 성능 저하가 발생.  
- 자연 이미지의 고해상도·다양한 스케일 정보 포착이 부족.  

## 제안 방법  
1. **입력 분할**  
   - 원본 이미지 $$I\in \mathbb{R}^{H\times W\times 3}$$를 $$n$$개의 패치 $$\{X_i\}_{i=1}^n$$, 크기 $$p\times p$$로 분할.  
   - 각 패치 $$X_i$$를 다시 $$m$$개의 하위 패치(단어) $$\{x_{i,j}\}_{j=1}^m$$, 크기 $$s\times s$$로 분할.  

2. **단어 임베딩**  

$$
     y_{i,j} = \mathrm{FC}(\mathrm{Vec}(x_{i,j}))\in\mathbb{R}^c,\quad 
     Y^i = [y_{i,1},\dots,y_{i,m}]
   $$

3. **내부 트랜스포머(Visual Word-level)**  

$$
     Y_{\ell}^{\prime i} = Y_{\ell-1}^{i} + \mathrm{MSA}(\mathrm{LN}(Y_{\ell-1}^i)),\quad
     Y_{\ell}^{i} = Y_{\ell}^{\prime i} + \mathrm{MLP}(\mathrm{LN}(Y_{\ell}^{\prime i}))
   $$

$$(\ell=1,\dots,L)$$  

4. **외부 트랜스포머(Visual Sentence-level)**  
   - 클래스 토큰 포함 문장 임베딩 $$Z_0=[Z_{\mathrm{cls}},Z_1^0,\dots,Z_n^0]$$.  
   - 단어 임베딩 투영 후 문장 임베딩에 합산:  
   
$$
       Z_i^{\ell-1} \gets Z_i^{\ell-1} + \mathrm{FC}(\mathrm{Vec}(Y_i^\ell))
     $$  
     
  - 그 후 standard transformer:  

$$
       Z_\ell' = Z_{\ell-1} + \mathrm{MSA}(\mathrm{LN}(Z_{\ell-1})),\quad
       Z_\ell = Z_\ell' + \mathrm{MLP}(\mathrm{LN}(Z_\ell'))
     $$

5. **모델 구조**  
   - Inner block과 Outer block을 $$L$$회 반복 쌓은 **TNT 블록**  
   - 최종 클래스 토큰에 FC 헤드를 적용하여 분류  

## 성능 향상 및 한계  
- **ImageNet**: TNT-S 81.5% Top-1(DeiT-S 79.8% 대비 +1.7%)  
- **파라미터 대비**: +8% 증가, **FLOPs 대비**: +14% 증가로 성능 이득  
- **전이 학습(Transfer)**: CIFAR-100, Oxford-Pets, Flowers, iNaturalist 등에서 DeiT 대비 평균 1% 이상 향상  
- **한계**:  
  1. **계산 복잡도 증가**: c≪d 이지만 자잘한 self-attention이 FLOPs와 메모리 소모 유발  
  2. **효율성–정확도 트레이드오프**: 경량화된 디바이스 적용 시 주의 필요  
  3. **단일 해상도 실험**: 다양한 입력 스케일 안정성 검증 추가 요구  

## 일반화 성능 향상 관점  
- **로컬 패치 내부 상호작용**을 explicit하게 학습함으로써, 작은 객체나 세밀한 세부 특징에서도 **일관된 표현 학습**이 가능해짐.  
- **Transfer Learning** 실험 결과, 데이터가 적거나 fine-grained한 데이터셋에서도 TNT가 더 안정적인 학습 곡선을 보이며, 과적합 위험이 감소.  
- **다양한 다운스트림 태스크**(객체 탐지·세그멘테이션)에서 순수 트랜스포머 백본으로서 경쟁력 입증.  

## 향후 영향 및 고려할 점  
- **후속 연구 영향**:  
  - 하위-상위 레벨 상호작용을 강조하는 멀티그레인 구조 연구 확장  
  - 패치 분할 방식 및 동적 단어 수 조절 기법 개발  
- **연구 시 고려사항**:  
  1. **효율성 최적화**: SE 모듈, 채널 어텐션, 경량화블록 적용을 통한 FLOPs 절감  
  2. **다중 스케일 처리**: 가변 패치 크기·단어 크기 스킴으로 다양한 해상도 대응  
  3. **실제 응용 검증**: 모바일·임베디드 환경에서 추론 속도 및 메모리 제약 하에서 성능 확인  

**Transformer in Transformer**는 로컬 디테일을 보존하는 새로운 트랜스포머 패러다임으로, 앞으로 비전 트랜스포머 연구에서 **다중 레벨 어텐션**과 **효율적 표현 학습**을 위한 핵심 아이디어를 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c43c0e44-fa65-400c-8e05-ffe7cdac6eca/2103.00112v3.pdf
