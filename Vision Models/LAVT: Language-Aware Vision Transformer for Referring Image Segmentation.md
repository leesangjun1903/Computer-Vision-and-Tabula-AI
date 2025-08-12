# LAVT: Language-Aware Vision Transformer for Referring Image Segmentation | Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
**핵심 주장**  
– 기존의 “비전·언어 독립 인코딩 후 디코더에서 융합” 패러다임을 버리고, 비전 트랜스포머의 인코딩 단계에서 언어 정보를 조기 융합함으로써 더 강력한 크로스모달 정렬이 가능하다.  

**주요 기여**  
1. 언어-인식 비주얼 인코딩(Language-Aware Visual Encoding): Swin Transformer 기반의 4단계 인코더 각 단계에서 픽셀-단어 어텐션(PWAM)을 통해 언어 특징을 시공간 위치별로 융합.  
2. 경량화 마스크 예측: 복잡한 크로스모달 디코더 불필요, 인코더 단계에서 통합된 특징만으로 간단한 디코더로도 정교한 분할 성능 달성.  
3. 최첨단 성능 달성: RefCOCO, RefCOCO+, G-Ref 벤치마크에서 oIoU 기준 기존 대비 최대 8.57%p 절대 향상.  

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능  

### 2.1 해결하고자 하는 문제  
- 자연어 표현(referring expression)으로 지정된 객체를 픽셀 단위로 분할하는 **Referring Image Segmentation**  
- 도전 과제: 자유자연어의 풍부한 구문·어휘 정보와 시각적 특징을 효과적으로 정렬·융합  

### 2.2 제안 방법  
1. **언어 특징 추출**  
   - BERT-base로 표현 임베딩 $$L\in\mathbb{R}^{C_t\times T}$$  
2. **언어-인식 비주얼 인코딩**  
   - 비전 인코더(Swin Transformer) 4단계 각 출력 $$V_i\in\mathbb{R}^{C_i\times H_i\times W_i}$$  
   - 픽셀-단어 어텐션 모듈(PWAM)로 단계별 언어 융합  

```math
       \begin{aligned}
         V_{i,q}&=\mathrm{flatten}(\omega_{i,q}(V_i)),\quad 
         L_{i,k}=\omega_{i,k}(L),\quad 
         L_{i,v}=\omega_{i,v}(L),\\
         G'_i&=\mathrm{softmax}\bigl(\tfrac{V_{i,q}^\top L_{i,k}}{\sqrt{C_i}}\bigr)L_{i,v}^\top,\quad
         G_i=\omega_{i,w}(\mathrm{unflatten}(G'_i)^\top),\\
         F_i&=\omega_{i,o}\bigl(\omega_{i,m}(V_i)\odot G_i\bigr).
       \end{aligned}
``` 

3. **언어 경로(Language Pathway)**  
   - PWAM 출력 $$F_i$$를 게이트 $$\gamma_i$$로 재조정한 후 잔차 합산  

$$
       S_i=\gamma_i(F_i),\quad E_i=S_i\odot F_i + V_i,\quad \gamma_i: \mathrm{Conv}{\text–}\mathrm{ReLU}{\text–}\mathrm{Conv}{\text–}\tanh.
     $$  

4. **경량 디코더**  
   - 다중 스케일 $$F_4\to F_1$$를 채널 연결·업샘플링 후 3×3 Conv로 예측 마스크 생성  

### 2.3 모델 구조  
- 비전 백본: Swin Transformer-B  
- 언어 백본: BERT-base (12층, 히든 768)  
- PWAM·언어 게이트 내 1×1 컨볼루션, 인스턴스 정규화, ReLU/tanh 사용  

### 2.4 성능 향상  
| 데이터셋    | RefCOCO val | RefCOCO+ val | G-Ref val (UMD) | G-Ref val (Google) |
|-------------|--------------|---------------|------------------|--------------------|
| 이전 최고   | 65.65%       | 55.50%        | 54.40%           | 51.93%             |
| **LAVT**    | **72.73%**   | **62.14%**    | **61.24%**       | **60.50%**         |
| 절대 향상   | +7.08%p      | +6.64%p       | +6.84%p          | +8.57%p            |

### 2.5 한계  
- 사전학습된 BERT의 언어 편향(bias) 문제[1]
- 복잡한 문장·표현에 대한 일반화 한계 (G-Ref에서 상대적 성능 저하 관찰)  
- 높은 연산·메모리 비용 (4단계 PWAM 추가)  

***

## 3. 모델의 일반화 성능 향상 가능성  
- **조기 융합(Early Fusion)**: 인코더 내부에서 반복적·계층적으로 언어 정보를 통합함으로써 학습된 표현이 다양한 문장 길이·구조에 더욱 견고하게 대응할 수 있음  
- **언어 게이트**: 층별 게이트 학습으로 과도한 언어 정보 유입을 제어, 도메인별 과적합 방지에 기여  
- **경량 디코더**: 특이한 언어 표현에도 encoder-only 학습으로 빠르게 적응 가능  
- **추가 기법 제안**:  
  - 교차도메인 언어-비전 데이터 증강  
  - 언어 편향 완화 기법 통합 (e.g., 편향 정규화)  
  - Self-supervised 크로스모달 프리트레이닝  

***

## 4. 향후 연구에 미치는 영향 및 고려 사항  
- **영향**:  
  - 비전·언어 융합 방식의 패러다임 전환 제시 (인코더 내 조기융합)  
  - 대규모 비전 언어 태스크에서 효율적·일관된 멀티모달 학습 가능성 확대  
- **고려 사항**:  
  1. **언어 모델 편향 완화**: BERT 등 사전학습 언어모델의 사회적 편향 문제 해결  
  2. **연산 효율성 개선**: PWAM 경량화·동적 활성화 기법 연구  
  3. **도메인 확장성 검증**: 의료·자율주행 등 특수 도메인 자연어 표현에 대한 일반화 성능 평가  
  4. **프리트레이닝 기법**: 대규모 크로스모달 데이터로 LAVT 구조 자체 프리트레이닝  

---  

**참고문헌**  
 Ahn & Oh, EMNLP 2021. “Mitigating language-dependent ethnic bias in BERT.”[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8b33d86e-b620-4611-944f-b2367dd66619/2112.02244v2.pdf
