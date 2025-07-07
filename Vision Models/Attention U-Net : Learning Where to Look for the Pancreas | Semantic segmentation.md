# Attention U-Net: Learning Where to Look for the Pancreas | Semantic segmentation

## 1. 핵심 주장 및 주요 기여  
**Attention U-Net**은 전통적 U-Net 기반 의료 영상 분할 모델에 *Attention Gate (AG)* 모듈을 삽입하여,  
- **ROI(localization) 단계 없이** 원하는 장기(예: 췌장)에 집중하면서  
- **불필요한 배경 특성을 억제**하고  
- **소형/모양 가변 대상**의 분할 정확도를 높인다.  

주요 기여:  
1. **Grid-based Soft Attention Gate**  
   - 고해상도 스킵 연결(feature map) 상에 국소적 공간 정보를 반영한 attention coefficient $$\alpha_i$$ 계산.  
2. **Cascade 모델 불필요**  
   - Multi-stage ROI 탐색 없이 단일 네트워크에서 end-to-end 학습.  
3. **성능·효율성 동시 개선**  
   - 파라미터 증가율 8% 미만으로 U-Net 대비 2–3% DSC(판크레아스) 개선.  
   - 추론 시간 약 7% 증가에 그침.  

## 2. 해결 문제 ‧ 제안 방법 ‧ 모델 구조 ‧ 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- **췌장은** CT 상에서 *저대비(low contrast)* , *형상 가변성* 이 커,  
- 전통적 U-Net은 다소 많은 False Positive/Negative를 양산하며,  
- 다수 연구가 ROI-crop 기반 cascade 모델을 사용하나 계산·파라미터 낭비가 큼.  

### 2.2 제안 방법  
Attention Gate(AG)는 스킵 연결 입력 $$x^l_i$$와, 더 깊은 레벨의 컨텍스트 신호 $$g_i$$를 받아,  
1) 1×1×1 conv로 선형 변환 후  
2) additive attention:
 
$$
   q^{l}\_{\text{att}} = \psi^T\bigl(\sigma_1(W_x^T x^l_i + W_g^T g_i + b_g)\bigr) + b_\psi
$$  

3) sigmoid로 활성화하여

$$
   \alpha^l_i = \sigma_2\bigl(q^{l}_{\text{att}}\bigr)
$$  

4) 스킵 연결 특성에 곱해

$$ \hat{x}^l\_i = \alpha^l\_i\, x^l\_i$$ 

로 출력.  

여기서  
- $$\sigma_1$$은 ReLU, $$\sigma_2$$는 sigmoid  
- $$W_x\in\mathbb R^{F_l\times F_{\text{int}}}$$, $$W_g\in\mathbb R^{F_g\times F_{\text{int}}}$$, $$\psi\in\mathbb R^{F_{\text{int}}\times1}$$.  

### 2.3 모델 구조  
- **Encoder–Decoder U-Net**에 각 스킵 연결 직전 AG 삽입 (Figure 1).  
- Deep Supervision으로 중간 출력에도 손실 적용.  
- CT 볼륨을 2 mm isotropic으로 downsample.  
- 3D convolution, Adam optimiser, Dice loss 사용.  

### 2.4 성능 향상  
| 실험 조건                | U-Net DSC (췌장) | Att-U-Net DSC (췌장) | p-value      |
|-------------------------|------------------|----------------------|--------------|
| CT-150 (120/30 split)   | 0.814 ± 0.116    | **0.840 ± 0.087**    | p = .005     |
| CT-150 (30/120 split)   | 0.741 ± 0.137    | **0.767 ± 0.132**    | p = .010     |
| TCIA CT-82 (fine-tuned) | 0.820 ± 0.043    | **0.831 ± 0.038**    | —            |

- Recall↑, Surface distance↓ 우수.  
- 파라미터 5.88 M → 6.40 M (+8%)로 2–3% 성능 개선[1].  

### 2.5 한계  
- 작은 배치 크기(2–4)로 학습 안정성 우려.  
- 입력 해상도 2 mm isotropic으로 downsampling된 저해상도 활용.  
- Hard-attention 대비 연산량 증가(1×1 conv overhead).  
- Post-processing(CRF 등) 미사용.  

## 3. 모델 일반화 성능 향상 가능성  
- **Grid-Attention**는 다양한 크기‧위치 분포 대상에 적응적 강조 가능.  
- Deep Supervision과 AG 결합으로 각 스케일이 강건한 표현 학습.  
- Transfer Learning: 사전 학습 U-Net weight로 초기화 후 AG만 fine-tuning 시 성능↑ 예상.  
- Residual/Gated Highway 네트워크와 결합 시 gradient 흐름 개선 및 더 부드러운 attention 가능.  
- Multi-organ(Right Kidney, Spleen 등) 동시 학습 시 상호 컨텍스트 공유로 췌장 분할 일반화↑.  

## 4. 향후 연구 기여 및 고려 사항  
- **영향**: 단일 네트워크 내 attention 기반 ROI 탐색 패러다임 제시로, 다양한 의료 분할·검출 연구에서 cascade 구조 대체 가능.  
- **고려점**:  
  - 고해상도(raw CT) 입력 처리와 더 큰 batch size 적용.  
  - Post-processing (CRF, shape prior) 통합 여부.  
  - Self-attention 및 non-local block 결합으로 장거리 의존성 캡처.  
  - Residual attention, multi-head attention 적용 실험.  
  - 다양한 장기‧질환 데이터셋 적응 시험을 통한 범용성 검증.  

[1] Oktay et al. “Attention U-Net: Learning Where to Look for the Pancreas,” MIDL 2018.  
 Table 1, Table 3 성능 지표.  
 Eqns (1)-(2) 및 Figure 1–2.  
 Ablation study (파라미터 대비 성능) and p-values.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f4d59042-2649-42c7-a820-97a2235410d8/1804.03999v3.pdf
