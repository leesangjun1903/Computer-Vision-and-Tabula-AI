# CodeFormer : Towards Robust Blind Face Restoration with Codebook Lookup Transformer | Image restoration, Blind Face restoration

# 핵심 주장 및 주요 기여 요약

**코드북 기반 불투명(Blind) 얼굴 복원 과제:**  
기존 연속형 잠재 공간(prior)은 심각한 열화(degradation) 상황에서 정확한 잠재 벡터 탐색이 어려워 현실감 있는 이미지는 생성하나 입력 표현과의 고충실도(fidelity)가 낮아지는 한계를 가집니다.

**Codebook Lookup Transformer(CodeFormer)의 제안:**  
불투명 얼굴 복원을 “코드 예측(code prediction)” 문제로 재정의하고, 크기가 제한된(1024개 항목) 이산 코드북(discrete codebook)과 고품질 복원 디코더(decoder)를 사전 학습해 보유한 뒤, 저화질(LQ) 입력에서 해당 코드 시퀀스를 트랜스포머로 예측하여 디코더로 복원하는 구조를 제안합니다.

주요 기여  
1. **이산 코드북 프라이어(discrete codebook prior)**:  
   - VQ-VAE 기반으로 1024개 코드, 차원 256의 이산 공간을 학습.  
   - 복원 불확실성을 크게 저감시키고 로컬 열화에도 강인한 표현 제공.  
2. **글로벌 문맥 모델링을 위한 트랜스포머:**  
   - LQ 특징을 토큰 시퀀스로 펼쳐(global composition) 9개 셀프-어텐션 블록을 거쳐 코드 인덱스를 예측.  
   - 국소 정보가 손실된 경우에도 전역 문맥 활용으로 정확도↑.  
3. **조절 가능한 특징 변환 모듈(CFT):**  
   - LQ 인코더 특징과 디코더 특징의 어파인 변환을 $$\hat F_d = F_d + w(\alpha\odot F_d+\beta)$$ 로 결합.  
   - $$w\in[1]$$ 로 복원 퀄리티와 충실도 간 탄력적 트레이드오프 가능.  

# 문제 정의, 제안 방법 및 모델 구조

## 1. 문제 정의  
입력 $$I_l$$이 블러·노이즈·JPEG·다운샘플링 등으로 손상되어 있을 때, 원본 고품질 얼굴 $$I_h$$를 예상 가능한 무한 해 공간 대신 유한한 이산 코드 시퀀스로 매핑하여 복원을 안정화하는 문제.

## 2. 제안 방법  
### 2.1 단계 I: 코드북 학습  
- HQ 이미지를 인코더 $$E_H$$로 $$\,Z_h\in\mathbb R^{m\times n\times d}$$ 임베딩.  
- 코드북 $$C=\{c_k\}_{k=0}^{N-1}$$의 최단 거리 매칭으로 양자화:  

$$
    Z_c^{(i,j)} = \underset{c_k\in C}{\arg\min}\,\|Z_h^{(i,j)}-c_k\|_2,\quad s^{(i,j)}=k,
  $$  
  
- 복원 디코더 $$D_H$$ 활용해 $$I_\text{rec}=D_H(Z_c)$$.  
- 손실: $$L_1+\lambda_\text{per}L_\text{per}+L_\text{feat}^\text{code}+\lambda_\text{adv}L_\text{adv}.$$

### 2.2 단계 II: 코드 예측 트랜스포머 학습  
- LQ 인코더 $$E_L$$로 얻은 $$Z_l\in\mathbb R^{m\times n\times d}$$를 토큰 시퀀스 $$Z_l^v\in\mathbb R^{(mn)\times d}$$로 펼침.  
- Transformer: 9개 Self-Attention 블록, positional embedding 포함.  
- 선형 층으로 $$(mn)\times N$$ 차원 예측 후 교차 엔트로피 $$L_\text{token}^\text{code}$$로 학습, 부가로 $$L_2$$ 특징 정합 손실 $$L_\text{feat}'{}^\text{code}=\|Z_l-\text{sg}(Z_c)\|_2^2$$.  
- 최종 $$L_\text{tf}=0.5L_\text{token}^\text{code}+L_\text{feat}'{}^\text{code}.$$

### 2.3 단계 III: CFT 모듈 학습  
- 디코더 도중 다중 해상도(32,64,128,256)에서 LQ 특징 $$F_e$$와 디코더 특징 $$F_d$$를  
  $$\hat F_d = F_d + w(\alpha\odot F_d + \beta)$$로 결합.  
- $$w=1$$로 학습, 추론 시 $$w$$를 조절해 fidelity↔quality 트레이드오프.

## 3. 모델 구조  
```
LQ Input → E_L → Z_l → Transformer → 코드 시퀀스 \hat s
                 ↓                                ↓
               CFT modules ← Decoder D_H ← 양자화된 Z_c
```
- 코드북과 디코더는 **고정**  
- 트랜스포머 전역 문맥, CFT 지역 보강

# 성능 향상 및 한계

## 성능 향상  
- **합성 데이터(CelebA-Test)**: LPIPS 0.299, IDS 0.60로 최상 성능[표1].  
- **실세계(WIDER-Test)**: FID·MUSIQ 우수, 특히 열화 심한 데이터에서 best[표2].  
- **아이덴티티 보존**: ArcFace 유사도 기반 IDS 최고점.  
- **추론 속도**: 0.07s/image, GPEN·PSFRGAN 수준.

## 한계  
1. **코드북 표현 범위 제한**: 악세서리·측면 얼굴 등 희귀 패턴 복원 어려움.  
2. **측면 얼굴 일반화 한계**: FFHQ 학습 분포에 측면 인물 적어 표현 학습 부족.  

# 일반화 성능 향상 가능성 중심 고찰

- **CFT 파라미터 $$w$$ 튜닝**: 다양한 열화 유형·강도에 대응해 적응적 정보 흐름 조절로 일반화 강화.  
- **코드북 다중 스케일 학습**: 단일 32배 압축 대신 16·32·64 배 다중 압축 코드북 도입 시 저·고주파 정보 균형 복원 가능성.  
- **측면 얼굴 데이터 확대**: FFHQ 외 추가 옆모습·다양인종 데이터로 코드북 학습 확대 시 극단적 포즈 일반화 기대.  
- **트랜스포머 구조 탐색**: Layer 수·head 수·입력 스케일별 어텐션 분리 등 아키텍처 다변화로 코드 예측 정확도↑.

# 향후 연구 방향 및 고려 사항

- **다중 스케일 코드북**: 세밀한 디테일과 구조적 정보를 균형 있게 양자화할 수 있도록 멀티-레졸루션 코드북 설계.  
- **데이터 다양성 확충**: 포즈·조명·인종 등 다양성을 강화한 데이터로 일반화, 특히 희귀 케이스(측면·극단적 표정) 복원 성능 향상.  
- **대형 트랜스포머 기반 글로벌 로컬 하이브리드**: 지역 어텐션과 전역 어텐션 결합으로 장·단기 문맥 모두 고려.  
- **적응형 CFT 스케줄링**: 이미지별 열화 진단 후 $$w$$ 자동 조절 메커니즘 연구로 사용자 개입 최소화.  
- **실시간·경량화**: 모바일·웹 환경 적용을 위한 모델 압축·경량화 기법 병용 검토.

이러한 방향성은 이산 코드북과 전역 문맥 모델링을 결합한 CodeFormer 패러다임을 확장·일반화하는 초석이 될 것입니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c7c10e2e-b2ac-40e0-ac4c-7478c4826baf/2206.11253v2.pdf
