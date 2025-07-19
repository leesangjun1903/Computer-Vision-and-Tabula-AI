# DiffTSR : Diffusion-based Blind Text Image Super-Resolution | Super resolution

## 1. 핵심 주장 및 주요 기여  
**DiffTSR**(Diffusion-based Blind Text Image Super-Resolution)은  
- **텍스트 이미지 복원**에 있어 **텍스트 충실도(text fidelity)** 와 **스타일 현실성(style realness)** 을 동시에 보장  
- 기존의 SR 기법들이 복원한 글자 구조 왜곡 혹은 획 누락 문제를 해결  
- **이미지 확산 모델(IDM)** 과 **텍스트 확산 모델(TDM)** 을 상호 보강하며,  
- 양자를 연결하는 **MoM(Mixture of Multi-modality) 모듈**을 도입해 매 디퓨전 단계에서 이미지와 텍스트 정보를 교환  

이를 통해 복잡한 획 구조, 심한 열화, 다양한 글꼴·스타일 상황에서도 고품질 HR 텍스트 이미지를 생성한다.

## 2. 문제 정의 및 제안 기법

### 2.1 해결하고자 하는 문제  
- **Blind SR**: 저해상도(LR) 텍스트 이미지가 갖는 복잡한 비선형 열화(blur, noise, 압축 아티팩트 등)를 사전 정보 없이 복원  
- **텍스트 특유 과제**:  
  - 획의 왜곡·누락·추가 시 **문자 의미 훼손**  
  - 글꼴·색상·포즈 등 스타일 부정확 시 **가독성·자연스러움 저하**

### 2.2 제안 방법 개요  
DiffTSR는 세 요소로 구성된다.  
  1. **IDM (Image Diffusion Model)**  
     - Stable Diffusion 기반 VAE–UNet 구조  
     - LR 이미지의 latent feature $$Z_{LR}=E(X_{LR})$$ 및 텍스트 조건 $$\mathbf{C}_{cond}^t$$를 입력으로 고해상도 latent $$Z_0$$ 복원  
  2. **TDM (Text Diffusion Model)**  
     - 다항 확산(multinomial diffusion)으로 텍스트 시퀀스 $$c$$ 복원  
     - UNet으로부터 전달받은 이미지 정보 $$\mathbf{I}_{cond}^t$$를 cross-attention으로 활용  
  3. **MoM (Mixture of Multi-modality)**  
     - t단계에서 IDM의 latent $$Z^t$$ · $$Z_{LR}$$ 와 이전 단계 텍스트 $$c^t$$를 융합  
     - 이미지 조건 $$\mathbf{I}\_{cond}^t$$ → TDM, 텍스트 조건 $$\mathbf{C}_{cond}^t$$ → IDM 생성  

#### 수식 요약  
- MoM: $$[\mathbf{I}\_{cond}^t,\ \mathbf{C}\_{cond}^t] = \mathrm{MoM}\_\phi([Z_{LR}, Z^t],\,c^t,\,t)$$  
- IDM reverse step (Stable Diffusion):  

$$
    Z^{t-1}
      = \frac{1}{\sqrt{\alpha_t}}
        \Bigl(Z^t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}\_t}}\,\epsilon_{\theta}(Z^t,[Z_{LR}],\mathbf{C}_{cond}^t)\Bigr)
        + \sigma_t z
  $$
  
- TDM reverse (multinomial):  

$$
    c^{t-1}\sim\mathrm{Cat}\bigl(\pi_{\mathrm{post}}(c^t,\,\hat{c}^t)\bigr)
  $$

## 3. 모델 구조  

```plaintext
LR Text Image X_LR
        │
      VAE Encoder E → Z_LR ───────────┐
        │                            │
    Text Recognizer P → initial c_T  │
        │                            │
    Random Noise Z_T                 │
        │                            │
   ┌──────────────────────────────────────────────────────────┐
   │  for t = T … 1:                                          │
   │    [I_cond^t, C_cond^t] = MoM([Z_LR, Z^t], c^t, t)       │
   │    Z^{t-1} = IDM_UNet(Z^t, Z_LR, C_cond^t, t)            │
   │    c^{t-1} = TDM_TransDec(c^t, I_cond^t, t)              │
   └──────────────────────────────────────────────────────────┘
        │
      VAE Decoder D → HR Text Image X_HR
```

- **IDM**: Stable Diffusion UNet + VAE  
- **TDM**: Transformer Decoder 기반 다항 확산  
- **MoM**: UNet(이미지) + Transformer Encoder(텍스트) 융합

## 4. 성능 향상 및 한계  

### 4.1 성능 향상  
- **합성 테스트(CTR-TSR-Test)**  
  - PSNR: 20.74 → **21.85** (↑1.11)  
  - LPIPS: 0.310 → **0.231** (↓0.079)  
  - ACC: 0.6179 → **0.8350** (↑0.2171)  
- **실세계 테스트(RealCE)**  
  - FID: 83.22 → **70.59** (↓12.63)  
  - NED: 0.8047 → **0.8747** (↑0.07)  

모든 지표(PSNR, LPIPS, FID, ACC, NED)에서 기존 방법 대비 우수한 성능 확인[Table 1][Table 2].

### 4.2 한계  
- **연산 비용**: 이중 확산 모델과 MoM 융합으로 **추론 속도 저하**  
- **텍스트 길이**: 24자 이하로 제한된 훈련 → **장문 텍스트** 처리 미확인  
- **언어·스크립트**: 중국어에 특화, 라틴·다른 스크립트 일반화는 추가 검증 필요  

## 5. 일반화 성능 향상 가능성  

- **MoM의 양방향 보강**을 통해 어려운 real-world 열화에도 텍스트 인식·이미지 복원 협력  
- **확산 모델**의 강력한 **분포 학습** 특성은  
  - 새로운 폰트·스타일·조명 조건에서도 **도메인 적응**에 유리  
  - 사전 정의된 코드북 방식보다 **미지 스타일 처리**에 유연  
- **추가 제언**:  
  - 다양한 언어·스크립트 학습 → 다국어 SR 적용  
  - **경량화** 및 **지연 축소** 연구 병행  

## 6. 영향 및 향후 연구 고려사항  

- **영향**:  
  - 텍스트 SR 분야에서 **확산 모델 활용**의 새로운 방향 제시  
  - 이미지-텍스트 멀티모달 협력 구조 설계 토대 마련  
- **향후 고려점**:  
  1. **추론 최적화**: 확산 단계 축소·지연 최소화  
  2. **긴 텍스트**·다양 스크립트 확대  
  3. **제한적 훈련 데이터** 넘어 **자기지도 학습** 기법 도입  
  4. **상업적 응용**을 위한 실시간 구현  

---  

**결론**: DiffTSR는 **IDM·TDM·MoM**의 상호보강을 통해 텍스트 SR에서 **고충실도·고현실성**을 달성하며, 확산 모델 기반 멀티모달 SR 연구의 새로운 이정표를 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8574c2c7-aa9d-43a5-bc7b-474cca178898/2312.08886v2.pdf
