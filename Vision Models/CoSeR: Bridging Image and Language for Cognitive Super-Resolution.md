# CoSeR: Bridging Image and Language for Cognitive Super-Resolution | Super resolution

## 1. 핵심 주장 및 주요 기여  
CoSeR은 저해상도(LR) 이미지 복원에 **인간의 인지 과정**을 모방하여, 이미지와 언어 정보를 결합한 **인지 임베딩(cognitive embedding)**을 활용함으로써  
- LR 이미지의 **글로벌 의미**를 이해  
- 대형 텍스트-이미지 확산(diffusion) 모델의 암묵적 프라이어(prior)를 활성화  
- **고품질 참조 이미지(reference image)**를 생성하여 SR 절차를 명시적으로 안내  

이를 위해  
1) **Cognitive Encoder**: CLIP 특징과 텍스트 토큰을 정렬하는 Q-Former 유사 구조로 다중 토큰 인지 임베딩 생성  
2) **참조 이미지 생성**: Stable Diffusion 모델에 인지 임베딩을 입력해 LR과 의미·텍스처가 일치하는 참조 이미지 생성  
3) **All-in-Attention(AiA)**: LR 입력, 인지 임베딩, 참조 이미지 제어 신호를 단일 어텐션 모듈에 통합하여 질감 재현과 원본 대비 충실도(fidelity) 모두 확보  

## 2. 해결 문제와 제안 방법

### 2.1 해결하고자 하는 문제  
기존 SR 모델은 주로 로컬 텍스처 복원에 집중하여  
- 장면 전반의 의미 정보 부족  
- 의미 오류 또는 비현실적 텍스처 생성  
라는 한계를 지님  

### 2.2 제안 방법  
① **Cognitive Encoder**  
- 입력: LR 이미지 → 경량 SRResNet으로 전처리 → CLIP 이미지 인코더  
- Q-Former 유사 구조의 어댑터가 $$T_e$$개의 학습 가능 쿼리로 CLIP 텍스트 임베딩 $$L\in\mathbb{R}^{T_l\times C_l}$$의 마지막 $$T_e$$ 토큰 $$L'$$을 감독(supervision)  
- 손실 함수:  

$$
\mathcal{L}_{CE} = \|E - L'\|_2^2
$$ 

(단, $$E\in\mathbb{R}^{T_e\times C_l}$$은 인지 임베딩)  

② **참조 이미지 생성 및 인코딩**  
- Stable Diffusion U-Net에 인지 임베딩을 투입해 참조 이미지 $$R$$ 생성  
- VQGAN으로 잠재 코드로 인코딩 후 ControlNet 방식으로 다중 스케일 제어 특징 $$\{R_i\},\{X_i\}$$ 생성  

③ **All-in-Attention(AiA) 모듈**  
- 기존 self-·cross-attention 모듈을 확장  
- LR 어텐션: 쿼리 $$Q$$ from U-Net, 키·값 $$K,V$$ from LR 제어 $$X_i$$  
- 참조 어텐션: $$Q$$ from $$X_i$$, $$K,V$$ from $$R_i$$  
- 교차 어텐션: $$K,V$$ from 인지 임베딩 $$E$$  
- “one-hot attention”으로 가장 유사한 참조 패치만 강조  

## 3. 모델 구조  
- 기반: Stable Diffusion 2.1–base  
- Cognitive Encoder: Q-Former 어댑터(50개 쿼리) + CLIP 이미지 인코더(동결)  
- ControlNet: U-Net 구조 복제, zero convolution 삽입  
- Denoising U-Net: 중간·디코더 블록 전부 AiA 모듈로 교체  

## 4. 성능 향상  
| 데이터셋                | FID↓ (Best=CoSeR) | CLIP-Score↑ | DISTS↓ | LPIPS↓ | MUSIQ↑ |
|-------------------------|-------------------|-------------|--------|--------|--------|
| ImageNet Test2000       | 19.41             | 0.8755      | 0.1482 | 0.2863 | 65.51  |
| RealSR                  | 80.82             | 0.8545      | 0.1826 | 0.2438 | 70.29  |
| DRealSR                 | 71.22             | 0.8766      | 0.1977 | 0.2702 | 70.18  |

- 기존 최고 모델 대비 FID 3–14% 절감, 의미·질감 복원력 및 비-reference 화질 지표에서 우위  
- 사용자 평가에서 23명 중 80% 비율로 최고 선호도 획득  

### 한계  
- 확산 모델 샘플링 비용(200 스텝)  
- 미세 조정 없이 매우 다양하거나 복잡한 도메인 일반화 검증 부족  

## 5. 일반화 성능 향상 가능성  
- 인지 임베딩이 **언어·시각 융합 정보**를 담아, LR 도메인 편차가 큰 실제 환경에서도 의미 일관성 유지  
- 참조 이미지 생성 방식이 **자동화된 외부 프라이어** 활용으로, 도메인 간 매뉴얼 참조 필요성 제거  
- AiA 모듈이 다중 조건을 유연 통합하여, 새로운 입력 조건(예: 추가 가이드맵)에도 쉽게 확장 가능  

## 6. 향후 연구에 미치는 영향 및 고려 사항  
- **확산-기반 복원 기법**에 인지적 차원의 융합을 제시하여, SR 외에도 디노이징·디블러링·인페인팅 등 전반적 복원 분야로 확장 가능  
- **샘플링 가속화**(fewer steps) 연구가 병행돼야 실제 응용 시스템 내 실시간 처리 달성  
- CLIP 등 멀티모달 임베딩 편향·제한 검토 필요: 사회문화적 편향이나 희귀 도메인 객체 인식 실패 위험  
- **참조 이미지 품질**에 대한 추가 평가 및 다중 참조 전략 최적화 연구로 더 탄탄한 일반화 성능 확보  

> **주요 시사점**: CoSeR은 멀티모달 인지 임베딩과 다중 조건 어텐션 설계를 통해 현실 세계 SR의 의미·텍스처 양립 문제를 해결하며, 차세대 복원 모델의 방향성을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/cee5e6a3-20fb-4dfe-84f7-6edbcaee594d/2311.16512v4.pdf
