# MAGVIT: Masked Generative Video Transformer | Video generation, Text-to-video generation

**주요 주장 및 기여 요약**  
MAGVIT은 단일 모델로 다양한 비디오 생성 및 편집 태스크를 효율적으로 수행하는 **세계 최고 수준(State-of-the-art) 비디오 생성 모델**이다.  
- **핵심 기여**  
  1. **3D-VQ 토크나이저**: 비디오를 시공간적 시퀀스로 양자화하여 압축률을 크게 높이면서 재구성 품질을 유지.  
  2. **COMMIT (Conditional Masked Modeling by Interior Tokens)**: 다양한 내부 조건(프레임, 공간 영역, 클래스)을 하나의 마스크화된 입력으로 결합해 멀티태스크 학습 가능.  
  3. **Masked Generative Video Transformer**: non-autoregressive 디코딩을 통해 12 스텝만에 16프레임 영상 생성.  

## 1. 해결 과제  
기존 비디오 생성 모델들은  
- **Diffusion**: 고품질이나 샘플링 속도가 매우 느림.  
- **Autoregressive**: 순차적 디코딩으로 비효율적.  
- **단일 태스크 전용**: 한 가지 생성/편집만 지원.  

**목표**:  
- 하나의 모델이 *다양한* 비디오 생성·편집(프레임 예측·보간·인페인팅·아웃페인팅·클래스 조건 생성 등) 태스크를  
- **고품질**, **고효율**(190 fps TPU), **고유연성**으로 수행  

## 2. 제안 방법

### 2.1 3D-VQ 토크나이저  
비디오 $$V \in \mathbb{R}^{T\times H\times W\times3}$$를 3D-VQGAN 인코더로 토크나이즈  

$$
z = f_T(V), \quad z \in \{1,\dots,|Z|\}^N,\quad N=\tfrac{T}{4}\times\tfrac{H}{16}\times\tfrac{W}{16}
$$  

- **인플레이션 초기화**: 2D-VQGAN 가중치를 중앙 슬라이스에 복사  
- **패딩**: 리플렉트 패딩으로 위치 일관성 향상  
- **코드북 크기**: $$|Z|=1024$$  
- **구조**: 얕은 레이어에 3D 다운샘플, 깊은 레이어엔 2D 다운샘플 혼합  

재구성 손실:  

$$
\mathcal{L}\_{\text{recons}} = \|V - f_T^{-1}(z)\|\_{p} + \lambda_{\text{perc}}\mathcal{L}_{\text{perceptual}}
$$  

### 2.2 COMMIT 멀티태스크 MTM  
MaskGIT의 이진 마스크 대신, **타스크별 내부 조건 토큰** $$\tilde z$$을 삽입한 다변량 마스크 $$m(z\mid \tilde z)$$ 사용:  

```math
\overline z_i = 
\begin{cases}
\tilde z_i, & s_i\le s^*\land \neg \mathrm{all\_padded}(\tilde z_i)\\
[MASK], & s_i\le s^*\land \mathrm{all\_padded}(\tilde z_i)\\
z_i, & s_i> s^*
\end{cases}
```

- $$s_i\sim U(0,1)$$, $$s^*=\lceil\gamma(r)N\rceil$$-th smallest  
- **Loss**:  

$$
  \mathcal{L}
  = \underbrace{\sum_{\overline z_i=\tilde z_i}-\log p_\theta(z_i\mid\cdot)}\_{\mathcal{L}\_{\mathrm{refine}}}
  +\underbrace{\sum_{\overline z_i=[MASK]}-\log p_\theta(z_i\mid\cdot)}\_{\mathcal{L}\_{\mathrm{mask}}}
  +\underbrace{\sum_{\overline z_i=z_i}-\log p_\theta(z_i\mid\cdot)}\_{\mathcal{L}_{\mathrm{recons}}}
$$  

### 2.3 Non-autoregressive 디코딩  
12 스텝 반복  

$$
\hat z^{(t+1)}\_i\sim p_\theta\bigl(z_i\mid [\rho,c,m(\hat z^{(t)}\!\mid\tilde z;s,s^*)]\bigr)
$$ 

- **스케줄** $$\gamma(t/K)$$로 마스크 비율 점진 감소  
- **속도**: V100 GPU에서 37 fps, TPU-v4i에서 190 fps  

## 3. 성능 및 비교

| 태스크      | UCF-101 CG FVD↓ | BAIR FP FVD↓ | Kinetics-600 FP FVD↓ |
|-------------|-----------------|--------------|----------------------|
| 이전 최고   | 332             | 84           | 16.2                 |
| MAGVIT-B    | 159             | 76           | 24.5                 |
| MAGVIT-L    | **76**          | **62**       | **9.9**              |

- **77%↓**, **39%↓** 절대 개선[1].  
- **IS**: 79.3 → 89.3 (↑13%) on UCF-101  
- **멀티태스크**(8–10개) 평균 FVD: 단일태스크 대비 60–80% 절감  

**한계**  
- *텍스트-투-비디오* 미지원  
- 128×128 해상도 한정 (추가 비용 없이 확장 가능성 보이나 아직 실험 부족)  
- BAIR 소규모 평가세트 FVD 추정 불안정성  

## 4. 일반화 성능 향상 가능성

- **멀티태스크 학습**: 태스크 간 공유된 마스크·조건 학습으로 *도메인 전이* 및 *새 태스크* 적응력 우수  
- **3D-VQ 토크나이저**: 비지도 사전학습으로 다양한 영상 도메인(자율주행, 웹, Objectron)에서도 높은 재구성→생성 성능 유지  
- **경량 디코더**: 적은 스텝·짧은 시퀀스 길이 덕분에 대규모 비디오 데이터에도 확장 가능  

## 5. 미래 영향 및 고려 사항

- **비디오 생성·편집 도구** 개발 가속: 업스케일·인페인팅·보간·아웃페인팅 등 단일 모델로 지원  
- **멀티모달 확장**: 텍스트, 오디오 조건 결합, 비디오-오디오-자막 통합 생성 연구에 유용  
- **고해상도 확장**: 3D-VQ 구조를 대형 코드북·더 깊은 아키텍처로 확장해 4K 이상 비디오 생성 가능성  
- **안정성**: GAN + 퍼셉추얼 손실 기반 VQ의 훈련 불안정성 완화 및 스케일업 시 리소스 고려 필요  

---  

MAGVIT의 **효율성**, **품질**, **유연성**은 비디오 생성 커뮤니티의 새로운 기준을 제시하며, 멀티태스크·비지도 사전학습 기반 모델 설계 방향을 주도할 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/028ae98a-13fc-440d-93bf-1a01f62766e1/2212.05199v2.pdf
