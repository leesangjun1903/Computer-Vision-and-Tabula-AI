# simple diffusion: End-to-end diffusion for high resolution images | Image generation

## 주요 주장 및 기여
1. **단일 스테이지(end-to-end) 고해상도 이미지 생성**  
   기존의 latent diffusion 혹은 cascaded diffusion 방식과 달리, 본 논문은 하나의 표준 denoising diffusion 모델만으로 고해상도(최대 512×512) 이미지를 직접 생성할 수 있음을 보인다.
2. **노이즈 스케줄 조정**  
   고해상도에서 낮은 해상도로 풀링(pooling) 시 SNR(signal-to-noise ratio)이 과도하게 높아지는 문제를 해결하기 위해, 기준 해상도(예: 64×64)의 cosine 스케줄을 전체 해상도로 “shift”시키거나 여러 스케줄을 보간(interpolate)한다.
3. **멀티스케일 학습 손실**  
   다운샘플된 여러 해상도(32×32, 64×64, …, d×d)에서의 재구성 손실을 가중합해(low-frequency에 더 높은 가중치) 학습함으로써 고해상도에서의 학습 안정성과 수렴 속도를 개선한다.
4. **아키텍처 확장 및 최적화**  
   – 주로 16×16 레벨의 U-Net 블록 수를 늘려 모델 용량을 확장  
   – 고해상도 feature map의 메모리 폭증을 막기 위해 DWT(dyadic wavelet) 또는 strided convolution patching을 통해 입력을 즉시 다운샘플링  
   – 저해상도 레이어에만 dropout 적용  
5. **U-ViT 제안**  
   16×16 해상도 이상에서 convolution 대신 self-attention+MLP 블록을 사용해 U-Net 구조에 transformer를 결합한 U-Vision Transformer를 도입.  
6. **성능**  
   – ImageNet 클래스 조건부 생성에서 512×512 해상도까지 단일 모델로 FID 4.28 달성 (eval)[Table 7]  
   – U-Net(256×256) 기준 FID 3.71, U-ViT(256×256) 기준 FID 3.75  
   – 텍스트→이미지(256×256) zero-shot COCO FID 8.30[Table 8]

## 1. 해결하고자 하는 문제
- **고해상도 이미지 생성의 복잡성**  
  기존 연구들은 (1) latent 공간으로 투영 후 diffusion, (2) cascade super-resolution, (3) mixture of experts 등으로 분할 학습하였으나, 여러 모델과 단계의 훈련/파이프라인 관리가 복잡해진다.

## 2. 제안 방법

### 2.1 노이즈 스케줄 조정
- 기준 해상도 $$d_0$$에 대해 cosine SNR 스케줄  

$$
    \log \mathrm{SNR}_{d_0}(t) = -2\log\tan\bigl(\tfrac{\pi t}{2}\bigr)
$$

- 전체 해상도 $$d$$로 shift:

$$
    \log\mathrm{SNR}\_{d\to d_0}(t)
    = \log\mathrm{SNR}_{d_0}(t) + 2\log\bigl(\tfrac{d_0}{d}\bigr)
$$

- 보간 스케줄(예: high-freq 보강):

$$
    t\,\log\mathrm{SNR}\_{\text{shift }d_L}(t)
    + (1-t)\,\log\mathrm{SNR}_{\text{shift }d_H}(t)
$$

### 2.2 멀티스케일 학습 손실
- 해상도 $$s\times s$$별 재구성 손실 $$L_{s}(x)$$ 계산 후 가중합:

$$
    \widetilde{L}\_d(x)=
    \sum_{s\in\{32,64,\dots,d\}}\frac{1}{s}\,
    \mathbb{E}\_{t,\epsilon}\|D_{s}[\,\epsilon\,]-D_{s}[\,\hat\epsilon_\theta(\alpha_t x+\sigma_t\epsilon)\,]\|^2
$$

### 2.3 아키텍처 구성  
- **U-Net 확장**  
  – 주로 16×16 레벨의 residual 블록 수 확장(예: 8→12)  
  – 저해상도에만 dropout(예: 16×16 이하) 적용  
- **고해상도 다운샘플**  
  – DWT(level-2) 또는 strided 4×4 convolution  
  – 메모리 및 compute 효율 향상[Table 5]
- **U-Vision Transformer (U-ViT)**  
  – 16×16 해상도부터 transformer 블록(MLP+Self-Attention) 적용  
  – 중간(16×16)에서 대규모 self-attention으로 글로벌 컨텍스트 학습

## 3. 성능 향상 및 한계

| 해상도 | 모델       | FID(eval) |
|-------:|-----------|----------:|
| 128×128 | U-Net     |    2.88  |
| 256×256 | U-Net     |    3.71  |
| 256×256 | U-ViT     |    3.75  |
| 512×512 | U-Net     |    4.28  |
| 512×512 | U-ViT     |    4.53  |

- **일반화**: train-eval FID 격차가 작아 과적합 억제 효과 관찰  
- **Dropout 효과**: 고해상도 레이어 제외 정규화가 오히려 성능 개선[Table 3]  
- **멀티스케일 손실**: 512×512에서 유의미한 FID 개선  
- **한계**:  
  – 노이즈 스케줄 shift는 과도한 guidance(η &gt; 0.2)에서 성능 저하 발생  
  – 1024×1024 이상으로 확장 시 매모리/컴퓨팅 비용 급증  
  – eval FID 최적화와 샘플링 속도 간 trade-off 존재

## 4. 일반화 성능 향상 관점
- **멀티스케일 손실**: 저/중주파영역 강조로 모델이 전역 구조를 안정적으로 학습  
- **선택적 dropout**: 고해상도 feature맵 과도한 regularization 방지  
- **노이즈 스케줄 보정**: 해상도별 SNR 균형 조정으로 과도한 저해상도 bias 제거  
- 이들 기법이 ensemble, semi-supervised, 도메인 적응 등 다양한 setting에서 일반화 능력 향상에 기여 가능

## 5. 향후 연구 영향 및 고려사항
- **단일 스테이지 고해상도 확장**: 복잡한 cascade 불필요성 입증, 후속 모델 단순화 가능  
- **스케줄 최적화**: 비정형 해상도·비균일 해상도에 대한 자동 최적화 연구  
- **멀티스케일 적용**: 비이미지 데이터(오디오, 비디오)로 확장  
- **효율화 및 실시간화**: distillation, dynamic step size, sparse attention 적용으로 실시간 응용  
- **일반화 검증**: 도메인 간 성능 이동 가능성, 소규모 데이터셋에서의 과적합 억제 효과 심층 평가

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/0f378852-5f7e-489f-a9e2-0d1bb5dc54f3/2301.11093v2.pdf
