# DiT : Scalable Diffusion Models with Transformers | Image generation

## 1. 핵심 주장 및 주요 기여  
**Scalable Diffusion Transformers (DiTs)**를 제안하여, 기존의 U-Net 기반 디퓨전 모델을 순수 트랜스포머 백본으로 대체해도 오히려 더 우수한 성능과 뛰어난 확장성(Scalability)을 달성할 수 있음을 보였다.  
- 대규모 트랜스포머(depth/width 증가)와 입력 토큰 수(패치 크기 축소)에 따라 **모델 복잡도(Gflops) 대비 FID가 꾸준히 감소**함을 실험적으로 입증.  
- 가장 큰 모델인 **DiT-XL/2**는 Class-conditional ImageNet 256×256에서 **FID 2.27**, 512×512에서 **FID 3.04**로 종전 최고 성능을 경신.  

## 2. 해결 과제, 제안 기법, 모델 구조, 성능 및 한계  

### 2.1 해결하고자 하는 문제  
- 디퓨전 모델의 주류 백본인 U-Net이 갖는 **국소적 합성곱 편향**(local convolutional inductive bias)이 필수적인지 검증  
- **모델 복잡도(Gflops) 대비 샘플 품질**(FID) 간의 확장 관계 분석 부족  

### 2.2 제안하는 방법  
1) **Latent Diffusion + Vision Transformer**  
   - Stable Diffusion VAE의 latent space(z) 위에서 DiT를 학습  
2) **Patchify & Transformer Blocks**  
   - $$I \times I \times C$$ latent → $$T=(I/p)^2$$ tokens, hidden size $$d$$  
   - 패치 크기 $$p\in\{2,4,8\}$$로 토큰 수 조절  
3) **Adaptive LayerNorm-Zero Conditioning**  
   - timestep $$t$$ 와 class label $$c$$ 임베딩으로 AdaLN-Zero 수행  
   - 잔차 연결 앞 스케일 파라미터 $$\alpha$$, 시프트 $$\beta$$를 예측, 블록 초깃값은 identity  
4) **확률적 디퓨전 학습**  
   - noise prediction $$\epsilon_\theta(xt)$$과 covariance $$\Sigma_\theta$$를 MSE 및 full KL로 학습  
   - Classifier-free guidance:  

$$\hat\epsilon_\theta(xt,c)=\epsilon_\theta(xt,\emptyset)+s\bigl(\epsilon_\theta(xt,c)-\epsilon_\theta(xt,\emptyset)\bigr)$$

### 2.3 모델 구조  
| 구성요소          | 세부사항                                                   |
|------------------|-------------------------------------------------------------|
| 패치 임베딩       | 선형 투영 + 2D 사인코사인 위치 임베딩                        |
| Transformer blocks | 4종(인-컨텍스트, 크로스-어텐션, AdaLN, AdaLN-Zero) 중 AdaLN-Zero 채택 |
| 모델 크기        | DiT-S/B/L/XL, N layers × hidden size $$d$$, attention heads   |
| 입력 토큰 수     | patch size $$p=8,4,2$$                                     |

### 2.4 성능 향상  
- **Gflops vs. FID**: 모델 Gflops가 늘어날수록 FID 비례 감소(상관계수 –0.93)  
- **256×256 ImageNet**: 종전 LDM-4-G(FID 3.60) → DiT-XL/2-G(FID 2.27)  
- **512×512 ImageNet**: 종전 ADM-G+U(FID 3.85) → DiT-XL/2-G(FID 3.04)  
- **테스트 시 샘플링 스텝**을 늘려도 작은 모델이 큰 모델을 넘지 못함  

### 2.5 한계  
- **VAE 한정**: 오프-더-쉘프 VAE latent space에 의존  
- **샘플 다양성 vs. 지침 강도**: Guidance scale 과다 시 다양성 저하  
- **자원 요구량**: DiT-XL/2 훈련에 수천 TPU v3 코어 필요  

## 3. 일반화 성능 향상 가능성  
- **Transformer 일반화 특성**: 전역어텐션으로 장거리 상관성 포착, 다양한 조건(input modalities) 수용  
- **스케일-업 이점**: 모델 크기 vs. 데이터 양 비례, 더 큰 데이터셋·다양한 도메인 전이학습 기대  
- **교차 도메인 적용**: 텍스트-이미지, 비디오 디퓨전에 DiT 백본 활용하여 멀티모달 일반화 강화 가능성  

## 4. 향후 연구 영향 및 고려사항  
- **아키텍처 통합**: U-Net→DiT 전환은 향후 모든 비전 디퓨전 연구 표준으로 자리잡을 전망  
- **더 큰 모델·토큰 수**: Gflops 증가에 따른 성능 향상 지속 관찰 필요  
- **VAE 개선**: 고해상도 VAE, end-to-end 학습으로 디코더 병목 해소  
- **샘플 다양성 제어**: guidance 기법 연구로 다양성과 품질 균형 고찰  
- **효율화**: Sparse/linear-attention, 지식 증류로 추론·훈련 비용 절감  

> **주요 시사점**: 순수 트랜스포머 백본으로도 디퓨전 모델의 성능과 확장성을 극대화할 수 있음을 입증했으며, 향후 디퓨전 아키텍처의 표준 축이 U-Net에서 DiT로 이동할 가능성이 크다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d40d703d-7e03-4503-bf50-ac45f7ae264b/2212.09748v2.pdf
