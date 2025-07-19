# Fine-Tuning Image-Conditional Diffusion Models Is Easier than You Think | Depth estimation

## 1. 핵심 주장 및 주요 기여  
이 논문은 **대규모 이미지-조건부 확산 모델(image-conditional diffusion models)** 을 기하 추정(깊이 및 법선 예측)에 재활용할 때,  
- 기존에 느린 것으로 알려진 확산 기반 추정기(diffusion-based estimators)가  
- **단일 단계(single-step) 추론** 으로도 동급의 성능을 보이며  
- 간단한 **end-to-end 미세조정(end-to-end fine-tuning)** 만으로 최첨단 성능을 달성할 수 있음을 보인다.

주요 기여:  
1. Marigold 및 유사 모델의 **DDIM 스케줄러 구현 오류**를 찾아내어 수정 → 단일 단계로 200× 이상 가속[1].  
2. 수정된 단일 단계 모델에 **작업별(task-specific) 손실**을 적용한 end-to-end 미세조정 프로토콜 제안 → 확산 기반 최상위 모델들을 단일 단계에서 능가[1].  
3. 오직 **Stable Diffusion** 만으로도 동일 프로토콜 적용 시 비슷한 성능 획득 → 확산-조건 생성보다 단순 미세조정이 더 효율적임을 입증[1].

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제  
- **기하 추정(Depth & Normal Estimation)**: 단안 이미지에서 픽셀별 깊이·법선을 예측.  
- 기존 확산 모델 기반 방법(Marigold 등)은 다단계 반복 추론으로 **속도 저하**(수 초~수십 초) 발생.  
- 일부 단일 단계 시도가 있지만 품질 저하 또는 복잡한 증류 기법 의존.

### 2.2 제안 방법 개요  
1) **오류 분석 및 수정**  
   - DDIM 스케줄러의 “leading schedule” 구현 오류로, 단일 단계 시 **노이즈-타임스텝 불일치** 발생.  
   - “trailing schedule” 사용 시 학습 시 노이즈 레벨과 정확히 일치하여 단일 단계에서도 의미 있는 예측 가능[1].

2) **단일 단계 End-to-End 미세조정**  
   - 확산 모델의 UNet만 남기고 VAE 인코더·디코더는 고정(frozen).  
   - 추론 단계 고정 $$t = T$$, 입력 노이즈 대신 **평균 노이즈(zeros)** 사용.  
   - 깊이 추정용 **affine-invariant 손실**:

  $$
       L_D = \frac{1}{HW}\sum_{i,j} \bigl|d^*\_{i,j} - (s\,d_{i,j} + t)\bigr|,
     $$
     
  여기서 $$s,t$$는 최적의 스케일·시프트 파라미터[1].  
   - 법선 추정용 **각도 기반 손실**:

```math
L_N = \frac{1}{HW}\sum_{i,j} \arccos \Bigl (\frac{n^*\_{i,j} \cdot \hat n\_{i,j}}{\|n^*\_{i,j}\|\|\hat n\_{i,j}\|}\Bigr).
```

3) **Stable Diffusion 직접 미세조정**  
   - Marigold 사전학습 없이도 동일한 프로토콜 적용 가능 → 확산 사전학습의 강력한 Priors 재확인.

### 2.3 모델 구조  
- 기반: **Stable Diffusion v2** UNet + 고정 VAE (인코더·디코더).  
- 미세조정 단계: UNet만 학습 대상, 입력ⓡ RGB 잠재(latent) 벡터에 평균 노이즈 추가, 타임스텝 고정.  
- 출력: VAE 디코더로 깊이/법선 맵 재구성.

## 3. 성능 향상 및 한계

### 3.1 성능 향상  
- **추론 속도**: 50단계×10앙상블(≈24초) → 1단계 단일 모델(≈0.12초)로 200× 가속, 성능 유지 또는 향상[1].  
- **깊이 예측** (NYUv2 AbsRel):  
  - 기존 Marigold(50,10): 5.5  
  - 수정 단일 단계: 5.7 → E2E FT: 5.2 → StableDiff+E2E FT: 5.4[1].  
- **법선 예측** (NYUv2 mean angular error):  
  - Marigold(50,10): 18.8°  
  - 수정 단일 단계: 17.4° → E2E FT: 16.2°[1].

### 3.2 한계  
- **데이터 규모 의존성**: discriminative 대규모 학습(수백만–수천만 샘플) 대비 사용한 합성 데이터는 수십만 건에 불과.  
- **Metric depth 복원**: affine-invariant 표현만 지원, 절대 깊이(실제 거리) 복원 시 추가 정보 필요.  
- **VAE 디코더 고정**: frozen decoder 한계로 세밀한 표현 한계 가능성(추후 fine-tuning 가능성 검토 필요).

## 4. 일반화 성능 향상 가능성  
- **Diffusion Priors**: 대규모 텍스트-이미지 학습된 강력한 표현력 보유 → in-the-wild 일반화에 유리.  
- **단일 단계 미세조정**: 전이 학습이 효율적, 소규모 고품질 데이터셋으로도 우수한 zero-shot 성능 달성.  
- **Self-training**: 빠른 추론 덕분에 의사 레이블(pseudo-label) 기반 자가학습 확장 가능(Depth Anything, Metric3D 유사)[1].  
- **모달리티 확장**: 추가 센서 데이터(RGB-D, LiDAR) 또는 태스크(세분화, 재구성)로도 유사 프로토콜 적용 가능.

## 5. 향후 연구 영향 및 고려 사항  
- **확산 모델 수정 점검**: DDIM 및 기타 스케줄러 오류 방지를 위해 구현 검증 중요.  
- **End-to-End 미세조정 대화**: 확산 대 discriminative 모델 간 trade-off 재평가.  
- **대규모 자가학습**: 빠른 단일 단계 추론으로 pseudo-label 기반 self-training 연구 활발해질 전망.  
- **절대 깊이 추정**: affine-invariant 한계 보완 위한 추가 데이터(캘리브레이션, sparse depth) 통합 연구 필요.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/c21009e7-d9ca-4530-b052-55cd19f5064f/2409.11355v2.pdf
