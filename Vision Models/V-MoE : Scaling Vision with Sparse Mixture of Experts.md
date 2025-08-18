# V-MoE : Scaling Vision with Sparse Mixture of Experts | Image classification

## 1. 핵심 주장 및 주요 기여  
이 논문은 Vision Transformer(ViT)에 **Sparse Mixture of Experts**(MoE) 기법을 도입한 V-MoE 모델을 제안한다.  
- **확장성**: MoE 레이어를 통해 파라미터 수는 수십억~조 단위로 키우면서도, 입력마다 활성화되는 전문가 수(k)가 매우 작아 연산량은 거의 일정하게 유지  
- **효율성**: 이미지 분류 성능은 최첨단 대형 ViT와 동등하거나 상회하되, 동일 성능 달성 시 최대 절반 이하의 추론 연산(FLOPs)  
- **유연성**: 학습된 V-MoE 모델에 대해 추론 시 희소도(sparsity) 및 k 값을 조절하여 성능–연산량 간 트레이드오프를 실시간으로 최적화  
- **Batch Prioritized Routing**: 입력 패치 중요도에 따라 전문가 할당을 우선순위화해 덜 유용한 패치를 과감히 생략, 추론 연산 절감  

## 2. 문제 정의 및 제안 방법  
### 2.1 해결하고자 하는 문제  
- 기존 대형 ViT는 **dense** 구조로 모든 파라미터가 모든 입력 패치에 소진되어, 모델 확장 시 연산 비용이 선형 이상으로 증가  
- MoE 도입 시 불균형(balancing)과 비차별적 패치 라우팅으로 일부 전문가만 과도하게 사용되는 현상  

### 2.2 수식 및 라우팅 메커니즘  
- MoE 레이어: 입력 토큰 $$x\in\mathbb{R}^D$$에 대해  

$$
    \mathrm{MoE}(x)=\sum_{i=1}^E g(x)_i\,e_i(x),
  $$
  
여기서 $$E$$는 전문가 수, $$e_i$$는 MLP 전문가, $$g(x)$$는 softmax 기반 라우터  
- **TOP-k gating**:  

$$
    g(x)=\mathrm{TOPk}\bigl(\mathrm{softmax}(Wx+\epsilon)\bigr),
    \quad \epsilon\sim\mathcal{N}(0,\tfrac{1}{E^2}),
  $$
  
k≪E 전문가만 활성화  

- **용량 비율** $$C$$ 제어: 전문가별 최대 처리 토큰 수  

```math
    B_e = \text{round}\Bigl(\tfrac{kNP}{E}C\Bigr)
```

### 2.3 모델 구조  
- 기본 ViT 블록 내 일부 MLP를 MoE 레이어로 대체  
- Every-2 (짝수 블록마다 MoE) 또는 Last-n (마지막 n개 블록만 MoE) 배치  
- Layer 수 $$L$$, 전문가 수 $$E$$, 선택 전문가 수 $$k$$, 용량비 $$C$$ 는 추론 단계에서 가변  

### 2.4 Batch Prioritized Routing (BPR)  
- 각 패치 $$p$$의 중요도 점수 $$s(p)=\max_i g(x)\_{p,i}$$ 혹은 $$\sum_i g(x)_{p,i}$$로 정의  
- 중요도 역순으로 패치 정렬 후 할당, 덜 중요한 패치는 할당 실패 시 즉시 생략  
- 작은 $$C$$ 환경에서 효과적으로 불필요 토큰 제거  

## 3. 성능 향상 및 한계  
### 3.1 성능 요약  
- **Upstream (JFT-300M)**: V-MoE-H/14 Every-2 모델이 ViT-H/14 대비 약 +4%p Precision@1 달성  
- **Few-shot (ImageNet 5-shot)**: 동일 연산량 대비 +3~4%p, 절반 FLOPs 시에도 동등 성능  
- **Fine-tuning (ImageNet)**: V-MoE-H/14 Every-2가 ViT-H/14 대비 +0.3%p  
- **추론 FLOPs 절감**: BPR 적용 시 C=0.5에서 dense 대비 20~30% 연산만으로 동등 성능  

### 3.2 일반화(Generalization) 향상  
- **스케일 효과**: 전문가 수 및 레이어 수 확장 시, 대형 데이터셋(JFT-3B) 학습 후 소량의 전이학습(1k 샘플)에도 90.35% ImageNet fine-tune 달성  
- **패치 선택**: BPR로 중요 패치만 처리해 background 패치 연산 생략, 오버피팅 방지 및 추론 효율 증대  
- **로우데이터 전이**: few-shot, VTAB(1k 샘플/task) 에서 V-MoE가 dense 대비 일관되게 우수 또는 동등  

### 3.3 한계  
- **불균형 로딩**: C≪1 시 일부 토큰 우선순위 부여 과정에서 잠재적 정보 손실  
- **메모리·통신 오버헤드**: MoE 레이어 간 디바이스 간 토큰 교환 비용  
- **적은 데이터 학습**: JFT 데이터 규모 축소 시(3% 이하), MoE 확장 이득 감소  
- **라우팅 불확실성**: stochastic noise 도입으로 결정 불안정, 추가 안정화 기법 필요  

## 4. 향후 연구 영향 및 고려 사항  
- **대규모 비전 모델**: MoE를 이용한 sparse scaling은 파라미터 수 조절과 연산량 절충을 자유롭게 할 수 있어, 멀티모달·비디오 모델 확장에 핵심  
- **라우터 학습 안정화**: 중요도 예측 정교화, learned routing score 조합 함수 탐색 필요  
- **효율적 하드웨어 구현**: 디바이스 간 통신 병목 해소, 동적 패치 스케줄링 지원  
- **데이터 효율성**: 소규모 데이터 또는 공개 데이터셋(ImageNet-21k) 학습 시 MoE 특성 최적화 및 데이터 증강 전략 강화  
- **안전·환경적 지속가능성**: sparse inference로 CO₂ 배출 저감, practical deployment 시 에너지 절감 효과 연구  

---  
이상으로 V-MoE 논문의 핵심 기여와 일반화 성능 관점에서의 의의를 간략하게 정리하였다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/fc8c8baf-1c31-453d-812e-1ab5d07198a6/2106.05974v1.pdf
