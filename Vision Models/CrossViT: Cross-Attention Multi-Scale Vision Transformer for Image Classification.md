# CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification | Image classification

## 1. 핵심 주장 및 주요 기여
“CrossViT” 논문은 **서로 다른 크기의 패치 토큰**(대·소 분기)을 병렬로 처리하고, **크로스-어텐션**을 통해 분기 간 정보를 효율적으로 교환함으로써 Vision Transformer의 **다중 스케일 특징 표현**을 학습하고 이미지 분류 성능을 크게 향상시킨다.  
주요 기여:
- Dual‐branch 구조로 서로 다른 패치 크기(Ps, Pl) 처리  
- CLS 토큰 기반의 **선형 시간** 크로스-어텐션 융합 모듈 제안  
- ImageNet1K에서 DeiT 대비 최대 +2.5% Top-1 정확도 향상  
- 효율성과 일반화 능력을 동시에 확보  

## 2. 문제 정의 및 제안 방법
### 2.1 해결 문제  
- Vision Transformer(ViT)는 단일 패치 크기로만 입력을 처리하여, 세밀함(fine-grained)과 계산 효율성(efficiency) 사이에서 균형을 맞추기 어려움.  
- 모든 토큰을 Self-Attention으로 융합하면 계산 및 메모리 복잡도가 $$O(n^2)$$로 불필요하게 증가.

### 2.2 모델 구조  
Dual-branch Multi-Scale Transformer  
- **L-Branch (Large)**: 패치 크기 $$P_l$$, 깊고 넓은 임베딩  
- **S-Branch (Small)**: 패치 크기 $$P_s$$, 얕고 좁은 임베딩  
- 각 분기마다 Transformer Encoder를 $$N$$, $$M$$회 적용  
- **크로스-어텐션 모듈**을 $$L$$회 삽입하여 분기 간 융합  

### 2.3 수식으로 보는 크로스-어텐션
1) 분기 l에서 작은 분기 패치 토큰과 CLS 결합:  

$$
x'\_l = [f_l(x^l_{cls}) \,\|\, x^s_{patch}]
$$

2) CLS만 쿼리로 사용하는 크로스-어텐션:  

$$
\begin{aligned}
q &= x'^l_{cls} W_q,\quad k = x'_l W_k,\quad v = x'_l W_v,\\
A &= \mathrm{softmax}\!\bigl(\tfrac{q k^T}{\sqrt{C/h}}\bigr),\quad
\mathrm{CA}(x'_l)=A\,v
\end{aligned}
$$

3) 잔차 연결 및 역투영:  

```math
y^l_{cls} = f_l(x^l_{cls}) + \mathrm{MCA}(\mathrm{LN}(x'_l)),\quad
z_l = [\,g_l(y^l_{cls}) \,\|\, x^l_{patch}\,]
```

### 2.4 성능 향상  
- ImageNet1K Top-1 정확도: DeiT-Ti 72.2% → CrossViT-Ti 73.4% (+1.2%), CrossViT-9† 77.1% (+3.2%)  
- FLOPs 및 파라미터 증가폭은 20–50% 수준에 그치며, 크로스-어텐션은 선형적 연산으로 효율적  
- Transfer Learning: CIFAR10/CIFAR100 등 5개 과제에서 DeiT와 동등한 일반화 성능 확보  

### 2.5 한계  
- 대분기(L-Branch)에 비해 보조 분기(S-Branch)의 기여도가 상대적으로 작아, S-Branch 구조 최적화 여지  
- Patch 크기 차이가 지나치게 클 경우 학습 불안정성 발생  
- 크로스-어텐션 모듈 빈도(L) 증가 시 성능 이득 미미  

## 3. 모델 일반화 성능 향상 가능성
- **Transfer Learning** 실험에서 다양한 소규모·의료 이미지 과제에서 DeiT 대비 동등 내지 우수한 성능 확인  
- 크로스-어텐션이 분기 간 풍부한 다중 스케일 특징을 학습하여, 데이터셋 변동에도 **특징 표현의 강건성** 확보  
- 향후 **객체 검출, 분할** 등 다른 비전 과제에 적용 시, 풍부한 스케일 정보로 일반화 성능 추가 개선 기대  

## 4. 향후 연구 영향 및 고려 사항
- **후속 연구**: CrossViT 구조를 확장해 **삼중 분기 이상**, 동적 패치 크기 조정, 토큰 중요도 기반 자동 스케일 선택 가능  
- **응용 영역**: 객체 검출·세그멘테이션·비디오 분석에 다중 스케일 Transformer 융합 모듈 적용  
- **고려점**:  
  - 분기 간 패치 크기 비율 및 깊이 비율 최적화  
  - 크로스-어텐션 모듈 삽입 빈도와 위치에 따른 연산-성능 균형  
  - 실시간 처리 제약이 있는 환경에서 효율성 극대화 방안  

***

**핵심 결론**: CrossViT는 다중 스케일 분기를 크로스-어텐션으로 효율적 융합하여 이미지 분류 성능과 일반화 능력을 동시에 향상시키는 새로운 비전 트랜스포머 아키텍처이다. 앞으로 다양한 비전 과제에 걸쳐 다중 스케일 특징 학습을 강화하는 연구가 기대된다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/4b55c3f2-385e-4ec6-af67-2f2801bc349f/2103.14899v2.pdf
