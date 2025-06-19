### LoRA: Low-Rank Adaptation of Large Language Models  

LoRA(Low-Rank Adaptation)는 대규모 언어 모델(LLM)을 효율적으로 미세 조정(fine-tuning)하기 위한 혁신적인 기법입니다. 기존의 전체 파라미터 재조정 방식과 달리, 사전 훈련된 모델 가중치를 고정한 상태에서 **저순위 행렬 분해(low-rank decomposition)**를 활용해 적은 수의 파라미터만 업데이트합니다. 이로 인해 GPT-3 175B 같은 초대형 모델의 경우 **훈련 가능 파라미터를 10,000배 이상 감소**시키면서도 동등하거나 더 나은 성능을 달성합니다[1][2].  

#### 1. **핵심 작동 원리**  
- **저순위 행렬 쌍 적용**: Transformer 레이어의 기존 가중치 행렬 $$W$$에 $$W + A \times B$$ 형태로 업데이트합니다. 여기서 $$A$$(차원 $$d \times r$$)와 $$B$$(차원 $$r \times d$$)는 **초기값이 0인 저순위 행렬**($$r \ll d$$)로, 훈련 중에만 조정됩니다[1][4].  
- **파라미터 효율성**: 예를 들어 $$d=1,000$$, $$r=8$$일 경우 기존 1M 파라미터 대비 $$2 \times 1,000 \times 8 = 16,000$$개만 훈련하면 됩니다[2].  

#### 2. **주요 장점**  
- **저장 공간 절감**: GPT-3 175B의 체크포인트 크기가 1TB에서 **25MB로 감소**[2].  
- **추론 지연 제거**: 훈련 후 $$A \times B$$를 원본 가중치에 병합(merge)하여 **추론 속도 저하 없음**[1][4].  
- **다중 작업 전환**: 작업별 LoRA 모듈(예: $$A_{\text{프랑스어}} \times B_{\text{프랑스어}}$$)을 실시간 교체하며 **다양한 태스크 지원**[2][4].  

#### 3. **실험 결과**  
- **성능 검증**: RoBERTa, DeBERTa, GPT-2, GPT-3에서 **전체 미세 조정과 동등하거나 우수한 성능** 달성[1].  
- **훈련 효율성**: GPU 메모리 사용량 **3배 감소**, 훈련 처리량(throughput) 향상[1].  

#### 4. **적용 가이드**  
- **순위($$r$$) 선택**: 낮은 순위(예: $$r=8$$)로 시작해 필요 시 점진적으로 증가[2].  
- **전체 미세 조정 필요 시점**: 사전 훈련 데이터와 완전히 다른 도메인(예: 영어→화성어) 작업 시 권장[2].  
- **범용성**: **선형 변환을 사용하는 모든 모델 구조**(CNN, Transformer 등)에 적용 가능[2][4].  

#### 5. **공학적 활용 사례**  
- **RAM 캐싱**: 여러 LoRA 모듈을 메모리에 저장해 **실시간 작업 전환** 지원[2].  
- **병렬 훈련**: 서로 다른 배치 데이터로 **동시에 다중 LoRA 모듈 훈련**[2].  

> LoRA는 대규모 모델의 효율적 적용을 가능케 하는 핵심 기술로, 오픈소스 구현체는 [Microsoft의 GitHub 저장소](https://github.com/microsoft/LoRA)에서 확인할 수 있습니다[1].  

이 기법은 모델의 과적합(over-parametrization) 특성을 활용해 적은 자원으로도 높은 성능을 이끌어내며, LLM의 실용적 배포를 가속화합니다[1][2][4].

[1] https://openreview.net/forum?id=nZeVKeeFYf9
[2] https://weaviate.io/papers/lora
[3] https://openreview.net/pdf?id=nZeVKeeFYf9
[4] https://www.ibm.com/docs/en/watsonx/w-and-w/2.1.0?topic=tuning-lora-fine
[5] https://www.ibm.com/think/topics/lora
[6] https://arxiv.org/abs/2502.14816
[7] https://dl.acm.org/doi/10.1145/3727582.3728688
[8] https://www.semanticscholar.org/paper/LoRA:-Low-Rank-Adaptation-of-Large-Language-Models-Hu-Shen/a8ca46b171467ceb2d7652fbfb67fe701ad86092
[9] https://portkey.ai/blog/lora-low-rank-adaptation-of-large-language-models-summary/
[10] https://arxiv.org/abs/2309.14717
[11] https://arxiv.org/abs/2406.01775
[12] https://arxiv.org/abs/2409.02119
[13] https://ieeexplore.ieee.org/document/10711229/
[14] https://ieeexplore.ieee.org/document/10946960/
[15] https://arxiv.org/abs/2410.16801
[16] https://arxiv.org/abs/2502.19747
[17] https://arxiv.org/abs/2402.10462
[18] https://arxiv.org/abs/2106.09685
[19] https://github.com/microsoft/LoRA
[20] https://aclanthology.org/2024.lrec-main.206.pdf
