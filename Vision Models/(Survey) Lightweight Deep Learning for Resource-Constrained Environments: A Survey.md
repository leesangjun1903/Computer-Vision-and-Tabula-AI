# Lightweight Deep Learning for Resource-Constrained Environments: A Survey
## 핵심 주장 및 기여 요약

“Lightweight Deep Learning for Resource-Constrained Environments: A Survey”는 **리소스 제약 환경**(모바일, IoT, MCU 등)에서 딥러닝 모델의 효율적 설계와 배포를 위한 종합적 가이드라인을 제시한다.  
주요 기여:
- **경량 아키텍처 계보 정리**: MobileNet, ShuffleNet, SqueezeNet 등 대표 모델을 계열별로 분류하고 발전 과정을 체계화  
- **압축·가속 기법 통합**: 양자화, 가지치기(pruning), 지식 증류(distillation), NAS, 하드웨어 가속(ASIC/FPGA/TPU) 기법을 포괄하여 상호 연관성 분석  
- **미래 전망 제시**: TinyML(초저전력 MCU 상 DL)과 경량화된 대규모 언어 모델(LLM)의 잠재력 및 도전 과제를 탐구  

# 문제 정의, 제안 기법 및 모델 구조

1. 문제 정의  
   - 고성능 DL 모델은 수백만~수조 개 파라미터를 가지며, 모바일·엣지 장치에서는 연산·메모리·전력 제약으로 배포가 어려움  

2. 제안 기법  
   1) **경량 CNN 아키텍처**  
      - Depthwise separable conv (MobileNet)  
      - Group conv + channel shuffle (ShuffleNet)  
      - Shift-conv, AdderNet(곱셈→덧셈)  
      - Compound scaling (EfficientNet):  

        $$\text{depth}=d^\phi,\ \text{width}=w^\phi,\ \text{res}=r^\phi $$  

# 2 LIGHTWEIGHT ARCHITECTURE DESIGN

경량 딥러닝 모델은 연산·메모리·전력 제약이 있는 모바일·엣지·IoT 장치에서 효율적으로 동작해야 한다. 2장에서는 이러한 **경량 CNN**과 **경량 트랜스포머** 설계를 다음과 같은 체계로 정리한다.

## 2.1 경량 모델의 기본 개념과 핵심 기법

1. **평가 지표**  
   - FLOPs(부동소수점 연산 수), MACs(곱셈-누산 연산 수)  
   - 메모리 접근 비용(Memory Access Cost)  
   - 추론 속도: Throughput(초당 인퍼런스 수), Latency(한 번 예측에 걸리는 시간)

2. **핵심 연산 블록**  
   ­∙ 1×1 합성곱(Pointwise convolution)  
   ­∙ 그룹 합성곱(Group convolution)  
   ­∙ 깊이별 분리 합성곱(Depthwise separable convolution)

이들 블록은 파라미터·연산량 절감의 근간이며, 다양한 변형을 통해 효율을 더욱 개선한다.

## 2.2 대표 경량 CNN 계열

1. **SqueezeNet 계열**  
   – *Fire 모듈*: 채널 축소(squeeze) 후 1×1·3×3 합성곱을 확장(expand)  
   – SqueezeNext: 저랭크 분해·단축 연결을 도입해 파라미터 112× 축소  

2. **ShuffleNet 계열**  
   – 그룹 합성곱 후 채널 셔플로 정보 교환  
   – V2: MAC 감소·메모리 효율 개선을 위한 설계 가이드라인 제시  

3. **CondenseNet 계열**  
   – DenseNet의 밀집 연결을 ‘학습형 그룹 합성곱(LGC)’으로 구조화된 가지치기  
   – V2: 동적 특성 재활성화 모듈로 가중치 연결 재학습  

4. **MobileNet 계열**  
   – MobileNetV1: 깊이별 분리 합성곱 도입  
   – V2: 역잔차 블록(inverted residual), 선형 병목 구조  
   – V3: NAS·채널 어텐션·H-swish 사용, 하드웨어 친화적 최적화

5. **Shift/Adder 계열**  
   – ShiftNet: 0 파라미터·0 FLOPs인 채널 이동 연산  
   – AddressNet: GPU 친화적 채널 이동 + 추가  
   – AdderNet: 곱셈 대신 덧셈만 사용하여 에너지·연산 절감  

6. **EfficientNet 계열**  
   – 복합 스케일링(depth·width·resolution)을 통한 균형 조정  
   – V2: 훈련 인지형 NAS·Fused-MBConv 도입, 적응적 정규화

이들 모델은 **파라미터·FLOPs 절감**과 **정확도 유지** 간의 다양한 trade-off를 보여 준다.

## 2.3 경량 트랜스포머 계열

1. **효율적 어텐션 모듈**  
   – Linformer: 어텐션 행렬 저차원 분해로 $$O(N^2)\to O(N)$$ 축소  
   – Reformer: LSH 해싱으로 $$O(N^2)\to O(N\log N)$$  
   – FAVOR+: 확률 피처로 선형화  

2. **토큰 축소(Token Sparsing)**  
   – DynamicViT/EViT: 중요 토큰만 추려 처리  
   – T2T-ViT: 이웃 패치를 매핑해 MLP 크기·메모리 절감  

3. **하이브리드 CNN+트랜스포머**  
   – MobileViT/Mobile-Former: MobileNet 블록 + 트랜스포머 블록 융합  
   – Conv-Transformer 융합으로 학습 용이성·일반화 성능 강화  

## 2.4 설계 시 고려 사항

- **하드웨어 호환성**: MAC·메모리 요구량과 디바이스 특성(ARM, FPGA 등)  
- **연산 vs. 메모리**: 깊이별 분리 합성곱은 낮은 FLOPs 대비 높은 MACs 유의  
- **정확도 vs. 효율**: 극단적 경량화 모델(ShiftNet/A-Net)은 작은 정확도 손실, 기존 모델은 더 높은 정확도  
- **NAS 활용**: 하이브리드 모델·하드웨어 제약 반영한 NAS로 최적 아키텍처 자동 탐색

**결론**  
2장에서는 경량 모델 설계의 전 범위를 다루며, 핵심 연산 블록부터 CNN·트랜스포머 계열, 모델 선택 시 하드웨어·성능 간 균형 맞추는 전략까지 체계적으로 설명한다. 이 가이드는 제약된 환경에서도 딥러닝 모델을 효과적으로 배포하기 위한 설계 로드맵을 제공한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7f26df44-b7a3-4fa0-a35c-3b2fab58edba/2404.07236v2.pdf

   2) **압축 기법**  
      - Pruning: 구조화(필터·채널) vs 비구조화  
      - Quantization: 32→8/4/2/1-bit, 대칭/비대칭, FakeQuant 기반 QAT  
      - Knowledge Distillation  

        $$\mathcal{L} = \alpha \mathcal{L}\_{\text{CE}}(y,p_s) + (1-\alpha)\,T^2\,\mathrm{KL}(p_t^{(T)}\|p_s^{(T)}) $$  

# 3 FUNDAMENTAL METHODS IN MODEL COMPRESSION

리소스 제약 환경에서 딥러닝 모델을 경량화하기 위해 주로 사용하는 네 가지 핵심 방법을 이해하기 쉽도록 정리하면 다음과 같습니다:  
1) 가지치기(Pruning)  
2) 양자화(Quantization)  
3) 지식 증류(Knowledge Distillation)  
4) 신경망 구조 탐색(NAS)

## 3.1 가지치기(Pruning)
*목표*: 불필요한 가중치를 제거해 모델 크기·연산량을 줄이고, 추론 속도를 높인다.  

1) 비구조화(unstructured) 가지치기  
  - 개별 가중치(weight)를 중요도 순으로 0으로 만듦  
  - 자유도가 높아 많은 파라미터를 줄일 수 있으나, 희소(sparse) 구조가 하드웨어 가속기와 비호환될 수 있음  
  - 대표 기법: Optimal Brain Damage/Sheron, Lottery Ticket 가설(핵심 서브네트워크 탐색)  

2) 구조화(structured) 가지치기  
  - 채널(channel)·필터(filter)·레이어 단위로 통째로 제거  
  - 온전한 밀집 구조(dense) 유지로 하드웨어 호환성 우수  
  - 필터 노름(norm) 기준, 기하학 중앙(geometric median) 기준, 레이어별 학습 가능한 문턱값(threshold) 등 다양  
  - 채널 가지치기: L1-norm, 헤시안 기반 중요도 계산  
  - 주요 장점: 프레임워크(TensorFlow, PyTorch) 내장 가속 지원  

## 3.2 양자화(Quantization)
*목표*: 32-bit 부동소수점을 저(低)비트 정수/고정소수점(fixed-point) 표현으로 바꿔 메모리 및 연산량 절감  

1) 정밀도 비트 수  
  - 8-bit, 4-bit, 심지어 1-bit(이진화)까지  
  - 비트 수가 줄어들수록 모델 용량과 연산량 감소폭↑, 정밀도 손실↑  

2) 대칭 vs 비대칭  
  - 대칭(symmetric): $$[-α, +α]$$ 구간을 고정 비율로 매핑  
  - 비대칭(asymmetric): $$[α, β]$$ 구간을 유연하게 매핑  

3) 사후 양자화(PTQ) vs 양자화 인식 훈련(QAT)  
  - PTQ: 이미 훈련된 모델을 그대로 양자화 후 미세조정  
  - QAT: 훈련 중 FakeQuant 연산으로 양자화 오차를 학습에 반영  

4) 혼합 정밀도(mixed-precision)  
  - 민감한 계층을 높은 비트(16-bit), 나머지는 낮은 비트(8-bit 이하)로 처리  

## 3.3 지식 증류(Knowledge Distillation)
*목표*: 큰(teacher) 모델이 학습한 지식을 작은(student) 모델로 옮겨, 경량 모델의 성능을 높임  

1) 오프라인(distillation)  
  - 사전 훈련된 teacher 모델 → student 모델에 soft label 전이  
  - 손실함수: $$L = \alpha\,L_\text{CE}(y,p_s)+(1-\alpha)T^2\,\mathrm{KL}(p_t^{(T)}\|p_s^{(T)})$$  

2) 온라인(mutual learning)  
  - 여러 모델을 동시에 학습하며 서로의 예측을 손실에 반영  
  - Deep Mutual Learning(DML) 등  

3) 자기 증류(self-distillation)  
  - 하나의 모델을 시점(레이어 간 혹은 epoch 간)별로 teacher/student로 활용  
  - 복잡도 상승 없이 간편 적용 가능  

## 3.4 신경망 구조 탐색(Neural Architecture Search; NAS)
*목표*: 사람의 개입 없이 최적의 경량 네트워크 구조를 자동으로 탐색  

1) 탐색 공간(search space)  
  - 커널 크기, 채널 수, 블록 연결 방식 등 가능한 조합  

2) 탐색 알고리즘(search algorithm)  
  - 강화학습(RL-NAS), 진화 알고리즘(EA-NAS), 연속화 기울기 기반(DARTS), 하드웨어 인식(Differentiable NAS) 등  

3) 평가 전략(performance evaluation)  
  - 정확도, 지연(latency), 에너지, 메모리 등 목적에 맞춘 보상(reward) 설계  

4) 하드웨어 인식 NAS  
  - 타깃 디바이스별 지연·전력 모델 포함  
  - 예: FBNet, NetAdapt, Once-for-All  

## 모델 압축 기법 간 비교 및 적용 가이드
| 기법              | 장점                                    | 단점                                           |
|-----------------|---------------------------------------|----------------------------------------------|
| 가지치기         | 구조화 시 하드웨어 친화적, 파라미터 대폭 감소   | 비구조화 시 비정형 스파스 구조로 인퍼런스 비효율            |
| 양자화           | 메모리·연산량 2–32배 절감                 | 비트 수 감소 시 정확도 하락                            |
| 지식 증류        | 성능 보존하며 경량화 가능                   | teacher 모델 훈련 비용(오프라인)                     |
| NAS             | 최적의 경량 아키텍처 자동 탐색              | 탐색 비용(특히 RL/EA) 높음; 연속화 기법은 효율적         |

- **간단 적용**: 하드웨어가 정수 연산 지원 시 먼저 8-bit 양자화, 그렇지 않으면 16-bit mixed-precision  
- **추가 경량화**: 구조화 가지치기 → 지식 증류 병합  
- **자동 최적화**: 하드웨어 예산이 충분하다면 NAS, 제한적 자원 시 DARTS-류 연속화 NAS  

이 네 가지 기법을 상황에 맞게 조합·응용하면, 모바일·임베디드·IoT 디바이스에서도 **저용량·저전력·고성능** 딥러닝 모델을 구현할 수 있습니다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7f26df44-b7a3-4fa0-a35c-3b2fab58edba/2404.07236v2.pdf

   3) **하드웨어 가속**  
      - Systolic array(TPU), spatial PE(FPGA), temporal PE(GPU)  
      - 데이터플로우 최적화(파이프라인·스트리밍·다디아나오 등)  

4. 모델 구조  
   - 각 아키텍처별 모듈화(block)로 구성, 계열(series)별 공통 패턴 체계화  
   - 예: MobileNetV2 inverted residual bottleneck, ShuffleNetV2 channel split+shuffle  

5. 성능 향상 및 한계  
   - 경량화 모델은 parameters·FLOPs 대폭 감소하며, 원본 대비 Top-1 정확도 1–5% 내외 저하  
   - **한계**:  
     1) MAC와 FLOPs 감소가 실제 latency 개선으로 이어지지 않는 경우  
     2) 비구조화 pruning 후 하드웨어 비호환성  
     3) NAS·RL 기반 탐색의 과도한 비용(수십~수백 GPU·수주 소요)  

# 일반화 성능 향상 관련 고찰

- **지식 증류**: Teacher→Student 간 부드러운 라벨(soft target) 전이로서 작은 모델의 일반화 개선  
- **Self-Distillation**: 동일 모델 내부 레이어 간 distill로 정규화 효과[PS-KD]  
- **NAS**: 데이터별·하드웨어별 맞춤 아키텍처 탐색으로 다양한 도메인 전이 능력 강화  
- **데이터 증강·교차 검증**: Mobile AI challenge 등 벤치마크 기반 교차 평가를 통해 범용성 확인  

# 향후 연구 영향 및 고려 사항

- **TinyML**: Ultra-low-power MCU(≤1 mW) 상 DL 배포, 표준화된 경량화 프레임워크·데이터셋 필요  
- **경량 LLM**: 대규모 언어 모델 109–12 파라미터의 후처리(prune/no-retrain) 및 구조 설계  
- **하드·소프트 co-design**: 아키텍처 탐색 시 하드웨어 제약을 동시 반영하여 실시간 처리·에너지 효율 고도화  
- **일반화 검증**: 실제 엣지 환경 배포를 통한 장기 안정성·보안·편향성 평가 병행  

경량 딥러닝은 엣지·모바일 AI의 보급을 가속화하여, **저전력·저지연·고정확도** 솔루션 제공을 가능케 한다. 미래 연구는 **하드·소프트 통합 설계**, **자율 학습형 TinyML**, **경량 LLM 최적화**에 집중하여야 한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7f26df44-b7a3-4fa0-a35c-3b2fab58edba/2404.07236v2.pdf
[2] https://dl.acm.org/doi/10.1145/3657282
[3] https://cspub-ijcisim.org/index.php/ijcisim/article/view/1025
[4] https://ieeexplore.ieee.org/document/10920425/
[5] https://ijarsct.co.in/Paper22305.pdf
[6] https://ieeexplore.ieee.org/document/10680036/
[7] https://www.mdpi.com/2079-9292/14/13/2583
[8] https://www.jstage.jst.go.jp/article/transinf/E108.D/7/E108.D_2024EDP7059/_article
[9] https://ijsrem.com/download/innovations-in-wildfire-and-smoke-detection-a-comprehensive-survey/
[10] https://arxiv.org/html/2404.07236v2
[11] https://huggingface.co/papers/2404.07236
[12] https://www.mdpi.com/1424-8220/20/21/6114
[13] https://www.themoonlight.io/en/review/lightweight-deep-learning-for-resource-constrained-environments-a-survey
[14] https://www.ijsat.org/papers/2025/2/6224.pdf
[15] https://www.sciencedirect.com/science/article/pii/S0167739X24004400
[16] https://www.emergentmind.com/papers/2404.07236
[17] https://dl.acm.org/doi/abs/10.1145/3657282
[18] https://www.nature.com/articles/s41598-025-97822-6
[19] https://arxiv.org/abs/2404.07236
[20] https://www.nature.com/articles/s41467-025-59516-5
[21] https://www.aimodels.fyi/papers/arxiv/lightweight-deep-learning-resource-constrained-environments-survey
[22] https://www.themoonlight.io/en/review/comparative-analysis-of-lightweight-deep-learning-models-for-memory-constrained-devices
[23] https://scholar.nycu.edu.tw/en/publications/lightweight-deep-learning-for-resource-constrained-environments-a
[24] https://arxiv.org/abs/2503.20516
[25] https://ieeexplore.ieee.org/document/11013043/
[26] https://arxiv.org/pdf/2404.07236.pdf
[27] https://arxiv.org/pdf/2501.15014.pdf
[28] https://arxiv.org/abs/2208.10498
[29] http://arxiv.org/pdf/2411.03350.pdf
[30] http://arxiv.org/pdf/2010.12309v3.pdf
[31] https://arxiv.org/abs/2106.08962
[32] https://arxiv.org/pdf/2111.05193.pdf
[33] https://arxiv.org/pdf/1710.09282.pdf
[34] https://www.hindawi.com/journals/misy/2020/8454327/
[35] https://arxiv.org/html/2405.12353
