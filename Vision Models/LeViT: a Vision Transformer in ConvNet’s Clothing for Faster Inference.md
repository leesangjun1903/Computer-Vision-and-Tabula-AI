# LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference | Image classification

**핵심 주장 및 주요 기여**  
LeViT는 이미지 분류 시 높은 처리 속도와 정확도 간의 균형을 최적화한 하이브리드 아키텍처로, 다음 네 가지 핵심 기여를 제안한다[1]:  
1. 다단계(transformer stages) 피라미드 구조를 도입하여 해상도를 줄이면서 특징 채널 수를 늘리는 **풀링 기반 다운샘플링**  
2. 입력 패치를 효율적으로 축소하는 **경량 컨볼루션 기반 임베딩**(PatchConv)  
3. 각 어텐션 블록에 상대적 위치 정보를 주입하는 **attention bias**  
4. MLP 확장비를 4→2로 줄이고, 어텐션 핵심 연산 뒤에 **Hardswish 활성화**를 추가하여 연산 균형 및 속도 최적화  

## 1. 해결하고자 하는 문제  
- 기존 Vision Transformer(ViT)와 ConvNet은 다음과 같은 한계를 지닌다:  
  - ViT: 높은 연산 복잡도 및 메모리 요구량  
  - ConvNet: 병렬 하드웨어에서 낮은 처리량(IO 바운드)  
- **목표:** GPU·CPU·ARM 등 다양한 하드웨어에서 **고속 추론**(throughput)과 **경량화**를 동시에 달성  

## 2. 제안하는 방법  

### 2.1 모델 구조 개요  
LeViT는 입력→PatchConv→Stage₁→ShrinkAttention→Stage₂→…→평균풀링→분류기로 구성된다(그림 참조)[1].  
- PatchConv: 3×3 stride=2 컨볼루션 4회, 채널 3→256  
- 각 Stage: N개의 Attention+MLP 블록 
- ShrinkAttention: Q 연산 전 해상도 ½ 축소, 출력 채널 증가, 잔류 연결 없이 정보 보존  
- 최종: 평균풀링으로 BCHW→벡터 변환  

### 2.2 어텐션 블록 세부  
- Q,K,V 크기 차별화: key 차원 D 작게(V는 2D)  
- Attention bias:  

$$ A^{(h)}\_{(x,y),(x',y')} = Q\_{(x,y)} \cdot K_{(x',y')} + B^{(h)}\_{|x-x'|,|y-y'|} $$ 

  모든 헤드 h마다 상대 위치 옵셋별 편향 학습  
- Attention output 후 Hardswish 적용  
- MLP: 1×1 conv, 확장비 2 (대비 ViT의 4)  

## 3. 성능 향상  
| 모델          | FLOPs (M) | CPU 속도 (im/s) | Top-1 (%) | 비교군              |
|---------------|-----------|-----------------|-----------|---------------------|
| LeViT-128S    | 305       | 39.1            | 76.6      | DeiT-Tiny (1220M)   |
| LeViT-192     | 658       | 24.2            | 80.0      | EfficientNet B2     |
| LeViT-256     | 1120      | 16.4            | 81.6      | EfficientNet B3     |

- **5–7× 속도 향상**(CPU) 및 **유사 정확도** 달성  
- ARM에서도 유사한 트레이드오프 우위  

## 4. 일반화 성능 및 한계  
- **데이터 효율적 학습:** DeiT식 distillation 활용, 레이블 노이즈에 덜 민감  
- **절대적 위치 인코딩 제거:** attention bias로 대체하여 모델 용량 절감  
- **한계:**  
  - 작은 데이터셋(Imagenet 이외)에서 distillation 의존도  
  - positional bias 범위(H,W)에 국한된 국소적 제약  
  - 고해상도 입력 처리 시 스테이지 조정 필요  

## 5. 향후 연구 영향 및 고려사항  
- **경량화 Transformer 설계:** PatchConv+bias 방식이 후속 ViT 계열에 적용 가능  
- **하드웨어 최적화:** BatchNorm fusion, NEON/MKL 특화 연산 모듈 설계  
- **일반화 강화:** 대규모 무레이블 데이터에 대한 self-supervised distillation 결합  
- **해상도 적응:** 동적 스테이지 배치 및 이미지 크기 별 구조 자동 탐색(AutoML)  

LeViT는 고속 추론과 Transformer의 유연성을 결합한 혁신적 설계로, 실시간 응용과 엣지 디바이스에 최적화된 비전 모델의 표준을 제시한다[1].

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8a238514-71b1-4220-a64e-d66661d59c5e/2104.01136v2.pdf
