# ViTDet : Exploring Plain Vision Transformer Backbones for Object Detection | Object detection

**핵심 요약 (Executive Summary)**  
Plain(비계층적) Vision Transformer(ViT) 백본을 그대로 객체 검출에 활용하면서, 복잡한 FPN이나 계층적 구조 없이도 최첨단 성능을 달성할 수 있음을 실험적으로 입증했다. Masked Autoencoder(MAE) 사전학습을 결합한 ViTDet 모델은 COCO에서 최대 61.3 $$\text{AP}\_\text{box}$$ , LVIS에서 48.1 $$\text{AP}\_\text{mask}$$ 를 기록하며, 기존 계층적 백본 기반 방법들과 동등하거나 그 이상의 결과를 보였다.

***

## 1. 해결하고자 하는 문제  
기존 객체 검출 연구는 ConvNet 계층적 백본과 이에 맞춘 FPN(neck) 설계를 전제해 왔으나,  
- Plain ViT는 단일 스케일(feature map 하나)만을 제공하여 멀티스케일 객체 검출에 직접 적용하기 어려움  
- 사전학습(pre-training) 단계에서부터 계층 구조를 도입해야만 검출에 적합한 성능을 얻음  
  
**목표**: 원본 ViT 아키텍처를 변경하지 않고, fine-tuning 시 최소한의 모듈만 추가하여 객체 검출 성능을 계층적 백본에 필적하도록 끌어올리는 것.

***

## 2. 제안 방법  
### 2.1. Simple Feature Pyramid  
– Plain ViT 최종 출력 $$F\in\mathbb{R}^{H\times W\times C}$$ (stride=16) 하나만 사용  
– 병렬적인 **stride/upsample convolution**을 통해 $$\{\tfrac1{32},\tfrac1{16},\tfrac1{8},\tfrac1{4}\}$$ 스케일 피라미드를 생성  

$$
  P_s = \text{Conv}_{s}(F),\quad s\in\{1/32,\,1/16,\,1/8,\,1/4\}.
  $$  

– lateral/top-down 연결 없이 단순 생성만으로 FPN 효과 달성  

### 2.2. Window Attention + Cross-Window Propagation  
– 고해상도 입력 시 전역 self-attention 대신, $$M\times M$$ 윈도우로 분할한 **window attention** 적용  
– 윈도우 간 정보 전파를 위해 **4개의 propagation block** 삽입  
  1. **Global propagation**: 일부 블록에서 전체 윈도우에 걸친 self-attention  
  2. **Convolutional propagation**: residual conv 블록(예: basic, bottleneck) 적용  
– 전체 블록 중 소수만 범윈도우(global) 연산을 수행하므로 연산/메모리 부담 최소화  

### 2.3. MAE 사전학습  
– ImageNet-1K MAE(Self-supervised)로 ViT-B/L/H 사전학습  
– 지도학습(supervised) 대비 검출 fine-tuning에서 3–4 $$\text{AP}\_\text{box}$$ 상승  

***

## 3. 주요 성능 및 비교  
| 모델            | COCO AP$$_{box}$$ (Mask R-CNN) | LVIS AP$$_{mask}$$ | 사전학습 방법     |
|----------------|-------------------------------|-------------------|-----------------|
| Swin-L (계층)  | 52.4                          | 41.7              | IN-21K Sup.     |
| MViTv2-L       | 53.6                          | 41.7              | IN-21K Sup.     |
| **ViT-L (Plain)** | **55.6**                      | **46.0**          | IN-1K MAE       |
| ViT-H (Plain)  | 56.7                          | 48.1              | IN-1K MAE       |

- ViT-L/H는 계층적 백본 대비 +2~3 AP 우위  
- FLOPs 및 inference time에서도 하드웨어 친화적  

***

## 4. 한계 및 일반화 성능  
1. **긴 꼬리 분포(LVIS Rare 클래스)**  
   – LVIS Rare 클래스에서 기존 계층적 백본 대비 개선폭이 상대적으로 작음  
2. **추가 inductive bias 부족**  
   – 완전한 plain 구조로 인해 작은 객체나 극단 해상도 변화에 민감할 수 있음  
3. **MAE vs. Hierarchical 학습**  
   – MAE 사전학습이 plain ViT에 더 큰 이점을 제공하나, 계층적 백본과의 조합 최적화는 미완  
4. **일반화 가능성**  
   – 넓은 모델 크기(규모 확대)에 따라 계층적 백본보다 더 좋은 scaling 특성  
   – 다양한 detection framework(One-stage, Two-stage)를 통해 재현성 및 확장성 확인  

***

## 5. 향후 연구 및 고려 사항  
- **경량화 및 실시간 검출**: Plain ViT 기반 경량화 후처리(neck/head)와 mobile 환경 적용  
- **MAE 기반 계층적 백본**: 계층적 ViT에 MAE pre-training 적용으로 성능 극대화  
- **윈도우 크기/배치 전략**: adaptive window·shift 전략 연구로 작은 객체 일반화 강화  
- **비지도 검출 모듈**: 검출 전용 inductive bias(Anchor-free, query-based)와 결합하여 plain backbone 효과 증대  

Plain ViT의 단순함과 강력한 MAE 학습이 객체 검출에도 충분히 경쟁력 있음을 보여주었으며, 백본과 검출 모듈의 분리를 통해 향후 다양한 Transformer 발전을 곧바로 검출에 활용할 수 있는 연구 방향을 제시한다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/aa5d713b-fe93-4de4-983c-3ce074a5f917/2203.16527v2.pdf
