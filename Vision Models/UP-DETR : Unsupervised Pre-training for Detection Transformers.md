# UP-DETR : Unsupervised Pre-training for Detection Transformers | Object detection

## 핵심 주장 및 주요 기여
“Unsupervised Pre-training for Detection Transformers” 논문은 DETR(Detection Transformer) 모델의 핵심인 트랜스포머 모듈을 **무(無)라벨 데이터**로 사전 학습(pre-training)함으로써,  
- 객체 검출 성능을 크게 향상시키고  
- 학습 수렴 속도를 획기적으로 단축할 수 있음을 보인다.  

주요 기여:
- **Random Query Patch Detection**: 입력 이미지에서 임의로 잘라낸 패치(query patch)를 디코더에 쿼리로 제공하여, 해당 패치의 원래 위치를 예측하는 새로운 자가 지도(pretext) 과제 제안.
- **Frozen Backbone**: 사전 학습 시 CNN 백본을 고정함으로써, 분류(feature discrimination)와 위치(localization) 학습 간 충돌을 방지.
- **Multi-query Localization**: M개의 패치를 N개의 object query에 할당하고 attention mask를 적용하여, 복수 쿼리 간 경쟁(NMS 유사 메커니즘)을 모방.
- **범용성**: 같은 모델 구조로 객체 검출(object detection), 원샷(one-shot) 검출, 파노픽 세그멘테이션(panoptic segmentation)에서 모두 강력한 성능 발휘.

## 해결하고자 하는 문제
DETR은 end-to-end set-prediction 방식으로 간결하지만  
1) 트랜스포머 모듈이 무작위 초기화되어 대용량 데이터와 긴 학습 스케줄이 필요  
2) 작은 데이터셋(PASCAL VOC)에서 수렴이 느리고 성능이 저조  
라는 한계를 지닌다.

## 제안하는 방법
### 1) Pre-training 과제: Random Query Patch Detection  
입력 이미지 $$I$$에서 무작위로 $$M$$개의 패치 $$\{P_j\}_{j=1}^M$$를 잘라낸 뒤,  
- CNN → GAP로 패치 특징 $$p_j\in\mathbb{R}^C$$ 추출  
- Object queries $$q_i\in\mathbb{R}^C$$와 합쳐서 트랜스포머 디코더에 입력  
- 디코더 출력 $$\hat{y}_i=(\hat{c}_i,\hat{b}_i,\hat{p}_i)$$으로  
  -  $$\hat{c}_i$$: 패치 매칭 여부(이진 분류)  
  -  $$\hat{b}_i=(x,y,w,h)$$: 박스 좌표 회귀  
  -  $$\hat{p}_i$$: 패치 특징 재구성  
- 헝가리안 매칭 후 다음 손실 계산:  

```math
    \mathcal{L} = \sum_{i=1}^N \Bigl[\lambda_{c_i}L_{\mathrm{cls}}(c_i,\hat{c}_{\sigma(i)}) + \mathbf{1}_{c_i=1}\,L_{\mathrm{box}}(b_i,\hat{b}_{\sigma(i)}) + \mathbf{1}_{c_i=1}\,L_{\mathrm{rec}}(p_i,\hat{p}_{\sigma(i)})\Bigr].
```
  
  - $$L_{\mathrm{cls}}$$: 이진 교차 엔트로피, $$\lambda_{1}=1,\;\lambda_{0}=M/N$$  
  - $$L_{\mathrm{box}}$$: $$\ell_1$$ + gIoU  
  - $$L_{\mathrm{rec}}=\|p_i/\|p_i\|\_2-\hat{p}\_{\sigma(i)}/\|\hat{p}_{\sigma(i)}\|_2\|_2^2$$

### 2) Frozen CNN Backbone  
Pre-training 단계에서 CNN 파라미터를 고정(frozen)하여,  
- 위치(localization) 학습이 분류(feature discrimination) 특성을 해치지 않도록 함.

### 3) Multi-query Localization & Attention Mask  
- $$N$$개의 object query를 $$M$$개 그룹으로 나누어 각 패치에 할당  
- 디코더 self-attention에 mask $$X_{i,j}=0$$ (같은 그룹), $$-\infty$$ (그 외) 적용  
- NMS 유사 억제 메커니즘 학습 유도  

### 모델 구조
사전 학습 후, DETR와 동일한 구조(ResNet-50 + 6-layer encoder + 6-layer decoder)로  
- **Object Detection**: learnable query만 입력  
- **One-shot Detection**: query patch 특징을 모든 쿼리에 추가 입력  
- **Panoptic Segmentation**: DETR에 mask head 추가

## 성능 향상
- **PASCAL VOC** (trainval07+12 → test2007)  
  -  150 에폭: AP 49.9→56.1 (+6.2), AP50 74.5→79.7 (+5.2)  
  -  300 에폭: AP 54.1→57.2 (+3.1)  
- **COCO** (train2017→val2017)  
  -  150 에폭: AP 39.7→40.5 (+0.8)  
  -  300 에폭: AP 42.1→43.1 (+1.0)  
- **One-shot Detection**: unseen 클래스 AP50 42.2→61.2 (+19.0) 수준 향상  
- **Panoptic Segmentation**: PQ 44.3→44.7, APseg 32.9→34.3  

## 한계 및 일반화 성능
- **사전 학습–미세 조정 갭**:  
  -  ImageNet(단일 객체, 작은 해상도) vs. COCO/PASCAL(다중 객체, 큰 해상도)  
  -  패치 수준(pretext) vs. 객체 수준(fine-tuning) 불일치  
- **FPN 부재**로 작은 객체 검출(APS) 성능이 일부 제한  
- **백본 고정**이 필수적이지만, end-to-end 학습을 위해 CNN+트랜스포머 통합 사전 학습 필요  

## 향후 연구 및 고려 사항
- **백본-트랜스포머 통합 사전 학습**: contrastive learning과 patch detection 과제를 결합한 end-to-end 방식  
- **Few-shot/Zero-shot** 검출 및 트래킹으로 확장  
- **다양한 백본 아키텍처**(Swin, ConvNeXt)와 조건부 쿼리 생성 전략 실험  
- **효율적 attention mask 설계**로 대규모 M, N 설정 가능성 검토

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/93509f91-1032-4d12-880d-d23bcb41903e/2011.09094v3.pdf
