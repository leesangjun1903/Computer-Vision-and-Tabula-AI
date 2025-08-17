# ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers | Image matting

**핵심 주장 및 주요 기여**  
ViTMatte는 *사전 학습된 plain ViT(비계층적 Vision Transformer)*를 이미지 매팅(image matting) 과제에 최소한의 적응(adaptation)만으로 적용하여, 기존의 복잡한 CNN 기반·하이브리드 구조보다 더 우수한 성능을 내는 최초의 매팅 시스템이다.  
- **하이브리드 어텐션**: 윈도우(window)·글로벌(global) 어텐션을 적절히 조합해 연산량을 절반가량 줄이면서도 성능 개선  
- **컨볼루션 넥(neck)**: 각 어텐션 그룹 후에 경량 ResBottleNeck 블록을 넣어 저주파 정보는 유지하고 고주파(경계·디테일) 정보 강화  
- **디테일 캡처 모듈(DCM)**: 3단계의 3×3 컨볼루션 스트림으로 고해상도 특징을 보완해 매팅 품질 크게 향상  
- Composition-1k 벤치마크에서 SAD 21.46→20.33, Connectivity 16.21→14.78, Distinctions-646에서 SAD 21.22→17.05 대폭 개선  

***

## 1. 해결하고자 하는 문제  
- **이미지 매팅**: 주어진 이미지와 사용자가 그린 trimap(전경·배경·미지정 영역)을 바탕으로 픽셀별 알파 매트(투명도)를 정확히 추정  
- 기존 CNN 기반 매팅은 계층적 피처 퓨전과 복잡한 디코더가 필수였으나, 비계층 plain ViT는 고해상도 경계 디테일 잡는 데 부적합  

## 2. 제안 방법  
### 2.1 입력 및 전체 구조  
- 입력: RGB 이미지 $$X\in\mathbb{R}^{H\times W\times3}$$와 trimap $$T\in\mathbb{R}^{H\times W\times1}$$을 채널 결합  
- 출력: 알파 매트 $$\hat\alpha\in\mathbb{R}^{H\times W\times1}$$  

```
Input (H×W×4) → ViT Backbone
  └─ Hybrid Attention + Convolution Neck → Feature F4 (H/16×W/16)
ConvStream (3×3 conv ×3) → Detail Maps D1, D2, D3 (해상도 1/2·1/4·1/8)
Fusion Module: Upsample(Fi) ⊕ Di → Conv → 점진적 복원 → α
```

### 2.2 Hybrid Attention  
- ViT 블록을 $$m$$개 그룹, 각 그룹 $$n$$개 블록으로 나눔  
- 그룹 내 $$n-1$$개 블록은 윈도우 어텐션, 마지막 1개는 글로벌 어텐션 적용  
- 계산 복잡도:  

$$
    O(HW\times HW\times C)\to
    O\bigl(k^2\times k^2\times C\bigr)
  $$  

- 윈도우 어텐션 4블록 구성 시 FLOPs 50% 절감, 오히려 SAD 1.52·MSE 0.74 개선  

### 2.3 Convolution Neck  
- 각 어텐션 그룹 후에 ResBottleNeck 블록 삽입(Residual 방식)  
- 트랜스포머가 주로 저주파에 집중하는 반면, 컨볼루션이 고주파(경계·텍스처) 캡처 강화  
- FLOPs 2% 증가 불과, SAD 28.31→27.24, MSE 5.64→5.14 개선  

### 2.4 Detail Capture Module (DCM)  
- 3단계 conv3×3 스트림으로 원본 해상도 절반·4분의1·8분의1 디테일 맵 생성  
- ViT 출력 F4와 단계별 디테일 맵을 결합·컨볼루션 통합  
- 파라미터 0.1M 미만 추가로 MSE 7.21 개선  

### 2.5 학습 및 손실  
- **손실 함수**:  
  $$\mathcal{L}_{total} = \mathcal{L}_{l1}^{known/unknown} + \mathcal{L}_{laplacian} + \mathcal{L}_{gradient\ penalty}$$  
- **파인튜닝**: FNA++ 방식으로 채널 확장, DINO/MAE 사전학습 weight 초기화, layer-wise LR 스케줄  

## 3. 성능 향상 및 한계  
### 성능  
| 벤치마크           | 기존 SOTA (MatteFormer) | ViTMatte-S | ViTMatte-B |
|--------------------|--------------------------|------------|------------|
| Composition-1k SAD | 23.80                    | **21.46**  | **20.33**  |
| Conn               | 18.90                    | **16.21**  | **14.78**  |
| Distinctions-646 SAD| 25.65                   | **21.22**  | **17.05**  |

- 모델 크기: ViTMatte-S 파라미터 25.8M(기존 44.8M→42.2%↓)  
- FLOPs: 고해상도(2048×2048) 기준 1.65T→1.02T (약 40%↓)  

### 한계  
- **고해상도 글로벌 어텐션 부담**: 추론 시 그리드 샘플링으로 메모리 66%↓ 가능하나 성능 소폭 저하  
- **동영상 매팅 및 real-time**: 트랜스포머 구조 특성상 실시간 처리 어려움  
- **더 작은 데이터셋 일반화**: 사전학습 효과 의존도 높음  

## 4. 일반화 성능 향상 관련  
- **다양한 사전학습 전략 지원**: ImageNet21k·DINO·iBOT·MAE 모두 무리 없이 초기화  
- **자가지도학습(MAE, DINO)이 최고의 결과**: supervised 대비 SAD 27.61→26.15 개선  
- **경량 디코더**: 디테일 캡처 모듈만으로도 다양한 환경에서 오버피팅 최소화  

이러한 구조적 모듈화와 사전학습 유연성 덕분에 새로운 도메인(예: 의료·위성영상) 매팅 과제에도 빠르게 적응 가능하며, 소규모 레이블링 데이터로도 강한 일반화 성능 기대  

***

## 5. 향후 연구 영향 및 고려사항  
- **파운데이션 모델 활용 증대**: 복잡한 디코더 설계보다 사전학습된 ViT 기반 미세조정이 더 효율적임을 입증  
- **경량 매팅 솔루션 산업 적용**: 모바일·웹 환경에서 매팅 기능 내장 시 서버 부담 절감  
- **추가 연구 고려점**  
  - 동영상 연속성 유지 및 실시간 처리 최적화  
  - 멀티모달(깊이, 뎁스 등) 입력 결합으로 디테일 및 경계 강화  
  - 더 작은 사전학습 데이터 활용을 위한 효율적 프롬프트·어댑터 설계  

ViTMatte는 *비계층 ViT의 잠재력*을 이미지 매팅에 처음으로 입증한 모델로, **파운데이션 모델 기반 경량 적응(adaptation) 패러다임**을 향후 매팅 및 기타 픽셀 수준 비전 과제의 표준으로 자리매김할 전망이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/e93ea0c8-b1fd-4790-aace-1c8e7638a622/2305.15272v2.pdf
