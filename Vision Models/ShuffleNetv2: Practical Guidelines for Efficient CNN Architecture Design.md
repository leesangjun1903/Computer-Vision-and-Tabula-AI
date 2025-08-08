# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design | Image classification, Object detection

## 1. 핵심 주장 및 주요 기여
이 논문은 **FLOPs(간접 지표)가 아닌 실제 속도(직접 지표)에 기반한 네트워크 설계**의 중요성을 강조하고, 이를 뒷받침하는 네 가지 실용적 지침(G1–G4)을 제안한다.  
주요 기여는 다음과 같다.  
- 플랫폼 특성(메모리 접근 비용·병렬성 등)을 고려한 **직접 속도 평가**를 네트워크 설계에 도입  
- 균형 채널 폭(Equal channel width), 과도한 그룹합성곱 억제, 연산 단편화 최소화, 불필요한 요소연산 감소 등의 **실용적 가이드라인 제시**  
- 가이드라인을 준수한 **새로운 경량 아키텍처 ShuffleNet V2** 설계 및 ImageNet 분류와 COCO 검출에서의 속도·정확도 우위 실증  

## 2. 문제 정의·제안 기법·모델 구조·성능 및 한계

### 2.1 해결 과제  
경량 CNN 설계 시 FLOPs만을 최적화하면 메모리 접근 비용(MAC), 라이브러리 최적화, 연산 병렬성 등을 반영하지 못해 실제 추론 속도와 차이가 발생한다.  

### 2.2 제안 방법  
논문은 다음 네 가지 지침을 이론 및 실험으로 도출한다.  
- G1. 입력 채널 수 c₁와 출력 채널 수 c₂를 같게 유지하여 MAC 최소화  
- G2. 그룹합성곱 수 g 과도 증가 시 MAC 증가하므로 적절한 g 선택  
- G3. 건물 블록 내 연산 단편화(Fragmentation) 억제하여 GPU 병렬성 극대화  
- G4. ReLU·Add 등 요소 연산 제거로 불필요한 메모리 접근 축소  

이 지침을 바탕으로 ShuffleNet V2의 기본 유닛은 채널 분할(Channel Split)→비그룹(pointwise) 1×1 합성곱 2회→Depthwise 3×3 합성곱→채널 셔플(Channel Shuffle)을 수행하며, 스트라이드 2 유닛은 병합(branch) 없이 채널 수를 두 배로 확장한다.

### 2.3 모델 구조  
논문 Table 5와 Appendix Table 2 참조: 입력 해상도 224×224 기준으로 0.5×, 1×, 1.5×, 2× 복합도 모델 설계  
- Conv1: 3×3, 24 채널  
- Stage 2–4: 기본 블록 반복(각 블록 채널 수는 복합도에 비례)  
- Conv5: 1×1, 1024–2048 채널  
- 최종 global pooling → FC(1000)  

### 2.4 성능 향상  
- ImageNet 분류: 동일 FLOPs 대비 MobileNet V2·ShuffleNet V1보다 Top-1 오류율 3–4%p 개선, GPU/ARM 모두에서 실제 속도 20–60% 향상  
- COCO 검출(mmAP): 경량 검출기 Light-Head R-CNN 백본으로 적용 시 모든 복합도에서 최상위 성능 달성  
- 대형 모델(2–12 GFLOPs): ResNet-50, ShuffleNet V1 대비 정확도 및 효율 우위  

### 2.5 한계  
- 자동화된 아키텍처 탐색(NAS) 모델에 비해 종단간 검색은 수행하지 않음  
- 특정 하드웨어(CUDNN·Neon 최적화) 환경에 의존하므로 일반화 필요  
- 매우 깊은 네트워크(100+ 레이어)에서는 보조적 잔차 경로(residual) 추가 요구  

## 3. 일반화 성능 및 향후 연구 고려 사항
ShuffleNet V2는 분류, 검출, 경량/대형 모델 모두에서 **플랫폼별 직접 속도와 정확도를 동시에 개선**하며, 채널 분할 및 셔플 전략으로 자연스러운 특성 재사용(feature reuse)을 구현했다.  

향후 연구에서는  
- **자동화된 NAS 기법**에 G1–G4 지침을 통합하여 속도 최적화된 아키텍처 탐색  
- 다양한 **하드웨어·라이브러리**(TPU, ONNX Runtime 등)에서 지침의 범용성 검증  
- **리셉티브 필드 확장**(ShuffleNet V2*)와 같이 고해상도 검출 성능 강화를 위한 모듈 탐색  
- **잔차 경로·SE 모듈** 조합을 통한 깊은 네트워크 안정화 및 추가적 정확도 개선  
를 고려할 필요가 있다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/eb465163-3887-405b-9955-373f4062ea98/1807.11164v1.pdf
