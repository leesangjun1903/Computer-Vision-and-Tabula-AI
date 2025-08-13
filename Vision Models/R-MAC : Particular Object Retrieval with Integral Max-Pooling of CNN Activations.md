# R-MAC : Particular Object Retrieval with Integral Max-Pooling of CNN Activations | Image retrieval

## 핵심 주장 및 주요 기여
“Particular Object Retrieval with Integral Max-Pooling of CNN Activations” 논문은 convolutional layer의 활성값을 활용해 객체 검색(initial retrieval)과 기하학적 재순위(geometric re-ranking)를 통합하는 새로운 파이프라인을 제안한다.  
- CNN의 마지막 convolutional layer 활성값에 대한 **MAC(Maximum Activations of Convolutions)** 및 이를 확장한 **R-MAC(Regional MAC)** 를 도입하여, 이미지 전역뿐 아니라 다중 스케일의 관심 영역(region)을 단일 벡터로 압축  
- 일반적인 sum-pooling 기반 integral image 기법을 확장해 **integral max-pooling** 을 구현, 후보 영역의 최대 응답을 효율적으로 근사  
- 이 근사(max-pooling via generalized mean)를 활용한 **AML(Approximate Max-pooling Localization)** 을 제안해, 탐색 영역을 제한 및 좌표 미세조정을 통해 객체 위치를 빠르고 정확하게 탐지  
- R-MAC + AML + QE(query expansion) 조합으로 전통적 로컬 피처 기법과 경쟁하거나 능가하는 검색 성능 달성

## 해결하고자 하는 문제
CNN 기반 특징 추출 기법은 강력하나,  
1) 객체 재순위 단계에서 기하학적 정합(geometric verification)이 어렵고  
2) 여러 크기·위치의 객체 영역을 탐색하기 위해 네트워크를 여러 번 전방전파해야 하는 비효율성  
→ 이를 해결해 **필터링(filtering)과 재순위(re-ranking) 모두** 동일한 CNN 활성값만으로 처리하고, 단일 전방전파(forward pass)로 다중 영역 표현을 얻는 것이 목표

## 제안하는 방법
### 1. MAC & R-MAC
- 입력 이미지 I에 대해 마지막 convolution layer의 K개 채널 활성값 맵 Xi(p) (p ∈ Ω) 추출  
- MAC: 전역 최대 풀링  

  $$ f_{\Omega,i} = \max_{p\in\Omega} X_i(p) $$  

- R-MAC: L개의 스케일 l별로 정사각형 영역 R_l 샘플링(40% 중첩), 각 영역에 대해 MAC 계산 후 ℓ₂ 정규화·PCA-whitening·ℓ₂ 정규화, 최종적으로 영역 벡터 합산 후 ℓ₂ 정규화

### 2. Integral Max-Pooling (Generalized Mean)
- max-pooling 근사를 위해 generalized mean 이용  

  $$\tilde f_{R,i} = \Bigl(\sum_{p\in R} X_i(p)^\alpha\Bigr)^{1/\alpha} \approx \max_{p\in R}X_i(p),\quad \alpha>1 $$  

- 각 채널별로 $X_i(p)^α$ 값에 대해 integral image 구성 → 4회 덧셈으로 근사 max-pooling 계산  
- α=10 사용 시, 영역수·크기 변화에도 높은 유사도 유지

### 3. AML(Approximate Max-pooling Localization)
- 전수 탐색 $O(W²H²)$ 대신, 영역 샘플링(step t) 및 종횡비 제한(s) 후, 좌표 하강법으로 상위 후보만 미세 조정  
- 최종적으로 query MAC 벡터 q와 각 영역 $$\tilde f_R$$ 유사도 최대화  

  $$\hat R = \arg\max_{R\subset\Omega}\frac{\tilde f_R^\top q}{\|\tilde f_R\|\|q\|} $$

### 4. 재순위 및 Query Expansion
- 필터링: MAC 또는 R-MAC 유사도 순으로 초기 랭킹  
- 재순위: 상위 N개에 AML 적용하여 영역별 유사도 재계산 후 재정렬  
- QE: 상위 5개 이미지의 벡터와 원본 q 평균을 구해 한 번 더 재정렬

## 모델 구조
VGG16 / AlexNet의 마지막 pooling layer 출력(512 또는 256 채널)을 특징으로 활용하며, 추가 학습(fine-tuning) 없이 사전 학습된 CNN만 사용한다.

## 성능 향상
- Oxford5k에서 R-MAC (512D)만으로 mAP 66.9%, AML+QE 적용 시 77.3% 달성  
- Paris6k에서는 R-MAC+AML+QE로 86.5% 기록, 기존 로컬 피처 기법과 동등 또는 상회  
- 1M 이미지 대규모 검색에서도 AML 재순위 시 mAP 약 13%p 개선  
- R-MAC 대비 MAC 단일 전역 풀링 시보다 10%p 이상의 대폭 향상

## 한계 및 고려점
- 근사 max-pooling(α-root 등)과 좌표 미세조정 과정이 여전히 상당한 계산 비용(리랭킹 1000건당 ≈2.9s)  
- AML의 거친 탐색은 정밀한 경계 추출에는 한계, 세밀한 localization이 필요한 응용에 부적합  
- CNN 특성 그대로 사용하므로, 도메인 간 데이터 분포 차이 시 일반화 성능 저하 위험

## 일반화 성능 향상 가능성
- R-MAC이 여러 스케일 영역을 통합함으로써 회전·스케일 변형에 강건하며, 다양한 객체 크기·위치에 대응  
- integral max-pooling은 활성값 통계 기반이므로 새로운 도메인에서도 추가 학습 없이 초매칭 가능  
- 그러나 PCA-whitening, query expansion 등 통계적 후처리 단계는 학습·검증 데이터에 민감하므로, 현장 도메인으로 확장 시 별도 재학습 또는 적응화가 필요

## 향후 연구 영향 및 고려 사항
- **경량화된 리랭킹 기법** 개발: AML 계산량 감소 및 실시간 검색 지원  
- **End-to-end 학습** 기반 R-MAC 파인튜닝: 손실 함수에 localization 성능 통합  
- **도메인 적응**: whitening, 영역 샘플링 전략을 도메인별 자동 최적화  
- **다중 모달 확장**: 텍스트·오디오·3D 포인트 클라우드 등 다양한 정보 융합 검색으로 일반화  
- **정밀 localization**: AML과 고정밀 분할(segment-level) 기법 결합으로 경계 상향  

이 논문은 CNN 활성값을 활용한 이미지 검색 파이프라인의 **효율성과 정확성**을 크게 개선하며, 이후 통합적 특징 추출·정합 기법의 연구에 중요한 전기를 마련했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d2422fc1-fd42-41f7-a2f4-cd0135831d5a/1511.05879v2.pdf
