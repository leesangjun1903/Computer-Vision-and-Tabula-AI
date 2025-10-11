# SimFL : Practical Federated Gradient Boosting Decision Trees

**핵심 주장 및 기여**  
“Practical Federated Gradient Boosting Decision Trees”(SimFL)는 수평적 연합 학습 환경에서 Gradient Boosting Decision Trees(GBDT)의 효율성과 정확도를 동시에 확보하기 위해 제안된 프레임워크이다.  
- 기밀성 보호를 위해 원시 데이터를 공유하지 않고, Locality-Sensitive Hashing(LSH)을 활용해 각 인스턴스의 유사도 정보만 교환  
- 유사도 기반으로 가중치화된 그래디언트(WGB)를 계산해 트리를 학습함으로써 효율적이면서도 높은 정확도 달성  
- 제안 방식은 비밀 공유나 동형 암호화 방식 대비 계산·통신 오버헤드가 낮고, 기존 Differential Privacy 설계 대비 모델 정확도가 우수  

***

## 1. 해결하고자 하는 문제  
수평적 연합 학습에서는 동일한 특징을 가진 데이터가 여러 당사자에 분산되어 있지만, 개인정보 보호 정책으로 원시 데이터를 공유할 수 없다.  
기존 연구들은  
- 동형 암호화·비밀 공유 방식: 계산 비용이 매우 높음  
- Differential Privacy 방식: 현저히 낮은 예측 정확도  
문제점: 연합 학습으로 인한 **실질적 정확도 향상**과 **실행 효율**을 동시에 확보하기 어려움  

***

## 2. 제안 방법  
### 2.1 프라이버시 모델  
- **Relaxed Privacy Model**: honest-but-curious 당사자가 일부 정보는 획득할 수 있으나, 실제 원시 데이터를 유추할 수 없도록 보장.[1]

### 2.2 전체 구조  
1. **전처리 단계**  
   - 각 당사자가 p-stable LSH 함수 $$F_{a,b}(v)=\lfloor\frac{a\cdot v+b}{r}\rfloor$$로 인스턴스 해시 생성.[1]
   - AllReduce로 전역 해시 테이블 구축 및 방송 → 각 인스턴스별 유사 인스턴스 ID 매핑 행렬 $$S_m\in\mathbb{R}^{N_m\times M}$$ 획득  

2. **학습 단계**  
   - 각 트리 순차 학습 시, 로컬 인스턴스 $$x^m_q$$의 1차·2차 그래디언트 $$g^m_q,h^m_q$$ 대신, 유사 인스턴스 집합으로부터 집계된 가중 그래디언트  

$$
       G^m_q=\sum_{n\in W^m_q} g^n_i,\quad H^m_q=\sum_{n\in W^m_q} h^n_i
     $$
   
   - 이를 이용해 GBDT의 분할 이득 함수  

$$
       \mathcal{L}_{\text{split}}=\frac{1}{2}\Bigl(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Bigr)-\gamma
     $$
     
  로 트리 구조 결정.[1]

***

## 3. 모델 구조  
- **LSH 기반 유사도 수집**: $$L$$개의 해시 함수로 각 인스턴스 해시 벡터 생성  
- **전역 해시 테이블**: AllReduce로 해시→인스턴스 ID 매핑 테이블 구축  
- **Weighted Gradient Boosting (WGB)**: 유사 인스턴스 수에 비례한 집계 그래디언트로 트리 학습  

***

## 4. 성능 향상 및 한계  
### 4.1 정확도  
- **Solo vs. SimFL**: 개별 학습 대비 약 4% 평균 테스트 오류 감소  
- **All-in vs. SimFL**: 통합 데이터 학습에 근접하는 정확도 달성  
- **기존 TFL**: Differential Privacy 기반 TFL은 오히려 정확도 저하. SimFL은 안정적 향상 보여줌.[1]

### 4.2 효율성  
- 전처리: $$O(N d + N L)$$ 계산, 통신량 $$O(M N L)$$  
- 학습: Vanilla GBDT 대비 $$<10\%$$ 추가 계산, 통신량 트리당 수십 MB 이내로 경량  
- GPU 가속 프레임워크(ThunderGBM) 적용 시 대규모 데이터에서도 실용적  

### 4.3 한계  
- **Privacy-Utility Tradeoff**: Relaxed 모델이므로 강력한 추론 공격 위험 존재. 해시 수 $$L$$ 조정 필요  
- **근사 오차**: 분할 이득 집계 과정에서 오차 $$\tilde{\epsilon}=O(N_m/N)$$ 발생, 파티 수 증가 시 일반화 성능 영향 가능성.  

***

## 5. 일반화 성능 향상 관점  
- 유사 인스턴스 기반 가중치 부여로 **소수 파티 데이터**도 전체 분포를 일부 반영  
- 모델 복잡도 규제 항목 $$\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum w^2$$는 그대로 유지하여 **과적합 억제**  
- 이론적 상한: 깊이 $$D$$, 전체 샘플 $$N$$ 증가 시 일반화 오차 감소 경향 유지  

***

## 6. 향후 연구 영향 및 고려할 점  
- **프라이버시 강화**: LSH 대신 secure sketch나 DP-noise 결합 연구  
- **유사도 측정 개선**: 콘텐츠 기반 심층 표현 학습+LSH로 유사도 정확도 향상  
- **이질적 데이터 처리**: 수평뿐 아니라 수직·전이 연합 학습 확장성 평가  
- **대규모 확장**: 수백 파티, 스트리밍 데이터 환경에서 통신·계산 최적화  

SimFL은 **효율적 연합 GBDT** 구현을 위한 실용적 해법으로, 향후 의료·금융 등 분산 데이터 협력 학습에 큰 기여가 기대된다.  

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0495ed89-a409-4879-b420-a84efdebbbda/5895-Article-Text-9120-1-10-20200513.pdf)
