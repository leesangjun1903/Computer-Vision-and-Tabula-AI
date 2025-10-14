# TransTab: Learning Transferable Tabular Transformers Across Tables

## 핵심 주장 및 주요 기여 요약  
**TransTab**은 테이블 구조가 다른 이질적(tabular) 데이터 간 지식을 전이하고 학습하는 데 초점을 맞춘 새로운 **가변 컬럼(tabular) 입력 처리 프로토콜**과 **대각 대비 학습(vertical-partition contrastive learning, VPCL)**을 도입한 모델이다.  
- **고정된 테이블 구조 제약**을 완화하여, 부분적으로만 컬럼이 겹치는 다수의 테이블에서 **사전학습(pretraining)**과 **미세조정(fine-tuning)**을 수행 가능  
- 컬럼 설명과 셀 데이터를 결합한 **시퀀스 기반 입력 프로세서**와 **게이트형 탭룰러 트랜스포머** 아키텍처 설계  
- **Supervised·Self-supervised VPCL**을 통해 사전학습을 수행하여, 평균 **AUC 2.3포인트 상승**을 달성[1]

## 문제 정의 및 제안 방법 상세  
TransTab이 해결하고자 하는 핵심 문제는 **훈련 시와 테스트 시의 테이블 컬럼 구조 불일치**에 따른 데이터 전처리 손실과 **다양한 테이블 간 지식 전이의 부재**이다.  
1. **문제**  
   - 서로 다른 스키마(schema)를 지닌 테이블 간 병합 전 대규모 컬럼·샘플 삭제 필요  
   - 신규 컬럼 등장, 서로 다른 도메인 테이블에서 예측 모델을 즉시 활용 불가  
2. **제안 방법**  
   - **입력 프로세서(Input Processor)**  
     - **Categorical/Textual**: “column name + cell value”를 토큰 시퀀스로 변환하여 임베딩  
     - **Binary**: 값 = 1일 때에만 토큰화하여 임베딩  
     - **Numerical**: 컬럼 임베딩과 값의 곱으로 수치 임베딩 생성  
   - **게이트형 탭룰러 트랜스포머(Gated Transformer)**  
     - 각 레이어별로  

$$Z^{(l)}_{\mathrm{att}} = \mathrm{MultiHeadAttn}(Z^{(l)})$$  

$$g^{(l)} = \sigma(Z^{(l)}_{\mathrm{att}} W_G)$$  

$$Z^{(l+1)} = \mathrm{Linear}(g^{(l)} \odot Z^{(l)}_{\mathrm{att}})$$  
     
  - 최종 **CLS 임베딩** $$z_{\mathrm{cls}}$$를 분류기(classifier) 및 프로젝터(projector)에 입력[1]
   - **VPCL(Vertical-Partition Contrastive Learning)**  
     - **Self-supervised**: 동일 샘플의 컬럼 파티션 간 양의 쌍, 타 샘플 파티션 간 음의 쌍으로 대비 학습  
       $$\mathcal{L}_{\mathrm{SSL}}=-\sum \log\frac{\exp(\cos(v^k_i,v^{k'}_i))}{\sum_j\exp(\cos(v^k_i,v^{k'}_j))}$$  
     - **Supervised**: 동일 클래스 샘플 간 파티션을 양의 쌍으로 구성하여, 하이퍼파라미터 민감도 완화 및 일반화 성능 제고[1]

## 성능 향상 및 한계  
TransTab은 **임상 시험 사망률 예측** 및 다수의 공개 테이블 데이터셋에서 다음과 같은 성능을 입증하였다:[1]
- **Supervised**: 모든 데이터셋에서 AUROC 1위  
- **Incremental learning**: 신규 컬럼 추가 시에도 지속적 학습으로 성능 유지·개선  
- **Transfer learning**: 부분 중첩 컬럼 테이블 간 사전학습→미세조정으로 성능 향상  
- **Zero-shot inference**: 전혀 겹치지 않는 컬럼의 신규 테이블에서도 감독학습 단일 테이블 모델 대비 높은 성능  
- **Pretraining**: VPCL 적용 시 평균 AUC +2.3 개선  

**한계**  
- 완전히 무관한 테이블 간 사전학습은 제한적 이득  
- 사전학습 데이터의 *테이블 표현(phenotypes)* 유사성 고려 필요[1]

## 모델의 일반화 성능 향상 가능성  
- **컬럼·셀 의미론적 컨텍스트화**: “gender is male” 형태의 시퀀스로 변환하여, 서로 다른 컬럼 네이밍 간 의미 연결  
- **VPCL**: 데이터셋별 클래스·샘플 분포 차이에 둔감한 대비 학습으로, **하이퍼파라미터 안정성** 및 **다양한 테이블에 대한 일반화**  
- **Zero-shot**: 완전 비중첩 컬럼 신규 테이블 예측에서도 기존 학습 지식 활용 가능[1]

## 향후 연구 영향 및 고려 사항  
TransTab은 **테이블 기반 파운데이션 모델**의 기초를 마련하며, 향후 다음을 고려해야 한다.  
- **테이블 표현량(phenotype) 분석**: 사전학습에 최적화된 테이블 그룹 자동 군집화  
- **대규모 공개 테이블 코퍼스 구축**: 범도메인 사전학습 자원 확보  
- **다양한 대비 학습 기법 융합**: 수평·수직 파티션 외 고급 샘플링 전략 도입  
- **실시간·온라인 학습**: 신규 컬럼·샘플 등장 시 지연 없는 모델 업데이트  

TransTab은 가변 스키마 테이블 학습의 새로운 패러다임을 제시하며, **다양한 산업·의료·금융 도메인**에서 테이블 간 지식 전이를 통한 모델 효율성·일반화를 획기적으로 향상시킬 전망이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/24fb8c9d-bc73-4002-b632-278dcca7d817/NeurIPS-2022-transtab-learning-transferable-tabular-transformers-across-tables-Paper-Conference.pdf)
