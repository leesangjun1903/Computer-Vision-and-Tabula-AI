# VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training | Video reconstruction

## 1. 핵심 주장 및 주요 기여 요약  
**핵심 주장**  
VideoMAE는 영상 데이터의 높은 시간적 중복성과 상관성을 활용해, 90–95%의 매우 높은 마스킹 비율과 튜브 마스킹(tube masking) 전략을 적용함으로써, 적은 데이터만으로도 효과적인 자기지도 영상 프리트레이닝이 가능함을 보인다.

**주요 기여**  
- **극히 높은 마스킹 비율(90–95%)** 적용으로 연산량 대폭 절감 및 재구성 난이도 제고  
- **튜브 마스킹**: 모든 프레임에 동일한 마스크를 적용해 인접 프레임에서의 정보 누수를 방지  
- **Vanilla ViT 백본 활용**: 추가 inductive bias 없이 순수 Vision Transformer만으로 영상 표현 학습  
- **데이터 효율성**: UCF101/HMDB51처럼 3–10K 규모 소규모 데이터셋만으로도 62–91% 정확도 달성  
- **성능**: Something-Something V2에서 75.4% Top-1, Kinetics-400에서 87.4% Top-1  

## 2. 문제 정의·제안 기법·구조·성과·한계 상세 설명  

### 2.1 해결하고자 하는 문제  
- 영상(Self-Supervised Video Pre-Training)은 이미지보다 작은 데이터셋에서 작동이 어렵고, 기존 MAE(이미지) 방식을 그대로 적용 시 시간적 중복(redundancy)과 상관성(correlation) 때문에 쉬운 “지름길” 복원이 발생  
- 고해상도·다수 프레임 입력 시 연산량 급증  

### 2.2 제안하는 방법  
1) **튜브 마스킹(tube masking)**  
   - 모든 시간축에 동일한 마스크 패턴 적용  
   - 정보 누수 차단  
2) **초고마스킹 비율(ρ ≈ 0.90–0.95)**  
   - 학습 입력의 90% 이상 토큰 제거  
   - 복원 난이도 강화 및 연산량 △10% 수준으로 절감  
3) **비대칭 인코더-디코더 구조**  
   - Encoder에만 visible tokens 투입  
   - Decoder는 shallow하게 설계  
4) **손실 함수**  
   - MSE loss:  

$$ L = \frac{1}{|\Omega|} \sum_{p \in \Omega} \|I(p) - \hat I(p)\|^2 $$  

### 2.3 모델 구조  
- 입력: 2×16×16 크기 큐브 임베딩 → T/2×H/16×W/16 토큰  
- Tube Mask → Encoder(ViT-B, joint space-time attention, 12 blocks)  
- Decoder: 4 Transformer 블록, hidden dim=384  
- 출력: masked cube 픽셀 직접 재구성  

### 2.4 성능 향상  
- **Something-Something V2**: from-scratch(32.6%) → VideoMAE(69.6%)  
- **Kinetics-400**: from-scratch(68.8%) → VideoMAE(80.0%)  
- **소규모 데이터**: UCF101 91.3%, HMDB51 62.6%  
- **전이 학습**: Kinetics→SSV2 전이에서 VideoMAE 68.5% vs. MoCo-v3 62.4%  

### 2.5 한계  
- **도메인 쉬프트**: 사전학습과 타깃 데이터 특성 불일치 시 성능 저하  
- **소형 물체·미세 모션 복원**: 매우 높은 마스킹 비율로 인한 중요 토큰 소실  
- **대규모 모델 확장 비용**: ViT-Huge급 확장 시 연산량·메모리 부담  

## 3. 일반화 성능 향상 관점  
- Tube masking으로 시간축 정보 누수 제거 → 고수준 시공간 패턴 학습  
- 고마스킹 전략: 실제 복원 난이도 조절 → representation에 일반적 피처 압축 유도  
- 소규모 데이터셋에서도 강건: overfitting 억제 및 전이 학습 기반 일반화력 강화  

## 4. 향후 연구 영향 및 고려 사항  
- **영향**:  
  - 영상 자기지도 학습에서 마스킹 전략의 중요성 재조명  
  - 데이터 효율적 프리트레이닝 방향 제시  
- **고려 사항**:  
  - 도메인 적응 기법 통합을 통한 쉬프트 완화  
  - 마스킹 비율·패턴의 적응적 조정 연구  
  - 오디오·텍스트 멀티모달 확장으로 표현력 확장  
  - 소형 객체·미세 모션 복원 보완을 위한 보조 과제 설계

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7d99a3f9-5cc8-4665-821b-9f76b91e2720/2203.12602v3.pdf)
