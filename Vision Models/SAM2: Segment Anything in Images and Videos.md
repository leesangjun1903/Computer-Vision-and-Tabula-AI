# SAM 2: Segment Anything in Images and Videos 

## 핵심 주장 및 주요 기여  
Segment Anything Model 2 (SAM 2)는 이미지와 비디오 양쪽에서 **프롬프트(promptable) 시각 분할**을 실시간으로 처리하는 범용 파운데이션 모델을 제안한다[1].  
-  **통합 모델**: 이미지(단일 프레임)와 비디오(스트리밍 프레임)를 동일 아키텍처로 처리  
-  **스트리밍 메모리**: 과거 프레임의 특징과 예측을 메모리 뱅크에 저장해 크로스 어텐션으로 활용  
-  **데이터 엔진**: 인터랙티브 모델-인-더-루프(annotation loop)를 통해 수집한 SA-V(35.5M 마스크, 50.9K 비디오) 데이터셋 구축  
-  **성능**:  
  – 비디오 분할에서 **3× 적은 상호작용**으로 동일 수준 이상의 정확도 달성[1]  
  – 이미지 분할에서 기존 SAM 대비 **6× 빠른 추론 속도**, 정확도 향상[1]  
-  **제거 불가능 객체 없는(segmentation-anything)** 범용성: 17개 비디오, 37개 이미지 제로샷 벤치마크에서 최고 성능  

# 문제 정의  
Segment Anything Model 2 (SAM 2)는 이미지 기반의 **Promptable Visual Segmentation**(PVS) 과제를 비디오 영역으로 확장하여, 단일 프레임(이미지)뿐 아니라 연속 프레임(비디오)에서도 임의의 프롬프트(점·박스·마스크)를 입력하면 해당 객체의 시공간적 분할(마스크릿)을 실시간으로 예측하는 **범용 분할 파운데이션 모델**을 제안한다[1].  
- 기존 이미지 세그멘테이션 모델들은 정적인 이미지에만 대응했고, 비디오 객체 분할(VOS) 모델들은 첫 프레임의 마스크를 전제한 세미-슈퍼바이즈드 방식에 한정되어 있었다.  
- 비디오는 객체 모양·외관 변화, 빠른 움직임, 블러·저해상도 문제, 효율적 처리 요구 등 고유의 도전 과제를 지닌다.  
- **목표**: 점·박스·마스크 프롬프트에 대응하면서, 모델-인-더-루프(annotation loop)를 활용한 대규모 데이터와 **스트리밍 메모리** 아키텍처로 이미지·비디오를 단일 모델로 처리하고, 제로샷 분할 성능을 극대화.

# 제안 방법  
SAM 2는 크게 4가지 주요 모듈로 구성되며, 학습 시 이미지(SA-1B)와 비디오(SA-V, VOS 등)를 **교차 학습**한다[1].  

1. **Image Encoder**  
   - MAE 사전학습된 Hiera 비전 트랜스포머(다중 스케일)  
   - 스트라이드 {4, 8, 16, 32} 피처 추출  
   - 윈도우 기반 절대 위치 인코딩 사용

2. **Prompt Encoder &amp; Mask Decoder**  
   - 클릭·박스·마스크 프롬프트 → 위치·타입별 임베딩 토큰화  
   - “Two-way” 트랜스포머 블록:  
     -  **Token→Image Self-Attention**  
     -  **Image→Token Cross-Attention**  
     -  **Image→Image Cross-Attention** → 마스크 예측  
   - **Occlusion Head**: 객체 비가시 여부 예측  
   - **다중 마스크 출력**으로 모호성 처리 및 IoU 점수 기반 선택

3. **Streaming Memory Encoder &amp; Bank**  
   - 각 프레임 예측 마스크↓컨볼루션↓이미지 임베딩 합산 → 메모리 뱅크(N 최근 프레임)  
   - **메모리 어텐션**: 현 프레임 자가-어텐션 후 메모리 크로스-어텐션 ⇒ 시공간 문맥 활용  
   - **객체 포인터**: 각 프레임 마스크 토큰을 64차원 벡터로 변환→메모리 크로스 어텐션에 추가  
   - **시간 위치 인코딩**: 메모리(최근 N)만 삽입, 프롬프트 프레임은 미삽입

4. **학습 손실**  
   - 마스크 예측:  
     $$L_{\text{mask}} = 20\,L_{\text{focal}} + 1\,L_{\text{dice}}$$  
   - IoU 예측:  
     $$L_{\text{IoU}} = L_{\ell_1}$$  
   - Occlusion 분류: 크로스엔트로피  
   - 다중 마스크 시, 최고 IoU 마스크만 L_mask 감독

# 모델 구조 요약  

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                         Video Frames                            │
└───────────────────────┬─────────────────────────────────────────┘
                        ↓
              ┌────────────────────┐
              │    Image Encoder   │
              │ (MAE-pretrained    │
              │  Hiera ViT)        │
              └────────────────────┘
                        ↓ (256×H×W 이미지 임베딩)
┌─────────┐    ┌─────────────────────────┐    ┌────────────┐
│ Prompt  │──▶ │   Prompt Encoder +      │─-─▶│ Mask       │
│ (점/박스/ │    │    Memory Attention     │    │ Decoder    │
│ 마스크)   │    └─────────────────────────┘    └────────────┘
└─────────-┘                │
                            ↓
                   ┌────────────────┐
                   │ Memory Encoder │
                   └────────────────┘
                            ↓
                   ┌────────────────┐
                   │  Memory Bank   │
                   └────────────────┘
```

# 성능 향상  
| 평가 항목                       | SAM (비교군) 대비 개선치                  |
|---------------------------------|-------------------------------------------|
| Video Segmentation (제로샷 9개 벤치) | J & F 평균 +8.6 pt, 클릭당 상호작용 3×↓       |
| Image Segmentation (SA-23)      | 1-click mIoU: 58.9→61.9 (+3.0 pt), 6× 빠름   |
| Interactive VOS (SA-V val/test) | J & F: 61.3→76.8/+15.5 pt(Val), 62.8→77.0/+14.2 pt(Test) |

# 한계  
1. **장면급변 및 샷전환**: 갑작스런 카메라 뷰 변경 후 오브젝트 식별 실패  
2. **긴 occlusion**: 긴 가림 후 재발견 지연  
3. **유사 객체·군중**: 밀집된 비슷한 객체 혼동  
4. **미세·빠른 물체**: 가는 구조, 고속 이동 객체 추적 불안정  
5. **다중 객체 비효율**: 객체별 독립 처리로 연산 중복  

이러한 한계는 **추가 모션(광학흐름·3D) 정보**, **다중 객체 상호작용 모듈**, **자동 실패 감지·정정** 연구로 보완 가능하다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f1da976f-162e-49b7-91a9-cfa5f595c88b/2408.00714v2.pdf

## 성능 향상  
| Task                | 개선 내용                 | 수치                         |
|---------------------|--------------------------|------------------------------|
| Video Segmentation  | 클릭당 상호작용 횟수      | 기존 대비 3× ↓[1]            |
| Image Segmentation  | 추론 속도                | 6× 빠름[1]                   |
| Zero-Shot Video     | J & F 평균               | 기존 대비 +8.6pt (17개 데이터)[1] |
| Zero-Shot Image     | mIoU (1-click 평균)      | 58.9 → 61.9 (+3.0pt)[1]       |

## 한계 및 일반화 성능  
-  **장면 급변·긴급 차단**: 급격한 카메라 뷰 변경, 긴 피사체 가림(occlusion) 후 추적 실패[2]  
-  **군중·유사 객체**: 유사 물체 밀집 시 혼동 발생  
-  **미세 구조**: 빠르게 이동하는 가는 디테일 객체 추적 불안정  
-  **다중 객체 비효율**: 객체별 독립 처리로 효율 저하  

이러한 한계에도 불구하고, SA-V의 방대한 데이터와 스트리밍 메모리로 기존 VOS·SAM 조합 대비 뛰어난 **제로샷 일반화** 성능을 입증[1].

## 향후 연구에 미치는 영향 및 고려 사항  
-  **범용 분할 파운데이션**: 이미지·비디오 양쪽 작업을 단일 모델로 통합한 접근은 **분할 파이프라인 재설계**의 전환점  
-  **데이터 엔진**: 모델-인-더-루프(annotation loop) 활용한 대규모 비디오 데이터셋 구축은 **데이터 효율성**을 높이는 새로운 패러다임  
-  **일반화 개선**:  
  – 모션 정보를 명시적 모델링(광학 흐름·3D 정보) 통합 → occlusion·유사 객체 영역 성능 향상  
  – 다중 객체 간 상호작용 모듈 추가 → 효율 및 일관성 증대  
-  **윤리적·공정성 고려**:  
  – 인종·성별·연령별 성능 검증 및 잠재 편향 완화  
  – 사용자 개입 최소화 위한 **자동 실패 감지·정정 메커니즘** 연구  
-  **실시간 응용**: AR/VR, 로보틱스, 의료·원격 진단, 비디오 편집 툴킷 등에서의 실전 배포 가능성  

> **주요 시사점**: SAM 2는 **프롬프트 기반 인터랙티브 분할**을 이미지와 비디오 전반에 확장하여, 제로샷 분할의 새 지평을 열었다. 향후 **모션·다중 객체·자동화**를 핵심 과제로 삼아, 분할 파운데이션 모델의 **일반화 및 실시간 적용성**을 더욱 강화해야 한다.

[1] https://arxiv.org/abs/2408.00714
[2] https://ai.meta.com/blog/segment-anything-2/
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/f1da976f-162e-49b7-91a9-cfa5f595c88b/2408.00714v2.pdf
[4] https://www.sec.gov/Archives/edgar/data/1549346/000154934625000028/sstk-20250331.htm
[5] https://www.sec.gov/Archives/edgar/data/758743/000143774924031205/vide20240831_10q.htm
[6] https://www.sec.gov/Archives/edgar/data/949870/000095017025026756/sam-20241228.htm
[7] https://www.sec.gov/Archives/edgar/data/1826011/000095017025072501/bnzi-20250331.htm
[8] https://www.sec.gov/Archives/edgar/data/1762417/000141057825000965/doyu-20241231x20f.htm
[9] https://www.sec.gov/Archives/edgar/data/1982448/000164117225010956/form20-f.htm
[10] https://www.sec.gov/Archives/edgar/data/2054979/0002054979-25-000001-index.htm
[11] https://arxiv.org/abs/2408.03286
[12] https://arxiv.org/abs/2408.00874
[13] https://arxiv.org/abs/2408.07931
[14] https://arxiv.org/abs/2408.00756
[15] https://arxiv.org/abs/2408.06305
[16] https://arxiv.org/abs/2412.01240
[17] https://arxiv.org/abs/2408.08315
[18] https://www.semanticscholar.org/paper/b6bd1c1028cd2f24470aacac1b6d590c35ff51ef
[19] https://arxiv.org/abs/2408.15224
[20] https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
[21] https://pmc.ncbi.nlm.nih.gov/articles/PMC10287018/
[22] https://datature.io/blog/beyond-sam-2-exploring-derivatives-for-better-performance
[23] https://meetcody.ai/blog/meta-sam-2-the-future-of-ai-image-segmentation/
[24] https://arxiv.org/html/2408.00714v2
[25] https://arxiv.org/html/2408.04212v2
[26] https://arxiv.org/html/2410.04960v2
[27] https://velog.io/@bluein/paper-27
[28] https://bdtechtalks.com/2024/08/05/meta-sam-2-object-segmentation-model/
