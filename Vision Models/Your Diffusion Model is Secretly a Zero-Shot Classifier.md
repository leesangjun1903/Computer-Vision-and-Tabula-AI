# Your Diffusion Model is Secretly a Zero-Shot Classifier

### 핵심 접근법  
기존 텍스트-이미지 확산 모델(예: Stable Diffusion)이 이미지 생성 외에 **조건부 확률 밀도 추정** 을 통해 제로샷 분류가 가능함을 입증했습니다[1][3]. 이 방법은 별도 학습 없이 사전 훈련된 모델의 density estimate를 활용해 클래스 간 상대적 가능도를 비교하며, 이를 **Diffusion Classifier** 로 명명했습니다[4][7].

### 주요 강점  
1. **다중모달 추론 능력**:  
   텍스트와 이미지 간 구성적 관계 이해에서 CLIP 등의 판별적 모델을 능가합니다(예: "빨간색 사과 vs 녹색 사과" 구분)[5][7].  
2. **벤치마크 성능**:  
   CIFAR-10(77.9%), Flowers(86.2%), ImageNet(58.9%) 등에서 기존 확산 모델 기반 분류기 대비 우수한 성적[2][5].  
3. **효율적 강건성**:  
   이미지넷 분류기 추출 시 약한 데이터 증강만으로도 분포 변화에 강인한 특성 보임[3][4].  

### 적용 사례  
- **이미지넷 분류기 변환**: 클래스 조건부 확산 모델을 전통적인 분류기로 변환 가능[4][7]  
- **합성 데이터 활용**: 확산 모델이 생성한 합성 데이터로 분류기 훈련 시 성능 향상[2][6]  

### 한계 및 비교  
- **계산 비용**: 실시간 추론에는 여전히 고비용(이미지당 1-2분 소요)[6]  
- **CLIP 대비 성능 격차**: 일부 벤치마크에서 OpenCLIP ViT-H/14 대비 10-15%p 낮은 정확도[2][5]  

이 연구는 생성 모델이 판별 작업에서도 유용함을 입증하며, 향후 다중모달 AI 시스템 개발에 새로운 방향성을 제시했습니다[1][7]. 특히 데이터 증강 없이도 분포 변화에 강인한 분류 가능성은 실제 응용 분야에서 주목할 만한 결과입니다[3][4].

[1] https://arxiv.org/abs/2303.16203
[2] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[3] https://openreview.net/forum?id=Ck3yXRdQXD
[4] https://github.com/diffusion-classifier/diffusion-classifier
[5] https://paperswithcode.com/paper/your-diffusion-model-is-secretly-a-zero-shot
[6] https://www.jetir.org/papers/JETIR2411561.pdf
[7] https://huggingface.co/papers/2303.16203
[8] https://papers.nips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[9] https://www.computer.org/csdl/proceedings-article/iccv/2023/071800c206/1TJjVcPg24g
[10] https://www.youtube.com/watch?v=t5Daou0eT-g
