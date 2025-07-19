# EMDiffuse : Diffusion-based deep learning method for augmenting ultrastructural imaging and volume electron microscopy | Image reconstruction

# 핵심 요약  
**EMDiffuse**는 확산 모델(diffusion model)을 기반으로 초고해상도 전자현미경(EM) 및 체적 전자현미경(vEM) 이미징을 크게 향상시키는 딥러닝 방법론이다[1].  
1. **노이즈 제거(EMDiffuse-n)**, **초분해(super-resolution, EMDiffuse-r)**, **비등방성→등방성 체적 재구성(vEMDiffuse-i/a)** 모듈을 한 패키지로 제공.  
2. **고해상도 세포 초미세 구조**(mitochondria cristae, ER 등)를 보존하면서 노이즈를 제거·복원.  
3. **단 1쌍(3메가픽셀)**의 fine-tuning 샘플만으로도 타 도메인(다른 조직·시편)으로 즉시 전이 가능.  
4. **등방성 체적(vEM) 재구성**: 등방성 학습 데이터 없이도 기존 비등방성 데이터를 3D로 보간하여 고해상도 등방성 체적으로 복원.  
5. **불확실도 평가(self-assessment)** 기능 탑재로 출력 신뢰도 정량화(불확실도 임계치 0.12)[2].  

# 1. 해결하고자 하는 문제  
- EM/vEM은 **이미징 속도↔화질** 간 필연적 트레이드오프로, 대형(㎜³) 등방성 체적 이미지를 얻기 어려움[1].  
- 기존 딥러닝 방식은 회귀 기반(regression-based)으로 저주파 정보에 치중해 **세부 구조가 과도하게 부드러워짐**.  
- CycleGAN 등은 기하학 변형에 취약, 불확실도 정량화 기능 부족.  

# 2. 제안하는 방법  
EMDiffuse는 **확산 기반 딥러닝**을 EM/vEM에 특화해 다음 4가지 태스크를 수행한다[1]:  
1) **EMDiffuse-n (Denoising)**  
   - 관측 이미지 $$c_r$$에서 노이즈 제거.  
   - 학습 손실:  

$$
     L
     = \frac{1}{N}\sum_{i}\Big\|\epsilon - \epsilon_\theta\big(\sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\,\epsilon,\,t,\,c_r\big)\Big\|^2_{2\,/\,\phi_i^2}
     + \log \phi_i
$$  
     
여기서 $$\phi_i$$는 “난이도 맵(difficulty map)”이며, 학습 도중 어려운 케이스 기여도를 가중치로 조정해 안정성↑.  
   - **추론**: $$K=2$$개의 샘플을 평균해 최종 출력, 분산을 불확실도로 사용.  
2) **EMDiffuse-r (Super-resolution)**  
   - 저해상도 $$c_n$$ → 고해상도 $$x_{gt}$$ 복원.  
   - 손실 함수는 denoising와 동일 형태로 low-res를 conditioning.  
3) **vEMDiffuse-i (Isotropic interpolation with isotropic training)**  
   - 소량의 등방성 체적 데이터(예: $$8\times8\times8$$ nm³)를 학습해, 비등방성 체적(예: $$8\times8\times48$$ nm³) 중간 층 복원.  
   - **채널 임베딩(channel embedding)**을 통해 복원할 층 인덱스 $$j\in[1,R]$$를 모델 입력으로 활용.  
4) **vEMDiffuse-a (Isotropic interpolation without isotropic training)**  
   - anisotropic 데이터만으로 XY축 정보를 XZ축으로 전이 학습.  
   - 실험에서 MICrONS, FANC 등 대규모 ANISO 체적에 적용해 voxel size를 $$z$$축으로까지 등방성 수준으로 강화.  

모델 구조는 **유니트(U-Net) 기반 UDiM(Ultrastructural Diffusion Model)** 아키텍처로, 다수 스케일의 컨볼루션 블록과 self-attention 모듈을 포함[1].  

# 3. 성능 향상 및 한계  
- **정성/정량 지표 모두 우수**: LPIPS↓, FSIM↑, 해상도 비율↑ [이미지 재구성 실험] [1].  
- **초분해**: 6.6 nm→3.3 nm로 해상도 2배↑, 촬영 속도 36× ↑ (specific setup) [1].  
- **불확실도 맵**: 99th percentile STD 사용, 임계치 0.12 이하일 때 **신뢰 출력**으로 분류[2].  
- **전이 학습(transfer learning)**: non-brain 조직(간·심장·골수) 및 HeLa 세포에서도 fine-tuning 한 쌍만으로 성능 회복, 직접 적용 대비 FSIM 20–30%↑[1].  
- **제한**:  
  - **노이즈 레벨 과도 시 복원 불가** 구간 존재(불확실도↑).  
  - **Axial resolution이 너무 낮으면**(e.g., 96 nm) 세부 구조 왜곡, 불확실도↑, FSIM↓ [vEMDiffuse-i].  
  - **전 이 임계치(0.12)**는 EM·vEM 전반에 걸쳐 보편적이지 않음.  

# 4. 향후 연구에 미치는 영향 및 고려사항  
- **Diffusion 모델의 EM 활용 확장**: 다른 생체현미경 기법(광학·조명학)에도 diffusion-based super-resolution, denoising 적용 가능성.  
- **대형 등방성 체적 재구성**: 수㎥ 단위 vEM 체적을 재구성하는 커넥토믹스 연구에 도입, 비용·시간 혁신.  
- **안정적인 확산 모델(Stable Diffusion)** 백본 도입 시 **추론 속도↑**, 대규모 데이터 처리 용이.  
- **불확실도 임계치 최적화**: 조직·시편 특성별로 threshold 재설정 및 시각화 워크플로우 구축 필요.  
- **3D 네트워크 연구**: 2D UDiM 대신 3D diffusion 네트워크 고찰로 **공간 연속성↑** 및 고해상도 복원력 강화.  
- **윤리·검증 프레임워크**: AI-생성 생물학 이미지 **검증 가이드라인** 마련 필요, 특히 논문·임상 적용 시.  

EMDiffuse는 고속·고해상도 vEM imaging의 기술적 제약을 해소하며, 대규모 3D nanoscale ultrastructure 분석을 가속화하는 **차세대 EM AI 툴킷**으로 자리매김할 전망이다.

[1] https://www.nature.com/articles/s41467-024-49125-z
[2] https://www.biorxiv.org/content/10.1101/2023.07.12.548636v1.full.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/7a786d11-c02b-40a2-916c-a7c777212ade/s41467-024-49125-z.pdf
[4] https://www.semanticscholar.org/paper/f963a88da7809888bfdf2939edc9fd90a952f517
[5] https://linkinghub.elsevier.com/retrieve/pii/S0302283823027318
[6] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11041/2522789/Automatic-paper-summary-generation-from-visual-and-textual-information/10.1117/12.2522789.full
[7] https://www.semanticscholar.org/paper/e6b2bf0d01c5ca8ce6bc621c0f89fe32d7cbcea8
[8] http://www.thieme-connect.de/DOI/DOI?10.1055/s-0042-1750385
[9] http://www.osti.gov/servlets/purl/1183728/
[10] https://linkinghub.elsevier.com/retrieve/pii/S0016510714013492
[11] http://link.springer.com/10.1007/s00464-014-3630-7
[12] https://www.sciencedirect.com/science/article/pii/S1361841523001809
[13] https://github.com/Luchixiang/EMDiffuse
[14] https://openaccess.thecvf.com/content/WACV2025/papers/Osuna-Vargas_Denoising_Diffusion_Models_for_High-Resolution_Microscopy_Image_Restoration_WACV_2025_paper.pdf
[15] https://academic.oup.com/jmicro/article/71/Supplement_1/i100/6530481
[16] https://www.biorxiv.org/content/10.1101/2023.07.12.548636v1
[17] https://arxiv.org/html/2409.16488v1
[18] https://www.nature.com/nature-index/article/10.1038/s41467-024-49125-z
[19] https://research-repository.uwa.edu.au/en/publications/diffusion-based-deep-learning-method-for-augmenting-ultrastructur
[20] https://www.sciencedirect.com/science/article/pii/S135964542300736X
[21] https://pmc.ncbi.nlm.nih.gov/articles/PMC7614724/
[22] https://www.everand.com/podcast/658798248/EMDiffuse-a-diffusion-based-deep-learning-method-augmenting-ultrastructural-imaging-and-volume-electron-microscopy
[23] https://www.embl.org/about/info/course-and-conference-office/wp-content/uploads/Larissa_EMBO_AI_in_bio_2024_compressed.pdf
[24] https://www.haibojianglab.com/emdiffuse
[25] https://ouci.dntb.gov.ua/en/works/42aXonX7/
[26] https://arxiv.org/abs/2304.01852
[27] https://www.semanticscholar.org/paper/4c0a31ff358e9269ae6529e4989a8c646e093397
[28] https://pmc.ncbi.nlm.nih.gov/articles/PMC11144272/
[29] https://arxiv.org/html/2407.01014v1
[30] https://arxiv.org/pdf/2410.21357.pdf
[31] https://arxiv.org/pdf/2211.01324.pdf
[32] https://arxiv.org/html/2405.16852v2
[33] https://www.mdpi.com/1424-8220/11/6/6297/pdf
[34] https://arxiv.org/pdf/2303.06555.pdf
[35] https://arxiv.org/html/2312.02256
[36] https://arxiv.org/abs/2102.12833
[37] https://arxiv.org/pdf/2308.01594.pdf
[38] https://sciety-labs.elifesciences.org/articles/article-recommendations/by?article_doi=10.1101%2F2023.07.12.548636
