# FVAE : Factorized Variational Autoencoders for Modeling Audience Reactions to Movies | Facial Expression Recognition

## Factorized Variational Autoencoders (FVAE) 설명  

Factorized Variational Autoencoders (FVAE)는 영화 관객의 표정 반응을 모델링하기 위해 개발된 딥러닝 기법입니다. 이 방법은 기존의 텐서 분해(Tensor Factorization)와 변이형 오토인코더(Variational Autoencoder, VAE)의 한계를 해결하며, 관객 반응 데이터의 복잡한 패턴을 효과적으로 분석합니다[1][2].  

### 1. **FVAE의 핵심 아이디어**  
FVAE는 두 가지 기술을 결합합니다:  
- **텐서 분해**: 데이터를 개별 관객($$U_i$$)과 시간($$V_t$$) 차원으로 분해하여 구조화[1][2].  
- **변이형 오토인코더**: 비선형 관계를 학습하는 신경망으로, 복잡한 표정 데이터를 저차원의 "잠재 공간"(latent space)으로 압축[3][2].  
이 결합 덕분에 FVAE는 데이터의 복잡성을 줄이면서도 의미 있는 패턴(예: 웃음, 놀람)을 추출할 수 있습니다[3][4].  

### 2. **실험 방법 및 데이터**  
연구진은 400석 규모의 극장에 4대의 적외선 카메라를 설치해 9개 영화의 150회 상영 동안 관객 3,179명의 표정을 추적했습니다. 수집된 데이터는 다음과 같습니다:  
- **16만 개 이상의 얼굴 랜드마크**  
- **초당 1회 샘플링**된 시간 경과 데이터  
- 약 13%의 결측치 포함[1][2].  
이 데이터는 $$N \times T \times D$$ 텐서(관객 수 × 시간 × 얼굴 특징)로 정리되었습니다[2].  

### 3. **주요 성능 결과**  
FVAE는 기존 방법보다 우수한 성능을 보였습니다:  
- **데이터 복원 정확도**: 더 적은 잠재 차원으로 높은 재구성 정확도 달성[1][5].  
- **결측치 예측**: 전체 데이터의 13% 결측치를 정확히 보완[1][2].  
- **장기 예측 능력**: 영화 시작 5% 지점(약 3분)만으로 관객의 **전체 영화 표정을 예측** 가능[1][5][4].  
예를 들어, 초반 웃음 패턴을 분석하면 영화 후반의 유머 장면에서의 반응도 예측할 수 있습니다[4][2].  

### 4. **해석 가능성**  
FVAE가 학습한 잠재 변수는 의미 있는 개념과 연결됩니다:  
- 웃음(laughing), 미소(smiling) 등의 표정이 자동으로 식별됨.  
- 이러한 패턴은 영화 내 유머 장면과 높은 상관관계를 보임[4][2].  
이는 FVAE가 단순한 예측을 넘어 관객 반응의 "왜"를 설명할 수 있음을 의미합니다.  

### 5. **응용 분야**  
- **영화 제작**: 테스트 상영에서 관객 반응 분석을 통해 장면 수정 지원.  
- **인공지능**: 애니메이션 캐릭터의 표정 생성에 활용 가능[3][4].  
- **확장성**: 기상 데이터, 소셜 미디어 트렌드 등 시계열 그룹 데이터 분석으로 적용 가능[4].  

FVAE는 대규모 데이터에서 의미 있는 패턴을 추출하는 동시에 장기 예측이 가능한 프레임워크로, 관객 반응 분석뿐 아니라 다양한 분야에 적용될 잠재력을 가집니다[1][3][2].

[1] https://studios.disneyresearch.com/wp-content/uploads/2019/04/FactorizedVariationalAutoencodersfor-ModelingAudienceReactionstoMovie-1.pdf
[2] https://openaccess.thecvf.com/content_cvpr_2017/papers/Deng_Factorized_Variational_Autoencoders_CVPR_2017_paper.pdf
[3] https://www.caltech.edu/about/news/neural-networks-model-audience-reactions-movies-79098
[4] https://www.techexplorist.com/neural-nets-model-audience-reactions-movies/6653/
[5] https://www.cs.sfu.ca/~mori/research/papers/deng-cvpr17.pdf
[6] https://rbcborealis.com/research-blogs/tutorial-5-variational-auto-encoders/
[7] https://www.datacamp.com/tutorial/variational-autoencoders
[8] https://d3.harvard.edu/platform-rctom/submission/disney-a-whole-new-world-of-machine-learning/
[9] https://www.youtube.com/watch?v=Myz8UPECgdI
[10] https://la.disneyresearch.com/publication/factorized-variational-autoencoder/

## Reference
https://nicedeveloper.tistory.com/entry/Disney-FVAE-%EA%B4%80%EB%9E%8C%EA%B0%9D-%ED%91%9C%EC%A0%95-%EB%B6%84%EC%84%9D-AI
