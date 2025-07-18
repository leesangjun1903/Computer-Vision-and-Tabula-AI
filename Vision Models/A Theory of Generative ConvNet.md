# A Theory of Generative ConvNet | Generation, EBM

**“A Theory of Generative ConvNet”**는 기존의 **판별형(Discriminative) ConvNet**을 **생성형(Generative) 모델**로 전환하고, 그 내부에 **오토인코더(자동부호화기)** 구조를 드러냄으로써 에너지 기반 모델 중 유일무이한 **조각별 가우시안(piecewise Gaussian)** 특성을 확보함을 보인다.  
1. **판별형 ⇄ 생성형 상호 유도**  
   - 다중 카테고리 분류용 ConvNet에 “기준 분포(base category)”로서 가우시안 화이트 노이즈를 도입하고, 소프트맥스 점수(fc(I;w))를 기준 분포에 지수부등(exponential tilting) 하면 분류 모델과 동형(동일한 파라미터로 상호 유도 가능)임을 증명.  
2. **조각별 가우시안 성질**  
   - ReLU 활성화와 가우시안 기준 분포 조합 시, 입력 공간이 ReLU 하이퍼플레인으로 2ᴷ 영역으로 분할되고, 각 조각별 확률밀도는 평균이 오토인코더 재구성 출력을 가지는 가우시안임을 보임.  
3. **내부 오토인코더 구조**  
   - **하향(Top-down) 디컨볼루션**의 기저함수로 상향(Bottom-up) 합성곱 필터를 재사용하고, ReLU 이진 활성화 변수(1(·)>0)가 기저 함수 계수로 작용하는 계층적 오토인코더를 규명.  
4. **Langevin 샘플링과 학습**  
   - Langevin dynamics가 오토인코더 **재구성 오차**로 구동되고, 대조적 발산(Contrastive Divergence)은 실제 이미지 복원 학습으로 해석됨.

# 해결 문제, 제안 방법, 모델 구조, 식별식 및 한계

## 1. 해결하고자 하는 문제  
- **제한된 레이블** 또는 **레이블 없는 데이터** 상황에서의 **생성 모델 학습**.  
- 판별용 ConvNet의 성공을 **생성·비지도 학습**까지 확장하려는 일관된 이론적 프레임워크 부재.

## 2. 제안 방법 및 수식  
- **분류용 점수 함수**  
  
$$
    f_c(I;w)\;=\;\sum_{k=1}^{N_L}w_{c,k}\,[F^{(L)}_k*I]\;+\;b_c
$$  

- **분류 모델↔생성 모델**  
  - 분류:  
    
$$
      p(c\mid I;w)=\frac{e^{f_c(I;w)}}{\sum_{c'}e^{f_{c'}(I;w)}}
$$  
 
- 생성:  
    
$$
      p(I\mid c;w)=\frac{1}{Z_c(w)}\,\exp\bigl[f_c(I;w)\bigr]\,q(I),
      \quad q(I)\propto e^{-\frac{1}{2}\|I\|^2}
$$  

- **조각별 선형화(piecewise linear)**  
  - ReLU $$h(r)=\max(0,r)$$ 도입으로  
    
$$
      f(I;w)=\alpha_{w,\delta}+\langle I,B_{w,\delta}\rangle
$$  
    
각 활성화 패턴 $$\delta$$에 따라 영역 분할.  
- **오토인코더 평균**  
  
$$
    B_{w,\delta}\;=\;\text{top-down deconv}\bigl(\delta;\,w\bigr)
  $$  

- **Langevin Dynamics**  

$$
    I_{t+1} = I_t - \tfrac{\epsilon^2}{2}\bigl(I_t - B_{w,\delta(I_t)}\bigr)
              + \epsilon\,\mathcal{N}(0,I)
$$  

- **대조적 발산과 복원 학습**  

$$
    \nabla_w\log p(I;w)
    \approx \tfrac1M\sum_{m}\nabla_w f(I_m;w)
    -\tfrac1{\tilde M}\sum_{\tilde I}\nabla_w f(\tilde I;w)
    \;\propto\;\nabla_w\|I - B_{w,\delta}\|^2
$$

## 3. 모델 구조  
- **Prototype**: 단층 $$h(\langle I,w_k\rangle+b_k)$$ → 조각별 가우시안.  
- **Convolutional**: 다층 필터 $$F^{(l)}_{k,x}$$과 ReLU $$h$$ → 계층적 오토인코더.  
- **Top-down Deconv**:  

$$
    B^{(l-1)}
    =\sum_{k,x}B^{(l)}\_{k}(x)\,\delta^{(l)}\_{k,x}\,w^{(l)}_{k,x}
$$

## 4. 성능 향상 및 한계  
- **장점**  
  - 소량의 데이터(단일 이미지)만으로 **질감(texture)**과 **정렬된 객체(object patterns)** 합성에 성공[1].  
  - 판별 모델 파라미터 재활용으로 **통합적 프레임워크** 제공.  
  - 오토인코더 기반 **직관적 샘플링** 및 **효과적 복원**.  
- **한계**  
  - **고해상도**·**대규모** 자연 이미지에는 **샘플링 비용** 과 Langevin 수렴 문제.  
  - **복잡도** 높은 네트워크로 확장 시, 활성화 조각 수 급증(piecewise regions exponential).  
  - **실시간 응용**에는 부적합한 느린 MCMC 기반 샘플링.

# 일반화 성능 향상 가능성

- **오토인코더 구조**가 사전 훈련된 판별 ConvNet 필터를 **비지도 방식으로 정교화** 가능.  
- **Contrastive Divergence**를 이용한 효율적 학습으로 **데이터 부족** 환경에서 오버피팅 완화.  
- **계층적 활성화 패턴** 분석으로 **전이 학습(Transfer Learning)** 에도 응용 여지: 상위 계층의 활성화 패턴만 동결하고 하위 계층 재학습.  
- **스파스 활성화 구현**으로 모델 규모와 샘플링 복잡도 최적화 가능.

# 향후 연구 영향 및 고려 사항

- **에너지 기반·오토인코더 통합** 연구 가속: 다양한 비선형성·기준 분포 조합 탐색  
- **고해상도 샘플링**을 위한 대체 MCMC 기법(예: Stochastic Gradient Langevin Dynamics) 도입  
- **모델 해석성**: 활성화 패턴별 조각 분포 연구로 **내부 표현 이해** 심화  
- **비지도 사전 학습**과 **판별 모델 파인튜닝** 연계로 **전이 학습** 최적화  

>Generative ConvNet은 판별·생성·비지도 학습의 통합적 이론을 제시함으로써, 이후 딥러닝 모델 설계에 있어 **에너지 기반 프레임워크**와 **오토인코더**의 결합 연구에 중대한 이정표를 마련했다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/6a716752-4505-4559-84b9-ee11fc694ebc/1602.03264v3.pdf
