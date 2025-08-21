# Towards Ghost-free Shadow Removal via Dual Hierarchical Aggregation Network and Shadow Matting GAN | Shadow removal, Shadow matting

## 1. 핵심 주장과 주요 기여
이 논문은 **그림자 제거 과정에서 발생하는 색 불일치(ghost) 문제**를 해결하기 위해 두 가지 핵심 기여를 제안한다.  
첫째, 경계 아티팩트 없이 자연스러운 그림자 제거를 달성하는 새로운 네트워크 구조인 **Dual Hierarchical Aggregation Network (DHAN)**을 설계하였다.  
둘째, 데이터셋의 장면 다양성 부족을 극복하기 위해 **Shadow Matting GAN (SMGAN)**을 이용하여 그림자 마팅(mask)으로부터 현실적인 합성 그림자 이미지를 생성, 데이터 증강을 수행하였다.

## 2. 문제 정의 및 제안 방법

### 2.1 해결하고자 하는 문제
- 기존 학습 기반 그림자 제거 모델은 경계 아티팩트(테두리 잔상)나 그림자 영역의 색 불연속(ghost)은 물론, 현실 장면에 대한 일반화(generalization)가 취약하다.
- 대규모 그림자–무그림자 쌍 데이터셋이 부족하여 네트워크가 다양한 장면과 그림자 유형을 학습하지 못함.

### 2.2 DHAN 모델 구조
- **Backbone**: 다운샘플링 없이 일련의 dilated convolution으로 멀티 컨텍스트를 보존하는 Context Aggregation Network(CAN) 기반.  
- **Dual Hierarchical Aggregation**:  
  - **Feature Aggregation Node (N)**와 **Attention Aggregation Node (AN)**를 통해 다중 레이어의 컨텍스트 특징 및 공간 주의(attention)를 트리 형태로 결합.  
  - 수식  

$$
      T_n = N(R^n_{n-1}(x), \dots, R^n_1(x), L^n_1(x), L^n_2(x)),\quad
      AT_n = AN(AR^n_{n-1}(x), \dots, AR^n_1(x), L^n_1(x), L^n_2(x))
    $$
  
  - 여기서 $$R^n_m(x)$$와 $$AR^n_m(x)$$는 재귀적으로 정의된 Aggregation 블록, $$L^n_k(x)$$는 dilated conv 블록 출력.  
- **Loss**:  
  - 픽셀 및 VGG 기반 multi-layer perceptual loss $$L_\Phi = \sum_{k=0}^5 \lambda_k \big\|\Phi_k(I'\_{\text{free}}) - \Phi_k(I_{\text{free}})\big\|_1$$  
  - Attention mask에 대한 binary cross-entropy $$L_m$$  
  - Conditional GAN adversarial loss $$L_{\text{cGAN}}$$  
  - 전체 최적화: $$\min_G \max_D L_{\text{cGAN}} + \lambda L_\Phi + \alpha L_m$$

### 2.3 SMGAN을 통한 데이터 증강
- **Shadow Matting GAN**: 그림자 마팅 $$I_{\text{matting}}$$을 학습하여  

$$
    I'\_{\text{shadow}} = G(I_{\text{free}}, M)\times I_{\text{free}}
  $$
  
  형태로 합성.  
- **Loss**: multi-scale perceptual loss 및 adversarial loss를 결합.  
- ISTD 데이터셋의 그림자–무그림자 쌍과 그림자 마스크를 이용해 GAN을 학습한 뒤, 새로운 장면과 마스크 조합으로 합성 그림자를 생성하여 학습 데이터에 추가.

## 3. 성능 향상 및 한계

### 3.1 성능 향상
- **숫자 지표**: ISTD 데이터셋에서 DHAN은 기존 최상위 모델 대비 그림자 영역 RMSE를 9.48→8.14로, SSIM-S를 96.66→98.29로 향상. SMGAN 증강 후에는 RMSE 7.52, SSIM-S 98.36 달성.  
- **시각 품질**: 경계 아티팩트 및 색 불일치(ghost) 현상 현저히 감소.  
- **검출 모델 일반화**: 합성 데이터 증강 후 SBU 검출 BER 4.56→4.29로 개선.

### 3.2 한계
- SMGAN은 **cast shadow**만 가정하며, 복잡한 조명·반사·부착형 그림자에는 적용 한계.  
- DHAN 구조가 복잡하여 학습 및 추론 속도가 느리고, 메모리 사용량이 커 GPU 자원 요구량이 높음.  
- **초고해상도** 이미지 처리 성능 평가 미비.

## 4. 일반화 성능 향상 관점
- SMGAN 기반 데이터 증강은 다양한 장면·마스크 조합을 만들어 **도메인 편차(domain shift)**를 줄여 일반화 성능을 크게 개선하였다.  
- DHAN의 hierarchical attention 구조는 멀티스케일 컨텍스트를 학습하여 **다양한 그림자 형태**에 대응할 수 있는 표현력을 제공한다.  
- 추가로 현실적인 그림자 합성(예: soft shadow, 반투명 그림자) 및 **도메인 적응(domain adaptation)** 기법을 접목하면 더욱 강력한 일반화가 가능할 것으로 기대된다.

## 5. 향후 연구 방향 및 고려 사항
- **복합 조명 환경**과 반사·투과성 그림자에 대응하기 위해 그림자 광원 모델링(light modeling) 및 물리 기반 렌더링을 통합.  
- SMGAN의 **고해상도 합성** 및 **다중 클래스 그림자 분할**을 통해 더 세밀한 마팅(mask) 제공.  
- DHAN 경량화 및 실시간 성능 최적화를 위한 **모델 압축(model compression)** 연구.  
- 합성 데이터와 실제 도메인 간 차이를 완화하는 **도메인 적응** 또는 **자기 지도 학습** 기법 적용.  
- 그림자 제거가 완료된 후 후속 컴퓨터 비전 과제(객체 검출·추적 등) 성능 향상 효과를 종합 평가.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/76ce4776-c083-4044-b6bc-9add1aa588b9/1911.08718v2.pdf)
