# Fast R-CNN | Object detection

**주요 주장 및 기여 요약**  
Fast R-CNN은 기존 R-CNN과 SPPnet의 단점을 극복하고, 객체 검출의 학습·추론 속도와 정확도를 동시에 향상시키는 단일 단계(end-to-end) 학습 방식을 제안한다.  
- **단일 단계 다중 과제 학습**(multi-task loss): 분류(softmax)와 경계 박스 회귀(bounding-box regression)를 하나의 네트워크에서 동시에 최적화.  
- **RoI 풀링 레이어**: 이미지 전체의 공통된 컨볼루션 특징 맵에서 각 제안(Region-of-Interest)을 고정 크기의 특징 벡터로 추출, 제안별 중복 계산을 제거.  
- **전 단계 특성 공유**: 학습 시 Mini-batch 내 다수 RoI가 같은 이미지에서 파생되어 계산·메모리 공유, SGD 속도 64× 향상.  
- **트렁케이션 SVD**: 완전 연결(fc) 레이어를 저차 행렬 분해로 압축, 추론 속도 30% 추가 향상.  
- VOC07/10/12 기준 VGG16 네트워크 사용 시 R-CNN 대비 학습 9×, 추론 213× 속도, mAP 2–4% 상승  

## 1. 해결 문제 및 제안 방법

### 1.1 문제 정의  
- 객체 검출은 *제안(proposal)* 수천 개를 처리하며, 각 제안에 대해 완전한 ConvNet 순전파를 수행해야 하므로 속도가 매우 느리고, 학습 파이프라인이 복잡(R-CNN: fine-tune → SVM → bbox regressor)하다.

### 1.2 제안 방법  
Fast R-CNN은 **단일 네트워크**로 제안 분류와 위치 보정을 동시에 학습하며, 전체를 end-to-end로 최적화한다.  

1) **네트워크 구조**  
   - 입력: 원본 이미지 + 객체 제안(RoI 리스트)  
   - 컨볼루션+풀링 → **RoI 풀링**(H×W) → FC 레이어 → 
     - softmax 분류 출력 $$p = (p_0,\dots,p_K)$$  
     - 클래스 별 bbox 회귀 출력 $$\mathbf{t}_k = (t_x,t_y,t_w,t_h)$$  

2) **다중 과제 손실함수**
 
$$
     L(p,u,\mathbf{t}\_u,\mathbf{v}) \;=\; L_{\mathrm{cls}}(p,u) \;+\; \lambda [u\ge1]\,L_{\mathrm{loc}}(\mathbf{t}_u,\mathbf{v})
$$
   
   - $$L_{\mathrm{cls}} = -\log p_u$$  
   - $$L_{\mathrm{loc}} = \sum_{i\in\{x,y,w,h\}}\mathrm{smooth}\_{L_1}(t_{u,i}-v_i)$$
 
$$ \mathrm{smooth}\_{L\_1}(x) = \begin{cases}0.5x^2, & if |x|<1 \\ |x|-0.5, & \text{otherwise}\end{cases} $$  

   - $$\lambda=1$$으로 분류·회귀 균형  

3) **학습 방식**  
   - Mini-batch: 이미지 $$N=2$$, RoI 총 $$R=128$$ (foreground: IoU≥0.5 25%, 배경: 0.1≤IoU<0.5)  
   - SGD: 초기 학습률 0.001 → 0.0001, 모멘텀 0.9, weight decay 0.0005  
   - 데이터 증강: 수평 뒤집기(50%)  

4) **추론 가속화**  
   - RoI 풀링으로 컨볼루션 공유  
   - fc6/fc7 층에 트렁케이션 SVD 적용  

## 2. 모델 구조 및 성능 향상

| 방법           | 학습 속도 | 추론 속도 (s/img) | VOC07 mAP |
|---------------|---------|------------------|-----------|
| R-CNN VGG16   | 84h     | 47.0             | 66.0%     |
| SPPnet VGG16  | 25.5h   | 2.3              | 63.1%     |
| Fast R-CNN    | **9.5h**| **0.32**         | **66.9%** |
| Fast R-CNN+SVD| 9.5h    | 0.22             | 66.6%     |

- 학습 9×, 추론 146× (213× with SVD) 속도 대폭 개선  
- 미세조정(fine-tune) 가능한 conv 계층(conv3_1 이상)으로 정확도 추가 확보  

## 3. 한계 및 일반화 성능

- **메모리 한계**: VGG16 다중 규모(multi-scale) 학습이 GPU 메모리 제약으로 구현 불가  
- **스케일 불변성**: 단일 스케일 학습(s=600)만으로도 우수하나, 다양한 물체 크기 분포에서는 multi-scale 고려 필요  
- **Proposal 의존성**: Selective Search 등 외부 제안 품질에 의존. 제안–분류 캐스케이드 구조가 일반화 성능 향상에 도움  
- **일반화 가능성**:  
  - RoI 풀링의 컨텍스트 포함 및 다중 과제 손실은 다양한 물체 검출·세분화(task)로 확장 가능  
  - End-to-end 학습 구조는 더 깊고 다양한 백본 네트워크(ResNet, EfficientNet 등)와 호환  

## 4. 향후 연구 영향 및 고려 사항

- **영향**  
  - 단일 단계 end-to-end 학습의 표준으로 자리매김, 이후 Faster R-CNN(Region Proposal Network 결합)·Mask R-CNN 등 후속 연구 발판  
  - RoI 풀링 아이디어는 다양한 영역별 특징 추출 기법(RoIAlign, 키포인트 검출)에 확장  

- **고려할 점**  
  1. **더 나은 제안 메커니즘**(RPN 통합)으로 학습·추론 통합  
  2. **정밀한 위치 보정**: RoIAlign, 다중 해상도 피처 맵 활용  
  3. **백본 네트워크 업그레이드**: 잔차 블록·효율적 컨볼루션 구조 적용  
  4. **제안-분류 공정 최적화**: 비최대 억제(NMS) 개선, Hard Negative Mining 자동화  

---  

Fast R-CNN은 단일 단계의 효율적 학습 구조와 RoI 풀링을 통해 객체 검출의 속도·정확도를 동시에 크게 향상시켰으며, 이후 초창기 딥러닝 기반 검출 기법 전반에 지대한 영향을 미쳤다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8b302f05-7355-4f1b-a3ae-833a1fb47e52/1504.08083v2.pdf
