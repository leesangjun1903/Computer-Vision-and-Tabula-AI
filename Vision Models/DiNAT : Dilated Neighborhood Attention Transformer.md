# 핵심 주장 및 주요 기여 요약  
Dilated Neighborhood Attention Transformer(DiNAT)는 기존의 국소적 슬라이딩 윈도우 주의(neighborhood attention, NA)가 지닌 장점(선형 복잡도, 병렬화 용이성)을 유지하면서, 희소(sparse) 글로벌 컨텍스트를 포착할 수 있도록 확장한 Dilated Neighborhood Attention(DiNA)를 제안한다. NA와 DiNA를 번갈아 쌓은 계층적 구조를 통해 수용 영역(receptive field)을 지수적으로(expontentially) 확장하면서도 계산 부담을 추가로 늘리지 않아, 이미지 분류·객체 검출·세분화 등 다양한 비전 태스크에서 성능을 유의미하게 향상시킨다[1].  

# 문제 정의  
기존 계층적 비전 트랜스포머(예: Swin Transformer)의 지역화된 윈도우 주의(local window attention)는 계산 효율을 확보하지만, 전역(Global) 상호의존성과 광역 수용 영역을 충분히 모델링하지 못해 긴 거리 의존성(long-range dependency)에 취약하다[1].  

# 제안 방법  
DiNA는 기존 NA의 슬라이딩 윈도우 내 이웃 선택을 ‘격자(dilation)’ 형태로 확장하여, 동일한 윈도우 크기 $$k$$하에서 간격을 늘린 이웃 $$\rho\_j^\delta(i)$$ 를 선택한다.  
수식으로, i번째 토큰에 대한 δ-팽창 이웃 주의는  

$$
A^{(k,\delta)}\_i = 
\begin{bmatrix}
Q_i K_{\rho_1^\delta(i)}^T + B(i,\rho_1^\delta(i)) \\
\vdots \\
Q_i K_{\rho_k^\delta(i)}^T + B(i,\rho_k^\delta(i))
\end{bmatrix}, 
\quad
\mathrm{DiNA}^\delta_k(i)=\mathrm{softmax}\bigl(A^{(k,\delta)}_i/\sqrt{d}\bigr)\,V^{(k,\delta)}_i
$$  

여기서 $$\rho_j^\delta(i)$$는 $$j\mod \delta = i\mod \delta$$를 만족하는 i의 j번째 이웃, $$B$$는 상대위치 바이어스, $$d$$는 임베딩 차원이다[1].  

# 모델 구조  
- 입력 해상도를 1/4로 초기 다운샘플링  
- 4단계(level)의 Transformer 블록: 각 단계는 NA 층과 DiNA 층을 교차 적용  
- 단계별로 공간 해상도를 절반으로, 채널 수를 두 배로 조정  
- DiNA 층의 팽창 계수(dilation)는 단계별로 (예: ImageNet-224 $$^2$$ 기준) 8→4→2→1로 점진적 설정[1]  

# 성능 향상  
- **ImageNet-1K 분류**: NAT 대비 대형 모델에서 Top-1 정확도 +0.1~0.4% 향상[1]  
- **COCO 객체 검출·인스턴스 분할**: Box AP +1.6%, Mask AP +1.4% 수준 개선  
- **ADE20K·Cityscapes 세멘틱·파노픽 분할**: mIoU·PQ에서 Swin·ConvNeXt 대비 1~2% 상승[1]  

# 한계  
- DiNA의 메모리 접근 패턴 분산으로 NA 대비 이론적 동일 복잡도에도 실경험 처리량이 다소 감소  
- 팽창 계수 최적값 탐색 비용 존재  
- 윈도우 크기 및 해상도에 따른 팽창 제약으로 정교한 하이퍼파라미터 튜닝 필요  

# 일반화 성능 향상 가능성  
DiNA는 고정된 윈도우 크기 내에서도 지수적 수용 영역 확대가 가능하므로, 다양한 입력 해상도·도메인에 걸쳐 장거리 의존성을 효과적으로 학습할 수 있다. 특히, 마스크 기반 분할·밀도 예측·비전-언어 멀티모달 대응 등 일반화가 중요한 태스크에서 강인성을 확보할 여지가 크다.  

# 향후 연구 영향 및 고려 사항  
향후 연구에서는 팽창 계수를 입력 해상도나 데이터 특성에 맞춰 자동으로 조정하는 동적 스킴(dynamic dilation)을 도입하거나, DiNA 커널의 하드웨어 최적화 구현을 통해 실효 처리량을 개선할 수 있다. 또한, 언어·음성·그래프 등의 비전 외 도메인으로의 확장 가능성, 그리고 DiNA와 전역·지역 주의를 결합한 하이브리드 모듈 설계가 주목받을 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/8fe9fd3e-b499-4a84-b49c-b88f8f1ca69a/2209.15001v3.pdf
