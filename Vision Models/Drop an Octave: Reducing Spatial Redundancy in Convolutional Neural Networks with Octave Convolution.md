# Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution | Image classification, Video action recognition

## 1. 핵심 주장 및 주요 기여
**주요 주장:**  
Convolutional Neural Networks(CNN)가 생성하는 피처 맵은 공간적 중복성을 포함하며, 이를 저·고(低·高) 주파수로 분리해 처리하면 메모리와 계산 효율을 크게 높일 수 있다.[1]

**주요 기여:**  
- *Octave Feature Representation* 제안: 채널을 저주파와 고주파 그룹으로 분리해 저주파는 해상도를 절반으로 줄여 저장  
- *Octave Convolution (OctConv)* 설계: 주파수 분리된 입력에 직접 작용하며, 네 가지 경로(H→H, L→L, H→L, L→H)를 통해 정보 업데이트 및 교환  
- 다양한 2D/3D CNN(ResNet, ResNeXt, DenseNet, MobileNet, I3D 등)에 플러그인 형태로 적용 가능  
- ImageNet, Kinetics 데이터셋에서 FLOPs와 메모리를 줄이면서 정확도 0.5–1.2% 향상  

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 향상 및 한계

### 문제 정의  
CNN의 각 위치는 독립적 피처를 저장해 공간적 중복이 발생하며, 특히 저주파 정보(글로벌 구조)는 해상도 감소가 가능한 반면 현행 convolution은 이를 반영하지 못함.[1]

### 제안 방법  
1. **Octave Feature Representation**  
   입력 $$X\in\mathbb{R}^{c\times h\times w}$$을  

$$
     X = \{X_H, X_L\},\quad X_H\in\mathbb{R}^{(1-\alpha)c\times h\times w},\; X_L\in\mathbb{R}^{\alpha c\times \tfrac{h}{2}\times \tfrac{w}{2}}
   $$  
   
   로 분리해 저주파 $$X_L$$는 해상도를 절반으로 줄임.[1]

2. **Octave Convolution (OctConv)**  
   출력 $$Y=\{Y_H,Y_L\}$$를 네 경로 합으로 계산:  

$$
     Y_H = f(X_H;W_{H\rightarrow H}) + \mathrm{upsample}\bigl(f(X_L;W_{L\rightarrow H}),2\bigr),
   $$  

$$
     Y_L = f(X_L;W_{L\rightarrow L}) + f\bigl(\mathrm{pool}(X_H,2);W_{H\rightarrow L}\bigr),
   $$  
   
   여기서 $$f(\cdot)$$은 convolution, pool은 average pooling, upsample은 nearest interpolation.[1]

3. **모델 구조 통합**  
   - 첫 번째 OctConv: $$\alpha_{\mathrm{in}}=0,\,\alpha_{\mathrm{out}}=\alpha$$  
   - 마지막 OctConv: $$\alpha_{\mathrm{in}}=\alpha,\,\alpha_{\mathrm{out}}=0$$  
   - 중간 블록: $$\alpha_{\mathrm{in}}=\alpha_{\mathrm{out}}=\alpha$$  
   - Group/Depth-wise convolution에도 확장 가능  

### 성능 향상  
- **ImageNet Classification:** ResNet-50 기준 FLOPs 4.1→2.4G, Top-1 +0.4% 향상.[1]
- **Video Action Recognition:** I3D 기준 FLOPs 28.1→25.6G, Top-1 +1.0% 향상.[1]
- **효율성:** $$\alpha=0.5$$일 때 이론적 FLOPs 44% 절감, 메모리 63% 절감.[1]

### 한계  
- α 값에 따라 저주파 정보 손실 가능성: α>0.5 시 정확도 감소  
- 추가 pooling/upsampling 연산으로 인한 구현 복잡도  
- 매우 작은 네트워크에서는 상대적 이득이 제한적  

## 3. 모델의 일반화 성능 향상 가능성  
- **확장된 수용 영역(Receptive Field):** 저주파 경로가 절반 해상도에서 처리되므로 원시 공간 기준 수용 영역이 2배로 확장되어 더 넓은 문맥 정보 포착  
- **주파수 분리 학습:** 고·저 주파수 피처를 독립 학습해 잡음에 강하며, 다양한 스케일 객체에 대한 인식력 향상  
- **다양한 백본 호환성:** ResNeXt, MobileNet, SE-Net 등 서로 다른 구조에서도 일관된 성능 향상 관측[1]
- **오버피팅 저감:** 저주파 정보가 압축 처리됨에 따라 모델이 세부 노이즈에 과도 적합되는 것을 방지  

## 4. 향후 연구에 미치는 영향 및 고려 사항
- **주파수 기반 네트워크 설계:** multi-frequency 처리 관점의 새로운 convolution 설계 제안  
- **Dynamic α 제어:** 학습 중 또는 레이어별로 α를 자동 조정하는 연구  
- **다중 옥타브 확장:** 두 옥타브 이상으로 주파수 그룹 확장 검토  
- **하드웨어 최적화:** OctConv 특화 하드웨어 및 컴파일러 지원 필요  
- **비주얼 도메인 외 응용:** 음성·자연어 등 다른 도메인에서 주파수 분리 구조 적용 가능성  

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/26adf7cc-2c60-4bd3-b825-f6604610e367/1904.05049v3.pdf)
