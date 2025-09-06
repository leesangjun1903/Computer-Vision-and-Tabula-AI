# Adaptive Histogram Equalization: 이미지 대비를 높이는 현명한 전처리 기법

딥러닝을 공부하는 대학생을 위해 **Adaptive Histogram Equalization(AHE)**와 그 변형인 **CLAHE(Contrast Limited AHE)**를 소개합니다. 이미지 전처리 단계에서 로컬 대비를 보정해 모델 성능을 높이는 핵심 기법입니다.

## 1. Histogram Equalization(HE) 복습  
Histogram Equalization(히스토그램 평활화, HE)은 이미지 전체의 픽셀 분포를 균일하게 바꿔 대비를 높입니다.  
하지만 이미지 일부 영역의 픽셀 분포가 다를 때, 전역(equal global) 히스토그램만 이용하면 왜곡이 발생합니다.  

## 2. Adaptive Histogram Equalization(AHE)  
AHE는 이미지를 여러 개의 작은 구역(grid)으로 나눕니다.  
각 구역마다 HE를 적용해 **로컬 대비**를 조정합니다.  
- 장점: 국소 대비를 세밀하게 보정해 어두운 영역이나 밝은 영역의 디테일이 살아납니다.  
- 단점: 노이즈가 뚜렷한 경우, 작은 픽셀이 크게 증폭되어 “노이즈 증폭(noise amplification)”이 발생합니다.  

### AHE 처리 과정  
1. 이미지를 M×N 크기의 그리드로 분할  
2. 각 그리드에 HE 적용  
3. 경계 픽셀은 반대편 데이터를 미러링해 처리  

## 3. CLAHE: 노이즈 증폭 방지를 위한 개선  
CLAHE는 AHE의 노이즈 증폭 문제를 해결하기 위해 **clip limit(클립 리밋)**를 도입합니다.  
히스토그램의 특정 높이 이상으로 픽셀이 몰리지 않도록 잘라내고(redistribution), 균등하게 재분배합니다.  

### CLAHE 처리 과정  
1. 각 그리드에서 히스토그램 계산  
2. 히스토그램의 높이를 clip limit 만큼 제한  
3. 잘려 나온 픽셀을 나머지 구간에 재분배  
4. 제한된 히스토그램으로 CDF(누적분포함수) 생성  
5. CDF에 따라 픽셀 값 변환  

이 과정을 통해 **CDF의 기울기가 너무 급격해지지 않으므로**, 노이즈 픽셀이 과도하게 강조되지 않습니다.  

## 4. 딥러닝에서의 활용  
- **데이터 증강 전처리**: 어두운 이미지나 밝은 이미지의 로컬 디테일을 살려 학습 데이터 품질 향상  
- **의료 영상 분석**: 병변 부위의 대비를 높여 특징 추출 성능 개선  
- **자율주행**: 도로 영상에서 그림자나 반사광 문제 완화  

## 5. 구현 팁  
- OpenCV: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`  
- scikit-image: `skimage.exposure.equalize_adapthist(image, clip_limit=0.01)`  
- clip limit와 grid 크기는 데이터 특성에 맞춰 실험적으로 조정하세요.  

***

Adaptive Histogram Equalization과 CLAHE는 로컬 대비 보정에 강력한 도구입니다. 모델 입력 이미지의 디테일을 살리고, 노이즈 증폭을 억제해 딥러닝 성능을 끌어올려 보세요.

[1](https://3months.tistory.com/407)
