https://www.kaggle.com/competitions/image-matching-challenge-2024/overview  

https://paperswithcode.com/sota/self-supervised-image-classification-on  
https://paperswithcode.com/dataset/imc-phototourism

# Baseline
https://www.kaggle.com/code/asarvazyan/imc-understanding-the-baseline

## Structure from Motion
Structure from Motion (SfM) is the name given to the procedure of reconstructing a 3D scene and simultaneously obtaining the camera poses of a camera w.r.t. the given scene. This means that, as the name suggests, we are creating the entire rigid structure from a set of images with different view points (or equivalently a camera in motion).
In this competition, the important aspect of SfM we are interested in is obtaining the camera poses of where each image was taken, described by a rotation matrix and translation vector from the origin.

ref)
- Structure from Motion : 2D 이미지는 어떻게 3D로 재구성될까? https://www.youtube.com/watch?v=LBW7a2UkRJI&t=97s
- https://mvje.tistory.com/92  
- A Deep Learning Approach to Camera Pose Estimation : https://github.com/fedeizzo/camera-pose-estimation?tab=readme-ov-file
- Structure From motion 설명 https://woochan-autobiography.tistory.com/944
- https://medium.com/@loboateresa/understanding-structure-from-motion-algorithms-fc034875fd0c
- https://cmsc426.github.io/sfm/
- https://velog.io/@shj4901/Structure-from-Motion-Revisited

- Dinov2 vs CLiP https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6
- Dinov2 https://mvje.tistory.com/143

## Estimate the Camera Poses

### Finding image pairs

### Computing keypoints

### Match and compute keypoint distances

### RANSAC

### Sparse Reconstruction

### Code

# Starter
https://www.kaggle.com/code/nartaa/imc2024-starter/notebook

## 39th place solution - SuperGlue + SIFT
https://github.com/gunesevitan/image-matching-challenge-2023/tree/main

## Feature detection
https://github.com/deepanshut041/feature-detection/tree/master/fast

## Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed
https://github.com/zju3dv/EfficientLoFTR/tree/main?tab=readme-ov-file

## Image Matching Challenge 2022 - EDA
https://www.kaggle.com/code/dschettler8845/image-matching-challenge-2022-eda

- Data Augmentation for Facial Keypoint Detection https://www.kaggle.com/code/balraj98/data-augmentation-for-facial-keypoint-detection
