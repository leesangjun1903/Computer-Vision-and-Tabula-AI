https://www.kaggle.com/competitions/image-matching-challenge-2024/overview  

https://paperswithcode.com/sota/self-supervised-image-classification-on  
https://paperswithcode.com/dataset/imc-phototourism

# Baseline
https://www.kaggle.com/code/asarvazyan/imc-understanding-the-baseline

## Structure from Motion
Structure from Motion (SfM) is the name given to the procedure of reconstructing a 3D scene and simultaneously obtaining the camera poses of a camera w.r.t. the given scene. This means that, as the name suggests, we are creating the entire rigid structure from a set of images with different view points (or equivalently a camera in motion).
In this competition, the important aspect of SfM we are interested in is obtaining the camera poses of where each image was taken, described by a rotation matrix and translation vector from the origin.

ref)
- https://mvje.tistory.com/92  
- A Deep Learning Approach to Camera Pose Estimation : https://github.com/fedeizzo/camera-pose-estimation?tab=readme-ov-file
- https://woochan-autobiography.tistory.com/944
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
