# Utilities

```
# General utilities
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from pathlib import Path
from time import time, sleep
from fastprogress import progress_bar
import gc
import numpy as np
import h5py
from IPython.display import clear_output
from collections import defaultdict
from copy import deepcopy
from typing import Any
import itertools
import pandas as pd

# CV/MLe
import cv2
import torch
from torch import Tensor as T
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import torch
from lightglue import match_pair
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd

# 3D reconstruction
import pycolmap

# Data importing into colmap
import sys
sys.path.append("/kaggle/input/colmap-db-import")

# Provided by organizers
from database import *
from h5_to_db import *

def arr_to_str(a):
    """Returns ;-separated string representing the input"""
    return ";".join([str(x) for x in a.reshape(-1)])

def load_torch_image(file_name: Path | str, device=torch.device("cpu")):
    """Loads an image and adds batch dimension"""
    img = K.io.load_image(file_name, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img

device = K.utils.get_cuda_device_if_available(0)
print(device)

DEBUG = len([p for p in Path("/kaggle/input/image-matching-challenge-2024/test/").iterdir() if p.is_dir()]) == 2
print("DEBUG:", DEBUG)

```

The provided code is a collection of import statements and utility functions for image processing and machine learning tasks. Here's a breakdown of what each part does:

1. **Import Statements**: The code imports various libraries and modules such as `matplotlib` for plotting, `tqdm` for progress bars, `pathlib` for file path handling, `time` for time-related functions, `fastprogress` for progress bars, `gc` for garbage collection, `numpy` for numerical operations, `h5py` for reading HDF5 files, `IPython.display` for clearing output in Jupyter notebooks, `collections` for data structures, `copy` for deep copying objects, `typing` for type hints, `itertools` for creating iterators, `pandas` for data manipulation, `cv2` for computer vision tasks, `torch` for PyTorch, `kornia` for computer vision operations, `PIL` for image processing, `transformers` for Hugging Face's transformer models, `lightglue` for feature matching, `pycolmap` for 3D reconstruction, and `database` and `h5_to_db` which are not standard Python libraries and are likely custom modules or scripts.

2. **Utility Functions**: The code defines two utility functions:
   - `arr_to_str(a)`: This function converts a numpy array into a string with elements separated by semicolons.
   - `load_torch_image(file_name, device=torch.device("cpu"))`: This function loads an image from a file, converts it to a PyTorch tensor, and adds a batch dimension.

3. **Device Selection**: The code sets the device to be used for PyTorch computations. If a CUDA-enabled GPU is available, it will use that; otherwise, it will use the CPU.

4. **Debugging Flag**: The code sets a `DEBUG` flag based on the number of directories in a specific test path. If there are exactly two directories, `DEBUG` is set to `True`; otherwise, it's `False`.

5. **Print Statements**: The code prints the selected device and the value of the `DEBUG` flag.

Here are a few suggestions for modifications:

- Ensure that the custom modules or scripts (`database` and `h5_to_db`) are correctly imported and used in the code.
- If you're not using the `DEBUG` flag for anything, you might want to remove it or replace it with a more descriptive flag name.
- If you're not using the `arr_to_str` function, you can remove it.
- If you're not using the `load_torch_image` function, you can remove it or modify it to better suit your needs.
- If you're not using the `device` variable, you can remove it or replace it with a more descriptive variable name.

Remember to test any modifications to ensure they don't break the code or alter its behavior unintentionally.

제공되는 코드는 이미지 처리 및 기계 학습 작업을 위한 수입 명세서 및 유틸리티 함수의 모음입니다. 다음은 각 파트가 수행하는 작업을 분석한 것입니다:

1. **ImportStatements**: 코드는 플롯을 위한 matplotlib, 프로그레스 바를 위한 tqdm, 파일 경로 처리를 위한 pathlib, 시간 관련 기능을 위한 타임, 프로그레스 바를 위한 빠른 진행, 가비지 컬렉션을 위한 gc, 수치 연산을 위한 numpy, HDF5 파일을 읽는 h5py, 주피터 노트북에 출력물을 클리어하기 위한 IPython.display, 데이터 구조를 위한 copy, 객체를 심층 복사하기 위한 copy, 유형 힌트를 위한 typing, 반복기를 만드는 이터툴, 데이터 조작을 위한 pandas, 컴퓨터 비전 작업을 위한 cv2, 컴퓨터 비전 작업을 위한 kornia, 이미지 처리를 위한 PIL, 허깅 페이스의 트랜스포머 모델을 위한 트랜스포머, 피처 매칭을 위한 lightglue, 3D 재구성을 위한 pycolmap, 표준 파이썬 라이브러리가 아닌 맞춤형 모듈 또는 스크립트일 가능성이 높은 데이터베이스 및 h5_to_db 등 다양한 라이브러리와 모듈을 가져옵니다.

2. **유틸리티 기능**: 코드는 두 가지 유틸리티 기능을 정의합니다:
   - arr_to_str(a)': 이 함수는 원소가 세미콜론으로 구분된 numpy 배열을 문자열로 변환합니다.
   - load_torch_image(file_name, device=torch.device (" cpu"): 이 함수는 파일에서 이미지를 로드한 후 PyTorch 텐서로 변환하고 배치 차원을 추가합니다.

3. **Device Selection**: 코드는 PyTorch 연산에 사용할 장치를 설정합니다. CUDA 지원 GPU를 사용할 수 있으면 그것을 사용하고 그렇지 않으면 CPU를 사용합니다.

4. **디버깅 플래그**: 코드는 특정 테스트 경로의 디렉터리 수를 기준으로 DEBUG 플래그를 설정합니다. 디렉터리가 정확히 두 개라면 DEBUG는 True이고, 그렇지 않으면 False입니다.

5. **PrintStatements**: 코드는 선택한 장치와 'DEBUG' 플래그의 값을 출력합니다.

다음은 수정을 위한 몇 가지 제안 사항입니다:

- 사용자 정의 모듈 또는 스크립트('database' 및 'h5_to_db')가 코드에 올바르게 가져와 사용되는지 확인합니다.
- 'DEBUG' 플래그를 사용하지 않는 경우 해당 플래그를 제거하거나 더 설명적인 플래그 이름으로 교체할 수 있습니다.
- arr_to_str' 기능을 사용하고 있지 않다면 제거할 수 있습니다.
- 'load_torch_image' 기능을 사용하지 않으신다면 제거하거나 필요에 맞게 수정할 수 있습니다.
- 'device' 변수를 사용하지 않는 경우 제거하거나 더 설명적인 변수 이름으로 대체할 수 있습니다.

수정 사항을 테스트하여 코드를 어기거나 의도치 않게 행동을 변경하지 않도록 해야 합니다.

# Finding image pairs 
```
def embed_images(
    paths: list[Path],
    model_name: str,
    device: torch.device = torch.device("cpu"),
) -> T:
    """Computes image embeddings.
    
    Returns a tensor of shape [len(filenames), output_dim]
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    embeddings = []
    
    for i, path in tqdm(enumerate(paths), desc="Global descriptors"):
        image = load_torch_image(path)
        
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs) # last_hidden_state and pooled
            
            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=-1, p=2)
            
        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)
```

The `embed_images` function you've provided is designed to compute image embeddings using a pre-trained transformer model from the Hugging Face `transformers` library. The function takes a list of `Path` objects representing the image file paths, the name of the model to use, and an optional `device` parameter specifying the device to run the computations on.

Here's a breakdown of the function:

1. **Model Initialization**: The function initializes an image processor and a transformer model from the specified `model_name`. The model is set to evaluation mode and moved to the specified device.

2. **Embedding Computation**: The function iterates over the provided list of `Path` objects, loading each image, processing it with the image processor, and passing it through the model to obtain the embeddings.

3. **Normalization and Pooling**: The function applies max pooling over all the hidden states of the model's output, excluding the first one (which is typically the special start token). It then normalizes the resulting tensor to have L2 norm of 1, which can help in subsequent distance computations.

4. **Embedding Storage**: The embeddings are detached from the computation graph and moved to the CPU, and then appended to a list of embeddings.

5. **Return Statement**: After processing all images, the function concatenates the list of embeddings along the first dimension (rows) to form a single tensor and returns it.

당신이 제공한 embed_images 함수는 허깅 페이스 트랜스포머 라이브러리에서 사전 훈련된 트랜스포머 모델을 사용하여 이미지 임베딩을 계산하도록 설계되었습니다. 이 함수는 이미지 파일 경로를 나타내는 Path 개체 목록, 사용할 모델 이름 및 계산을 실행할 장치를 지정하는 선택적 장치 매개 변수를 가져옵니다.

다음은 기능을 분석한 것입니다:

Model Initialization(모델 초기화): 지정한 model_name에서 이미지 프로세서와 트랜스포머 모델을 초기화합니다. 모델이 평가 모드로 설정되어 지정한 장치로 이동합니다.

Embedding Computation: 함수는 제공된 Path 개체 목록을 반복하여 각 이미지를 로드하고 이미지 프로세서로 처리한 다음 모델을 통과하여 임베딩을 얻습니다.

정규화 및 풀링: 함수는 모델 출력의 첫 번째 상태(일반적으로 특수 시작 토큰)를 제외한 모든 숨겨진 상태에 대해 최대 풀링을 적용합니다. 그런 다음 결과 텐서를 정규화하여 L2 놈이 1이 되도록 하여 후속 거리 계산에 도움이 될 수 있습니다.

임베딩 스토리지: 임베딩은 계산 그래프에서 분리되어 CPU로 이동한 다음 임베딩 목록에 추가됩니다.

반환문: 모든 이미지를 처리한 후 함수는 첫 번째 차원(행)을 따라 임베딩 목록을 연결하여 단일 텐서를 형성하고 반환합니다.
