# The Guide to Fine-tuning Stable Diffusion with Your Own Images
https://www.edge-ai-vision.com/2023/10/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images/

한 줄 요약: DreamBooth를 활용하면 Stable Diffusion을 **소수 이미지**로 개인화하여 특정 인물/객체를 다양한 컨텍스트로 자연스럽게 생성할 수 있습니다. 본 글은 원리, 데이터 준비, Colab/로컬 학습, 추론과 WebUI 활용, 그리고 완전한 예시 코드까지 제공합니다.[1][2][3][4]

## DreamBooth 개요
DreamBooth는 텍스트-이미지 확산 모델을 소수의 주제 이미지로 미세조정해, 고유 토큰과 클래스 명사를 묶어 개인화된 생성을 가능하게 하는 기법입니다. 텍스트 임베딩만 학습하는 Textual Inversion과 달리, DreamBooth는 모델 가중치까지 업데이트하여 주제 재현력이 높습니다. 이로써 “a [토큰] man” 같은 프롬프트만으로 해당 주제를 다양한 장면에 합성할 수 있습니다.[5][2][6][1]

```
DreamBooth는 기존 텍스트-이미지 모델을 개인화하기 위해 소수(3~5장)의 특정 대상 이미지로 미세 조정(fine-tuning)하는 딥러닝 생성 모델입니다. 2022년 구글 리서치와 보스턴 대학교 연구팀이 개발했으며, 주로 텍스트-이미지 확산 모델의 UNet 부분을 훈련하여 특정 대상의 세밀한 특징을 다양한 컨텍스트에서 생성할 수 있도록 합니다.

기술적 특징은 다음과 같습니다:

3~5장의 대상 이미지와 대상 클래스(예: car)와 고유 식별자를 조합한 텍스트 프롬프트를 사용해 훈련합니다.
기존 모델이 갖는 클래스 일반화 능력을 유지하면서 특정 대상에 특화된 표현을 가능하게 하기 위해 prior preservation loss를 적용합니다.
고해상도 및 저해상도 이미지 쌍으로 초해상도 컴포넌트를 미세 조정해 세부사항 유지가 이루어집니다.
사용자 친화적 버전도 있어, getimg.ai 같은 플랫폼에서는 10~20장의 사진만 업로드하면 별도의 코딩 없이도 쉽게 맞춤 DreamBooth 모델을 생성할 수 있습니다.

즉, DreamBooth는 소량 데이터만으로도 텍스트-이미지 생성 모델에 특정 인물, 대상, 스타일을 자연스럽게 반영하도록 개인화하는 강력한 도구입니다.
```

## 핵심 이슈와 해결
소수 데이터로의 미세조정은 과적합과 언어 드리프트 문제가 발생합니다. DreamBooth는 클래스별 prior-preservation loss를 추가해, 일반 클래스의 생성 능력을 유지하면서 주제의 특징을 학습하도록 정규화합니다. 과도한 스텝은 품질 저하를 초래할 수 있어 적정 스텝과 정규화 이미지 수가 중요합니다.[7][6][1]

## 데이터 준비 가이드
권장 샷 수는 8~12장 수준이며, 다양한 각도, 표정, 배경을 포함해 주제의 시각적 변이를 확보합니다. 프롬프트에는 희귀 토큰(예: sks)과 클래스 명사(예: man, dog)를 함께 사용해 모델의 시각적 사전(prior)을 활용합니다. 낮은 해상도나 모션 블러는 그대로 학습되므로, 고해상도·선명한 이미지를 사용합니다.[2][8][1][5]

## 환경과 리소스
- Hugging Face diffusers의 공식 DreamBooth 스크립트/노트북을 활용하면 재현성이 높습니다.[2][4]
- 최신 모델군(SD 1.5, SDXL, SD3) 별 DreamBooth 예제가 공개되어 있어 버전에 맞게 선택합니다.[9][3][10]

## 최소 예시: SD 1.5 DreamBooth
**Stable Diffusion v1.5**는 텍스트 입력으로부터 512x512 크기의 고품질 포토리얼리스틱 이미지를 생성하는 잠재 공간(latent space) 기반 텍스트-투-이미지 딥러닝 모델입니다. 이 모델은 CLIP의 ViT-L/14 텍스트 인코더로 문장을 임베딩하여, U-Net 구조의 잠재 확산(latent diffusion) 방식을 통해 이미지를 생성합니다.

주요 특징은 다음과 같습니다:
- 텍스트-투-이미지 생성: 상세한 텍스트 설명을 바탕으로 사실감 넘치는 이미지를 만듭니다.
- 이미지 편집: 기존 이미지에 텍스트 기반 수정 및 인페인팅 가능.
- 경량화된 구조: 860M 파라미터 U-Net과 123M 텍스트 인코더를 사용해, 10GB VRAM 이상의 GPU에서 효율적 구동.
- 안전장치 탑재: 유해하거나 부적절한 콘텐츠 생성을 차단하는 안전 검사 기능 내장.
- 학습 데이터: LAION-5B 데이터셋 일부와 laion-aesthetics v2 5+ 데이터셋 위주로 약 59만 스텝 이상 미세조정(fine-tuning).

Stable Diffusion v1.5는 오픈소스 커뮤니티와 Stability AI, LAION의 협력으로 개발되었으며, 실사용에 적합한 신뢰성과 유연성을 갖춰 현재까지도 널리 활용되고 있습니다.

**sd15-dreambooth** 는 Stable Diffusion 1.5 모델을 바탕으로 한 DreamBooth 방식의 맞춤형 텍스트-이미지 생성 모델입니다. DreamBooth는 소수의 학습 이미지(3~5장)만으로 특정 인물, 사물 등을 모델에 학습시켜 다양한 스타일과 환경에서 해당 대상을 생성할 수 있게 해줍니다.

DreamBooth는 주로 Stable Diffusion v1.5 모델(즉, sd15)에 적용되어, 사용자가 본인의 얼굴, 반려동물, 사물 등을 모델에 넣어 자신만의 맞춤형 생성 모델을 만들 수 있습니다. 이를 위해 Colab 노트북 등 쉽고 빠른 툴로 학습이 가능하며, 코딩 지식이 없어도 사용할 수 있습니다. Hugging Face에 sd15-dreambooth라는 이름으로 업로드 된 커스텀 모델도 존재합니다.

요약하면, sd15-dreambooth는 Stable Diffusion 1.5 기반의 DreamBooth 맞춤 학습 모델이며, 적은 수의 이미지로 특정 대상을 모델에 학습하여 자연스러운 이미지 생성을 가능하게 하는 기술입니다. Stable Diffusion의 유연성과 DreamBooth의 개인화 모델링이 결합된 형태라 볼 수 있습니다.

아래는 Hugging Face diffusers DreamBooth 노트북 흐름을 간추린 예시입니다. Colab에서 실행 가능한 형태의 핵심 단계 코드입니다.[4][2]

- 사전 요구  
  - Hugging Face 계정/토큰 준비 및 모델 약관 동의(SD1.5: runwayml/stable-diffusion-v1-5).[5][4]
  - 학습용 이미지 8~12장 업로드(예: /content/data/subject).[1][5]

- 런타임 초기화
```bash
# Diffusers/Accelerate 설치
pip install -q diffusers==0.24.0 transformers accelerate safetensors xformers
pip install -q datasets huggingface_hub
```


```python
# Hugging Face 로그인
from huggingface_hub import notebook_login
notebook_login()
```


- 경로/변수 설정
```python
import os

MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # SD 1.5
INSTANCE_DIR = "/content/data/subject"         # 주제 이미지 폴더(8~12장)
OUTPUT_DIR = "/content/output-sd15-dreambooth" # 결과 가중치 저장 폴더
INSTANCE_PROMPT = "a photo of sks man"         # 희귀 토큰+클래스 명사
CLASS_PROMPT = "a photo of a man"              # prior 이미지 생성을 위한 클래스 프롬프트

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```


- prior-preservation 이미지 생성 및 학습 실행  
공식 예제 스크립트는 prior 이미지를 자동 생성하고 학습을 수행합니다. Accelerate 런처로 실행합니다.[3][4]
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/content/data/subject"
export OUTPUT_DIR="/content/output-sd15-dreambooth"

accelerate launch \
  /usr/local/lib/python3.10/dist-packages/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks man" \
  --class_prompt="a photo of a man" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --num_class_images=1200
```


- 추론 파이프라인으로 확인
```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    OUTPUT_DIR, torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

prompt = "portrait of sks man wearing a leather jacket, studio lighting, 85mm photography"
neg = "blurry, distorted, deformed, low quality"

images = pipe(prompt, negative_prompt=neg, num_inference_steps=30, guidance_scale=5.0, num_images_per_prompt=2).images
for i, img in enumerate(images):
    img.save(f"/content/result_{i}.png")
```


## 파라미터 튜닝 팁
- max_train_steps: 이미지 1장당 100~200 스텝부터 시도하고 품질/과적합을 보며 조정합니다.[7][1]
- num_class_images: 주제 이미지 수의 약 200배가 권장이나, 리소스 한도에 맞춰 200×N 대비 축소도 가능합니다.[6][1]
- learning_rate: 1e-6~1e-5 범위를 주로 사용하며, 과적합 시 더 낮춰봅니다.[1][3]
- guidance_scale/steps(추론): CFG 3~7, steps 20~40부터 탐색합니다.[1][4]

## WebUI(AUTOMATIC1111)로 창작 워크플로우 확장
학습 결과 ckpt/디퓨저스 포맷을 WebUI에 탑재하면 txt2img, img2img, inpainting으로 빠른 실험이 가능합니다. X/Y plot로 seed 고정 후 Steps·CFG 스윕을 자동으로 생성해 비교할 수 있어 생산성이 높습니다. 로컬 또는 Colab 런의 제약을 고려해 세션 시간을 관리합니다.[8][1][4]

## SDXL/SD3로 확장하기
- SDXL: 해상도와 품질이 향상된 모델로, 전용 DreamBooth 예제가 공개되어 있습니다.[10]
- SD3: 새 아키텍처에 맞춘 DreamBooth 스크립트/LoRA 예제가 제공되며, 게이트 동의 및 로그인 절차가 필요합니다.[9]
- 버전별로 스크립트 인자와 권장 설정이 달라 문서를 확인하고 적용합니다.[10][9]

## 문제 해결 체크리스트
- 주제 재현력이 낮음: 프롬프트에 클래스 명사 포함, 스텝 소폭 증가, 데이터 다양성(각도/표정/배경) 보강.[5][1]
- 훈련 이미지에 과도하게 종속: 스텝 감소, 이미지 수/다양성 확충, prior 이미지 수 상향.[6][1]
- 일반 클래스 품질 저하: prior-preservation 활성화 여부, num_class_images 재확인, 스텝 과다 여부 점검.[7][6]
- 자원 부족: 배치 1 + gradient_accumulation, fp16, 해상도 512, 학습 스텝 축소.[3][4]

## 완전 예시: SDXL DreamBooth(요약 명령)
SDXL은 해상도가 높아 파라미터가 일부 다릅니다. 공식 예제를 참고해 아래처럼 실행 가능합니다(개념용).[10]
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="./data/subject"
export OUTPUT_DIR="./trained-sdxl"

accelerate launch train_dreambooth_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks man" \
  --class_prompt="a photo of a man" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --max_train_steps=1200 \
  --num_class_images=1200
```


## 참고 자료
- Tryolabs 블로그: DreamBooth 동기/실무 팁, 데이터 수/프롬프트 예시 등 실전 가이드가 정리되어 있습니다.[1]
- Hugging Face diffusers: DreamBooth 문서/노트북/예제 스크립트로 최신 워크플로우를 제공합니다.[3][2][4]
- 보완 학습: DreamBooth 개념·수식·정규화 손실의 배경과 장단점은 논문과 리뷰 자료로 학습하면 좋습니다.[6][5][7]

[1](https://tryolabs.com/blog/2022/10/25/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images)
[2](https://huggingface.co/docs/diffusers/training/dreambooth)
[3](https://huggingface.co/docs/diffusers/v0.11.0/training/dreambooth)
[4](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
[5](https://www.machinelearningmastery.com/training-stable-diffusion-with-dreambooth/)
[6](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf)
[7](https://arxiv.org/html/2407.05312v1)
[8](https://learnopencv.com/dreambooth-using-diffusers/)
[9](https://huggingface.co/spaces/nvidia/Cosmos-Predict2/blob/845427f3cce3d3def8499d2c6eacc866b7d14c4a/diffusers_repo/examples/dreambooth/README_sd3.md)
[10](https://huggingface.co/spaces/nvidia/Cosmos-Predict2/blob/845427f3cce3d3def8499d2c6eacc866b7d14c4a/diffusers_repo/examples/dreambooth/README_sdxl.md)
[11](https://www.edge-ai-vision.com/2023/10/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images/)
[12](https://www.youtube.com/watch?v=v89kB4OScOA)
[13](https://wandb.ai/geekyrakshit/dreambooth-keras/reports/Fine-Tuning-Stable-Diffusion-Using-Dreambooth-in-Keras--VmlldzozNjMzMzQ4)
[14](https://www.reddit.com/r/StableDiffusion/comments/wvzr7s/tutorial_fine_tuning_stable_diffusion_using_only/)
[15](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dreambooth/)
[16](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/multimodal/text2img/dreambooth.html)
[17](https://jang-inspiration.com/dreambooth)
[18](https://kkm0476.tistory.com/5)
[19](https://www.digitalocean.com/community/tutorials/dreambooth-stable-diffusion-tutorial-1)
[20](https://hsejun07.tistory.com/300)
[21](https://www.reddit.com/r/StableDiffusion/comments/xqe9bz/dreambooth_training_and_inference_using/)

한 줄 답: 아래는 DreamBooth 학습·추론 셀에서 자주 쓰는 코드/커맨드를 줄단위로 해설한 안내입니다. 주석, 인자, 동작 원리를 간결하게 설명해 초심자도 따라할 수 있도록 구성했습니다.[1][2][3]

## 패키지 설치 셀 해설
```bash
pip install -q diffusers==0.24.0 transformers accelerate safetensors xformers
```
- diffusers: Stable Diffusion 등 확산 모델 학습/추론을 위한 핵심 라이브러리입니다.[1]
- transformers: 텍스트 인코더(CLIP 등) 로딩과 토큰화에 필요합니다.[1]
- accelerate: 분산/혼합정밀/장치 관리 유틸로 학습 런칭에 사용됩니다.[1]
- safetensors: 안전하고 빠른 가중치 저장 포맷을 지원합니다.[1]
- xformers: 메모리 효율적 어텐션(MEA)을 활성화해 VRAM을 절약합니다.[1]

```bash
pip install -q datasets huggingface_hub
```
- datasets: 데이터셋 로딩/전처리에 사용됩니다.[1]
- huggingface_hub: 모델/데이터 다운로드·인증 등 Hugging Face 허브 연동을 담당합니다.[1]

## 허깅페이스 로그인 셀 해설
```python
from huggingface_hub import notebook_login
notebook_login()
```
- notebook_login(): 노트북 환경에서 토큰 입력 UI를 띄워 허브에 인증합니다.[1]
- 모델 카드 약관 동의가 필요한 베이스 가중치(SD 1.5 등) 접근에 필요합니다.[1]

## 경로/변수 설정 셀 해설
```python
import os

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_DIR = "/content/data/subject"
OUTPUT_DIR = "/content/output-sd15-dreambooth"
INSTANCE_PROMPT = "a photo of sks man"
CLASS_PROMPT = "a photo of a man"

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
- MODEL_NAME: 베이스 모델 리포 이름으로 허브에서 체크포인트를 자동 로드합니다.[3]
- INSTANCE_DIR: 학습 대상(주제) 이미지가 들어갈 로컬 폴더 경로입니다.[3]
- OUTPUT_DIR: 파인튜닝 결과(가중치, 스텝별 체크포인트 등)를 저장할 폴더입니다.[3]
- INSTANCE_PROMPT: 희귀 토큰(sks) + 클래스 명사(man)를 결합해 대상과 클래스 사전을 연결합니다.[1]
- CLASS_PROMPT: prior-preservation용 일반 클래스 이미지를 생성·라벨링할 프롬프트입니다.[1]
- os.makedirs(..., exist_ok=True): 폴더가 없으면 생성하고, 있으면 오류 없이 넘어갑니다.[3]

## 학습 실행 커맨드 셀 해설
```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/content/data/subject"
export OUTPUT_DIR="/content/output-sd15-dreambooth"
```
- 쉘 환경변수로 동일 값을 재사용하기 쉽게 지정합니다.[3]

```bash
accelerate launch \
  /usr/local/lib/python3.10/dist-packages/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks man" \
  --class_prompt="a photo of a man" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --num_class_images=1200
```
- accelerate launch: Accelerate 런처로 다중 GPU/혼합정밀 등 환경을 자동 구성해 스크립트를 실행합니다.[1]
- train_dreambooth.py: Diffusers 제공 DreamBooth 학습 스크립트 경로입니다(노트북 설치 경로 기준).[3]
- --pretrained_model_name_or_path: 허브 리포나 로컬 경로로 베이스 모델을 지정합니다.[1]
- --instance_data_dir: 주제 이미지가 들어있는 폴더입니다.[3]
- --output_dir: 학습 산출물을 저장할 폴더입니다.[3]
- --instance_prompt: 주제용 프롬프트(희귀 토큰 + 클래스)로 주제 정체성 학습을 유도합니다.[1]
- --class_prompt: prior 이미지 생성·라벨에 쓰는 일반 클래스 프롬프트입니다.[1]
- --with_prior_preservation: 클래스 일반성을 유지하기 위한 prior-preservation loss를 활성화합니다.[1]
- --prior_loss_weight=1.0: prior 손실 가중치로, 주제 재현성과 클래스 일반성의 균형을 조정합니다.[1]
- --resolution=512: 학습 시 이미지 입력 해상도입니다(SD1.x 기본 스케일에 적합).[1]
- --train_batch_size=1: VRAM 제약을 고려한 배치 크기입니다.[3]
- --gradient_accumulation_steps=4: 누적 그래디언트로 유효 배치를 4배로 만들어 안정화/메모리 절약을 돕습니다.[1]
- --learning_rate=1e-6: 미세한 업데이트를 위한 낮은 LR로 과적합을 줄입니다.[1]
- --lr_scheduler="constant": 학습 내내 일정 학습률을 유지합니다.[1]
- --lr_warmup_steps=0: 워밍업 없이 즉시 설정된 LR을 사용합니다.[1]
- --max_train_steps=2000: 전체 최적화 스텝 수로, 데이터 수×100~200 정도를 출발점으로 삼습니다.[1]
- --num_class_images=1200: prior용 클래스 샘플 수(대상 이미지 수×200 권장)를 자동 생성하게 합니다.[1]

추가 메모리 절약 옵션 예시:
- --mixed_precision=bf16 또는 fp16: 텐서 정밀도를 낮춰 VRAM을 절약합니다.[1]
- --enable_xformers_memory_efficient_attention: xformers 기반 MEA를 사용합니다.[1]
- --use_8bit_adam: 옵티마이저 메모리를 줄입니다(환경에 따라 설치 필요).[4]

## 추론 파이프라인 셀 해설
```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
```
- StableDiffusionPipeline: 텍스트→이미지 파이프라인을 손쉽게 호출하는 고수준 API입니다.[2]
- DDIMScheduler: 기본 스케줄러 대안으로, 빠르고 안정적인 샘플링을 제공합니다.[5]

```python
pipe = StableDiffusionPipeline.from_pretrained(
    OUTPUT_DIR, torch_dtype=torch.float16
).to("cuda")
```
- from_pretrained(OUTPUT_DIR): 방금 학습된 가중치를 로드해 파이프라인을 구성합니다.[2]
- torch_dtype=torch.float16: FP16로 VRAM 사용량을 줄이고 속도를 높입니다.[1]
- .to("cuda"): GPU로 올려 추론 속도를 크게 향상시킵니다.[2]

```python
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```
- 현재 파이프라인의 스케줄러 구성을 그대로 가져와 DDIM으로 교체합니다.[5]
- 필요시 DPMSolverMultistepScheduler 등으로 교체해 더 적은 스텝으로 빠르게 생성할 수 있습니다.[5]

```python
prompt = "portrait of sks man wearing a leather jacket, studio lighting, 85mm photography"
neg = "blurry, distorted, deformed, low quality"
```
- prompt: 학습된 희귀 토큰(sks)을 포함해 대상 정체성을 호출합니다.[2]
- negative_prompt: 흔한 노이즈/왜곡을 억제해 품질을 안정화합니다.[2]

```python
images = pipe(
    prompt,
    negative_prompt=neg,
    num_inference_steps=30,
    guidance_scale=5.0,
    num_images_per_prompt=2
).images
```
- num_inference_steps: 디노이징 횟수로, 20~40에서 품질·속도 균형을 찾습니다.[5]
- guidance_scale(CFG): 텍스트 조건 반영 강도이며 3~7 구간이 무난합니다.[2]
- num_images_per_prompt: 한 번에 여러 이미지를 생성해 탐색 효율을 높입니다.[2]
- .images: PIL 이미지 리스트를 반환합니다.[2]

```python
for i, img in enumerate(images):
    img.save(f"/content/result_{i}.png")
```
- 생성 결과를 파일로 저장합니다. 반복 실행으로 다양한 시드를 시험하면 탐색 폭이 넓어집니다.[2]

## WebUI 연동 팁
- 학습 결과를 ckpt 또는 Diffusers 디렉터리 형태로 변환·저장 후 AUTOMATIC1111 WebUI의 models/Stable-diffusion 폴더에 배치합니다.[2]
- txt2img/img2img/inpainting과 X/Y Plot으로 seed 고정 뒤 Steps·CFG 스윕을 자동 비교하면 최적 설정을 빠르게 찾을 수 있습니다.[2]

## 파라미터 튜닝 요령
- 스텝 증가 전, 데이터 다양성(각도/표정/배경)과 prior 활성화 여부를 먼저 확인합니다.[1]
- 과적합 징후(훈련 이미지 복제·표정/포즈 고정)가 보이면 스텝을 줄이고 num_class_images를 늘려봅니다.[1]
- VRAM 부족 시 batch=1, gradient_accumulation, fp16/bf16, xformers, 8bit Adam을 조합합니다.[4][1]

## 참고 문서
- Hugging Face DreamBooth 공식 가이드: 스크립트 인자 설명, VRAM 절약 팁, Accelerate 설정을 제공합니다.[1]
- Colab 노트북: end-to-end 예제와 셀 구성 흐름을 확인할 수 있습니다.[3]
- 실전 튜토리얼: 추론 파이프라인, 프롬프트/네거티브 프롬프트 사용 예를 확인하세요.[2]

[1](https://huggingface.co/docs/diffusers/training/dreambooth)
[2](https://learnopencv.com/dreambooth-using-diffusers/)
[3](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
[4](https://www.machinelearningmastery.com/training-stable-diffusion-with-dreambooth/)
[5](https://huggingface.co/mishig/llms-txt/raw/main/diffusers.txt)
[6](https://huggingface.co/docs/diffusers/v0.11.0/training/dreambooth)
[7](https://huggingface.co/blog/dreambooth)
[8](https://www.reddit.com/r/StableDiffusion/comments/xqe9bz/dreambooth_training_and_inference_using/)
[9](https://docs.proximl.ai/tutorials/gan/stable-diffusion-2-custom-model-training/)
[10](https://huggingface.co/docs/diffusers/v0.18.0/training/dreambooth)
[11](https://huggingface.co/docs/diffusers/ko/training/dreambooth)
[12](https://github.com/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
[13](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_inference.ipynb)
[14](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py)
[15](https://colab.research.google.com/drive/1muBY-yFCu_jAxnxeNW0RGwwNQhWQsE24)
[16](https://learnopencv.com/fine-tuning-stable-diffusion-3-5m/)
[17](https://huggingface.co/blog/sdxl_lora_advanced_script)
[18](https://cocalc.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
[19](https://pypi.org/project/diffusers/0.7.2/)
[20](https://syncedreview.com/2023/02/13/hugging-face-releases-lora-scripts-for-efficient-stable-diffusion-fine-tuning/)
