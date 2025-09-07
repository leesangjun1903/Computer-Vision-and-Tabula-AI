# Representation, Feature, Latent Space, Encoder

## 한눈에 핵심
- Representation은 데이터를 n차원 벡터로 “표현”하는 방법이며, 좋은 표현이 곧 성능입니다.[2][1]
- Feature는 표현 벡터의 요소이며, 이 벡터들이 모인 공간이 바로 Latent Space입니다.[1][2]
- Encoder는 입력에서 유용한 Feature를 추출해 Latent Space로 맵핑하는 신경망 하위 구조입니다.[17][12]

## 왜 ‘표현’이 중요한가
딥러닝은 원시 픽셀 대신 압축되고 의미 있는 **표현**을 학습합니다. 같은 클래스의 이미지들은 Latent Space에서 가깝게 모이고, 다른 클래스는 멀리 떨어지며, 이 분리도가 분류 성능을 좌우합니다.[8][1]
즉, 좋은 표현은 차원을 낮추면서도 의미 정보를 보존하고, 거리(코사인/유클리드)가 유사도를 반영하도록 합니다.[2][1]

## 용어 정리
- Representation: 데이터를 n차원 **벡터로 표현**한 결과로, 학습된 내부 표현입니다.[8][1]
- Feature: 표현 벡터의 좌표 성분으로, 모델이 중요하다고 본 특성입니다.[7][1]
- Latent Space: 표현 벡터들이 위치하는 잠재 공간으로, 의미적 유사도가 거리에 반영됩니다.[1][2]
- Encoder: 입력을 Latent Space의 벡터로 변환하는 부분(예: CNN 백본, Autoencoder의 인코더)입니다.[12][17]

## 2D로 보는 Latent Space: t‑SNE

t-SNE (t-distributed Stochastic Neighbor Embedding)는 고차원 데이터를 2차원 또는 3차원 같은 저차원 공간으로 변환하여 시각화하는 비선형 차원 축소 기법입니다. 주로 데이터의 클러스터링이나 복잡한 구조적 관계를 시각적으로 이해하는 데 사용됩니다.

t-SNE는 고차원 공간에서 데이터 간의 유사성을 조건부 확률로 표현하고, 이를 저차원 공간에서 비슷한 확률 분포가 되도록 임베딩하여 원래 데이터의 군집이나 구조를 보존합니다. 이때 유사성 계산에 가우시안 분포를 이용하지만, 저차원 임베딩에서는 꼬리가 두터운 t-분포를 사용해 데이터가 너무 가까워지거나 멀어지는 문제를 완화합니다.

특징을 정리하면 다음과 같습니다:

- 고차원 데이터의 국소적 구조(이웃 관계)를 잘 보존하여 복잡한 데이터 구조를 시각화하는 데 강점이 있습니다.
- 클러스터링 알고리즘은 아니며, t-SNE 결과에 k-means 등의 클러스터링을 추가하는 경우가 많습니다.
- PCA와 달리 저차원 변수 자체에 직접적인 해석 의미는 없습니다.
- 퍼플렉시티(Perplexity)라는 하이퍼파라미터를 통해 각 포인트 주변 이웃 수를 조절해 유사성 계산의 민감도를 조정할 수 있습니다.
- 요약하면, t-SNE는 데이터 간 비선형 관계를 저차원 공간에 잘 보존해 시각적 탐색에 매우 유용한 차원 축소 기법입니다. 특히 고차원 데이터의 군집이나 패턴을 직관적으로 이해하는 데 널리 사용됩니다.

고차원 표현을 2D로 시각화할 때는 t‑SNE가 널리 쓰입니다. 국소 구조(근접 이웃)를 잘 보존해 클래스 클러스터를 직관적으로 확인하기 좋습니다.[9][15]
실무에선 scikit‑learn의 TSNE를 사용하며, 전처리와 샘플 수 조절, perplexity 탐색이 품질에 영향을 줍니다.[4][9]

## 실습 1: 사전학습 CNN으로 Feature 추출 → t‑SNE 시각화
사전학습 모델(예: ResNet) 중간/펜얼티메이트 레이어를 **Encoder**처럼 사용해 Feature를 뽑고, t‑SNE로 2D 시각화를 합니다.[4][7]

- 포인트  
  - Feature 추출은 분류 헤드를 제거하거나 forward hook을 사용합니다.[14][7]
  - t‑SNE 전에는 표준화/차원 축소(PCA 50D) 후 t‑SNE를 자주 사용합니다.[15][9]

예시 코드(요약):
- 데이터: 커스텀 ImageFolder
- 백본: torchvision.models.resnet18(pretrained=True)
- 추출: avgpool 뒤 벡터
- 시각화: sklearn TSNE

코드:
```python
# env: pip install torch torchvision scikit-learn matplotlib
import torch, torchvision as tv
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

dataset = datasets.ImageFolder("path/to/data", transform=transform)  # class 폴더 구조
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# 1) Encoder처럼 쓰기: 분류기 제거
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
encoder = nn.Sequential(*(list(backbone.children())[:-1]))  # avgpool까지
encoder.eval()

features, labels = [], []
with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        f = encoder(x)              # (B, 512, 1, 1)
        f = f.view(f.size(0), -1)   # (B, 512)
        features.append(f.cpu().numpy())
        labels.append(y.numpy())
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 2) (선택) PCA로 50D 축소 후 t‑SNE
features_50 = PCA(n_components=50, random_state=0).fit_transform(features)
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=0)
Z = tsne.fit_transform(features_50)

# 3) 시각화
plt.figure(figsize=(7,6))
num_classes = len(dataset.classes)
colors = plt.cm.tab20(np.linspace(0,1,num_classes))
for c in range(num_classes):
    idx = labels == c
    plt.scatter(Z[idx,0], Z[idx,1], s=5, color=colors[c], label=dataset.classes[c], alpha=0.7)
plt.legend(markerscale=3, bbox_to_anchor=(1.05,1), loc='upper left')
plt.title("t-SNE of CNN Features")
plt.tight_layout()
plt.show()
```
이렇게 얻은 플롯에서 클래스별 클러스터가 깔끔히 분리될수록 Latent Space 품질이 높다고 볼 수 있습니다.[9][4]

## 실습 2: Convolutional Autoencoder로 Encoder/Decoder 설계
Autoencoder는 입력을 압축(latent)했다가 복원하는 구조로, 인코더가 Feature를 학습하고 Latent Space를 형성합니다.[17][12]
재구성 손실을 최소화하면 일반적 구조를 학습하고, 다운스트림 분류기나 이상탐지로 확장할 수 있습니다.[16][12]

예시 코드(요약):
- 인코더: Conv → ReLU → Pool 반복
- 디코더: ConvTranspose로 업샘플
- 손실: BCE/ MSE

코드:
```python
# env: pip install torch torchvision
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("path/to/data", transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

class ConvAE(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(True),  # 32x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True), # 16x16
            nn.Conv2d(64, latent_channels, 3, stride=2, padding=1), nn.ReLU(True) # 8x8
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True), # 16x16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),               # 32x32
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()                  # 64x64
        )

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z

model = ConvAE().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()  # 픽셀 회귀이므로 MSE 자주 사용

for epoch in range(20):
    model.train()
    running = 0.0
    for x, _ in loader:
        x = x.to(device)
        xhat, z = model(x)
        loss = crit(xhat, x)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()*x.size(0)
    print(f"epoch {epoch+1}: recon_loss={running/len(dataset):.4f}")
```
학습 후 enc 출력 z를 평균/평탄화해 분류기의 입력 Feature로 사용하면 전이학습처럼 활용 가능합니다.[5][12]

## 실습 3: Autoencoder 인코더를 분류에 재활용
학습된 인코더를 고정(freeze)하고, 위에 얕은 분류기를 붙여 소량 라벨로 빠르게 적응할 수 있습니다.[5][12]
이는 Encoder가 이미 일반적 구조를 학습했다는 가정 하에 효과적이며, 과적합을 막기 위해 드롭아웃·데이터 증강을 함께 사용합니다.[12][5]

코드(요약):
```python
# AE 학습 후
for p in model.enc.parameters():
    p.requires_grad = False

clf = nn.Sequential(
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64, 128), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
).to(device)

params = [p for p in clf.parameters() if p.requires_grad] + \
         [p for p in model.enc.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=1e-3)
ce = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in labeled_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            z = model.enc(x)
        logits = clf(z)
        loss = ce(logits, y)
        optim.zero_grad(); loss.backward(); optim.step()
```
이 접근은 자기지도 사전학습과도 유사한 흐름이며, 작은 데이터셋에서 강력한 초기화를 제공합니다.[5][12]

## 실습 4: Torchvision로 중간 Feature 추출하기
hook 또는 feature_extraction 유틸로 임의 레이어 출력을 편히 수집할 수 있습니다.[7][14]
레이어별 Feature를 비교하며 t‑SNE를 그려보면, 깊어질수록 클래스 분리가 좋아지는 경향을 관찰할 수 있습니다.[4][7]

코드(요약):
```python
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18, ResNet18_Weights

m = resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
# avgpool 직전, layer4 끝 등 원하는 노드 지정
return_nodes = {"layer2.1.relu_1": "l2", "layer4.1.relu_1": "l4", "avgpool": "pool"}
extractor = create_feature_extractor(m, return_nodes=return_nodes)

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        feats = extractor(x)  # dict: {'l2':..., 'l4':..., 'pool':...}
        # feats['pool'] 등으로 t‑SNE 진행
```
이 방식은 레이어 선택이 명시적이고 재현성이 좋으며, 모델 해석에도 유용합니다.[14][7]

## Latent Space 품질을 높이는 팁
- 데이터: 클래스 균형, 다양성 확보, 증강으로 분리도를 높입니다.[9][4]
- 모델·훈련: 적절한 깊이/정규화/스케줄링과 라벨 품질이 Latent 구조를 개선합니다.[2][1]
- 시각화: t‑SNE의 perplexity, 학습률, 초기화(PCA) 등을 스윕해 안정적인 패턴을 찾습니다.[15][9]

## 주의: t‑SNE 해석
t‑SNE는 국소 구조에 민감하여 전역 거리는 왜곡될 수 있고, 하이퍼파라미터·초기값에 따라 모양이 변합니다.[15][9]
따라서 “클러스터 간 거리의 절대값”보다 “클러스터 분리 유무”와 “근접 이웃 보존”에 초점을 맞추는 것이 안전합니다.[9][15]

## 확장 학습 주제
- VAE: 확률적 Latent를 학습하여 연속적 조작과 샘플링이 가능합니다.[18][19]
- Latent 안정성: 분포 이동 시 Latent Feature의 일관성·안정성 평가 연구가 진행 중입니다.[3][19]

## 마무리 체크리스트
- Encoder가 만든 **Feature**로 Latent Space를 시각화했는가?[7][4]
- 클래스별 클러스터가 형성되는가, 분리도가 개선되고 있는가?[4][9]
- 사전학습 백본/AE 인코더를 적절히 재사용했는가?[12][7]

참고: 본 글의 개념적 설명은 잠재공간의 정의와 CNN/Autoencoder에서의 역할, t‑SNE 활용법에 대한 공개 자료들을 바탕으로 구성되었습니다.[8][1][2][15][4][9]

[1](https://www.baeldung.com/cs/dl-latent-space)
[2](https://www.geeksforgeeks.org/deep-learning/latent-space-in-deep-learning/)
[3](https://arxiv.org/pdf/2402.11404.pdf)
[4](https://learnopencv.com/t-sne-for-feature-visualization/)
[5](https://discuss.pytorch.org/t/how-to-use-parameters-from-autoencoder-to-cnn-for-classification/46725)
[6](https://www.appsilon.com/post/r-tsne)
[7](https://docs.pytorch.org/vision/stable/feature_extraction.html)
[8](https://www.ibm.com/think/topics/latent-space)
[9](https://blog.paperspace.com/dimension-reduction-with-t-sne/)
[10](https://github.com/E008001/Autoencoder-in-Pytorch)
[11](https://www.youtube.com/watch?v=D9bdJm1GYFY)
[12](https://www.geeksforgeeks.org/machine-learning/implement-convolutional-autoencoder-in-pytorch-with-cuda/)
[13](https://gaussian37.github.io/ml-concept-t_sne/)
[14](https://stackoverflow.com/questions/75605946/extract-features-by-a-cnn-with-pytorch-not-working)
[15](https://github.com/oreillymedia/t-SNE-tutorial)
[16](https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/)
[17](https://www.geeksforgeeks.org/machine-learning/auto-encoders/)
[18](https://www.youtube.com/watch?v=FslFZx08beM)
[19](https://www.sciencedirect.com/science/article/pii/S0010465525002309)
[20](https://unist.tistory.com/4)
