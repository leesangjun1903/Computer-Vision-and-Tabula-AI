# CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders | Image generation

**핵심 주장 및 주요 기여**  
CLIPDraw는 사전 학습된 CLIP 언어·이미지 인코더를 평가 시점의 최적화(objective-based optimization) 지표로 활용하여, 별도의 학습 없이 자연어 프롬프트에 부합하는 벡터 기반 드로잉을 생성하는 방법이다.  
1. **학습 불필요성**: 제안 기법은 모델 학습 단계 없이, 사전 학습된 CLIP를 유사도 측정기로만 사용해 그림을 합성한다.  
2. **벡터 스트로크 최적화**: 픽셀 대신 RGBA 베지어 곡선(strokes)을 최적화함으로써, 인간이 인지하기 쉬운 단순한 형태와 선화(line art)를 유도한다.  
3. **다양한 스타일·복합성 제어**: 프롬프트의 형용사 변경(“watercolor”, “3D rendering”)이나 스트로크 수 조정(16→256) 만으로 스타일과 세부 묘사의 정도를 조절할 수 있다.  
4. **부정 프롬프트(Negative Prompt)**: 추가적인 부정적 텍스트를 손실에 반영하여, 원치 않는 요소(예: “purple”, “many sailboats”)를 억제할 수 있음을 보였다.  

## 1. 문제 정의  
- **기존 문제**: 텍스트-투-이미지 생성 분야는 보통 GAN 또는 대규모 autoregressive 모델(DALL·E 등)을 학습시켜야 하며, 학습 비용과 데이터 요구량이 크다.  
- **해결 목표**: 학습 없이, 단순 벡터 그래픽스 제약을 통해 자연어 설명과 일치하는 “드로잉”을 빠르게 합성하는 방법 제안.  

## 2. 제안 방법  
### 2.1. 최적화 목표  
프롬프트 $$\mathrm{desc} $$ 를 CLIP 텍스트 인코더에 통과시켜 얻은 임베딩 $$\mathbf{e}\_\mathrm{txt} $$ 와, 생성된 그림의 CLIP 이미지 임베딩 $$\mathbf{e}_\mathrm{img} $$ 사이의 코사인 유사도를 최대화한다.  

$$
\mathcal{L}\_{\text{match}} = - \cos\bigl(\mathbf{e}\_\mathrm{txt},\mathbf{e}_\mathrm{img}\bigr).
$$  

부정 프롬프트 $$\{\mathrm{neg}\_k\} $$ 가 주어지면, 각 부정 임베딩 $$\mathbf{e}_{\mathrm{neg}_k} $$ 과의 코사인 거리를 음의 가중치 $$\lambda_k$$ 로 더해 다음과 같이 확장한다:  

```math
\mathcal{L} = \mathcal{L}\_{\text{match}} + \sum_k \lambda_k\,\cos\bigl(\mathbf{e}\_{\mathrm{neg}_k},\mathbf{e}_\mathrm{img}\bigr).
``` 

### 2.2. 벡터 드로잉 표현  
- **곡선 파라미터**: $$N$$개의 베지어 곡선, 각각 3–5개 제어점, 선 두께, RGBA 색상  
- **렌더링**: Differentiable Renderer(Li et al., 2020)로 벡터→픽셀 변환  
- **이미지 증강**: 랜덤 원근 변환·크롭 앤 리사이즈를 $$D$$ 배 적용하여, 다양한 뷰에서도 프롬프트와 매칭하도록 강제  

### 2.3. 최적화 절차 (알고리즘 1)  
1. $$\mathbf{P} \leftarrow$$ CLIP 텍스트 인코딩(프롬프트)  
2. 곡선 파라미터 $$\Theta_0 \sim \mathcal{U}$$ 로 초기화  
3. for $$i=1$$ to $$I$$ do  
   a. $$I_i \leftarrow$$ DifferentiableRender $$(\Theta_{i-1})$$  
   b. $$\{I_i^j\}\_{j=1}^D \leftarrow$$ RandomAugment $$(I_i)$$  
   c. $$\mathbf{E}\_i \leftarrow\frac1D\sum_j\mathrm{CLIP_{img}}(I_i^j)$$  
   d. $$\mathcal{L}\_i\leftarrow -\cos(\mathbf{P},\mathbf{E}\_i)+(\text{negatives})$$  
   e. $$\Theta_i\leftarrow \Theta_{i-1}-\eta\nabla_{\Theta}\mathcal{L}_i$$  

## 3. 모델 구조 및 구현  
- **Backbone**: 사전 학습된 CLIP(ViT- or ResNet 기반)  
- **Differentiable Renderer**: Li et al.(2020)의 벡터 래스터화  
- **Hyperparameters**:  
  -  Iterations $$I$$=250  
  -  곡선 수 $$N$$=16–256 (실험별)  
  -  증강 배수 $$D$$=8  

## 4. 성능 향상 및 일반화 가능성  
1. **다중 증강(Augmentations)**: 랜덤 원근·크롭이 과적합(adversarial artifact) 생성 억제, 인간 인지 가능성 향상  
2. **벡터 제약**: 픽셀 최적화 대비 모호한 텍스처 대신 구조적 선화 유지  
3. **스타일 및 복잡도 제어**: 프롬프트 형용사·스트로크 수 조정만으로 간편하게 조절 가능 → 다양한 도메인(UI 아이콘↔콘셉트 아트)에서 일반화  
4. **부정 프롬프트**: 모델 행동을 세밀하게 조율하여, 원치 않는 편향 억제 가능 (다만 일관된 효과는 여전히 도전 과제)  

## 5. 한계  
- **해상도 및 세부 묘사**: 고해상도·정교한 포토리얼리즘에는 한계  
- **세부 위치 제어**: “배 위치를 오른쪽으로” 같은 정확한 배치 제어 어려움  
- **부정 프롬프트 불안정성**: 일관된 억제 효과 확보가 쉽지 않음  
- **CLIP의 사회적 편향**: 사전 학습 데이터에 기인한 편향이 결과에 반영될 수 있음  

## 6. 향후 연구 및 고려 사항  
- **고해상도 확장**: GAN 판별기나 고해상도 벡터 렌더러와 결합  
- **미세 제어 메커니즘**: 위치·구성 요소별 손실(term) 도입  
- **편향 완화**: CLIP 편향 분석 결과를 반영한 공정성(fairness) 제약 추가  
- **학습 없는 벡터 생성 일반화**: 드로잉 외 도식도·차트·3D 스케치로 확대  

CLIPDraw는 학습 없이도 자연어 기반 드로잉 합성을 가능케 하는 간결하고 확장성 있는 프레임워크로, 벡터 제약을 통한 인간 친화적 결과물과 프롬프트 기반 제어성을 조합한 점이 향후 언어-비주얼 융합 연구 및 AI 예술 도구 개발에 중요한 참고 사례가 될 것이다.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/5457fee0-cd6b-4792-a530-be0c3333683e/2106.14843v1.pdf
