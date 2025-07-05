# VAE : Auto-Encoding Variational Bayes | Image generation
### Autoencoder 와 Variational Autoencoder의 직관적인 이해 
https://medium.com/@hugmanskj/autoencoder-%EC%99%80-variational-autoencoder%EC%9D%98-%EC%A7%81%EA%B4%80%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-171b3968f20b
### VAE(Varitional Auto-Encoder)를 알아보자
https://velog.io/@hong_journey/VAEVaritional-Auto-Encoder%EB%A5%BC-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90
### 
https://wikidocs.net/152474

### [논문] VAE(Auto-Encoding Variational Bayes) 직관적 이해
https://taeu.github.io/paper/deeplearning-paper-vae/

# Auto-Encoding Variational Bayes (AEVB, VAE) 

## 1. 개요  
Auto-Encoding Variational Bayes(AEVB)는 2013년 Diederik P. Kingma와 Max Welling이 제안한 알고리즘으로, 연속 잠재 변수(latent variable)를 갖는 확률 모델에서 효율적 추론과 학습을 가능하게 한다[1]. 주요 기여는 두 가지이다. 첫째, 변분 하한(ELBO)의 재파라미터화(reparameterization)를 통해 확률적 경사하강법으로 직접 최적화할 수 있게 한 점이다[1]. 둘째, 인식 모델(recognition model)을 학습시켜 각 데이터 포인트의 복잡한 사후분포(posterior)를 효율적으로 근사하도록 한 점이다[1].

---

## 2. 배경  
### 2.1 변분 추론(Variational Inference)  
- 실제 사후분포 $$p(z|x)$$는 계산이 어렵기 때문에, 근사분포 $$q_\phi(z|x)$$를 도입하여 $$\mathrm{KL}(q\|p)$$를 최소화한다[1].  
- 이때 최적화 대상은 Evidence Lower Bound(ELBO)로,

$$
\log p(x) \ge \mathbb{E}\_{q_\phi}[\log p(x,z) - \log q_\phi(z|x)]
$$
  
[1].

### 2.2 전통적 문제  
- ELBO의 기댓값 항은 종종 적분 불가능하거나 샘플링 기반 추정이 느리다는 문제가 있다[2].  
- Mean-field 방식(factoring)은 분포 단순화를 가정하지만 복잡한 분포에 적용하기 어렵다[2].  
- 단순 몬테카를로 샘플링은 분산이 커서 효율이 떨어진다[2].

---

## 3. 재파라미터화 트릭(Reparameterization Trick)  
AEVB는 샘플링 연산을 미분 가능한 함수로 변환하여 경사하강법에 적용한다[1].  
1. 잠재 변수 $$z$$를 인식 네트워크 $$q_\phi(z|x)$$ 대신,
   
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$
   
   와 같이 표준정규분포 $$\epsilon$$와 조합해 생성한다[1].  
3. 이렇게 하면 ELBO 내 기댓값을 샘플로 근사해도 $$\phi$$에 대한 미분이 가능해진다[1].

---

## 4. AEVB 학습 절차  
1. **인코더(Encoder)**: 입력 $$x$$를 받아 잠재 변수 분포 파라미터 $$\mu_\phi(x)$$, $$\sigma_\phi(x)$$를 출력한다[1].  
2. **샘플링**: 재파라미터화 트릭을 통해 $$z$$를 얻는다[1].  
3. **디코더(Decoder)**: $$z$$로부터 재구성 확률 $$p_\theta(x|z)$$를 계산한다[1].  
4. **ELBO 최적화**:

$$\mathcal{L}(\theta,\phi;x) = -\mathrm{KL}\bigl(q_\phi(z|x)\,\|\,p(z)\bigr) + \mathbb{E}\_{q_\phi}[\log p_\theta(x|z)] $$  

   를 미니배치 확률적 경사하강법으로 최적화한다[1].

---

## 5. 주요 이점 및 결과  
- **확장성**: 대규모 데이터셋에서도 효율적으로 학습 가능하다[1].  
- **범용성**: 다양한 잠재 변수 모델에 적용할 수 있다[3].  
- **생성 성능**: MNIST 등 이미지 생성 실험에서 뛰어난 성능을 보였다[4].  
- **이론적 단순함**: 기존 변분 기법보다 구현이 간단하다[3].

---

## 6. 결론  
AEVB는 복잡한 사후분포를 인식 네트워크로 근사하고, 재파라미터화 트릭으로 경사 계산 문제를 해결하여 **효율적이며 직관적인 생성 모델 학습** 방법을 제시한다[1]. 이 기법은 Variational Autoencoder(VAE)의 근간이 되었으며, 이후 많은 분야에서 활용되고 있다.

---

## 참고 문헌  
[1] D. P. Kingma & M. Welling, "Auto-Encoding Variational Bayes," *ICLR*, 2014.  
[3] J. Doersch, "Tutorial on Variational Autoencoders," arXiv:1606.05908, 2016.  
[4] D. Stutz, "Auto-Encoding Variational Bayes," 2013.  
[2] M. Hasegawa-Johnson, "Lecture on Variational Autoencoder," Univ. Illinois, 2020.

[1] https://www.semanticscholar.org/paper/5f5dc5b9a2ba710937e2c413b37b053cd673df02
[2] https://courses.grainger.illinois.edu/ece417/fa2020/slides/lec22.pdf
[3] https://arxiv.org/abs/2208.07818
[4] https://davidstutz.de/auto-encoding-variational-bayes-kingma-and-welling/
[5] https://ceasjournal.com/index.php/CEAS/article/view/33
[6] https://www.semanticscholar.org/paper/af5447b2908681ee9cff2b3c66ea2c1be8a13882
[7] https://www.aclweb.org/anthology/2020.coling-main.458
[8] https://www.semanticscholar.org/paper/25a90a913a5e7124da3da3f650442af79b64b5a6
[9] https://ieeexplore.ieee.org/document/10252337/
[10] https://www.semanticscholar.org/paper/54906484f42e871f7c47bbfe784a358b1448231f
[11] https://arxiv.org/abs/1312.6114
[12] https://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html
[13] http://arxiv.org/pdf/1312.6114.pdf
[14] https://openreview.net/forum?id=33X9fd2-9FyZd
[15] https://dataconomy.com/2025/05/07/what-is-a-variational-autoencoder-vae/
[16] https://www.ibm.com/think/topics/variational-autoencoder
[17] https://www.semanticscholar.org/paper/3d5643bac10972c75656c5426a96abc52278d93c
[18] https://www.semanticscholar.org/paper/ef4f5a50837a7c1b3e87b9300ffc7ba00d461a0f
[19] https://www.semanticscholar.org/paper/Auto-Encoding-Variational-Bayes-Kingma-Welling/5f5dc5b9a2ba710937e2c413b37b053cd673df02
[20] https://en.wikipedia.org/wiki/Variational_autoencoder
