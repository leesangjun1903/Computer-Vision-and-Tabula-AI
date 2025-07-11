# Score-Based Generative Modeling through Stochastic Differential Equations | Image generation, Density estimation, Image inpainting

확률적 생성 모델에는 성공적인 분야가 두 가지 있는데, 하나는 Score matching with Langevin dynamics (SMLD)이고, 다른 하나는 DDPM (논문리뷰)이다.  
두 클래스 모두 천천히 증가하는 noise로 학습 데이터를 순차적으로 손상시키고 이를 되돌리는 방법을 생성 모델이 학습한다.  
SMLD는 score(ex. 로그 확률 밀도의 기울기)를 각 noise scale에서 학습하고 Langevin dynamics로 샘플링한다.  
DDPM은 학습이 tractable하도록 reverse distribution을 정의하여 각 step을 reverse한다. 연속적인 state space의 경우 DDPM의 목표 함수는 암시적으로 각 noise sclae에서 점수를 계산한다.  
따라서 이 두 모델 클래스를 함께 score 기반 생성 모델이라고 한다.

Score 기반 생성 모델은 이미지나 오디오 등의 다양한 생성 task에 효과적인 것으로 입증되었다.  
저자들은 확률적 미분 방정식(Stochastic Differential Equations, SDE)을 통해 이전 접근 방식을 일반화하는 통합 프레임워크를 제안한다.  
이를 통해 새로운 샘플링 방법이 가능하고 score 기반 생성 모델의 기능이 더욱 확장된다.

특히, 저자들은 유한 개의 noise 분포 대신 diffusion process에 의해 시간이 지남에 따라 진화하는 분포의 연속체(continuum)를 고려하였다.  
이 프로세스는 점진적으로 데이터 포인트를 random noise로 확산시키고 데이터에 의존하지 않고 학습 가능한 parameter가 없는 SDE에 의해 가능하다.  
이 프로세스를 reverse하여 random noise를 샘플 생성을 위한 데이터로 만들 수 있다.  
결정적으로, 이 reverse process는 reverse-time SDE를 충족하며, 이는 시간의 함수로서 주변 확률 밀도의 score가 주어진 forward SDE에서 유도할 수 있다.  
따라서 시간에 의존하는 신경망을 학습시켜 score를 추정한 다음 numerical SDE solver로 샘플을 생성하여 reverse-time SDE를 근사할 수 있다.



# Reference
- https://dlaiml.tistory.com/entry/Score-Based-Generative-Modeling-through-Stochastic-Differential-Equations
- https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/sbgm/
