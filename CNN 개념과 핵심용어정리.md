# 0714_python_CNN
<CNN (Convolutional Neural Network) 개념 및 용어 정리>  
# 1. 핵심 구성 요소  
## A) 컨볼루션 레이어 (Convolution Layer)    
- 필터/커널 (Filter/Kernel): 입력에 적용되는 가중치 행렬 (보통 3×3, 5×5 크기)  
- 스트라이드 (Stride): 필터가 이동하는 간격    
- 패딩 (Padding): 입력 주변에 추가하는 값 (보통 0)  
  . Valid padding: 패딩 없음  
  . Same padding: 출력 크기를 입력과 동일하게 유지  
- 특성 맵 (Feature Map): 컨볼루션 연산의 출력 결과  
## B) 풀링 레이어 (Pooling Layer)  
- 최대 풀링 (Max Pooling): 영역 내 최댓값 선택  
- 평균 풀링 (Average Pooling): 영역 내 평균값 계산  
- 글로벌 풀링 (Global Pooling): 전체 특성 맵에 대한 풀링  
* 활성화 함수 (Activation Functions)  
- ReLU (Rectified Linear Unit): f(x) = max(0, x)  
- Leaky ReLU: 음수 영역에서 작은 기울기 유지  
- Sigmoid: S자 형태의 함수 (0~1 출력)  
- Tanh: 쌍곡탄젠트 함수 (-1~1 출력)  
## C) 정규화 기법  
- 배치 정규화 (Batch Normalization)    
  . 각 배치의 평균과 분산을 정규화  
  . 학습 안정성과 속도 향상  
-  드롭아웃 (Dropout)  
  . 훈련 시 일부 뉴런을 무작위로 비활성화  
  . 과적합 방지  
   
# 2. CNN 아키텍처 유형    
## A) 기본 구조    
- LeNet: 최초의 CNN 구조  
- AlexNet: ImageNet 대회 우승으로 딥러닝 붐 시작  
- VGGNet: 작은 필터(3×3)를 깊게 쌓은 구조  
## B) 고급 구조  
- ResNet: 잔차 연결(Residual Connection)로 깊은 네트워크 학습  
- DenseNet: 각 레이어가 이전 모든 레이어와 연결  
- Inception: 다양한 크기의 필터를 병렬로 사용  
- MobileNet: 모바일 최적화를 위한 경량 구조  
## C) 특수 기법  
- 전이 학습 (Transfer Learning)  
  . 사전 훈련된 모델을 새로운 작업에 적용  
  . 적은 데이터로도 효과적 학습 가능  
- 데이터 증강 (Data Augmentation)  
  . 회전, 크기 조정, 밝기 변경 등으로 데이터 확장  
  . 과적합 방지 및 일반화 성능 향상  
- 어텐션 메커니즘 (Attention Mechanism)  
  . 중요한 영역에 집중하는 기법  
  . CBAM (Convolutional Block Attention Module) 등  
## D) 성능 지표    
- 정확도 (Accuracy): 전체 중 올바르게 분류한 비율  
- 정밀도 (Precision): 양성 예측 중 실제 양성 비율  
- 재현율 (Recall): 실제 양성 중 올바르게 예측한 비율  
- F1 Score: 정밀도와 재현율의 조화평균  
## E) 최적화 기법    
- 옵티마이저 (Optimizer)  
  . SGD (Stochastic Gradient Descent): 확률적 경사하강법  
  . Adam: 적응적 학습률 조정  
  . RMSprop: 경사의 제곱 이동평균 사용  
- 학습률 스케줄링  
  . 학습률 감소 (Learning Rate Decay)  
  . 주기적 학습률 (Cyclical Learning Rate)  
- 손실 함수 (Loss Functions)  
  . 교차 엔트로피 (Cross-Entropy): 분류 문제용  
  . 평균 제곱 오차 (MSE): 회귀 문제용  
  . 초점 손실 (Focal Loss): 불균형 데이터셋용  
  
# 3.실용적 고려사항    
## A) 하이퍼파라미터    
- 학습률, 배치 크기, 에폭 수  
- 필터 개수, 네트워크 깊이  
## B) 하드웨어 최적화  
- GPU 가속: CUDA, cuDNN 활용  
- 혼합 정밀도 (Mixed Precision): 메모리 효율성  
- 모델 압축: 양자화, 프루닝
