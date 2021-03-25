---
layout: post
title: "tensorflow 시작하기"
---

# TensorFlow 시작하기

## 노드 만들기


```python
import tensorflow as tf
# 그래프 노드 만들고 출력하기.
node1= tf.constant(3.0, dtype=tf.float32)
node2= tf.constant(4.0)
print(node1,node2)


#2개의 노드값을 더한 노드 생성하기
node3 = tf.add(node1,node2)
print(node3)
```

    tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(4.0, shape=(), dtype=float32)
    tf.Tensor(7.0, shape=(), dtype=float32)
    

- tf.constant()란?
```
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const')
```
tf.constant는 상수 텐서를 선언한다. 각각의 인자는 아래와 같다.<br>
- value : 상수값이며 직접 지정하거나 shape 형태로 채울 값을 지정할 수 있습니다.
- dtype : 데이터 타입(e.g.tf.float32, tf.int32, tf.bool)
- shape : 상수 데이터의 형태
- name: 텐서의 이름(optional)
tf constant() 예시.


```python
tensor1 = tf.constant([1,2,3,4,5,6,7])
print(tensor1)
tensor2 = tf.constant(-1.0,shape=[2,3])
print(tensor2)
```

    tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
    tf.Tensor(
    [[-1. -1. -1.]
     [-1. -1. -1.]], shape=(2, 3), dtype=float32)
    

## 선형 회귀(Linear Regression) 및 경사하강법(Gradient Descent) 알고리즘

- 회귀(Regression)란?
    회귀란 어떤 실수값(예를 들면, 1.7, 10.5, 12.3)을 예측하는 문제를 뜻한다.
- 선형 회귀(Linear Regression)이란?
    선형 회귀란 선형함수를 이용해서 회귀를 수행하는 기법이다.
- 모든 머신러닝 모델은 다음 3가지의 과정을 거친다.
1. 학습하고자하는 가정(Hypothesis)(H(x))를 수학적 표현식으로 나타낸다.
2. 가설의 성늘을 측정할 수 있는 손실 함수(Loss/Cost Function)를 정의한다.
3. 손실 함수를 최소화할 수 있는 학습 알고리즘을 설계한다..


```python
import tensorflow as tf
import numpy as np
#X and Y data
x_train = [1,2,3]
y_train = [1,2,3]

W=tf.Variable(tf.random.normal([1]), name='weight')
b=tf.Variable(tf.random.normal([1]), name='bias')
# hypothesis = Wx+b
hypothesis= x_train * W + b

#cost/Loss function 
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

# GradientDescent를 이용한 cost를 Minimize하기
""" tensorflow 1.x에서 사용하던 코드라서 2.x에서는 실행 되지않는다.
optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

for step in range(2001):
    train
    if step % 20 == 0:
        print(step,cost,W,b)
"""
#2.x에 맞게 케라스 사용.
sgd=tf.keras.optimizers.SGD(learning_rate=0.01)
# 실행해보기
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1))
#훈련 환경 설정
model.compile(loss='mean_squared_error',optimizer=sgd)
# 훈련 시작
model.fit(x_train,y_train,epochs=1000)
print(model.predict(np.array([5])))
print(model.evaluate(x_train,y_train))
```

    Epoch 1/1000
    1/1 [==============================] - 0s 0s/step - loss: 1.9519
    Epoch 2/1000
    1/1 [==============================] - 0s 0s/step - loss: 1.5446
    Epoch 3/1000
    1/1 [==============================] - 0s 0s/step - loss: 1.2226
    Epoch 4/1000
    1/1 [==============================] - 0s 0s/step - loss: 0.9680
    .
    .
    .
    Epoch 999/1000
    1/1 [==============================] - 0s 0s/step - loss: 6.6813e-05
    Epoch 1000/1000
    1/1 [==============================] - 0s 0s/step - loss: 6.6493e-05
    [[4.974238]]
    1/1 [==============================] - 0s 0s/step - loss: 6.6173e-05
    6.617304461542517e-05
    

## 위에서 사용한  keras 함수 정리
- keras.optimizers.SGD(learning late): 확률적 경사 하강법(Stochastic Gradient Descent)
- tf.keras.models.Sequential(): layer를 차례대로 쌓을 수 있게 한다.
- model.add(tf.keras.layers.Dense()): layer를 추가한다 keras.layers.Dense를 사용하여 레이어의 특성을 정의한다.
- tf.keras.layers.Dense(1, input_dim=1): 출력 뉴런의 수가 1, 입력 뉴런의 수(입력의 차원)=1을 의미 세번째 인자에 activation 활성화 함수를 의미한다.
- model.compile(): 훈련 과정을 설정한다. optimizer 옵티마이저와 loss 손실 함수 metrics 훈련을 모니터링하기 위한 지표를 선택할 수 있다.
- model.fit(x_train,y_train,epochs=1000): 1000번의 에포크동안 훈련한다.
- model.predict(np.array([5])): 5를 입력했을때의 예측값을 출력해준다.
- model.evaluate(X_test, Y_test): X_test(테스트 데이터),Y_test(레이블 테스트 데이터)를 이용하여 정확도를 평가합니다.
