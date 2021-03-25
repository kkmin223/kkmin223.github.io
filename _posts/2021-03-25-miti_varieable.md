---
layer: post
title: "Multi-variable linear regression 구현하기"
---

# Multi-variable linear regression 구현해보기



```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
# x and y data
X=[[73., 80., 75.], [93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70]]
Y=[152.,185.,180.,196.,142.]


W=tf.Variable(tf.random.normal([3,1]),name= "weight")
b=tf.Variable(tf.random.normal([1]), name="bias")

# hypothesis 정의하기 행렬곱으로 여러개의 데이터를 입력.
hypothesis = tf.matmul(X,W)+b
# GradientDescent를 이용한 cost Minimize
sgd=tf.keras.optimizers.SGD(learning_rate=1e-5)

model = tf.keras.models.Sequential()
# 출력이 1 입력이 3 
model.add(tf.keras.layers.Dense(1, input_dim=3))
#훈련 환경 loss funtcion & optimizer 설정.
model.compile(loss="mean_squared_error",optimizer=sgd)
model.fit(X,Y,epochs=1000)
# 학습결과 보기 
model.evaluate(X,Y)

```

    Epoch 1/1000
    1/1 [==============================] - 0s 0s/step - loss: 92021.1875
    Epoch 2/1000
    1/1 [==============================] - 0s 0s/step - loss: 28849.1191
    Epoch 3/1000
    1/1 [==============================] - 0s 0s/step - loss: 9048.0332
    .
    .
    .
    Epoch 999/1000
    1/1 [==============================] - 0s 0s/step - loss: 4.7175
    Epoch 1000/1000
    1/1 [==============================] - 0s 0s/step - loss: 4.7152
    1/1 [==============================] - 0s 0s/step - loss: 4.7128
    
    1.0682244300842285

